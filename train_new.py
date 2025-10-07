import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer
from PIL import Image
import numpy as np
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode

from config import Config
from dataset import PrivacyProtectionDataset, collate_fn
from generator import NoiseGenerator
from utils import calculate_psnr, calculate_linf_norm


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


# ===== NEW: InternVL 专用输入构建（含 image_flags） =====
IMG_START_TOKEN  = "<img>"
IMG_END_TOKEN    = "</img>"
IMG_CONTEXT_TOKEN = "<IMG_CONTEXT>"

def build_internvl_inputs(question, answer, tokenizer, model, max_len=1024, num_patches=1):
    """
    返回: input_ids, attention_mask, labels, image_flags  (均在 CPU 上, 调用方再 .to(device))
    逻辑：
      1) 将 <image> 替换为 <img> + <IMG_CONTEXT>* (num_image_token * num_patches) + </img>
      2) labels 仅监督答案段；问题段 + 所有视觉占位符置 -100
      3) image_flags 为 [B, T, 1]，标记 <IMG_CONTEXT> 位置为 1
    """
    # 取得 img_context_token_id，并写回模型（InternVL forward 会用到）
    img_ctx_id = tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
    if not hasattr(model, "img_context_token_id") or model.img_context_token_id is None:
        model.img_context_token_id = img_ctx_id

    # 需要的视觉 token 个数（例如 256）
    num_img_tokens = int(getattr(model, "num_image_token", 256)) * int(num_patches)

    # 1) 准备用户问题与答案文本
    user_text = f"<image>\n{question}".strip()
    assistant_text = f" {answer}"

    # 2) 展开 <image> 为视觉 token 块
    image_block = IMG_START_TOKEN + (IMG_CONTEXT_TOKEN * num_img_tokens) + IMG_END_TOKEN
    user_with_visual = user_text.replace("<image>", image_block, 1)

    # 3) 拼接 “问题 + 答案”
    full_text = user_with_visual + assistant_text

    # 4) 分别 tokenize：用于确定问题长度（mask 边界）
    q_enc = tokenizer(
        user_with_visual, return_tensors='pt', padding=False, truncation=True, max_length=max_len
    )
    q_len = q_enc["input_ids"].shape[1]

    enc = tokenizer(
        full_text, return_tensors='pt', padding=False, truncation=True, max_length=max_len
    )
    input_ids = enc["input_ids"]      # [1, T]
    attention_mask = enc["attention_mask"]

    # 5) labels：默认拷贝，再把问题段 & 视觉 token 段置 -100
    labels = input_ids.clone()
    labels[:, :q_len] = -100  # mask 掉 <img> 块 + 问题 tokens

    # 视觉特殊 token 也需 mask
    img_start_id = tokenizer.convert_tokens_to_ids(IMG_START_TOKEN)
    img_end_id   = tokenizer.convert_tokens_to_ids(IMG_END_TOKEN)
    special_mask = (input_ids == img_ctx_id) | (input_ids == img_start_id) | (input_ids == img_end_id)
    labels[special_mask] = -100

    # 6) image_flags: [B, T, 1]，在 <IMG_CONTEXT> 处为 1
    image_flags = torch.ones(1, num_img_tokens, dtype=torch.long)


    return input_ids, attention_mask, labels, image_flags


def compute_normal_task_loss(images, qa_pairs, tokenizer, model, device):
    total_loss, num_samples = 0.0, 0
    batch_size = images.shape[0]
    model_dtype = next(model.parameters()).dtype  # 与模型对齐 (fp16)

    for i in range(batch_size):
        qa_list = qa_pairs[i]
        if not qa_list:
            continue
        qa = qa_list[np.random.randint(len(qa_list))]
        question, answer = qa['question'], qa['answer']

        # 预处理图像
        image = images[i:i+1]
        image_resized = T.Resize((448, 448), interpolation=InterpolationMode.BICUBIC)(image)
        pixel_values = T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)(image_resized)
        pixel_values = pixel_values.to(dtype=model_dtype, device=device)

        # ！！！关键：用 InternVL 专用构建器 + 传 image_flags
        input_ids, attention_mask, labels, image_flags = build_internvl_inputs(
            question, answer, tokenizer, model, max_len=1024, num_patches=1
        )
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        labels = labels.to(device)
        image_flags = image_flags.to(device)

        try:
            model.train()  # 需要 loss
            outputs = model(
                pixel_values=pixel_values,
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                image_flags=image_flags,     # ！！！必须
                output_hidden_states=False,
                output_attentions=False,
                return_dict=True
            )
            model.eval()

            loss = outputs.loss
            if loss is None or torch.isnan(loss):
                # 手动 fallback（几乎不会再触发）
                logits = outputs.logits
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                active = shift_labels.view(-1) != -100
                if active.any():
                    loss_fct = nn.CrossEntropyLoss()
                    loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1))[active],
                                    shift_labels.view(-1)[active])
                else:
                    loss = torch.tensor(0.0, device=device, requires_grad=True)

            total_loss += loss
            num_samples += 1

        except Exception as e:
            print(f"警告: 正常任务损失计算出错: {e}")
            continue

    if num_samples == 0:
        return torch.tensor(0.01, device=device, requires_grad=True)
    return total_loss / num_samples



def compute_privacy_task_loss(images, qa_pairs, tokenizer, model, device):
    total_loss, num_samples = 0.0, 0
    batch_size = images.shape[0]
    model_dtype = next(model.parameters()).dtype

    for i in range(batch_size):
        qa_list = qa_pairs[i]
        if not qa_list:
            continue
        qa = qa_list[np.random.randint(len(qa_list))]
        question, answer = qa['question'], qa['answer']

        image = images[i:i+1]
        image_resized = T.Resize((448, 448), interpolation=InterpolationMode.BICUBIC)(image)
        pixel_values = T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)(image_resized)
        pixel_values = pixel_values.to(dtype=model_dtype, device=device)

        input_ids, attention_mask, labels, image_flags = build_internvl_inputs(
            question, answer, tokenizer, model, max_len=1024, num_patches=1
        )
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        labels = labels.to(device)
        image_flags = image_flags.to(device)

        try:
            model.train()
            outputs = model(
                pixel_values=pixel_values,
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                image_flags=image_flags,     # ！！！必须
                output_hidden_states=False,
                output_attentions=False,
                return_dict=True
            )
            model.eval()

            loss = outputs.loss
            if loss is None or torch.isnan(loss):
                logits = outputs.logits
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                active = shift_labels.view(-1) != -100
                if active.any():
                    loss_fct = nn.CrossEntropyLoss()
                    loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1))[active],
                                    shift_labels.view(-1)[active])
                else:
                    loss = torch.tensor(0.0, device=device, requires_grad=True)

            total_loss -= loss  # 关键：最大化隐私损失
            num_samples += 1

        except Exception as e:
            print(f"警告: 隐私任务损失计算出错: {e}")
            continue

    if num_samples == 0:
        return torch.tensor(0.0, device=device, requires_grad=True)
    return total_loss / num_samples



class AdversarialTrainer:
    
    def __init__(self, config):
        self.config = config
        self.device = torch.device(config.device if torch.cuda.is_available() else "cpu")
        
        print(f"使用设备: {self.device}")
        
        # 加载 surrogate MLLM (InternVL2-1B)
        print(f"Loading surrogate MLLM: {config.surrogate_model_name}")
        self.model = AutoModel.from_pretrained(
            config.surrogate_model_name,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            trust_remote_code=True
        ).to(self.device)
        
        # 冻结模型参数，只训练生成器
        for param in self.model.parameters():
            param.requires_grad = False
        self.model.eval()
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            config.surrogate_model_name,
            trust_remote_code=True
        )
        
        print("初始化噪声生成器")
        self.generator = NoiseGenerator(
            in_channels=3,
            out_channels=3,
            epsilon=config.epsilon
        ).to(self.device)
        
        # 优化器（只优化生成器）
        self.optimizer = optim.Adam(
            self.generator.parameters(),
            lr=config.learning_rate
        )
        

        os.makedirs(config.checkpoint_dir, exist_ok=True)
        os.makedirs(config.log_dir, exist_ok=True)
        

        self.history = {
            'epoch': [],
            'loss_total': [],
            'loss_normal': [],
            'loss_privacy': [],
            'psnr': [],
            'linf_norm': []
        }
    
    def train_step(self, batch):
        images = batch['images'].to(self.device)
        privacy_qa_list = batch['privacy_qa_list']
        normal_qa_list = batch['normal_qa_list']
        
        # 生成对抗噪声
        delta = self.generator(images)
        x_adv = torch.clamp(images + delta, 0.0, 1.0)
        
        # ============ 损失1: 正常任务损失 (最小化) ============
        loss_normal = compute_normal_task_loss(
            x_adv,
            normal_qa_list,
            self.tokenizer,
            self.model,
            self.device
        )
        
        # ============ 损失2: 隐私任务损失 (反向优化) ============
        # 注意：compute_privacy_task_loss 内部已经取负
        loss_privacy = compute_privacy_task_loss(
            x_adv,
            privacy_qa_list,
            self.tokenizer,
            self.model,
            self.device
        )
        
        # ============ 总损失 ============
        # 正常任务：最小化损失（让模型回答正确）
        # 隐私任务：已经取负，优化时会最大化原始损失（让模型回答错误）
        loss_total = self.config.alpha * loss_normal + self.config.beta * loss_privacy
        
        # 反向传播
        self.optimizer.zero_grad()
        loss_total.backward()
        self.optimizer.step()
        
        # 计算指标
        psnr = calculate_psnr(images, x_adv)
        linf_norm = calculate_linf_norm(delta)
        
        return {
            'loss_total': loss_total.item(),
            'loss_normal': loss_normal.item(),
            'loss_privacy': loss_privacy.item(),
            'psnr': psnr,
            'linf_norm': linf_norm
        }
    
    def train_epoch(self, dataloader, epoch):
        """训练一个 epoch"""
        self.generator.train()
        
        epoch_metrics = {
            'loss_total': 0.0,
            'loss_normal': 0.0,
            'loss_privacy': 0.0,
            'psnr': 0.0,
            'linf_norm': 0.0
        }
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch}/{self.config.num_epochs}")
        
        for batch_idx, batch in enumerate(pbar):
            metrics = self.train_step(batch)
            
            # 累加指标
            for key in epoch_metrics.keys():
                epoch_metrics[key] += metrics[key]
            
            # 更新进度条
            pbar.set_postfix({
                'loss': f"{metrics['loss_total']:.4f}",
                'L_normal': f"{metrics['loss_normal']:.4f}",
                'L_privacy': f"{metrics['loss_privacy']:.4f}",
                'PSNR': f"{metrics['psnr']:.2f}",
                'L_inf': f"{metrics['linf_norm']:.6f}"
            })
        
        # 计算平均指标
        num_batches = len(dataloader)
        for key in epoch_metrics.keys():
            epoch_metrics[key] /= num_batches
        
        return epoch_metrics
    
    def save_checkpoint(self, epoch):
        """保存检查点"""
        checkpoint_path = os.path.join(
            self.config.checkpoint_dir,
            f"generator_epoch_{epoch}.pth"
        )
        torch.save({
            'epoch': epoch,
            'generator_state_dict': self.generator.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config.__dict__
        }, checkpoint_path)
        print(f"保存检查点: {checkpoint_path}")
    
    def save_history(self):
        """保存训练历史"""
        history_path = os.path.join(self.config.log_dir, "train_history.json")
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=2)
        print(f"保存训练历史: {history_path}")
    
    def train(self, dataloader):
        """完整训练流程"""
        print("\n" + "="*50)
        print("开始训练（改进版 - 使用完整 VLM forward）")
        print("="*50 + "\n")
        
        for epoch in range(1, self.config.num_epochs + 1):
            # 训练一个 epoch
            metrics = self.train_epoch(dataloader, epoch)
            
            # 记录历史
            self.history['epoch'].append(epoch)
            self.history['loss_total'].append(metrics['loss_total'])
            self.history['loss_normal'].append(metrics['loss_normal'])
            self.history['loss_privacy'].append(metrics['loss_privacy'])
            self.history['psnr'].append(metrics['psnr'])
            self.history['linf_norm'].append(metrics['linf_norm'])
            
            # 打印统计信息
            print(f"\nEpoch {epoch} 统计:")
            print(f"  总损失: {metrics['loss_total']:.4f}")
            print(f"  正常任务损失: {metrics['loss_normal']:.4f}")
            print(f"  隐私任务损失: {metrics['loss_privacy']:.4f}")
            print(f"  PSNR: {metrics['psnr']:.2f} dB")
            print(f"  L-inf 范数: {metrics['linf_norm']:.6f}")
            
            # 保存检查点
            if epoch % self.config.save_interval == 0:
                self.save_checkpoint(epoch)
            
            # 保存训练历史
            self.save_history()
        
        print("\n" + "="*50)
        print("训练完成")
        print("="*50 + "\n")
        
        # 保存最终模型
        self.save_checkpoint(self.config.num_epochs)


def main():
    """主函数"""
    # 加载配置
    config = Config()
    
    print("配置信息:")
    print(f"  数据根目录: {config.data_root}")
    print(f"  Surrogate 模型: {config.surrogate_model_name}")
    print(f"  Batch size: {config.batch_size}")
    print(f"  Epochs: {config.num_epochs}")
    print(f"  Learning rate: {config.learning_rate}")
    print(f"  Epsilon: {config.epsilon}")
    print(f"  Alpha (正常任务权重): {config.alpha}")
    print(f"  Beta (隐私任务权重): {config.beta}")
    
    # 加载数据集
    print("\n加载数据集...")
    dataset = PrivacyProtectionDataset(
        data_root=config.data_root,
        image_size=config.image_size,
        app_filter=config.test_single_app
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0
    )
    
    # 创建训练器
    trainer = AdversarialTrainer(config)
    
    # 开始训练
    trainer.train(dataloader)


if __name__ == "__main__":
    main()

