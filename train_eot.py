import os
import json
import math
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer
import torchvision.transforms.functional as F
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode

from config import Config
from dataset import PrivacyProtectionDataset, collate_fn
from generator import NoiseGenerator
from utils import calculate_psnr, calculate_linf_norm
from attention import SaliencyAttention
from set_seed_example import set_seed


# ===== InternVL 专用输入构建（含 image_flags） =====
IMG_START_TOKEN  = "<img>"
IMG_END_TOKEN    = "</img>"
IMG_CONTEXT_TOKEN = "<IMG_CONTEXT>"


def build_internvl_inputs(question, answer, tokenizer, model, max_len=1024, num_patches=1):
    """
    返回: input_ids, attention_mask, labels, image_flags
    逻辑：
      1) 将 <image> 替换为 <img> + <IMG_CONTEXT>* (num_image_token * num_patches) + </img>
      2) labels 仅监督答案段；问题段 + 所有视觉占位符置 -100
      3) image_flags: [B, num_img_tokens]，标记 <IMG_CONTEXT> 位置为 1
    """
    img_ctx_id = tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
    if not hasattr(model, "img_context_token_id") or model.img_context_token_id is None:
        model.img_context_token_id = img_ctx_id

    num_img_tokens = int(getattr(model, "num_image_token", 256)) * int(num_patches)

    user_text = f"<image>\n{question}".strip()
    assistant_text = f" {answer}"

    image_block = IMG_START_TOKEN + (IMG_CONTEXT_TOKEN * num_img_tokens) + IMG_END_TOKEN
    user_with_visual = user_text.replace("<image>", image_block, 1)

    full_text = user_with_visual + assistant_text

    q_enc = tokenizer(
        user_with_visual, return_tensors='pt', padding=False, truncation=True, max_length=max_len
    )
    q_len = q_enc["input_ids"].shape[1]

    enc = tokenizer(
        full_text, return_tensors='pt', padding=False, truncation=True, max_length=max_len
    )
    input_ids = enc["input_ids"]
    attention_mask = enc["attention_mask"]

    labels = input_ids.clone()
    labels[:, :q_len] = -100

    img_start_id = tokenizer.convert_tokens_to_ids(IMG_START_TOKEN)
    img_end_id   = tokenizer.convert_tokens_to_ids(IMG_END_TOKEN)
    special_mask = (input_ids == img_ctx_id) | (input_ids == img_start_id) | (input_ids == img_end_id)
    labels[special_mask] = -100

    image_flags = torch.ones(1, num_img_tokens, dtype=torch.long)

    return input_ids, attention_mask, labels, image_flags


def _vlm_forward_loss(images, qa_pairs, tokenizer, model, device, print_debug=False, debug_prefix=""):
    """
    与 train_new 中 compute_normal_task_loss/compute_privacy_task_loss 一致的核心前向，
    通过返回每个 QA 的正向 loss 列表，便于在 EoT 中复用。
    """
    losses = []
    batch_size = images.shape[0]
    model_dtype = next(model.parameters()).dtype

    for i in range(batch_size):
        qa_list = qa_pairs[i]
        if not qa_list:
            continue
        for qa in qa_list:
            question, answer = qa['question'], qa['answer']

            image = images[i:i+1]
            image_resized = T.Resize((448, 448), interpolation=InterpolationMode.BICUBIC)(image)
            pixel_values = T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))(image_resized)
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
                    image_flags=image_flags,
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
                losses.append(loss)
            except Exception as e:
                print(f"警告: VLM 前向失败: {e}")
                continue

    if len(losses) == 0:
        return [torch.tensor(0.0, device=device, requires_grad=True)]
    return losses


def compute_normal_task_loss(images, qa_pairs, tokenizer, model, device):
    losses = _vlm_forward_loss(images, qa_pairs, tokenizer, model, device)
    return torch.stack(losses).mean()


def compute_privacy_task_loss(images, qa_pairs, tokenizer, model, device):
    losses = _vlm_forward_loss(images, qa_pairs, tokenizer, model, device)
    return -torch.stack(losses).mean()


class EoTTransforms(nn.Module):
    """
    期望变换分布 D：
    - 轻微仿射（旋转/平移/缩放/剪切）
    - 颜色抖动（亮度/对比度/饱和度/色调）
    - 高斯模糊
    注意：仅使用对梯度友好的变换，保持张量维度不变。
    """
    def __init__(self,
                 p_affine=0.9,
                 p_color=0.6,
                 p_blur=0.3,
                 max_rotate_deg=10.0,
                 max_translate=0.07,
                 min_scale=0.95,
                 max_scale=1.05,
                 max_shear_deg=5.0,
                 brightness=0.1,
                 contrast=0.1,
                 saturation=0.1,
                 hue=0.02):
        super().__init__()
        self.p_affine = p_affine
        self.p_color = p_color
        self.p_blur = p_blur
        self.max_rotate_deg = max_rotate_deg
        self.max_translate = max_translate
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.max_shear_deg = max_shear_deg
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue

    @torch.no_grad()
    def _sample_affine_params(self, h, w):
        angle = random.uniform(-self.max_rotate_deg, self.max_rotate_deg)
        trans_x = random.uniform(-self.max_translate, self.max_translate) * w
        trans_y = random.uniform(-self.max_translate, self.max_translate) * h
        scale = random.uniform(self.min_scale, self.max_scale)
        shear = random.uniform(-self.max_shear_deg, self.max_shear_deg)
        return angle, (trans_x, trans_y), scale, [shear, 0.0]

    def forward(self, images):
        """
        Args:
            images: Tensor [B, C, H, W], 值域 [0,1]
        Returns:
            Tensor [B, C, H, W]
        """
        device = images.device
        dtype = images.dtype
        b, c, h, w = images.shape

        transformed_list = []
        for i in range(b):
            x = images[i]
            # 仿射
            if random.random() < self.p_affine:
                angle, translate, scale, shear = self._sample_affine_params(h, w)
                x = F.affine(
                    x,
                    angle=angle,
                    translate=(int(translate[0]), int(translate[1])),
                    scale=scale,
                    shear=shear,
                    interpolation=InterpolationMode.BILINEAR,
                    fill=0.0,
                    center=None
                )
            # 颜色抖动
            if random.random() < self.p_color:
                if self.brightness > 0:
                    factor = 1.0 + random.uniform(-self.brightness, self.brightness)
                    x = F.adjust_brightness(x, factor)
                if self.contrast > 0:
                    factor = 1.0 + random.uniform(-self.contrast, self.contrast)
                    x = F.adjust_contrast(x, factor)
                if self.saturation > 0:
                    factor = 1.0 + random.uniform(-self.saturation, self.saturation)
                    x = F.adjust_saturation(x, factor)
                if self.hue > 0:
                    factor = random.uniform(-self.hue, self.hue)
                    x = F.adjust_hue(x, factor)
            # 高斯模糊
            if random.random() < self.p_blur:
                k = random.choice([3, 5])
                sigma = random.uniform(0.1, 1.0)
                x = F.gaussian_blur(x, kernel_size=[k, k], sigma=sigma)

            x = x.clamp(0.0, 1.0)
            transformed_list.append(x)

        return torch.stack(transformed_list, dim=0).to(device=device, dtype=dtype)


class AdversarialTrainerEoT:
    def __init__(self, config):
        self.config = config
        self.device = torch.device(config.device if torch.cuda.is_available() else "cpu")
        self.global_step = 0

        # EoT 相关
        self.eot_samples = int(getattr(config, 'eot_samples', 4))
        self.transforms = EoTTransforms()

        # 日志/权重目录（独立于普通训练）
        self.log_dir = getattr(config, 'log_dir_eot', "./logs_eot")
        self.checkpoint_dir = getattr(config, 'checkpoint_dir_eot', "./checkpoints_eot")
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        print(f"使用设备: {self.device}")
        print(f"EoT 样本数: {self.eot_samples}")

        # 加载 surrogate VLM
        print(f"Loading surrogate MLLM: {config.surrogate_model_name}")
        self.model = AutoModel.from_pretrained(
            config.surrogate_model_name,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            trust_remote_code=True
        ).to(self.device)
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
            epsilon=config.epsilon,
            attn_gamma=getattr(config, 'attn_gamma', 1.0),
            attn_threshold=getattr(config, 'attn_threshold', 0.0),
            attn_topk_percent=getattr(config, 'attn_topk_percent', 0.0),
            attn_mix=getattr(config, 'attn_mix', 1.0)
        ).to(self.device)

        # 注意力提取器（用于隐私任务显著图）
        self.attn_extractor = SaliencyAttention(
            model=self.model,
            tokenizer=self.tokenizer,
            device=self.device,
            save_dir=(self.config.attention_dir if getattr(self.config, 'save_attention', False) else None),
            method=getattr(self.config, 'attn_method', 'xattn_grad')
        )

        self.optimizer = optim.Adam(
            self.generator.parameters(),
            lr=config.learning_rate
        )

        self.history = {
            'epoch': [],
            'loss_total': [],
            'loss_normal': [],
            'loss_privacy': [],
            'psnr': [],
            'linf_norm': []
        }

    def train_step(self, batch, batch_idx):
        images = batch['images'].to(self.device)
        privacy_qa_list = batch['privacy_qa_list']
        normal_qa_list = batch['normal_qa_list']

        # 计算隐私相关注意力图（对比注意力可选）：
        with torch.enable_grad():
            attention_map = self.attn_extractor.get_attention_map(images, privacy_qa_list, normal_qa_list)
        # 生成扰动（用注意力图进行空间加权）
        delta = self.generator(images, attention_map=attention_map)
        x_adv = torch.clamp(images + delta, 0.0, 1.0)

        # 正常路：不使用 EoT（直接在 x_adv 上优化“答对正常任务”）
        loss_normal = compute_normal_task_loss(
            x_adv, normal_qa_list, self.tokenizer, self.model, self.device
        )

        # 隐私路：仅对隐私任务使用 EoT，取期望
        loss_privacy_accum = 0.0
        for _ in range(self.eot_samples):
            x_eot = self.transforms(x_adv)
            loss_privacy_accum = loss_privacy_accum + compute_privacy_task_loss(
                x_eot, privacy_qa_list, self.tokenizer, self.model, self.device
            )
        loss_privacy = loss_privacy_accum / self.eot_samples
        loss_total = self.config.alpha * loss_normal + self.config.beta * loss_privacy

        self.optimizer.zero_grad()
        loss_total.backward()
        self.optimizer.step()

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
        self.generator.train()
        epoch_metrics = {
            'loss_total': 0.0,
            'loss_normal': 0.0,
            'loss_privacy': 0.0,
            'psnr': 0.0,
            'linf_norm': 0.0
        }

        pbar = tqdm(dataloader, desc=f"EoT Epoch {epoch}/{self.config.num_epochs}")
        for batch_idx, batch in enumerate(pbar):
            metrics = self.train_step(batch, batch_idx)
            for key in epoch_metrics.keys():
                epoch_metrics[key] += metrics[key]
            pbar.set_postfix({
                'loss': f"{metrics['loss_total']:.4f}",
                'L_normal': f"{metrics['loss_normal']:.4f}",
                'L_privacy': f"{metrics['loss_privacy']:.4f}",
                'PSNR': f"{metrics['psnr']:.2f}",
                'L_inf': f"{metrics['linf_norm']:.6f}"
            })
            self.global_step += 1

        num_batches = len(dataloader)
        for key in epoch_metrics.keys():
            epoch_metrics[key] /= num_batches
        return epoch_metrics

    def save_checkpoint(self, epoch):
        checkpoint_path = os.path.join(self.checkpoint_dir, f"generator_epoch_{epoch}.pth")
        torch.save({
            'epoch': epoch,
            'generator_state_dict': self.generator.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config.__dict__
        }, checkpoint_path)
        print(f"保存检查点: {checkpoint_path}")

    def save_history(self):
        history_path = os.path.join(self.log_dir, "train_history_eot.json")
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=2)
        print(f"保存训练历史: {history_path}")

    def train(self, dataloader):
        print("\n" + "="*50)
        print("开始训练（EoT 版 - Expectation over Transformations）")
        print("="*50 + "\n")

        for epoch in range(1, self.config.num_epochs + 1):
            metrics = self.train_epoch(dataloader, epoch)
            self.history['epoch'].append(epoch)
            self.history['loss_total'].append(metrics['loss_total'])
            self.history['loss_normal'].append(metrics['loss_normal'])
            self.history['loss_privacy'].append(metrics['loss_privacy'])
            self.history['psnr'].append(metrics['psnr'])
            self.history['linf_norm'].append(metrics['linf_norm'])

            print(f"\nEpoch {epoch} 统计:")
            print(f"  总损失: {metrics['loss_total']:.4f}")
            print(f"  正常任务损失: {metrics['loss_normal']:.4f}")
            print(f"  隐私任务损失: {metrics['loss_privacy']:.4f}")
            print(f"  PSNR: {metrics['psnr']:.2f} dB")
            print(f"  L-inf 范数: {metrics['linf_norm']:.6f}")

            if epoch % self.config.save_interval == 0:
                self.save_checkpoint(epoch)
            self.save_history()

        print("\n" + "="*50)
        print("训练完成（EoT）")
        print("="*50 + "\n")
        self.save_checkpoint(self.config.num_epochs)


def main():
    set_seed(42)

    config = Config()
    # 为 EoT 运行打印关键信息
    print("配置信息 (EoT):")
    print(f"  数据根目录: {config.data_root}")
    print(f"  Surrogate 模型: {config.surrogate_model_name}")
    print(f"  Batch size: {config.batch_size}")
    print(f"  Epochs: {config.num_epochs}")
    print(f"  Learning rate: {config.learning_rate}")
    print(f"  Epsilon: {config.epsilon}")
    print(f"  Alpha: {config.alpha}")
    print(f"  Beta: {config.beta}")
    print(f"  EoT samples: {getattr(config, 'eot_samples', 4)}")

    print("\n加载数据集...")
    dataset = PrivacyProtectionDataset(
        data_root=config.data_root,
        image_size=config.image_size,
        app_filter=config.test_single_app,
        split='train',
        split_ratio=getattr(config, 'train_split_ratio', 0.8)
    )
    dataloader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0
    )

    trainer = AdversarialTrainerEoT(config)
    trainer.train(dataloader)


if __name__ == "__main__":
    main()


