"""
训练脚本：使用双重损失训练对抗噪声生成器
- 损失1: 保证正常任务不出错 (最小化交叉熵)
- 损失2: 使隐私任务出错 (反向优化，最大化交叉熵)
"""

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


def build_transform(input_size):
    """构建 InternVL2 的图像变换"""
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])
    return transform


class AdversarialTrainer:
    """对抗训练器"""
    
    def __init__(self, config):
        self.config = config
        self.device = torch.device(config.device if torch.cuda.is_available() else "cpu")
        
        print(f"使用设备: {self.device}")
        
        # 加载 surrogate MLLM (InternVL2-1B)
        print(f"加载模型: {config.surrogate_model_name}")
        self.model = AutoModel.from_pretrained(
            config.surrogate_model_name,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            trust_remote_code=True
        ).to(self.device)
        
        # 冻结模型参数，只训练生成器
        for param in self.model.parameters():
            param.requires_grad = False
        self.model.eval()  # 保持 eval 模式但允许 forward pass
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            config.surrogate_model_name,
            trust_remote_code=True
        )
        
        # InternVL2 的图像预处理（官方标准）
        self.image_transform = build_transform(input_size=448)
        
        # 生成配置
        self.generation_config = dict(max_new_tokens=50, do_sample=False)
        
        # 初始化噪声生成器
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
        
        # 交叉熵损失函数
        self.ce_loss = nn.CrossEntropyLoss()
        
        # 创建保存目录
        os.makedirs(config.checkpoint_dir, exist_ok=True)
        os.makedirs(config.log_dir, exist_ok=True)
        
        # 训练历史
        self.history = {
            'epoch': [],
            'loss_total': [],
            'loss_normal': [],
            'loss_privacy': [],
            'psnr': [],
            'linf_norm': []
        }
    
    def compute_mllm_loss(self, images, questions, answers):
        """
        计算 MLLM 的损失 - 简化版本
        直接通过图像特征与答案的对齐来计算损失
        
        Args:
            images: Tensor (B, C, H, W)，值域 [0, 1]，带梯度
            questions: List[str]
            answers: List[str]
        
        Returns:
            loss: 可微分的损失
        """
        batch_size = images.shape[0]
        losses = []
        
        for i in range(batch_size):
            # 准备单个样本
            image = images[i:i+1]  # (1, C, H, W)，保持梯度
            question = questions[i]
            target_answer = answers[i]
            
            # 重新应用图像预处理，保持梯度
            # Resize
            image_resized = T.Resize((448, 448))(image)
            # Normalize
            image_normalized = T.Normalize(
                mean=IMAGENET_MEAN, 
                std=IMAGENET_STD
            )(image_resized)
            pixel_values = image_normalized.to(dtype=torch.float16)
            
            try:
                with torch.set_grad_enabled(True):
                    # 这里提取图像特征嵌入
                    vit_embeds = self.model.extract_feature(pixel_values)  # (1, seq_len, hidden_size)
    
                    # 编码答案
                    answer_ids = self.tokenizer.encode(
                        target_answer,
                        return_tensors='pt',
                        max_length=20,
                        truncation=True
                    ).to(self.device)
                    
                    # 文本嵌入
                    answer_embeds = self.model.language_model.get_input_embeddings()(answer_ids)
                    
                    vit_mean = vit_embeds.mean(dim=1)
                    answer_mean = answer_embeds.mean(dim=1) 
                    
                    # 余弦相似度损失
                    cosine_sim = nn.functional.cosine_similarity(
                        vit_mean, 
                        answer_mean, 
                        dim=1
                    )
                    
                    # 损失 = 1 - 相似度
                    loss = 1.0 - cosine_sim.mean()
                    
                    losses.append(loss)
            
            except Exception as e:
                print(f"警告: 计算损失时出错: {e}")
                import traceback
                traceback.print_exc()
                # 使用带梯度的默认损失
                dummy_loss = (images[i:i+1] * 0).mean() + 0.5
                losses.append(dummy_loss)
        
        # 计算平均损失
        if len(losses) == 0:
            dummy_loss = (images * 0).mean() + 0.5
            return dummy_loss
        
        return torch.stack(losses).mean()
    
    def train_step(self, batch):
        """
        单步训练
        
        Args:
            batch: 一个 batch 的数据
        
        Returns:
            dict: 包含各种损失和指标
        """
        images = batch['images'].to(self.device)
        privacy_qa_list = batch['privacy_qa_list']
        normal_qa_list = batch['normal_qa_list']
        
        batch_size = images.shape[0]
        
        # 生成对抗噪声
        delta = self.generator(images)
        x_adv = torch.clamp(images + delta, 0.0, 1.0)
        
        # ============ 损失1: 正常任务损失 (最小化) ============
        # 随机选择一个正常任务 QA 对
        normal_questions = []
        normal_answers = []
        for i in range(batch_size):
            qa_pairs = normal_qa_list[i]
            if len(qa_pairs) > 0:
                # 随机选一个
                qa = qa_pairs[np.random.randint(len(qa_pairs))]
                normal_questions.append(qa['question'])
                normal_answers.append(qa['answer'])
            else:
                # 默认问答
                normal_questions.append("这是什么？")
                normal_answers.append("应用截图")
        
        loss_normal = self.compute_mllm_loss(
            x_adv,
            normal_questions,
            normal_answers
        )
        
        # ============ 损失2: 隐私任务损失 (反向优化，最大化) ============
        # 随机选择一个隐私任务 QA 对
        privacy_questions = []
        privacy_answers = []
        for i in range(batch_size):
            qa_pairs = privacy_qa_list[i]
            if len(qa_pairs) > 0:
                # 随机选一个
                qa = qa_pairs[np.random.randint(len(qa_pairs))]
                privacy_questions.append(qa['question'])
                privacy_answers.append(qa['answer'])
            else:
                # 默认问答
                privacy_questions.append("截图中的人名是什么？")
                privacy_answers.append("未知")
        
        loss_privacy = self.compute_mllm_loss(
            x_adv,
            privacy_questions,
            privacy_answers
        )
        
        # ============ 总损失 ============
        # loss_total = alpha * loss_normal - beta * loss_privacy
        # 注意：loss_privacy 前面是负号，因为我们要反向优化（最大化）
        loss_total = self.config.alpha * loss_normal - self.config.beta * loss_privacy
        
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
        print("开始训练")
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
        app_filter=config.test_single_app  # 如果设置了，只加载指定 app
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0  # 设为 0 避免多进程问题
    )
    
    # 创建训练器
    trainer = AdversarialTrainer(config)
    
    # 开始训练
    trainer.train(dataloader)


if __name__ == "__main__":
    main()

