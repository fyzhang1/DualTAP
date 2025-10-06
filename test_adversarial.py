"""
测试对抗噪声的效果
加载训练好的生成器，对图像添加对抗噪声，并测试对正常任务和隐私任务的影响
"""

import os
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from transformers import AutoModel, AutoTokenizer
import torchvision.transforms as T

from config import Config
from generator import NoiseGenerator
from utils import calculate_psnr, calculate_linf_norm, visualize_noise


class AdversarialTester:
    """对抗噪声测试器"""
    
    def __init__(self, config, checkpoint_path):
        self.config = config
        self.device = torch.device(config.device if torch.cuda.is_available() else "cpu")
        
        print(f"使用设备: {self.device}")
        
        # 加载模型
        print(f"加载模型: {config.surrogate_model_name}")
        self.model = AutoModel.from_pretrained(
            config.surrogate_model_name,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            trust_remote_code=True
        ).to(self.device).eval()
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            config.surrogate_model_name,
            trust_remote_code=True
        )
        
        # 加载生成器
        print(f"加载生成器: {checkpoint_path}")
        self.generator = NoiseGenerator(
            in_channels=3,
            out_channels=3,
            epsilon=config.epsilon
        ).to(self.device).eval()
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.generator.load_state_dict(checkpoint['generator_state_dict'])
        
        # 图像预处理
        self.transform = transforms.Compose([
            transforms.Resize((config.image_size, config.image_size)),
            transforms.ToTensor(),
        ])
    
    def generate_response(self, image_tensor, question):
        """
        使用 MLLM 生成回答
        
        Args:
            image_tensor: Tensor (1, C, H, W)，值域 [0, 1]
            question: str
        
        Returns:
            response: str
        """
        # 转换为 PIL Image
        image_np = (image_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        pil_image = Image.fromarray(image_np)
        
        # 准备输入
        pixel_values = self.model.vision_model.image_processor(
            pil_image,
            return_tensors='pt'
        ).pixel_values.to(self.device).to(torch.bfloat16)
        
        # 构建对话
        question_text = f"<image>\n{question}"
        input_ids = self.tokenizer.encode(
            question_text,
            return_tensors='pt',
            add_special_tokens=True
        ).to(self.device)
        
        # 生成回答
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=input_ids,
                pixel_values=pixel_values,
                max_new_tokens=100,
                do_sample=False
            )
        
        # 解码
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # 移除问题部分
        if question in response:
            response = response.split(question)[-1].strip()
        
        return response
    
    def test_single_image(self, image_path, normal_qa, privacy_qa, save_dir=None):
        """
        测试单张图像
        
        Args:
            image_path: 图像路径
            normal_qa: List[Dict], 正常任务 QA 对
            privacy_qa: List[Dict], 隐私任务 QA 对
            save_dir: 保存目录 (可选)
        """
        print("\n" + "="*70)
        print(f"测试图像: {image_path}")
        print("="*70)
        
        # 加载图像
        image = Image.open(image_path).convert('RGB')
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        # 生成对抗样本
        with torch.no_grad():
            delta = self.generator(image_tensor)
            x_adv = torch.clamp(image_tensor + delta, 0.0, 1.0)
        
        # 计算指标
        psnr = calculate_psnr(image_tensor, x_adv)
        linf_norm = calculate_linf_norm(delta)
        
        print(f"\n图像质量指标:")
        print(f"  PSNR: {psnr:.2f} dB")
        print(f"  L-inf 范数: {linf_norm:.6f}")
        
        # 测试正常任务
        print(f"\n{'='*70}")
        print("正常任务测试:")
        print(f"{'='*70}")
        for i, qa in enumerate(normal_qa[:3], 1):  # 测试前3个
            question = qa['question']
            ground_truth = qa['answer']
            
            # 原始图像
            response_clean = self.generate_response(image_tensor, question)
            
            # 对抗图像
            response_adv = self.generate_response(x_adv, question)
            
            print(f"\n[正常任务 {i}]")
            print(f"  问题: {question}")
            print(f"  标准答案: {ground_truth}")
            print(f"  原图回答: {response_clean}")
            print(f"  对抗回答: {response_adv}")
        
        # 测试隐私任务
        print(f"\n{'='*70}")
        print("隐私任务测试:")
        print(f"{'='*70}")
        for i, qa in enumerate(privacy_qa[:3], 1):  # 测试前3个
            question = qa['question']
            ground_truth = qa['answer']
            
            # 原始图像
            response_clean = self.generate_response(image_tensor, question)
            
            # 对抗图像
            response_adv = self.generate_response(x_adv, question)
            
            print(f"\n[隐私任务 {i}]")
            print(f"  问题: {question}")
            print(f"  真实答案: {ground_truth}")
            print(f"  原图回答: {response_clean}")
            print(f"  对抗回答: {response_adv}")
        
        # 保存结果
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            
            # 保存原始图像
            clean_pil = transforms.ToPILImage()(image_tensor.squeeze(0).cpu())
            clean_pil.save(os.path.join(save_dir, "clean.png"))
            
            # 保存对抗图像
            adv_pil = transforms.ToPILImage()(x_adv.squeeze(0).cpu())
            adv_pil.save(os.path.join(save_dir, "adversarial.png"))
            
            # 保存噪声可视化
            noise_vis = visualize_noise(delta.squeeze(0).cpu(), scale=10.0)
            noise_vis.save(os.path.join(save_dir, "noise_visualized.png"))
            
            print(f"\n结果已保存到: {save_dir}")


def main():
    """主函数"""
    config = Config()
    
    # 检查点路径
    checkpoint_path = os.path.join(config.checkpoint_dir, "generator_epoch_50.pth")
    
    if not os.path.exists(checkpoint_path):
        print(f"错误: 检查点不存在: {checkpoint_path}")
        print("请先运行 train.py 进行训练")
        return
    
    # 创建测试器
    tester = AdversarialTester(config, checkpoint_path)
    
    # 测试示例
    # 这里需要根据你的数据修改
    app_name = "amazon"
    image_name = "amazon_0"
    
    image_path = os.path.join(config.data_root, app_name, "images", f"{image_name}.png")
    
    # 示例 QA (你需要根据实际数据修改)
    normal_qa = [
        {"question": "这是什么应用？", "answer": "Amazon"},
        {"question": "界面是什么语言？", "answer": "英文"},
    ]
    
    privacy_qa = [
        {"question": "截图中的人名是什么？", "answer": "John Doe"},
        {"question": "显示的地址是什么？", "answer": "123 Main St"},
    ]
    
    # 运行测试
    save_dir = "./test_results"
    tester.test_single_image(image_path, normal_qa, privacy_qa, save_dir)


if __name__ == "__main__":
    main()

