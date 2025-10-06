"""
推理脚本
对单张图像生成对抗样本
"""

import torch
from PIL import Image
from torchvision import transforms
import argparse
import os

from config import Config
from generator import NoiseGenerator


def load_generator(checkpoint_path, config):
    """加载噪声生成器"""
    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    
    generator = NoiseGenerator(
        in_channels=3,
        out_channels=3,
        epsilon=config.epsilon
    ).to(device)
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    generator.load_state_dict(checkpoint['generator_state_dict'])
    generator.eval()
    
    return generator, device


def generate_adversarial_image(image_path, generator, device, image_size=224):
    """
    为单张图像生成对抗样本
    
    Args:
        image_path: 输入图像路径
        generator: 噪声生成器
        device: 设备
        image_size: 图像尺寸
    
    Returns:
        original_image: PIL Image，原始图像
        adversarial_image: PIL Image，对抗样本
        noise: Tensor，生成的噪声
    """
    # 加载图像
    original_image = Image.open(image_path).convert('RGB')
    
    # 图像预处理
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
    ])
    
    image_tensor = transform(original_image).unsqueeze(0).to(device)  # (1, C, H, W)
    
    # 生成对抗样本
    with torch.no_grad():
        noise = generator(image_tensor)
        adversarial_tensor = generator.generate_adversarial(image_tensor)
    
    # 转换回 PIL Image
    to_pil = transforms.ToPILImage()
    adversarial_image = to_pil(adversarial_tensor.squeeze(0).cpu())
    
    # 调整回原始尺寸
    adversarial_image = adversarial_image.resize(original_image.size, Image.LANCZOS)
    
    return original_image, adversarial_image, noise


def main():
    parser = argparse.ArgumentParser(description="生成对抗样本")
    parser.add_argument(
        '--image',
        type=str,
        required=True,
        help='输入图像路径'
    )
    parser.add_argument(
        '--checkpoint',
        type=str,
        required=True,
        help='噪声生成器检查点路径'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='adversarial.jpg',
        help='输出图像路径'
    )
    
    args = parser.parse_args()
    
    # 检查输入文件
    if not os.path.exists(args.image):
        print(f"错误: 输入图像不存在 {args.image}")
        return
    
    if not os.path.exists(args.checkpoint):
        print(f"错误: 检查点不存在 {args.checkpoint}")
        return
    
    # 加载配置
    config = Config()
    
    # 加载生成器
    print("加载噪声生成器...")
    generator, device = load_generator(args.checkpoint, config)
    
    # 生成对抗样本
    print(f"处理图像: {args.image}")
    original_image, adversarial_image, noise = generate_adversarial_image(
        args.image,
        generator,
        device,
        config.image_size
    )
    
    # 保存结果
    adversarial_image.save(args.output)
    print(f"对抗样本已保存至: {args.output}")
    
    # 计算噪声统计信息
    noise_max = noise.abs().max().item()
    noise_mean = noise.abs().mean().item()
    print(f"\n噪声统计:")
    print(f"  最大值: {noise_max:.6f}")
    print(f"  平均值: {noise_mean:.6f}")
    print(f"  epsilon: {config.epsilon:.6f}")


if __name__ == "__main__":
    main()

