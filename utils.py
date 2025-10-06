"""
工具函数
"""

import torch
import numpy as np
from PIL import Image


def tensor_to_numpy(tensor):
    """
    将 Tensor (C, H, W) 转换为 numpy array (H, W, C)
    """
    if tensor.dim() == 4:
        tensor = tensor.squeeze(0)
    return tensor.permute(1, 2, 0).cpu().numpy()


def numpy_to_tensor(array, device='cpu'):
    """
    将 numpy array (H, W, C) 转换为 Tensor (C, H, W)
    """
    tensor = torch.from_numpy(array).permute(2, 0, 1).float()
    return tensor.to(device)


def calculate_psnr(img1, img2):
    """
    计算两张图像之间的 PSNR（峰值信噪比）
    
    Args:
        img1: Tensor (C, H, W) 或 (B, C, H, W)，值域 [0, 1]
        img2: Tensor (C, H, W) 或 (B, C, H, W)，值域 [0, 1]
    
    Returns:
        psnr: float，PSNR 值（单位：dB）
    """
    mse = torch.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    
    max_pixel = 1.0
    psnr = 20 * torch.log10(max_pixel / torch.sqrt(mse))
    return psnr.item()


def calculate_linf_norm(tensor):
    """
    计算 Tensor 的 L-infinity 范数
    
    Args:
        tensor: Tensor
    
    Returns:
        linf_norm: float
    """
    return torch.max(torch.abs(tensor)).item()


def visualize_noise(noise, scale=10.0):
    """
    可视化噪声
    
    Args:
        noise: Tensor (C, H, W)，噪声
        scale: float，放大倍数
    
    Returns:
        PIL Image，可视化后的噪声
    """
    # 放大噪声以便可视化
    noise_vis = (noise * scale + 0.5).clamp(0, 1)
    
    # 转换为 PIL Image
    from torchvision.transforms import ToPILImage
    to_pil = ToPILImage()
    return to_pil(noise_vis)


def create_data_template():
    """
    创建数据目录模板
    """
    import os
    import json
    
    data_root = "./data"
    
    # 创建示例应用目录
    app_names = ["wechat", "alipay", "taobao"]
    
    for app_name in app_names:
        app_dir = os.path.join(data_root, app_name)
        images_dir = os.path.join(app_dir, "images")
        
        # 创建目录
        os.makedirs(images_dir, exist_ok=True)
        
        # 创建示例 privacy_qa.json
        privacy_qa = {
            "example.jpg": [
                {
                    "question": "截图中的人名是什么？",
                    "answer": "张三"
                },
                {
                    "question": "电话号码是多少？",
                    "answer": "138xxxx1234"
                }
            ]
        }
        
        privacy_qa_path = os.path.join(app_dir, "privacy_qa.json")
        with open(privacy_qa_path, 'w', encoding='utf-8') as f:
            json.dump(privacy_qa, f, indent=2, ensure_ascii=False)
        
        # 创建示例 normal_qa.json
        normal_qa = {
            "example.jpg": [
                {
                    "question": "这是什么应用？",
                    "answer": app_name
                },
                {
                    "question": "界面上有几个按钮？",
                    "answer": "3个"
                }
            ]
        }
        
        normal_qa_path = os.path.join(app_dir, "normal_qa.json")
        with open(normal_qa_path, 'w', encoding='utf-8') as f:
            json.dump(normal_qa, f, indent=2, ensure_ascii=False)
        
        print(f"已创建模板: {app_dir}")
    
    print(f"\n数据目录模板已创建在: {data_root}")
    print("请将图像文件放置在各应用的 images/ 目录下，并更新 QA 文件。")


if __name__ == "__main__":
    # 创建数据目录模板
    create_data_template()

