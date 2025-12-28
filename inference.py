"""
推理脚本
对单张图像生成对抗样本
"""

import torch
from PIL import Image
from torchvision import transforms
import argparse
import json
import os
import matplotlib.pyplot as plt
import numpy as np

from config import Config
from generator import NoiseGenerator
from attention import SaliencyAttention
from transformers import AutoModel, AutoTokenizer


def load_generator(checkpoint_path, config):
    device = torch.device(config.device if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    saved_cfg = checkpoint.get('config', {}) or {}
    gen_kwargs = dict(
        in_channels=3,
        out_channels=3,
        epsilon=saved_cfg.get('epsilon', config.epsilon),
        attn_gamma=saved_cfg.get('attn_gamma', 1.0),
        attn_threshold=saved_cfg.get('attn_threshold', 0.0),
        attn_topk_percent=saved_cfg.get('attn_topk_percent', 0.0),
        attn_mix=saved_cfg.get('attn_mix', 1.0),
        attn_dilate_kernel=saved_cfg.get('attn_dilate_kernel', 1),
        attn_renorm=saved_cfg.get('attn_renorm', False),
        attn_as_epsilon=saved_cfg.get('attn_as_epsilon', False),
        attn_integration=saved_cfg.get('attn_integration', 'film'),
        film_hidden=saved_cfg.get('film_hidden', 32),
        film_strength=saved_cfg.get('film_strength', 1.0),
    )

    generator = NoiseGenerator(**gen_kwargs).to(device)

    state = checkpoint['generator_state_dict']

    if any(k.endswith('.up.weight') for k in state.keys()):
        new_state = state.copy()
        for name in ['up1', 'up2', 'up3', 'up4']:
            w_key = f'{name}.up.weight'
            b_key = f'{name}.up.bias'
            if w_key in state:
                W = state[w_key] 
                W_reduce = W.mean(dim=(2, 3)).permute(1, 0).unsqueeze(-1).unsqueeze(-1)
                new_state[f'{name}.reduce.weight'] = W_reduce
                if b_key in state:
                    new_state[f'{name}.reduce.bias'] = state[b_key]
                new_state.pop(w_key, None)
                new_state.pop(b_key, None)
        generator.load_state_dict(new_state, strict=False)
    else:
        try:
            generator.load_state_dict(state)
        except RuntimeError:
            generator.load_state_dict(state, strict=False)

    generator.eval()

    for k in [
        'surrogate_model_name', 'attn_method', 'use_attention', 'image_size',
        'attn_gamma', 'attn_threshold', 'attn_topk_percent', 'attn_mix',
        'attn_dilate_kernel', 'attn_renorm', 'attn_as_epsilon', 'attn_integration',
        'film_hidden', 'film_strength'
    ]:
        if k in saved_cfg:
            try:
                setattr(config, k, saved_cfg[k])
            except Exception:
                pass

    return generator, device


def save_visualization(image_tensor, adversarial_tensor, noise, save_path, image_name, attention_map=None):
    try:
        # 准备数据
        img_np = image_tensor[0].detach().cpu().permute(1, 2, 0).numpy().clip(0, 1)
        x_adv_np = adversarial_tensor[0].detach().cpu().permute(1, 2, 0).numpy().clip(0, 1)
        delta_np = noise[0].detach().cpu().abs().mean(dim=0).numpy()
        
        # 差异图
        diff_np = (x_adv_np - img_np).mean(axis=2)
        diff_np = (diff_np - diff_np.min()) / (diff_np.max() - diff_np.min() + 1e-8)
        
        # 创建图像
        fig, axes = plt.subplots(1, 5, figsize=(20, 4))
        
        # 1. 原始图像
        axes[0].imshow(img_np)
        axes[0].set_title(f'Original Image\n{image_name}', fontsize=10)
        axes[0].axis('off')
        
        # 2. 注意力图（若提供）
        if attention_map is not None:
            if attention_map.shape[1] == 1:
                attn_np = attention_map[0, 0].detach().cpu().numpy()
            else:
                attn_np = attention_map[0].detach().cpu().mean(dim=0).numpy()
            im1 = axes[1].imshow(attn_np, cmap='jet', vmin=0, vmax=1)
            axes[1].set_title('Attention Map', fontsize=10)
            axes[1].axis('off')
            plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
        else:
            axes[1].text(0.5, 0.5, 'Attention Map\nNot Provided', 
                       ha='center', va='center', fontsize=10, transform=axes[1].transAxes)
            axes[1].set_title('Attention Map', fontsize=10)
            axes[1].axis('off')
        
        # 3. 噪声
        im2 = axes[2].imshow(delta_np, cmap='hot')
        axes[2].set_title(f'Noise\nL∞={delta_np.max():.4f}', fontsize=10)
        axes[2].axis('off')
        plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)
        
        # 4. 加噪后图像
        axes[3].imshow(x_adv_np)
        axes[3].set_title('Protected Image\n(with noise)', fontsize=10)
        axes[3].axis('off')
        
        # 5. 差异图
        im4 = axes[4].imshow(diff_np, cmap='hot')
        axes[4].set_title('Difference\n(amplified)', fontsize=10)
        axes[4].axis('off')
        plt.colorbar(im4, ax=axes[4], fraction=0.046, pad=0.04)
        
        plt.suptitle(f'Adversarial Example Visualization', fontsize=14, y=0.98)
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        
        print(f"Figure are saved: {save_path}")
        
    except Exception as e:
        print(f"Warning: {e}")


def generate_adversarial_image(image_path, generator, device, image_size=448, attention_map=None):
    original_image = Image.open(image_path).convert('RGB')
    
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
    ])
    
    image_tensor = transform(original_image).unsqueeze(0).to(device)  # (1, C, H, W)
    
    with torch.no_grad():
        use_attn = False
        if attention_map is not None:
            try:
                mask_probe = generator.shape_attention_map(
                    attention_map,
                    target_size=image_tensor.shape[-2:],
                    out_channels=1
                )
                use_attn = bool(float(mask_probe.max().item()) > 1e-6)
                if not use_attn:
                    print("注意: 注意力掩码近乎全零，回退为无注意力加噪。")
            except Exception:
                use_attn = False

        if use_attn:
            delta = generator(image_tensor, attention_map=attention_map)
            adversarial_tensor = (image_tensor + delta).clamp(0.0, 1.0)
            noise = delta
        else:
            noise = generator(image_tensor)
            adversarial_tensor = generator.generate_adversarial(image_tensor)
    
    # 转换回 PIL Image（保持训练尺寸，不要resize回原始尺寸）
    to_pil = transforms.ToPILImage()
    original_image_resized = to_pil(image_tensor.squeeze(0).cpu())
    adversarial_image = to_pil(adversarial_tensor.squeeze(0).cpu())
    
    return original_image_resized, adversarial_image, image_tensor, adversarial_tensor, noise


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
        '--use-attention',
        action='store_true',
        help='使用与训练一致的注意力图进行加噪（需提供QA）'
    )
    parser.add_argument(
        '--qa-question',
        type=str,
        default=None,
        help='用于生成注意力图的隐私问题（与训练一致）'
    )
    parser.add_argument(
        '--qa-answer',
        type=str,
        default=None,
        help='用于生成注意力图的参考答案（与训练一致）'
    )
    parser.add_argument(
        '--qa-json',
        type=str,
        default=None,
        help='包含 privacy_qa_list 与可选 normal_qa_list 的JSON文件路径'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='adversarial.png',
        help='输出加噪图像路径'
    )
    parser.add_argument(
        '--vis',
        type=str,
        default=None,
        help='可视化图保存路径（可选，默认为 output_vis.png）'
    )
    
    args = parser.parse_args()
    
    # 检查输入文件
    if not os.path.exists(args.image):
        print(f"错误: 输入图像不存在 {args.image}")
        return
    
    if not os.path.exists(args.checkpoint):
        print(f"错误: 检查点不存在 {args.checkpoint}")
        return
    
    config = Config()
    
    generator, device = load_generator(args.checkpoint, config)
    
    attention_map = None
    if args.use_attention:
        privacy_qa_list = []
        normal_qa_list = []
        if args.qa_json and os.path.exists(args.qa_json):
            try:
                with open(args.qa_json, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                privacy_qa_list = data.get('privacy_qa_list', [])
                normal_qa_list = data.get('normal_qa_list', [])
            except Exception as e:
                print(f"警告: 读取QA JSON失败: {e}")
        elif args.qa_question and args.qa_answer:
            privacy_qa_list = [[{"question": args.qa_question, "answer": args.qa_answer}]]
            normal_qa_list = [[]]
        else:
            print("警告: 开启 --use-attention 但未提供 QA；将回退为无注意力模式")
        if privacy_qa_list:
            attn_model_name = getattr(config, 'surrogate_model_name', 'OpenGVLab/InternVL3_5-2B')
            surrogate = AutoModel.from_pretrained(
                attn_model_name,
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True,
                trust_remote_code=True
            ).to(device)
            surrogate.eval()
            tokenizer = AutoTokenizer.from_pretrained(
                attn_model_name,
                trust_remote_code=True
            )
            attn_extractor = SaliencyAttention(
                model=surrogate,
                tokenizer=tokenizer,
                device=device,
                save_dir=None,
                method=getattr(config, 'attn_method', 'xattn_grad')
            )
            img = Image.open(args.image).convert('RGB')
            transform = transforms.Compose([
                transforms.Resize((config.image_size, config.image_size)),
                transforms.ToTensor(),
            ])
            img_tensor = transform(img).unsqueeze(0).to(device)
            with torch.enable_grad():
                try:
                    attention_map = attn_extractor.get_attention_map(
                        img_tensor, privacy_qa_list, normal_qa_list or [[]]
                    )
                except Exception as e:
                    attention_map = None


    original_image, adversarial_image, image_tensor, adversarial_tensor, noise = generate_adversarial_image(
        args.image,
        generator,
        device,
        config.image_size,
        attention_map=attention_map
    )
    
    adversarial_image.save(args.output)

    save_original_size = True 
    original_input_size = Image.open(args.image).size
    original_size_path = None
    if save_original_size and original_input_size != adversarial_image.size:
        base_name = os.path.splitext(args.output)[0]
        ext = os.path.splitext(args.output)[1]
        original_size_path = f"{base_name}_original_size{ext}"
        
        # 将加噪图像resize回原始尺寸
        from PIL import Image as PILImage
        adversarial_original_size = adversarial_image.resize(
            original_input_size,
            PILImage.LANCZOS
        )
        adversarial_original_size.save(original_size_path)
    
    if args.vis is None:
        base_name = os.path.splitext(args.output)[0]
        vis_path = f"{base_name}_visualization.png"
    else:
        vis_path = args.vis
    
    image_name = os.path.basename(args.image)
    save_visualization(image_tensor, adversarial_tensor, noise, vis_path, image_name, attention_map=attention_map)

if __name__ == "__main__":
    main()

