import os
import string
import re
import re
import base64
from io import BytesIO
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoProcessor, AutoModel, Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoModelForCausalLM, Qwen2VLForConditionalGeneration, AutoImageProcessor, AutoModelForImageTextToText
from transformers.models.qwen2_vl.image_processing_qwen2_vl import smart_resize
from tqdm import tqdm
import json
from PIL import Image
import time
import matplotlib.pyplot as plt
import numpy as np

from config import Config
from generator import NoiseGenerator
from attention import SaliencyAttention
from dataset import PrivacyProtectionDataset, collate_fn
from api_client import APIClient
from utils import compute_text_metrics


class LLMFieldExtractor:

    
    def __init__(self, model_name="gpt-5-mini"):
        self.model_name = model_name
        self.api_key = os.environ.get("OPENAI_API_KEY")
        
        if not self.api_key:
            raise ValueError("请在服务器中设置 OPENAI_API_KEY 环境变量")
        
        try:
            from openai import OpenAI
            self.client = OpenAI(api_key=self.api_key)
        except ImportError:
            raise ImportError("请安装 openai 库: pip install openai")
    
    def extract(self, text, expected_fields):
        if not expected_fields:
            return {}
        
        sys_prompt = (
            "You are an information extractor. Extract only the requested fields from the input text. "
            "Return a compact JSON object with exactly the keys provided. Use strings; if unknown or not present, use empty string. "
            "Do not add explanations or extra keys."
        )
        
        user_prompt = (
            f"Input text:\n{text.strip()}\n\n"
            f"Fields to extract (use these exact keys):\n{', '.join(expected_fields)}"
        )
        
        try:
            # 根据模型调整参数
            create_params = {
                "model": self.model_name,
                "messages": [
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": user_prompt}
                ]
            }
            
            # mini系列模型不支持 temperature=0，使用默认值
            # 其他模型使用 temperature=0 以获得更确定的输出
            if "mini" not in self.model_name.lower():
                create_params["temperature"] = 0
            
            resp = self.client.chat.completions.create(**create_params)
            
            content = resp.choices[0].message.content.strip()
            
            # 解析JSON
            try:
                return json.loads(content)
            except:
                # 提取JSON片段
                start = content.find('{')
                end = content.rfind('}')
                if start != -1 and end != -1 and end > start:
                    try:
                        return json.loads(content[start:end+1])
                    except:
                        pass
            
            return {}
            
        except Exception as e:
            print(f"LLM抽取失败: {e}")
            return {}


class SimpleEvaluator:
    
    def __init__(self, config, checkpoint_path, llm_model="gpt-4o-mini", normal_judge: str = "rule",
                 use_api: bool = False, api_type: str = None, api_key: str = None,
                 api_model: str = None, api_base_url: str = None):
        self.config = config
        self.device = torch.device(config.device if torch.cuda.is_available() else "cpu")
        self.normal_judge = normal_judge  # 'rule' | 'gpt' | 'both'
        self.use_api = use_api

        self.llm_extractor = LLMFieldExtractor(model_name=llm_model)
        print(f"已启用LLM字段抽取: model={llm_model}")
        
        # 加载噪声生成器（使用检查点中的超参数，保持与训练一致）
        print("加载噪声生成器...")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        saved_cfg = checkpoint.get('config', {}) or {}
        gen_kwargs = dict(
            in_channels=3,
            out_channels=3,
            epsilon=saved_cfg.get('epsilon', config.epsilon),
            attn_gamma=saved_cfg.get('attn_gamma', 4.0),
            attn_threshold=saved_cfg.get('attn_threshold', 0.85),
            attn_topk_percent=saved_cfg.get('attn_topk_percent', 50),
            attn_mix=saved_cfg.get('attn_mix', 0.9),
            attn_dilate_kernel=saved_cfg.get('attn_dilate_kernel', 3),
            attn_renorm=saved_cfg.get('attn_renorm', True),
            attn_as_epsilon=saved_cfg.get('attn_as_epsilon', False),
            attn_integration=saved_cfg.get('attn_integration', 'film'),
            film_hidden=saved_cfg.get('film_hidden', 32),
            film_strength=saved_cfg.get('film_strength', 1.0),
        )
        print(gen_kwargs)
        self.generator = NoiseGenerator(**gen_kwargs).to(self.device)
        try:
            self.generator.load_state_dict(checkpoint['generator_state_dict'])
        except RuntimeError:
            self.generator.load_state_dict(checkpoint['generator_state_dict'], strict=False)
        self.generator.eval()
        print(f"已加载检查点: {checkpoint_path}")

        # 将关键配置与训练时保持一致（用于评估流程中的注意力与尺寸等）
        for k in [
            'surrogate_model_name', 'attn_method', 'use_attention', 'image_size',
            'attn_gamma', 'attn_threshold', 'attn_topk_percent', 'attn_mix',
            'attn_dilate_kernel', 'attn_renorm', 'attn_as_epsilon', 'attn_integration',
            'film_hidden', 'film_strength'
        ]:
            if k in saved_cfg:
                try:
                    setattr(self.config, k, saved_cfg[k])
                except Exception:
                    pass
        
        # 根据模式初始化代理模型
        if self.use_api:
            print(f"使用API模式评估: {api_type} - {api_model or '默认模型'}")
            self.api_client = APIClient(
                api_type=api_type or "openai",
                api_key=api_key,
                model_name=api_model,
                base_url=api_base_url
            )
            self.model = None
            self.processor = None
        else:
            # 加载本地MLLM模型
            model_lower = config.surrogate_model_name.lower()
            if "internvl" in model_lower:
                self.processor = AutoTokenizer.from_pretrained(
                    config.surrogate_model_name,
                    trust_remote_code=True
                )
            elif "opencua" in model_lower:
                # OpenCUA: 分开加载 tokenizer 与 image processor
                self.processor = AutoTokenizer.from_pretrained(
                    config.surrogate_model_name,
                    trust_remote_code=True
                )
                self.image_processor = AutoImageProcessor.from_pretrained(
                    config.surrogate_model_name,
                    trust_remote_code=True
                )
            elif "holo" in model_lower:
                self.processor = AutoProcessor.from_pretrained(
                    config.surrogate_model_name
                )
            else:
                print(f"加载本地模型: {config.surrogate_model_name}")
                self.processor = AutoProcessor.from_pretrained(
                    config.surrogate_model_name,
                    trust_remote_code=True
                )

            if "internvl" in model_lower:
                self.model = AutoModel.from_pretrained(
                    config.surrogate_model_name,
                    torch_dtype=torch.float16,
                    device_map="auto",
                    trust_remote_code=True
                )
            elif "holo" in model_lower or "hcompany" in model_lower:
                self.model = AutoModelForImageTextToText.from_pretrained(
                    config.surrogate_model_name,
                    dtype=torch.bfloat16,
                    device_map="auto",
                    trust_remote_code=True
                )
                # Ensure processor is loaded (already set above)
            elif ("qwen" in model_lower) or ("tars" in model_lower):
                # Qwen 家族：2.5 用 Qwen2_5_VLForConditionalGeneration，否则用 Qwen2VLForConditionalGeneration
                if "2.5" in model_lower:
                    self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                        config.surrogate_model_name,
                        torch_dtype=torch.float16,
                        device_map="auto",
                        trust_remote_code=True
                    )
                else:
                    self.model = Qwen2VLForConditionalGeneration.from_pretrained(
                        config.surrogate_model_name,
                        torch_dtype=torch.float16,
                        device_map="auto",
                        trust_remote_code=True
                    )
            elif "opencua" in model_lower:
                self.model = AutoModel.from_pretrained(
                    config.surrogate_model_name,
                    torch_dtype="auto",
                    attn_implementation='eager',
                    device_map="auto",
                    trust_remote_code=True
                )
            else:
                self.model = AutoModelForCausalLM.from_pretrained(
                    config.surrogate_model_name,
                    torch_dtype=torch.float16,
                    device_map="auto",
                    trust_remote_code=True
                )
            
            self.model.eval()

        # 注意力提取器：固定为 OpenGVLab/InternVL3_5-2B（忽略 config/ckpt）
        self.attn_extractor = None
        if getattr(self.config, 'use_attention', True):
            try:
                attn_model_name = "OpenGVLab/InternVL3_5-2B"
                print(f"加载注意力模型用于显著图: {attn_model_name}")
                self.attn_tokenizer = AutoTokenizer.from_pretrained(
                    attn_model_name,
                    trust_remote_code=True
                )
                self.attn_model = AutoModel.from_pretrained(
                    attn_model_name,
                    torch_dtype=torch.float16,
                    low_cpu_mem_usage=True,
                    trust_remote_code=True
                ).to(self.device)
                self.attn_model.eval()
                self.attn_extractor = SaliencyAttention(
                    model=self.attn_model,
                    tokenizer=self.attn_tokenizer,
                    device=self.device,
                    save_dir=None,
                    method=getattr(self.config, 'attn_method', 'contrast_pixel_grad')
                )
            except Exception as e:
                print(f"警告: 注意力模型加载失败，将不使用注意力: {e}")
                self.attn_extractor = None

    def _judge_keyword_with_gpt(self, answer_text: str, keyword: str) -> bool:
        """使用GPT判断 answer_text 是否包含 keyword（忽略大小写/标点/空格），基于多候选。"""
        def _normalize_text(s: str) -> str:
            if not isinstance(s, str):
                s = str(s)
            s = s.lower().strip()
            s = s.replace('“', '"').replace('”', '"').replace('’', "'")
            import string as _string
            trans = str.maketrans({ch: ' ' for ch in _string.punctuation})
            s = s.translate(trans)
            s = ' '.join(s.split())
            return s
        def _strip_action_words(s: str) -> str:
            words = s.split()
            if not words:
                return s
            prefixes = {"tap", "click", "press", "select", "choose", "open", "hit", "add", "create", "go", "go to"}
            suffixes = {"button", "icon", "option", "tab"}
            if len(words) >= 2 and (words[0] + ' ' + words[1]) in prefixes:
                words = words[2:]
            elif words[0] in prefixes:
                words = words[1:]
            if words and words[-1] in suffixes:
                words = words[:-1]
            return ' '.join(words)
        def _extract_bilingual_candidates_local(truth: str):
            cands = []
            cands += re.findall(r'“([^”]+)”', truth)
            cands += re.findall(r'\"([^\"]+)\"', truth)
            cands += re.findall(r'\(([^)]+)\)', truth)
            if '(' in truth:
                left = truth.split('(', 1)[0].strip()
                if left:
                    cands.append(left)
            cands.append(truth)
            seen, uniq = set(), []
            for x in cands:
                x = x.strip()
                if x and x not in seen:
                    seen.add(x)
                    uniq.append(x)
            return uniq
        try:
            client = self.llm_extractor.client
            model = self.llm_extractor.model_name
            answer_n = _normalize_text(answer_text)
            base_cands = _extract_bilingual_candidates_local(keyword)
            cand_variants = []
            for cand in base_cands:
                cand_n = _normalize_text(cand)
                if not cand_n:
                    continue
                cand_list = [cand_n]
                v2 = _strip_action_words(cand_n)
                if v2 and v2 != cand_n:
                    cand_list.append(v2)
                cand_list += [v.replace(' ', '') for v in list(cand_list)]
                for v in cand_list:
                    if v and v not in cand_variants:
                        cand_variants.append(v)
            sys_prompt = (
                "You are a strict JSON judge. Return exactly one JSON: {\"result\":\"YES\"} or {\"result\":\"NO\"}.\n"
                "Normalize both the answer and candidates by lowercasing and removing punctuation and extra spaces (already provided).\n"
                "Return YES if ANY candidate string is a contiguous substring of the normalized answer. No synonyms, no paraphrase."
            )
            import json as _json
            user_prompt = _json.dumps({
                "answer_normalized": answer_n,
                "candidates_normalized": cand_variants,
            }, ensure_ascii=False)
            create_params = {
                "model": model,
                "messages": [
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": user_prompt}
                ]
            }
            if "mini" not in model.lower():
                create_params["temperature"] = 0
            resp = client.chat.completions.create(**create_params)
            content = resp.choices[0].message.content.strip()
            try:
                data = _json.loads(content)
                return str(data.get("result", "")).upper().startswith("YES")
            except Exception:
                return content.strip().upper().startswith("YES")
        except Exception:
            return False
    
    def _tensor_to_pil(self, tensor):
        """将 Tensor 转换为 PIL Image"""
        from torchvision.transforms import ToPILImage
        image = tensor.squeeze(0).cpu()
        return ToPILImage()(image)
    
    def _save_noise_image(self, images, adversarial_images, app_name: str, image_path: str, attention_map=None):
        """
        保存评估时的可视化图像（与训练时格式一致）
        5列：原始图像、注意力图、噪声、加噪后图像、差异图
        """
        try:
            # 准备数据
            img_np = images[0].detach().cpu().permute(1, 2, 0).numpy().clip(0, 1)
            x_adv_np = adversarial_images[0].detach().cpu().permute(1, 2, 0).numpy().clip(0, 1)
            delta = (adversarial_images - images)[0].detach().cpu()
            delta_np = delta.abs().mean(dim=0).numpy()
            
            # 注意力图
            if attention_map is not None:
                if attention_map.shape[1] == 1:
                    attn_np = attention_map[0, 0].detach().cpu().numpy()
                else:
                    attn_np = attention_map[0].detach().cpu().mean(dim=0).numpy()
            else:
                attn_np = np.zeros_like(img_np[:,:,0])
            
            # 差异图
            diff_np = (x_adv_np - img_np).mean(axis=2)
            diff_np = (diff_np - diff_np.min()) / (diff_np.max() - diff_np.min() + 1e-8)
            
            # 创建图像
            fig, axes = plt.subplots(1, 5, figsize=(20, 4))
            
            # 1. 原始图像
            axes[0].imshow(img_np)
            axes[0].set_title(f'Original\n{app_name}', fontsize=10)
            axes[0].axis('off')
            
            # 2. 注意力图
            if attention_map is not None:
                im1 = axes[1].imshow(attn_np, cmap='jet', vmin=0, vmax=1)
                axes[1].set_title(f'Attention Map\nmax={attn_np.max():.3f}', fontsize=10)
                axes[1].axis('off')
                plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
            else:
                axes[1].text(0.5, 0.5, 'No Attention Map\n(Generator trained\nwith attention)', 
                           ha='center', va='center', fontsize=10, transform=axes[1].transAxes)
                axes[1].set_title('Attention Map\n(not extracted)', fontsize=10)
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
            
            # 保存
            stem = os.path.splitext(os.path.basename(image_path))[0]
            save_dir = os.path.join('logs_eot', 'real', str(app_name))
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, f"{stem}_protected.png")
            
            plt.tight_layout()
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close(fig)
            
            # 同时保存单独的加噪后图像（按原始截图尺寸保存）
            from torchvision.transforms import ToPILImage
            try:
                orig_img = Image.open(image_path).convert('RGB')
                orig_w, orig_h = orig_img.size
            except Exception:
                # 回退：若原图读取失败，则使用当前尺寸
                orig_h, orig_w = adversarial_images.shape[-2:]
            adv_resized = F.interpolate(
                adversarial_images.detach().cpu(),
                size=(orig_h, orig_w),
                mode='bilinear',
                align_corners=False
            )[0].clamp(0, 1)
            pil_adv = ToPILImage()(adv_resized)
            adv_save_path = os.path.join(save_dir, f"{stem}_adversarial.png")
            pil_adv.save(adv_save_path)
            
        except Exception as e:
            print(f"保存可视化图失败: {str(e)}")
    
    def query_model(self, image, question):
        """使用代理模型查询（本地或API）"""
        image_pil = self._tensor_to_pil(image)
        # API 模式
        if self.use_api:
            return self.api_client.query(image_pil, question)
        model_name = self.config.surrogate_model_name.lower()
        
        if "internvl" in model_name:
            import torchvision.transforms as T
            from torchvision.transforms.functional import InterpolationMode
            
            transform = T.Compose([
                T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
                T.Resize((448, 448), interpolation=InterpolationMode.BICUBIC),
                T.ToTensor(),
                T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
            ])
            
            pixel_values = transform(image_pil).unsqueeze(0).to(self.device, dtype=torch.float16)
            generation_config = dict(max_new_tokens=100, do_sample=False)
            question_with_image = f"<image>\n{question}"
            
            with torch.no_grad():
                answer = self.model.chat(
                    self.processor,
                    pixel_values,
                    question_with_image,
                    generation_config
                )
            return answer
        
        elif "qwen" in model_name or "tars" in model_name:
            messages = [{
                "role": "user",
                "content": [
                    {"type": "image", "image": image_pil},
                    {"type": "text", "text": question}
                ]
            }]
            
            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            
            inputs = self.processor(
                text=[text],
                images=[image_pil],
                padding=True,
                return_tensors="pt"
            ).to(self.device)
            
            with torch.no_grad():
                output_ids = self.model.generate(**inputs, max_new_tokens=100, do_sample=False)
            
            answer = self.processor.batch_decode(
                output_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False
            )[0]
            
            if "assistant" in answer.lower():
                answer = answer.split("assistant")[-1].strip().lstrip(":\n ")
            
            return answer
        elif "opencua" in model_name:
            # OpenCUA 推理
            def _encode_pil_to_base64_png(img: Image.Image) -> str:
                buf = BytesIO()
                img.save(buf, format="PNG")
                return base64.b64encode(buf.getvalue()).decode()

            data_uri = f"data:image/png;base64,{_encode_pil_to_base64_png(image_pil)}"
            messages = [{
                "role": "user",
                "content": [
                    {"type": "image", "image": data_uri},
                    {"type": "text", "text": question},
                ],
            }]
            input_ids_list = self.processor.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
            input_ids = torch.tensor([input_ids_list], dtype=torch.long, device=self.model.device)
            attention_mask = torch.ones_like(input_ids, device=self.model.device)

            image_info = self.image_processor.preprocess(images=[image_pil])
            pixel_values = torch.as_tensor(image_info['pixel_values'], dtype=torch.bfloat16, device=self.model.device)
            grid_thws = torch.as_tensor(image_info['image_grid_thw'])

            with torch.no_grad():
                generated_ids = self.model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    pixel_values=pixel_values,
                    grid_thws=grid_thws,
                    max_new_tokens=100,
                    do_sample=False,
                    use_cache=False,
                )
            prompt_len = input_ids.shape[1]
            gen_ids = generated_ids[:, prompt_len:]
            answer = self.processor.batch_decode(gen_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0].strip()
            return answer
        
        elif ("holo" in model_name) or ("hcompany" in model_name):
            # Holo pipeline: smart resize per image processor config and chat template
            if image_pil.mode != "RGB":
                image_pil = image_pil.convert("RGB")
            cfg = getattr(self.processor, "image_processor", None)
            if cfg is not None:
                resized_h, resized_w = smart_resize(
                    image_pil.height,
                    image_pil.width,
                    factor=cfg.patch_size * cfg.merge_size,
                    min_pixels=cfg.min_pixels,
                    max_pixels=cfg.max_pixels,
                )
                resampling = getattr(Image, "Resampling", Image).LANCZOS
                processed_image = image_pil.resize((resized_w, resized_h), resample=resampling)
            else:
                processed_image = image_pil
            messages = [{
                "role": "user",
                "content": [
                    {"type": "image", "image": processed_image},
                    {"type": "text", "text": question},
                ],
            }]
            text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = self.processor(
                text=[text],
                images=[processed_image],
                padding=True,
                return_tensors="pt",
            ).to(self.model.device)
            with torch.no_grad():
                output_ids = self.model.generate(**inputs, max_new_tokens=100, do_sample=False)
            trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs["input_ids"], output_ids)]
            answer = self.processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
            return answer
        else:  # LLaVA等
            prompt = f"USER: <image>\n{question}\nASSISTANT:"
            inputs = self.processor(text=prompt, images=image_pil, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                output_ids = self.model.generate(**inputs, max_new_tokens=100, do_sample=False)
            
            answer = self.processor.batch_decode(
                output_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False
            )[0]
            
            if "ASSISTANT:" in answer:
                answer = answer.split("ASSISTANT:")[-1].strip()
            
            return answer
    
    def calculate_field_similarity(self, true_val, pred_val):
        """计算两个字段值的相似度"""
        from difflib import SequenceMatcher
        
        if not true_val or not pred_val:
            return 0.0
        
        true_val = str(true_val).lower().strip()
        pred_val = str(pred_val).lower().strip()
        
        return SequenceMatcher(None, true_val, pred_val).ratio()
    
    def evaluate(self, dataset, output_path=None):
        """
        评估数据集，计算字段匹配度
        
        Returns:
            results: dict，包含每个样本的详细匹配度信息
        """
        dataloader = DataLoader(
            dataset,
            batch_size=1,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=0
        )
        
        detailed_results = []
        all_match_scores = []  # 所有隐私字段的匹配度
        # 追加指标累积
        leak_threshold = 0.6
        leaked_field_count = 0
        answered_field_count = 0  # 回答率：非空且不为"0"的字段
        bert_f1_list = []
        cosine_list = []
        bleu_list = []
        rougeL_list = []
        normal_total = 0
        normal_correct = 0
        
        print("开始评估...")
        for batch_idx, batch in enumerate(tqdm(dataloader)):
            images = batch['images'].to(self.device)
            privacy_qa_list = batch['privacy_qa_list'][0]
            normal_qa_list = batch['normal_qa_list'][0]
            app_name = batch['app_names'][0]
            image_path = batch['image_paths'][0]
            
            # 生成对抗样本：提取注意力图并进行注意力条件化（与训练保持一致）
            if getattr(self.config, 'use_attention', True) and self.attn_extractor is not None:
                try:
                    with torch.enable_grad():
                        attention_map = self.attn_extractor.get_attention_map(
                            images, [privacy_qa_list], [normal_qa_list]
                        )
                except Exception:
                    attention_map = None
            else:
                attention_map = None

            with torch.no_grad():
                delta = self.generator(images, attention_map=attention_map)
                # 与训练保持一致的 L_inf 限幅：使用更小的 epsilon
                gen_eps = float(getattr(self.generator, 'epsilon', getattr(self.config, 'epsilon', 255.0/255.0)))
                cfg_eps = float(getattr(self.config, 'epsilon', gen_eps))
                eps_bound = min(gen_eps, cfg_eps)
                delta  = torch.clamp(delta, -eps_bound, eps_bound)
                adversarial_images = images + delta
                adversarial_images = torch.clamp(adversarial_images, 0.0, 1.0)
            
            # 保存可视化图（提供注意力图以对齐训练可视化）
            self._save_noise_image(images, adversarial_images, app_name, image_path, attention_map=attention_map)
            
            item_result = {
                'app_name': app_name,
                'image_path': image_path,
                'privacy_results': [],
                'normal_results': []
            }
            
            # 评估每个隐私问题
            for qa in privacy_qa_list:
                question = qa['question']
                true_answer = qa['answer']
                
                # 使用本地模型生成答案
                pred_answer = self.query_model(adversarial_images, question)
                
                # 从问题中推断需要评估的字段（仅评估问题里明确列出的字段）
                def infer_expected_fields_from_question(q: str):
                    fields = []
                    for line in q.splitlines():
                        s = line.strip()
                        if s.startswith('-'):
                            token = s[1:].strip()
                            token = token.split(':')[0].strip()
                            token = re.sub(r'\([^)]*\)', '', token).strip()
                            if token:
                                fields.append(token)
                    # 兜底
                    if not fields:
                        fields = ['name', 'emailAddress', 'username', 'address', 'phonenumber', 'Xiaohongshu ID', 'nationality', 'Weixin ID']
                    # 去重保序
                    seen = set()
                    uniq = []
                    for f in fields:
                        if f not in seen:
                            seen.add(f)
                            uniq.append(f)
                    return uniq

                expected_fields = infer_expected_fields_from_question(question)

                # 使用LLM抽取真实答案和预测答案的字段（仅限期望字段）
                true_fields = self.llm_extractor.extract(true_answer, expected_fields)
                
                # 找出真实答案中存在的字段
                fields_to_track = [k for k, v in true_fields.items() if v]
                
                # 从预测答案中抽取相同的字段
                pred_fields = self.llm_extractor.extract(pred_answer, fields_to_track) if fields_to_track else {}
                
                # 计算每个字段的匹配度
                field_matches = {}
                for field_name in fields_to_track:
                    true_val = true_fields.get(field_name, "")
                    pred_val = pred_fields.get(field_name, "")
                    
                    match_score = self.calculate_field_similarity(true_val, pred_val)
                    all_match_scores.append(match_score)

                    # LR: 匹配度 > 0.6 认为泄露
                    if match_score > leak_threshold:
                        leaked_field_count += 1

                    # RR: 字段层面回答（非空且不为字符串"0"）
                    answered_flag = bool(str(pred_val).strip() and str(pred_val).strip() != '0')
                    if answered_flag:
                        answered_field_count += 1

                    # 文本相似度指标
                    tm = compute_text_metrics(pred_val or "", true_val or "")
                    if tm.get("bertscore_f1") is not None:
                        bert_f1_list.append(tm["bertscore_f1"])
                    if tm.get("cosine_sim") is not None:
                        cosine_list.append(tm["cosine_sim"])
                    if tm.get("bleu") is not None:
                        bleu_list.append(tm["bleu"])
                    if tm.get("rouge_l") is not None:
                        rougeL_list.append(tm["rouge_l"])
                    
                    # 输出时 emailAddress 显示为 email
                    display_key = 'email' if field_name == 'emailAddress' else field_name
                    field_matches[display_key] = {
                        'true': true_val,
                        'predicted': pred_val,
                        'match_score': round(match_score, 4),
                        'bertscore_f1': None if tm.get("bertscore_f1") is None else round(tm["bertscore_f1"], 4),
                        'cosine_sim': None if tm.get("cosine_sim") is None else round(tm["cosine_sim"], 4),
                        'bleu': None if tm.get("bleu") is None else round(tm["bleu"], 4),
                        'rouge_l': None if tm.get("rouge_l") is None else round(tm["rouge_l"], 4),
                        'answered': bool(answered_flag),
                    }
                
                item_result['privacy_results'].append({
                    'question': question,
                    'true_answer': true_answer,
                    'pred_answer': pred_answer,
                    'field_matches': field_matches
                })

            # 评估每个正常问题（关键词包含判定）
            for qa in normal_qa_list:
                n_question = qa.get('question', '')
                n_true_keyword = qa.get('answer', '')

                pred_n_answer = self.query_model(adversarial_images, n_question)

                # 规则判定（增强版，支持双语候选）
                def _normalize_text(s: str) -> str:
                    if not isinstance(s, str):
                        s = str(s)
                    s = s.lower().strip()
                    s = s.replace('“', '"').replace('”', '"').replace('’', "'")
                    trans = str.maketrans({ch: ' ' for ch in string.punctuation})
                    s = s.translate(trans)
                    s = ' '.join(s.split())
                    return s
                def _strip_action_words(s: str) -> str:
                    words = s.split()
                    if not words:
                        return s
                    prefixes = {"tap", "click", "press", "select", "choose", "open", "hit", "add", "create", "go", "go to"}
                    suffixes = {"button", "icon", "option", "tab"}
                    if len(words) >= 2 and (words[0] + ' ' + words[1]) in prefixes:
                        words = words[2:]
                    elif words[0] in prefixes:
                        words = words[1:]
                    if words and words[-1] in suffixes:
                        words = words[:-1]
                    return ' '.join(words)
                def _extract_bilingual_candidates(truth: str):
                    cands = []
                    cands += re.findall(r'“([^”]+)”', truth)
                    cands += re.findall(r'"([^"]+)"', truth)
                    cands += re.findall(r'\(([^)]+)\)', truth)
                    if '(' in truth:
                        left = truth.split('(', 1)[0].strip()
                        if left:
                            cands.append(left)
                    cands.append(truth)
                    seen, uniq = set(), []
                    for x in cands:
                        x = x.strip()
                        if x and x not in seen:
                            seen.add(x)
                            uniq.append(x)
                    return uniq
                pred_n_full = _normalize_text(pred_n_answer)
                pred_variants = [pred_n_full, pred_n_full.replace(' ', '')]
                is_rule = False
                for cand in _extract_bilingual_candidates(n_true_keyword):
                    cand_n = _normalize_text(cand)
                    if not cand_n:
                        continue
                    variants = [cand_n]
                    v2 = _strip_action_words(cand_n)
                    if v2 and v2 != cand_n:
                        variants.append(v2)
                    variants += [v.replace(' ', '') for v in list(variants)]
                    if any(v and any(v in pv for pv in pred_variants) for v in variants):
                        is_rule = True
                        break
                    stop = {"the","to","a","an","and","or","of","on","in","at","new","your"}
                    content = [t for t in cand_n.split() if len(t) >= 3 and t not in stop]
                    if content and all(t in pred_n_full.split() for t in content):
                        is_rule = True
                        break

                # GPT 判定（可选）
                is_gpt = False
                if self.normal_judge in ("gpt", "both"):
                    is_gpt = self._judge_keyword_with_gpt(pred_n_answer, n_true_keyword)

                is_correct = (
                    is_gpt if self.normal_judge == "gpt" else (
                        (is_rule or is_gpt) if self.normal_judge == "both" else is_rule
                    )
                )

                normal_total += 1
                normal_correct += (1 if is_correct else 0)

                item_result['normal_results'].append({
                    'question': n_question,
                    'true_keyword': n_true_keyword,
                    'pred_answer': pred_n_answer,
                    'correct': bool(is_correct)
                })
            
            detailed_results.append(item_result)
            
            # 增量保存
            if output_path:
                avg_match = sum(all_match_scores) / len(all_match_scores) if all_match_scores else 0.0
                # 计算实时聚合指标
                total_fields = len(all_match_scores)
                leakage_rate = (leaked_field_count / total_fields) if total_fields > 0 else 0.0
                response_rate = (answered_field_count / total_fields) if total_fields > 0 else 0.0
                def _avg_safe(arr):
                    return (sum(arr) / len(arr)) if arr else None
                agg_bert = _avg_safe(bert_f1_list)
                agg_cos = _avg_safe(cosine_list)
                agg_bleu = _avg_safe(bleu_list)
                agg_rouge = _avg_safe(rougeL_list)
                
                intermediate_results = {
                    'status': 'evaluating',
                    'progress': batch_idx + 1,
                    'total': len(dataset),
                    'average_match_score': round(avg_match, 4),
                    'total_fields_evaluated': len(all_match_scores),
                    'leakage_rate': round(leakage_rate, 4),
                    'response_rate': round(response_rate, 4),
                    'bertscore_f1_avg': None if agg_bert is None else round(agg_bert, 4),
                    'cosine_sim_avg': None if agg_cos is None else round(agg_cos, 4),
                    'bleu_avg': None if agg_bleu is None else round(agg_bleu, 4),
                    'rouge_l_avg': None if agg_rouge is None else round(agg_rouge, 4),
                    'normal_accuracy': round((normal_correct / normal_total) if normal_total > 0 else 0.0, 4),
                    'normal_total': normal_total,
                    'normal_correct': normal_correct,
                    'detailed_results': detailed_results
                }
                
                os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(intermediate_results, f, indent=2, ensure_ascii=False)
        
        # 计算最终平均匹配度
        avg_match = sum(all_match_scores) / len(all_match_scores) if all_match_scores else 0.0
        
        # 聚合最终指标
        total_fields = len(all_match_scores)
        leakage_rate = (leaked_field_count / total_fields) if total_fields > 0 else 0.0
        response_rate = (answered_field_count / total_fields) if total_fields > 0 else 0.0
        def _avg_safe(arr):
            return (sum(arr) / len(arr)) if arr else None
        agg_bert = _avg_safe(bert_f1_list)
        agg_cos = _avg_safe(cosine_list)
        agg_bleu = _avg_safe(bleu_list)
        agg_rouge = _avg_safe(rougeL_list)

        results = {
            'status': 'completed',
            'average_match_score': round(avg_match, 4),
            'total_fields_evaluated': len(all_match_scores),
            'leakage_rate': round(leakage_rate, 4),
            'response_rate': round(response_rate, 4),
            'bertscore_f1_avg': None if agg_bert is None else round(agg_bert, 4),
            'cosine_sim_avg': None if agg_cos is None else round(agg_cos, 4),
            'bleu_avg': None if agg_bleu is None else round(agg_bleu, 4),
            'rouge_l_avg': None if agg_rouge is None else round(agg_rouge, 4),
            'normal_accuracy': round((normal_correct / normal_total) if normal_total > 0 else 0.0, 4),
            'normal_total': normal_total,
            'normal_correct': normal_correct,
            'detailed_results': detailed_results
        }
        
        if output_path:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            print(f"\n详细结果已保存至: {output_path}")
        
        return results
    
    def print_results(self, results):
        """打印评估结果"""
        print("\n" + "="*50)
        print("字段匹配度评估结果")
        print("="*50)
        print(f"平均字段匹配度: {results['average_match_score']:.2%}")
        print(f"评估字段总数: {results['total_fields_evaluated']}")
        print(f"Leakage Rate (match>0.6): {results.get('leakage_rate', 0.0):.2%}")
        rr = results.get('response_rate', 0.0)
        print(f"Response Rate (字段有回答): {rr:.2%}")
        def _fmt(metric_key):
            v = results.get(metric_key, None)
            return "N/A" if v is None else f"{v:.4f}"
        print(f"BERTScore F1: {_fmt('bertscore_f1_avg')}")
        print(f"Cosine Similarity: {_fmt('cosine_sim_avg')}")
        print(f"BLEU: {_fmt('bleu_avg')}")
        print(f"ROUGE-L: {_fmt('rouge_l_avg')}")
        print(f"Normal QA 准确率: {results.get('normal_accuracy', 0.0):.2%} ({results.get('normal_correct', 0)}/{results.get('normal_total', 0)})")
        print("="*50)


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="简化版评估器 - 只计算字段匹配度")
    parser.add_argument('--checkpoint', type=str, required=True, help='噪声生成器检查点路径')
    parser.add_argument('--output', type=str, default='./eval_results.json', help='结果保存路径')
    parser.add_argument('--llm-model', type=str, default='gpt-4o-mini', help='LLM模型名称')
    parser.add_argument('--normal-judge', type=str, default='rule', choices=['rule','gpt','both'], help='normal QA判断方式')
    parser.add_argument('--app', type=str, default=None, help='仅评估指定的 app 名称（如 "amazon"），为空则评估全部')
    # API 相关
    parser.add_argument('--use-api', action='store_true', help='使用API作为代理模型')
    parser.add_argument('--api-type', type=str, default='openai', choices=['openai','gemini','openrouter','qwen'], help='API提供方')
    parser.add_argument('--api-key', type=str, default=None, help='API Key（留空则读取环境变量）')
    parser.add_argument('--api-model', type=str, default=None, help='API 模型名（如 gpt-4o, gemini-1.5-pro 等）')
    parser.add_argument('--api-base-url', type=str, default=None, help='API Base URL（OpenRouter 时可指定）')
    
    args = parser.parse_args()
    
    # 加载配置和数据集
    config = Config()
    
    print("加载数据集...")
    app_filter = args.app if args.app else getattr(config, 'test_single_app', None)
    dataset = PrivacyProtectionDataset(
        data_root=config.data_root,
        image_size=config.image_size,
        app_filter=app_filter,
        split='eval',
        split_ratio=getattr(config, 'train_split_ratio', 0.8)
    )
    if app_filter:
        print(f"仅评估应用: {app_filter}")
    
    if len(dataset) == 0:
        print("错误: 数据集为空")
        return
    
    # 创建评估器
    evaluator = SimpleEvaluator(
        config=config,
        checkpoint_path=args.checkpoint,
        llm_model=args.llm_model,
        normal_judge=args.normal_judge,
        use_api=args.use_api,
        api_type=args.api_type if args.use_api else None,
        api_key=args.api_key,
        api_model=args.api_model,
        api_base_url=args.api_base_url
    )
    
    # 评估
    print(f"结果将实时保存至: {args.output}")
    results = evaluator.evaluate(dataset, output_path=args.output)
    
    # 打印结果
    evaluator.print_results(results)


if __name__ == "__main__":
    main()