import os
import torch
from torch.utils.data import DataLoader
from transformers import AutoProcessor, AutoModel
from tqdm import tqdm
import json
from PIL import Image
import base64
import io
import time
import re
from difflib import SequenceMatcher

from config import Config
from generator import NoiseGenerator
from dataset import PrivacyProtectionDataset, collate_fn
from privacy_metrics import PrivacyMetrics


"""
python evaluate.py \
  --checkpoint /home/ecs-user/Agent_VLM/checkpoints/generator_epoch_50.pth \
  --output ./eval_results/eval_results_qwen.json
"""


class APIClient:
    """
    统一的API客户端，支持多种视觉语言模型API
    """
    
    def __init__(self, api_type="openai", api_key=None, model_name=None):
        """
        Args:
            api_type: API类型，支持 'openai', 'claude', 'gemini'
            api_key: API密钥
            model_name: 模型名称
        """
        self.api_type = api_type.lower()
        self.api_key = api_key or os.environ.get(f"{api_type.upper()}_API_KEY")
        self.model_name = model_name
        
        if not self.api_key:
            raise ValueError(f"请设置 {api_type.upper()}_API_KEY 环境变量或传入 api_key 参数")
        
        # 初始化对应的客户端
        if self.api_type == "openai":
            try:
                from openai import OpenAI
                self.client = OpenAI(api_key=self.api_key)
                self.model_name = model_name or "gpt-4o"  # 默认使用 gpt-4o
            except ImportError:
                raise ImportError("请安装 openai 库: pip install openai")
        
        elif self.api_type == "claude":
            try:
                import anthropic
                self.client = anthropic.Anthropic(api_key=self.api_key)
                self.model_name = model_name or "claude-3-5-sonnet-20241022"
            except ImportError:
                raise ImportError("请安装 anthropic 库: pip install anthropic")
        
        elif self.api_type == "gemini":
            try:
                import google.generativeai as genai
                genai.configure(api_key=self.api_key)
                self.client = genai
                self.model_name = model_name or "gemini-1.5-pro"
            except ImportError:
                raise ImportError("请安装 google-generativeai 库: pip install google-generativeai")
        
        else:
            raise ValueError(f"不支持的API类型: {api_type}")
    
    def _pil_to_base64(self, image_pil):
        """将PIL图像转换为base64编码"""
        buffered = io.BytesIO()
        image_pil.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        return img_str
    
    def query(self, image_pil, question, max_retries=3):
        """
        查询API
        
        Args:
            image_pil: PIL图像
            question: 问题文本
            max_retries: 最大重试次数
        
        Returns:
            answer: 模型的回答
        """
        for attempt in range(max_retries):
            try:
                if self.api_type == "openai":
                    return self._query_openai(image_pil, question)
                elif self.api_type == "claude":
                    return self._query_claude(image_pil, question)
                elif self.api_type == "gemini":
                    return self._query_gemini(image_pil, question)
            except Exception as e:
                if attempt < max_retries - 1:
                    print(f"API调用失败，重试 {attempt + 1}/{max_retries}: {e}")
                    time.sleep(2 ** attempt)  # 指数退避
                else:
                    print(f"API调用最终失败: {e}")
                    return "Error: API调用失败"
    
    def _query_openai(self, image_pil, question):
        """调用OpenAI API"""
        base64_image = self._pil_to_base64(image_pil)
        
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": question},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{base64_image}",
                                "detail": "high"  # 高细节模式，提升UI截图/OCR识别能力
                            }
                        }
                    ]
                }
            ],
            max_tokens=300
        )
        
        return response.choices[0].message.content
    
    def _query_claude(self, image_pil, question):
        """调用Claude API"""
        base64_image = self._pil_to_base64(image_pil)
        
        message = self.client.messages.create(
            model=self.model_name,
            max_tokens=300,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/png",
                                "data": base64_image,
                            },
                        },
                        {
                            "type": "text",
                            "text": question
                        }
                    ],
                }
            ],
        )
        
        return message.content[0].text
    
    def _query_gemini(self, image_pil, question):
        """调用Gemini API"""
        model = self.client.GenerativeModel(self.model_name)
        response = model.generate_content([question, image_pil])
        return response.text


class Evaluator:
    """
    评估器
    评估指标：
    1. 隐私保护率：MLLM 在对抗样本上回答隐私问题的错误率
    2. 正常任务准确率：MLLM 在对抗样本上回答正常问题的准确率
    3. 图像质量：PSNR, SSIM（可选）
    
    支持本地模型和API评估
    """
    
    def __init__(self, config, checkpoint_path, use_api=False, api_type=None, 
                 api_key=None, api_model_name=None):
        """
        Args:
            config: 配置对象
            checkpoint_path: 噪声生成器检查点路径
            use_api: 是否使用API进行评估
            api_type: API类型 ('openai', 'claude', 'gemini')
            api_key: API密钥
            api_model_name: API模型名称
        """
        self.config = config
        self.device = torch.device(config.device if torch.cuda.is_available() else "cpu")
        self.use_api = use_api
        
        # 初始化隐私指标计算器
        self.privacy_metrics = PrivacyMetrics()
        
        # 加载噪声生成器
        print("加载噪声生成器...")
        self.generator = NoiseGenerator(
            in_channels=3,
            out_channels=3,
            epsilon=config.epsilon
        ).to(self.device)
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.generator.load_state_dict(checkpoint['generator_state_dict'])
        self.generator.eval()
        print(f"已加载检查点: {checkpoint_path}")
        
        # 根据评估模式初始化
        if self.use_api:
            # 使用API模式
            print(f"使用API模式评估: {api_type} - {api_model_name or '默认模型'}")
            self.api_client = APIClient(
                api_type=api_type,
                api_key=api_key,
                model_name=api_model_name
            )
            self.surrogate_model = None
            self.processor = None
        else:
            # 使用本地模型模式
            print(f"加载本地模型: {config.surrogate_model_name}")
            # 使用 AutoModelForVision2Seq 自动识别模型类型（支持Qwen2VL, LLaVA等）
            self.surrogate_model = AutoModel.from_pretrained(
                config.surrogate_model_name,
                torch_dtype=torch.float16,
                device_map="auto",  # 自动设备映射
                trust_remote_code=True  # Qwen模型需要
            )
            self.processor = AutoProcessor.from_pretrained(
                config.surrogate_model_name,
                trust_remote_code=True
            )
            self.surrogate_model.eval()
            self.api_client = None
    
    def _tensor_to_pil(self, tensor):
        """将 Tensor (1, C, H, W) 转换为 PIL Image"""
        from torchvision.transforms import ToPILImage
        to_pil = ToPILImage()
        image = tensor.squeeze(0).cpu()
        return to_pil(image)
    
    def query_mllm(self, image, question):
        """
        查询 MLLM（支持本地模型和API）
        
        Args:
            image: Tensor (1, C, H, W)
            question: str
        
        Returns:
            answer: str，MLLM 的回答
        """
        # 转换为PIL图像
        image_pil = self._tensor_to_pil(image)
        
        if self.use_api:
            # 使用API查询
            answer = self.api_client.query(image_pil, question)
            return answer
        else:
            # 使用本地模型查询
            # 检查是否是 InternVL2
            model_name = self.config.surrogate_model_name.lower()
            
            if "internvl" in model_name:
                # InternVL2 使用 chat 方法
                import torchvision.transforms as T
                from torchvision.transforms.functional import InterpolationMode
                
                # 图像预处理
                transform = T.Compose([
                    T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
                    T.Resize((448, 448), interpolation=InterpolationMode.BICUBIC),
                    T.ToTensor(),
                    T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
                ])
                
                pixel_values = transform(image_pil).unsqueeze(0).to(self.device, dtype=torch.float16)
                
                # 使用 InternVL2 的 chat 方法
                generation_config = dict(max_new_tokens=100, do_sample=False)
                question_with_image = f"<image>\n{question}"
                
                with torch.no_grad():
                    answer = self.surrogate_model.chat(
                        self.processor,  # tokenizer
                        pixel_values,
                        question_with_image,
                        generation_config
                    )
                
                return answer
            
            elif "qwen" in model_name:
                # Qwen2.5-VL格式
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": image_pil},
                            {"type": "text", "text": question}
                        ]
                    }
                ]
                
                text = self.processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                
                inputs = self.processor(
                    text=[text],
                    images=[image_pil],
                    padding=True,
                    return_tensors="pt"
                ).to(self.device)
                
                # 生成回答
                with torch.no_grad():
                    output_ids = self.surrogate_model.generate(
                        **inputs,
                        max_new_tokens=100,
                        do_sample=False
                    )
                
                # 解码
                answer = self.processor.batch_decode(
                    output_ids,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False
                )[0]
                
                # 清理输出
                if "assistant" in answer.lower():
                    answer = answer.split("assistant")[-1].strip().lstrip(":\n ")
                
                return answer
            
            else:
                # LLaVA等其他模型格式
                prompt = f"USER: <image>\n{question}\nASSISTANT:"
                inputs = self.processor(
                    text=prompt,
                    images=image_pil,
                    return_tensors="pt"
                ).to(self.device)
                
                # 生成回答
                with torch.no_grad():
                    output_ids = self.surrogate_model.generate(
                        **inputs,
                        max_new_tokens=100,
                        do_sample=False
                    )
                
                # 解码
                answer = self.processor.batch_decode(
                    output_ids,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False
                )[0]
                
                # 清理输出
                if "ASSISTANT:" in answer:
                    answer = answer.split("ASSISTANT:")[-1].strip()
                
                return answer
    
    def check_answer_correctness(self, pred_answer, true_answer):
        """
        检查答案是否正确
        简单的字符串匹配（可根据需要改进）
        
        Args:
            pred_answer: 预测的答案
            true_answer: 真实答案
        
        Returns:
            bool: 是否正确
        """
        # 简单的包含关系判断
        return true_answer.lower() in pred_answer.lower()
    
    def evaluate(self, dataset, output_path=None):
        """
        在数据集上进行评估，支持增量保存
        
        Args:
            dataset: 评估数据集
            output_path: 输出文件路径（如果提供，则边评估边保存）
        
        Returns:
            results: dict，评估结果
        """
        dataloader = DataLoader(
            dataset,
            batch_size=1,  # 评估时使用 batch_size=1
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=0
        )
        
        # 统计变量
        privacy_total = 0
        privacy_protected = 0  # 隐私问题回答错误的数量
        
        normal_total = 0
        normal_correct = 0  # 正常问题回答正确的数量
        
        # 匹配度统计
        name_match_scores = []
        email_match_scores = []
        all_match_scores = []  # 所有字段的match_score
        
        # 详细结果
        detailed_results = []
        
        # 如果提供了输出路径，创建目录并初始化文件
        if output_path:
            output_dir = os.path.dirname(output_path)
            if output_dir and output_dir != '.':  # 只有当目录路径非空且非当前目录时才创建
                os.makedirs(output_dir, exist_ok=True)
            # 初始化输出文件
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump({
                    'status': 'evaluating',
                    'progress': 0,
                    'total': len(dataset),
                    'detailed_results': []
                }, f, indent=2, ensure_ascii=False)
        
        print("开始评估...")
        for batch_idx, batch in enumerate(tqdm(dataloader)):
            images = batch['images'].to(self.device)  # (1, C, H, W)
            privacy_qa_list = batch['privacy_qa_list'][0]
            normal_qa_list = batch['normal_qa_list'][0]
            app_name = batch['app_names'][0]
            image_path = batch['image_paths'][0]
            
            # 生成对抗样本
            with torch.no_grad():
                adversarial_images = self.generator.generate_adversarial(images)
            
            item_result = {
                'app_name': app_name,
                'image_path': image_path,
                'privacy_results': [],
                'normal_results': []
            }
            
            # 评估隐私任务
            for qa in privacy_qa_list:
                question = qa['question']
                true_answer = qa['answer']
                
                # 查询 MLLM（使用对抗样本）
                pred_answer = self.query_mllm(adversarial_images, question)
                
                # 动态按应用/答案内容选择字段（支持 name/email/username 等）
                true_fields = self.privacy_metrics.extract_all_fields(true_answer)
                pred_fields = self.privacy_metrics.extract_all_fields(pred_answer)
                fields_to_track = [k for k, v in true_fields.items() if v]

                # 使用隐私指标计算精确的泄露程度（内部已按字段统计）
                privacy_eval = self.privacy_metrics.evaluate_privacy_leakage(
                    pred_answer, true_answer, threshold=0.8
                )
                
                # 统计：如果保护率 > 0.5，认为隐私被保护
                is_protected = privacy_eval['is_protected']
                
                privacy_total += 1
                if is_protected:
                    privacy_protected += 1
                
                # 构建匹配度信息（按字段动态计算）
                extracted_info = {}
                for field_name in fields_to_track:
                    true_val = true_fields.get(field_name)
                    pred_val = pred_fields.get(field_name)
                    score = 0.0
                    found = False
                    if true_val and pred_val:
                        score = self.privacy_metrics.field_similarity(true_val, pred_val)
                        found = score > 0.0
                    elif true_val and self.privacy_metrics.simple_search_in_text(true_val, pred_answer):
                        score = 1.0
                        found = True

                    # 统计聚合分数（仅对 name / emailAddress 做汇总，兼容原有统计）
                    if field_name == 'name' and found:
                        name_match_scores.append(score)
                    if field_name == 'emailAddress' and found:
                        email_match_scores.append(score)
                    # 统计所有字段，不区分类型
                    all_match_scores.append(score)

                    # 输出字段名兼容：emailAddress 在结果中展示为 email，其它保持原样
                    display_key = 'email' if field_name == 'emailAddress' else field_name
                    extracted_info[display_key] = {
                        'true': true_val,
                        'predicted': pred_val,
                        'found': found,
                        'match_score': round(score, 4)
                    }
                
                item_result['privacy_results'].append({
                    'question': question,
                    'true_answer': true_answer,
                    'pred_answer': pred_answer,
                    'protected': is_protected,
                    'extracted_info': extracted_info
                })
            
            # 评估正常任务
            for qa in normal_qa_list:
                question = qa['question']
                true_answer = qa['answer']
                
                # 查询 MLLM
                pred_answer = self.query_mllm(adversarial_images, question)
                is_correct = self.check_answer_correctness(pred_answer, true_answer)
                
                normal_total += 1
                if is_correct:
                    normal_correct += 1  # 回答正确 = 正常功能保持
                
                item_result['normal_results'].append({
                    'question': question,
                    'true_answer': true_answer,
                    'pred_answer': pred_answer,
                    'correct': is_correct
                })
            
            detailed_results.append(item_result)
            
            # 增量保存：每处理完一个样本就更新文件
            if output_path:
                # 计算当前指标
                privacy_protection_rate = privacy_protected / privacy_total if privacy_total > 0 else 0.0
                normal_accuracy = normal_correct / normal_total if normal_total > 0 else 0.0
                
                intermediate_results = {
                    'status': 'evaluating',
                    'progress': batch_idx + 1,
                    'total': len(dataset),
                    'privacy_protection_rate': privacy_protection_rate,
                    'normal_accuracy': normal_accuracy,
                    'privacy_total': privacy_total,
                    'privacy_protected': privacy_protected,
                    'normal_total': normal_total,
                    'normal_correct': normal_correct,
                    'detailed_results': detailed_results
                }
                
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(intermediate_results, f, indent=2, ensure_ascii=False)
        
        # 计算最终指标
        privacy_protection_rate = privacy_protected / privacy_total if privacy_total > 0 else 0.0
        normal_accuracy = normal_correct / normal_total if normal_total > 0 else 0.0
        
        # 计算平均匹配度
        avg_name_match = sum(name_match_scores) / len(name_match_scores) if name_match_scores else 0.0
        avg_email_match = sum(email_match_scores) / len(email_match_scores) if email_match_scores else 0.0
        avg_overall_match = sum(all_match_scores) / len(all_match_scores) if all_match_scores else 0.0
        
        results = {
            'status': 'completed',
            'mode': 'adversarial (with noise)',
            'privacy_protection_rate': privacy_protection_rate,
            'normal_accuracy': normal_accuracy,
            'privacy_total': privacy_total,
            'privacy_protected': privacy_protected,
            'normal_total': normal_total,
            'normal_correct': normal_correct,
            'average_match_scores': {
                'name': round(avg_name_match, 4),
                'email': round(avg_email_match, 4),
                'overall': round(avg_overall_match, 4),  # 全字段平均
                'name_count': len(name_match_scores),
                'email_count': len(email_match_scores),
                'all_count': len(all_match_scores)
            },
            'detailed_results': detailed_results
        }
        
        # 最终保存
        if output_path:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            print(f"\n详细结果已保存至: {output_path}")
        
        return results
    
    def print_results(self, results):
        """打印评估结果"""
        print("\n" + "="*50)
        print("对抗样本评估结果")
        print("="*50)
        print(f"隐私保护率: {results['privacy_protection_rate']:.2%}")
        print(f"  - 隐私问题总数: {results['privacy_total']}")
        print(f"  - 成功保护（回答错误）: {results['privacy_protected']}")
        print()
        print(f"正常任务准确率: {results['normal_accuracy']:.2%}")
        print(f"  - 正常问题总数: {results['normal_total']}")
        print(f"  - 回答正确: {results['normal_correct']}")
        print()
        print("平均匹配度 (对抗样本上的泄露程度):")
        if 'average_match_scores' in results:
            avg_scores = results['average_match_scores']
            print(f"  - Name匹配度: {avg_scores['name']:.2%} (共{avg_scores['name_count']}个)")
            print(f"  - Email匹配度: {avg_scores['email']:.2%} (共{avg_scores['email_count']}个)")
            print(f"  - 总体匹配度: {avg_scores['overall']:.2%}")
        print("="*50)
    
    def save_results(self, results, output_path):
        """
        保存评估结果（独立保存方法，用于兼容性）
        注意：evaluate方法已支持增量保存，推荐直接传output_path给evaluate
        """
        # 自动创建目录
        output_dir = os.path.dirname(output_path)
        if output_dir and output_dir != '.':
            os.makedirs(output_dir, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\n详细结果已保存至: {output_path}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="评估噪声生成器")
    parser.add_argument(
        '--checkpoint',
        type=str,
        required=True,
        help='噪声生成器检查点路径'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='./eval_results.json',
        help='评估结果保存路径'
    )
    
    # API相关参数
    parser.add_argument(
        '--use-api',
        action='store_true',
        help='使用API进行评估（而非本地模型）'
    )
    parser.add_argument(
        '--api-type',
        type=str,
        choices=['openai', 'claude', 'gemini'],
        default='openai',
        help='API类型: openai, claude, gemini'
    )
    parser.add_argument(
        '--api-key',
        type=str,
        default=None,
        help='API密钥（如果未设置，将从环境变量读取）'
    )
    parser.add_argument(
        '--api-model',
        type=str,
        default=None,
        help='API模型名称（如gpt-4o, claude-3-5-sonnet-20241022等，默认使用各API的推荐模型）'
    )
    
    args = parser.parse_args()
    
    # 加载配置
    config = Config()
    
    # 加载数据集
    print("加载数据集...")
    dataset = PrivacyProtectionDataset(
        data_root=config.data_root,
        image_size=config.image_size
    )
    
    if len(dataset) == 0:
        print("错误: 数据集为空，请检查数据目录")
        return
    
    # 创建评估器
    evaluator = Evaluator(
        config=config,
        checkpoint_path=args.checkpoint,
        use_api=args.use_api,
        api_type=args.api_type if args.use_api else None,
        api_key=args.api_key,
        api_model_name=args.api_model
    )
    
    # 进行评估（边评估边保存）
    print(f"结果将实时保存至: {args.output}")
    results = evaluator.evaluate(dataset, output_path=args.output)
    
    
    # 打印最终结果
    evaluator.print_results(results)


if __name__ == "__main__":
    main()