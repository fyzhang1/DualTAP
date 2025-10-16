import os
import torch
from torch.utils.data import DataLoader
from transformers import AutoProcessor, AutoTokenizer, AutoModelForCausalLM, AutoModel, LlavaOnevisionForConditionalGeneration, Qwen2_5_VLForConditionalGeneration
from tqdm import tqdm
import json
from PIL import Image
import base64
import io
import time
import re

from config import Config
from dataset import PrivacyProtectionDataset, collate_fn
from privacy_metrics import PrivacyMetrics
from qwen_vl_utils import process_vision_info   

"""
python evaluate_baseline.py \
  --output ./eval_results/eval_results_qwen_baseline.json
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
                self.model_name = model_name or "gpt-4o"
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
        """查询API"""
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
                    time.sleep(2 ** attempt)
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


class BaselineEvaluator:
    """
    Baseline评估器（使用原始图像，不加噪声）
    """
    
    def __init__(self, config, use_api=False, api_type=None, 
                 api_key=None, api_model_name=None):
        """
        Args:
            config: 配置对象
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
        model_name = self.config.surrogate_model_name.lower()
        # 根据评估模式初始化
        if self.use_api:
            # 使用API模式
            print(f"使用API模式评估: {api_type} - {api_model_name or '默认模型'}")
            self.api_client = APIClient(
                api_type=api_type,
                api_key=api_key,
                model_name=api_model_name
            )
            self.model = None
            self.processor = None
        else:
            # 使用本地模型模式
            if  "minicpm" in model_name.lower():
                print(f"加载minicpm模型: {config.surrogate_model_name}")
                self.model = AutoModel.from_pretrained(
                    config.surrogate_model_name,
                    attn_implementation='sdpa',
                    torch_dtype=torch.bfloat16,
                    device_map="auto",
                    trust_remote_code=True
                )

                self.processor = AutoTokenizer.from_pretrained(
                    config.surrogate_model_name,
                    trust_remote_code=True
                )
            elif "llava" in model_name.lower():
                print(f"加载LLaVA-OneVision模型: {config.surrogate_model_name}")
                self.model = LlavaOnevisionForConditionalGeneration.from_pretrained(
                    config.surrogate_model_name,
                    torch_dtype=torch.float16,
                    device_map="auto",
                    trust_remote_code=True
                )
                self.processor = AutoProcessor.from_pretrained(
                    config.surrogate_model_name,
                    trust_remote_code=True
                )
            elif "qwen" in model_name.lower():
                print(f"加载Qwen2-VL模型: {config.surrogate_model_name}")
                self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                    config.surrogate_model_name,
                    torch_dtype=torch.float16,
                    device_map="auto",
                    trust_remote_code=True
                )
                self.processor = AutoProcessor.from_pretrained(
                    config.surrogate_model_name,
                    trust_remote_code=True
                )

            else:
                print(f"加载本地模型: {config.surrogate_model_name}")
                self.model = AutoModelForCausalLM.from_pretrained(
                    config.surrogate_model_name,
                    torch_dtype=torch.float16,
                    device_map="auto",
                    trust_remote_code=True
                )
                self.processor = AutoProcessor.from_pretrained(
                    config.surrogate_model_name,
                    trust_remote_code=True
                )

            self.model.eval()
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
        修复了 LLaVA-OneVision 的支持
        """
        # 转换为PIL图像
        image_pil = self._tensor_to_pil(image)
        # 确保为RGB（MiniCPM官方要求）
        if image_pil.mode != "RGB":
            image_pil = image_pil.convert("RGB")
        
        if self.use_api:
            answer = self.api_client.query(image_pil, question)
            return answer
        else:
            model_name = self.config.surrogate_model_name.lower()
            
            # 检测是否为 LLaVA-OneVision
            if "llavaonevision" in model_name or "llava-onevision" in model_name:
                
                # LLaVA-OneVision 格式
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
                
                image_inputs, video_inputs = process_vision_info(messages)
                
                inputs = self.processor(
                    text=[text],
                    images=image_inputs,
                    videos=video_inputs,
                    padding=True,
                    return_tensors="pt"
                ).to(self.device)
                
            elif "qwen" in model_name:
                # Qwen2-VL 格式
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
            
            elif "internvl" in model_name.lower():
                # InternVL2 使用专用的 chat 方法，不需要手动构建 inputs
                # 将图像转换为 tensor，并匹配模型的数据类型（float16）
                import torchvision.transforms as T
                transform = T.Compose([
                    T.Resize((448, 448)),
                    T.ToTensor(),
                    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])
                pixel_values = transform(image_pil).unsqueeze(0)  # (1, 3, 448, 448)
                pixel_values = pixel_values.to(self.device, dtype=torch.float16)  # 转换为 float16
                
                # 使用 InternVL2 的 chat 方法
                generation_config = {
                    'max_new_tokens': 100,
                    'do_sample': False,
                }
                
                # 调用 chat 方法（直接返回文本）
                answer = self.model.chat(
                    tokenizer=self.processor,
                    pixel_values=pixel_values,
                    question=question,
                    generation_config=generation_config
                )
                
                return answer
        

            
            elif "minicpm" in model_name.lower():
                # MiniCPM-V 使用专用的 chat 方法
                # 根据官方示例：image和question都放在msgs的content中
                msgs = [{'role': 'user', 'content': [image_pil, question]}]
                
                # MiniCPM-V-2_6 的 chat 方法
                # 注意：image参数传None，图像在msgs中
                answer = self.model.chat(
                    image=None,
                    msgs=msgs,
                    tokenizer=self.processor,
                    sampling=False,
                    max_new_tokens=100
                )
                
                return answer
            
            else:
                # LLaVA 等其他模型
                prompt = f"USER: <image>\n{question}\nASSISTANT:"
                inputs = self.processor(
                    text=prompt,
                    images=image_pil,
                    return_tensors="pt"
                ).to(self.device)
            
            # 生成
            with torch.no_grad():
                output_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=100,
                    do_sample=False
                )
            
            # 解码 - 所有模型统一使用 processor 的 batch_decode
            answer = self.processor.batch_decode(
                output_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False
            )[0]
            
            # 清理输出
            if "llavaonevision" in model_name or "qwen" in model_name:
                if "assistant" in answer.lower():
                    answer = answer[answer.lower().rfind("assistant") + len("assistant"):]
                    answer = answer.lstrip(":\n ")
            elif "internvl" in model_name.lower():
                # InternVL2 输出清理
                # 移除输入的 prompt 部分
                if "<image>" in answer:
                    answer = answer.split("<image>")[-1]
                # 移除问题部分（如果在输出中）
                answer = answer.strip()
            elif "minicpm" in model_name:
                # MiniCPM-V 输出已经是干净的，不需要额外清理
                answer = answer.strip()
            else:
                if "ASSISTANT:" in answer:
                    answer = answer.split("ASSISTANT:")[-1].strip()
            
            return answer
    
    def check_answer_correctness(self, pred_answer, true_answer):
        """检查答案是否正确（简单的包含关系判断）"""
        return true_answer.lower() in pred_answer.lower()
    
    def evaluate(self, dataset, output_path=None):
        """
        在数据集上进行评估（使用原始图像）
        
        Args:
            dataset: 评估数据集
            output_path: 输出文件路径（如果提供，则边评估边保存）
        
        Returns:
            results: dict，评估结果
        """
        dataloader = DataLoader(
            dataset,
            batch_size=1,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=0
        )
        
        # 统计变量
        privacy_total = 0
        privacy_protected = 0
        
        normal_total = 0
        normal_correct = 0
        
        # 匹配度统计
        name_match_scores = []
        email_match_scores = []
        all_match_scores = []  # 所有字段的match_score
        
        # 详细结果
        detailed_results = []
        
        # 如果提供了输出路径，创建目录并初始化文件
        if output_path:
            output_dir = os.path.dirname(output_path)
            if output_dir and output_dir != '.':
                os.makedirs(output_dir, exist_ok=True)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump({
                    'status': 'evaluating',
                    'mode': 'baseline (clean images, no noise)',
                    'progress': 0,
                    'total': len(dataset),
                    'detailed_results': []
                }, f, indent=2, ensure_ascii=False)
        
        print("开始Baseline评估（原始图像，无噪声）...")
        for batch_idx, batch in enumerate(tqdm(dataloader)):
            images = batch['images'].to(self.device)  # (1, C, H, W) 原始图像
            privacy_qa_list = batch['privacy_qa_list'][0]
            normal_qa_list = batch['normal_qa_list'][0]
            app_name = batch['app_names'][0]
            image_path = batch['image_paths'][0]
            
            # 注意：这里直接使用原始图像，不进行噪声添加
            
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
                
                # 查询 MLLM（使用原始图像）
                pred_answer = self.query_mllm(images, question)
                
                # 动态按应用/答案内容选择字段（支持 name/email/username 等）
                true_fields = self.privacy_metrics.extract_all_fields(true_answer)
                pred_fields = self.privacy_metrics.extract_all_fields(pred_answer)
                fields_to_track = [k for k, v in true_fields.items() if v]

                # 使用隐私指标计算精确的泄露程度（内部已按字段统计）
                privacy_eval = self.privacy_metrics.evaluate_privacy_leakage(
                    pred_answer, true_answer, threshold=0.8
                )
                
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
                
                # 查询 MLLM（使用原始图像）
                pred_answer = self.query_mllm(images, question)
                is_correct = self.check_answer_correctness(pred_answer, true_answer)
                
                normal_total += 1
                if is_correct:
                    normal_correct += 1
                
                item_result['normal_results'].append({
                    'question': question,
                    'true_answer': true_answer,
                    'pred_answer': pred_answer,
                    'correct': is_correct
                })
            
            detailed_results.append(item_result)
            
            # 增量保存
            if output_path:
                privacy_protection_rate = privacy_protected / privacy_total if privacy_total > 0 else 0.0
                normal_accuracy = normal_correct / normal_total if normal_total > 0 else 0.0
                
                intermediate_results = {
                    'status': 'evaluating',
                    'mode': 'baseline (clean images, no noise)',
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
            'mode': 'baseline (clean images, no noise)',
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
        print("Baseline评估结果（原始图像，无噪声）")
        print("="*50)
        print(f"隐私保护率: {results['privacy_protection_rate']:.2%}")
        print(f"  - 隐私问题总数: {results['privacy_total']}")
        print(f"  - 成功保护（未泄露）: {results['privacy_protected']}")
        print()
        print(f"正常任务准确率: {results['normal_accuracy']:.2%}")
        print(f"  - 正常问题总数: {results['normal_total']}")
        print(f"  - 回答正确: {results['normal_correct']}")
        print()
        print("平均匹配度:")
        if 'average_match_scores' in results:
            avg_scores = results['average_match_scores']
            print(f"  - Name匹配度: {avg_scores['name']:.2%} (共{avg_scores['name_count']}个)")
            print(f"  - Email匹配度: {avg_scores['email']:.2%} (共{avg_scores['email_count']}个)")
            print(f"  - 总体匹配度: {avg_scores['overall']:.2%}")
        print("="*50)


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Baseline评估（原始图像，不加噪声）")
    parser.add_argument(
        '--output',
        type=str,
        default='./eval_results/baseline_results.json',
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
        help='API模型名称'
    )
    parser.add_argument(
        '--app',
        type=str,
        default=None,
        help='仅评估指定的应用子集（如: email, ins, amazon）'
    )
    
    args = parser.parse_args()
    
    # 加载配置
    config = Config()
    
    # 加载数据集
    print("加载数据集...")
    dataset = PrivacyProtectionDataset(
        data_root=config.data_root,
        image_size=config.image_size,
        app_filter=args.app
    )
    
    if len(dataset) == 0:
        print("错误: 数据集为空，请检查数据目录")
        return
    
    # 创建评估器
    evaluator = BaselineEvaluator(
        config=config,
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

