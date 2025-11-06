import os
import json
import re
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

from config import Config
from dataset import PrivacyProtectionDataset, collate_fn
from attention import SaliencyAttention
from generator import NoiseGenerator
from utils import compute_text_metrics


class LLMFieldExtractor:
    """使用LLM从文本中抽取结构化字段（与 eval.py 保持一致）"""
    
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
        """从文本中抽取指定字段"""
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
            create_params = {
                "model": self.model_name,
                "messages": [
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": user_prompt}
                ]
            }
            
            if "mini" not in self.model_name.lower():
                create_params["temperature"] = 0
            
            resp = self.client.chat.completions.create(**create_params)
            content = resp.choices[0].message.content.strip()
            
            try:
                return json.loads(content)
            except:
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


class MaskedAttentionBaseline:

    def __init__(self, config: Config, 
                 checkpoint_path: str = None,
                 use_api: bool = False,
                 api_type: str = None,
                 api_key: str = None,
                 api_model_name: str = None,
                 llm_model: str = None,
                 normal_judge: str = "rule",
                 attn_method: str = None,
                 attn_topk_percent: float = None,
                 attn_threshold: float = None,
                 attn_gamma: float = None,
                 attn_dilate_kernel: int = None,
                 # 遮挡样式参数
                 mask_style: str = 'black',
                 mosaic_block: int = None,
                 noise_sigma: float = None,
                 blur_kernel: int = None,
                 blur_sigma: float = None,
                 save_masks_dir: str = None,
                 use_attention_mask: bool = True):
        self.config = config
        self.device = torch.device(config.device if torch.cuda.is_available() else "cpu")
        self.device_str = ("cuda" if torch.cuda.is_available() and "cuda" in str(self.device) else "cpu")
        self.use_attention_mask = use_attention_mask

        # 如果提供了 checkpoint，从中读取训练时的配置（与 eval.py 保持一致）
        saved_cfg = {}
        if checkpoint_path:
            print(f"从 checkpoint 加载训练配置: {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            saved_cfg = checkpoint.get('config', {}) or {}
            print(f"  已加载配置键: {list(saved_cfg.keys())[:10]}...")
            
            # 将关键配置覆盖到 config（与 eval.py 148-159行保持一致）
            for k in [
                'surrogate_model_name', 'attn_method', 'use_attention', 'image_size',
                'attn_gamma', 'attn_threshold', 'attn_topk_percent', 'attn_mix',
                'attn_dilate_kernel', 'attn_renorm', 'attn_as_epsilon', 'attn_integration',
                'film_hidden', 'film_strength'
            ]:
                if k in saved_cfg:
                    try:
                        old_val = getattr(config, k, None)
                        setattr(self.config, k, saved_cfg[k])
                        if old_val != saved_cfg[k]:
                            print(f"  覆盖 config.{k}: {old_val} -> {saved_cfg[k]}")
                    except Exception:
                        pass

        # 查询器：复用 baseline_eval.SimpleBaselineEvaluator 的查询与解码逻辑
        from eval_original import SimpleBaselineEvaluator  # 延迟导入以减少依赖问题
        self.query_backend = SimpleBaselineEvaluator(
            config=config,
            llm_model=(llm_model if llm_model is not None else 'gpt-5-mini'),
            use_api=use_api,
            api_type=api_type if use_api else None,
            api_key=api_key,
            api_model=api_model_name,
            api_base_url=None
        )

        # 注意力提取器：只在使用注意力掩膜时加载
        if self.use_attention_mask:
            from transformers import AutoModel, AutoTokenizer
            attn_model_name = "OpenGVLab/InternVL3_5-2B"
            print(f"加载注意力模型用于显著图: {attn_model_name}")
            self.attn_model = AutoModel.from_pretrained(
                attn_model_name,
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True,
                trust_remote_code=True
            ).to(self.device)
            for p in self.attn_model.parameters():
                p.requires_grad = False
            self.attn_model.eval()
            self.attn_tokenizer = AutoTokenizer.from_pretrained(attn_model_name, trust_remote_code=True)
        else:
            print("跳过注意力模型加载（使用全图遮挡模式）")
            self.attn_model = None
            self.attn_tokenizer = None

        # 注意力方法与整形参数（只在使用注意力掩膜时初始化）
        if self.use_attention_mask:
            self.attn_method = (
                attn_method if attn_method is not None 
                else saved_cfg.get('attn_method', getattr(config, 'attn_method', 'pixel_grad'))
            )
            
            # 从 checkpoint 或 config 或命令行参数获取注意力整形参数
            final_gamma = (
                attn_gamma if attn_gamma is not None 
                else saved_cfg.get('attn_gamma', getattr(config, 'attn_gamma', 4.0))
            )
            final_threshold = (
                attn_threshold if attn_threshold is not None 
                else saved_cfg.get('attn_threshold', getattr(config, 'attn_threshold', 0.85))
            )
            final_topk = (
                attn_topk_percent if attn_topk_percent is not None 
                else saved_cfg.get('attn_topk_percent', getattr(config, 'attn_topk_percent', 50))
            )
            final_dilate = (
                attn_dilate_kernel if attn_dilate_kernel is not None 
                else saved_cfg.get('attn_dilate_kernel', getattr(config, 'attn_dilate_kernel', 3))
            )
            
            print(f"使用注意力参数: method={self.attn_method}, gamma={final_gamma}, "
                  f"threshold={final_threshold}, topk={final_topk}, dilate={final_dilate}")

            # 使用 NoiseGenerator 的注意力整形工具，便于与训练侧一致
            self.mask_shaper = NoiseGenerator(
                in_channels=3,
                out_channels=3,
                epsilon=1.0,  # 与噪声无关，仅复用 shape_attention_map
                attn_gamma=final_gamma,
                attn_threshold=final_threshold,
                attn_topk_percent=final_topk,
                attn_mix=0.9,  # 生成掩膜时不与全局混合
                attn_dilate_kernel=final_dilate,
                attn_renorm=False,
                attn_as_epsilon=False,
            ).to(self.device)

            self.attn_extractor = SaliencyAttention(
                model=self.attn_model,
                tokenizer=self.attn_tokenizer,
                device=self.device,
                save_dir=None,
                method=self.attn_method,
            )
        else:
            self.attn_method = None
            self.mask_shaper = None
            self.attn_extractor = None

        # 遮挡样式设置
        self.mask_style = (mask_style or 'black').lower()
        self.mosaic_block = int(mosaic_block) if mosaic_block is not None else 12
        self.noise_sigma = float(noise_sigma) if noise_sigma is not None else 0.2
        self.blur_kernel = int(blur_kernel) if blur_kernel is not None else 11
        if self.blur_kernel % 2 == 0:
            self.blur_kernel += 1
        self.blur_sigma = float(blur_sigma) if blur_sigma is not None else 3.0
        print(f"遮挡样式: {self.mask_style} | mosaic_block={self.mosaic_block}, noise_sigma={self.noise_sigma}, blur_kernel={self.blur_kernel}, blur_sigma={self.blur_sigma}")

        # 初始化LLM字段抽取器（与 eval.py 保持一致）
        self.llm_extractor = LLMFieldExtractor(model_name=llm_model if llm_model else 'gpt-5-mini')
        print(f"已启用LLM字段抽取: model={llm_model if llm_model else 'gpt-5-mini'}")
        
        # Normal QA 判断方式（与 eval.py 保持一致：'rule' | 'gpt' | 'both'）
        self.normal_judge = normal_judge
        
        # 仅保存遮挡后的图像
        self.save_masks_dir = save_masks_dir
    
    def calculate_field_similarity(self, true_val, pred_val):
        """计算两个字段值的相似度（与 eval.py 保持一致）"""
        from difflib import SequenceMatcher
        
        if not true_val or not pred_val:
            return 0.0
        
        true_val = str(true_val).lower().strip()
        pred_val = str(pred_val).lower().strip()
        
        return SequenceMatcher(None, true_val, pred_val).ratio()

    def _judge_keyword_with_gpt(self, answer_text: str, keyword: str) -> bool:
        """使用GPT判断 answer_text 是否包含 keyword（与 eval.py 保持一致）"""
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

    # 复制 baseline_eval 中的关键词匹配（规则法），用于 Normal QA 正确性判断
    def _normalize_text(self, s: str) -> str:
        import string as _string
        if not isinstance(s, str):
            s = str(s)
        s = s.lower().strip()
        s = s.replace('“', '"').replace('”', '"').replace('’', "'")
        trans = str.maketrans({ch: ' ' for ch in _string.punctuation})
        s = s.translate(trans)
        s = ' '.join(s.split())
        return s

    def _strip_action_words(self, s: str) -> str:
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

    def _extract_bilingual_candidates(self, truth: str):
        import re as _re
        cands = []
        cands += _re.findall(r'“([^”]+)”', truth)
        cands += _re.findall(r'"([^"]+)"', truth)
        cands += _re.findall(r'\(([^)]+)\)', truth)
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

    def _is_keyword_matched_rule(self, pred: str, truth: str) -> bool:
        """强调“主动作”匹配：
        - 仅在预测答案的首句/首段中判断（忽略后续解释段落）
        - 抽取首段中的主动作目标（Tap/Click/... 后的对象，或首个引号内按钮名）
        - 与真实关键词（含双语候选）做严格短语匹配（规范化、连续子串）
        """
        import re as _re

        # 取首段：到第一个换行或句号为止，避免“Explanation”等噪声
        pred_segment = pred.split('\n', 1)[0]
        # 如果首句很短且下一句紧随而来，允许取前 ~160 字符作为首段
        if len(pred_segment) < 40 and len(pred) > 40:
            pred_segment = pred[:160]

        # 规范化全段与首段
        seg_norm = self._normalize_text(pred_segment)

        # 1) 优先：提取引号内短语（按钮/选项常在引号中）
        quoted = []
        for pat in [r'“([^”]{1,50})”', r'"([^"\n]{1,50})"', r"'([^'\n]{1,50})'"]:
            quoted += _re.findall(pat, pred_segment)
        quoted_norm = [self._normalize_text(q) for q in quoted if q.strip()]

        # 2) 其次：提取动词 + 目标短语（tap/click/press/select/choose/open/hit/add/create/go/go to）
        actions = []
        verb_pat = r"\b(tap|click|press|select|choose|open|hit|add|create|go\s+to|go)\s+([\w\s\-\'\"“”]{1,60})"
        for m in _re.finditer(verb_pat, pred_segment, flags=_re.IGNORECASE):
            obj = m.group(2)
            # 截断在常见尾缀
            obj = _re.split(r"\b(button|icon|option|tab|menu|section)\b|[\.!?,\n]", obj, maxsplit=1)[0]
            actions.append(obj.strip())
        actions_norm = [self._normalize_text(a) for a in actions if a]

        # 候选“主动作目标”集合
        pred_targets = []
        pred_targets += quoted_norm
        pred_targets += actions_norm
        # 兜底：若未抽到任何候选，则用首段本身参与匹配
        if not pred_targets:
            pred_targets = [seg_norm]
        # 加入无空格变体
        pred_targets_expanded = []
        for p in pred_targets:
            if p:
                pred_targets_expanded.append(p)
                p2 = p.replace(' ', '')
                if p2 and p2 != p:
                    pred_targets_expanded.append(p2)

        # 构造真实关键词候选（含双语/引号/括号）
        truth_cands = self._extract_bilingual_candidates(truth)
        truth_norms = []
        for c in truth_cands:
            cn = self._normalize_text(c)
            if cn:
                truth_norms.append(cn)
                c2 = cn.replace(' ', '')
                if c2 and c2 != cn:
                    truth_norms.append(c2)

        # 严格匹配：真实候选需作为连续子串出现在“主动作目标”任一候选中
        for t in truth_norms:
            for p in pred_targets_expanded:
                if t and p and t in p:
                    return True
        return False

    @torch.no_grad()
    def _apply_black_mask(self, image_bchw: torch.Tensor, mask_bchw: torch.Tensor) -> torch.Tensor:
        """
        将 mask 为1的区域置为黑色（0）。
        image_bchw: [1,3,H,W] in [0,1]
        mask_bchw:  [1,1 or 3,H,W] in {0,1}
        """
        if mask_bchw.shape[1] == 1:
            mask_bchw = mask_bchw.repeat(1, image_bchw.shape[1], 1, 1)
        masked = image_bchw * (1.0 - mask_bchw)
        return masked.clamp(0.0, 1.0)

    @torch.no_grad()
    def _apply_mosaic_mask(self, image_bchw: torch.Tensor, mask_bchw: torch.Tensor, block_size: int) -> torch.Tensor:
        if mask_bchw.shape[1] == 1:
            mask_bchw = mask_bchw.repeat(1, image_bchw.shape[1], 1, 1)
        _, _, H, W = image_bchw.shape
        bs = max(1, int(block_size) if block_size is not None else 12)
        small_h = max(1, H // bs)
        small_w = max(1, W // bs)
        down = F.interpolate(image_bchw, size=(small_h, small_w), mode='nearest')
        pixelated = F.interpolate(down, size=(H, W), mode='nearest')
        out = image_bchw * (1.0 - mask_bchw) + pixelated * mask_bchw
        return out.clamp(0.0, 1.0)

    @torch.no_grad()
    def _apply_gaussian_noise_mask(self, image_bchw: torch.Tensor, mask_bchw: torch.Tensor, sigma: float) -> torch.Tensor:
        if mask_bchw.shape[1] == 1:
            mask_bchw = mask_bchw.repeat(1, image_bchw.shape[1], 1, 1)
        noise = torch.randn_like(image_bchw) * float(sigma)
        noisy = (image_bchw + noise).clamp(0.0, 1.0)
        out = image_bchw * (1.0 - mask_bchw) + noisy * mask_bchw
        return out.clamp(0.0, 1.0)

    def _build_gaussian_kernel(self, kernel_size: int, sigma: float, device, dtype):
        k = int(kernel_size)
        if k % 2 == 0:
            k += 1
        coords = torch.arange(k, device=device, dtype=dtype) - (k - 1) / 2.0
        g = torch.exp(-0.5 * (coords / float(sigma)) ** 2)
        g = g / g.sum()
        kernel2d = torch.outer(g, g)
        return kernel2d

    @torch.no_grad()
    def _apply_blur_mask(self, image_bchw: torch.Tensor, mask_bchw: torch.Tensor, kernel_size: int, sigma: float) -> torch.Tensor:
        if mask_bchw.shape[1] == 1:
            mask_bchw = mask_bchw.repeat(1, image_bchw.shape[1], 1, 1)
        _, C, H, W = image_bchw.shape
        kernel2d = self._build_gaussian_kernel(kernel_size, sigma, image_bchw.device, image_bchw.dtype)
        weight = kernel2d.view(1, 1, kernel2d.shape[0], kernel2d.shape[1]).repeat(C, 1, 1, 1)
        pad = kernel2d.shape[0] // 2
        x = F.pad(image_bchw, (pad, pad, pad, pad), mode='reflect')
        blurred = F.conv2d(x, weight, groups=C)
        out = image_bchw * (1.0 - mask_bchw) + blurred * mask_bchw
        return out.clamp(0.0, 1.0)

    def _build_binary_mask(self, attn_map: torch.Tensor, image: torch.Tensor) -> torch.Tensor:
        """
        使用与训练一致的注意力整形，将注意力转为与图像等尺寸的二值掩膜。
        attn_map: [1,1,H,W] in [0,1]
        image:    [1,3,H,W]
        返回:      [1,1,H,W] in {0,1}
        """
        target_hw = image.shape[-2:]
        shaped = self.mask_shaper.shape_attention_map(
            attention_map=attn_map,
            target_size=target_hw,
            out_channels=1,
        )  # [1,1,H,W], 已经在阈值/Top-K后可能为0/1
        # 保险起见再次二值化
        binary = (shaped >= 0.5).float()
        return binary

    

    def evaluate(self, dataset: PrivacyProtectionDataset, output_path: str = None):
        dataloader = DataLoader(
            dataset,
            batch_size=1,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=0
        )

        all_match_scores = []  # 所有隐私字段的匹配度
        # 追加指标累积（与 eval.py 保持一致）
        leak_threshold = 0.6
        leaked_field_count = 0
        answered_field_count = 0  # 回答率：非空且不为"0"的字段
        bert_f1_list = []
        cosine_list = []
        bleu_list = []
        rougeL_list = []
        normal_total = 0
        normal_correct = 0
        
        detailed_results = []

        if output_path:
            out_dir = os.path.dirname(output_path)
            if out_dir and out_dir != '.':
                os.makedirs(out_dir, exist_ok=True)
            mode_str = f"baseline ({'attention' if self.use_attention_mask else 'full-image'} masked: {self.mask_style})"
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump({
                    'status': 'evaluating',
                    'mode': mode_str,
                    'progress': 0,
                    'total': len(dataset),
                    'detailed_results': []
                }, f, indent=2, ensure_ascii=False)

        mode_desc = "基于注意力遮挡" if self.use_attention_mask else "全图遮挡"
        print(f"开始{mode_desc}的Baseline评估，样式: {self.mask_style} ...")
        for idx, batch in enumerate(tqdm(dataloader)):
            images = batch['images'].to(self.device)  # [1,3,H,W], [0,1]
            privacy_qa_list = batch['privacy_qa_list'][0]
            normal_qa_list = batch['normal_qa_list'][0]
            app_name = batch['app_names'][0]
            image_path = batch['image_paths'][0]

            # 生成遮挡掩膜
            if self.use_attention_mask:
                # 使用注意力图生成掩膜
                attn_map = self.attn_extractor.get_attention_map(images, [privacy_qa_list], None)[0:1]
                binary_mask = self._build_binary_mask(attn_map, images)
            else:
                # 不使用注意力图，对整个图像应用遮挡（全1掩膜）
                binary_mask = torch.ones(1, 1, images.shape[2], images.shape[3], 
                                        device=images.device, dtype=images.dtype)
            style = self.mask_style
            if style == 'black':
                masked_images = self._apply_black_mask(images, binary_mask)
            elif style == 'mosaic':
                masked_images = self._apply_mosaic_mask(images, binary_mask, self.mosaic_block)
            elif style in ('gaussian', 'noise', 'gaussian_noise'):
                masked_images = self._apply_gaussian_noise_mask(images, binary_mask, self.noise_sigma)
            elif style == 'blur':
                masked_images = self._apply_blur_mask(images, binary_mask, self.blur_kernel, self.blur_sigma)
            else:
                masked_images = self._apply_black_mask(images, binary_mask)

            # 可选保存遮挡后的图像与二值掩膜
            if self.save_masks_dir is not None:
                try:
                    import torchvision.utils as vutils
                    os.makedirs(self.save_masks_dir, exist_ok=True)
                    base = os.path.splitext(os.path.basename(image_path))[0]
                    prefix = f"{app_name}_{base}"
                    masked_path = os.path.join(self.save_masks_dir, f"{prefix}_masked.png")
                    mask_path = os.path.join(self.save_masks_dir, f"{prefix}_mask.png")
                    vutils.save_image(masked_images.clamp(0.0, 1.0), masked_path)
                    # 将 [1,1,H,W] 的二值掩膜保存为单通道图像
                    vutils.save_image(binary_mask.clamp(0.0, 1.0), mask_path)
                except Exception as e:
                    print(f"保存遮挡图/掩膜失败: {e}")

            item_result = {
                'app_name': app_name,
                'image_path': image_path,
                'privacy_results': [],
                'normal_results': []
            }

            # 隐私任务评估（与 eval.py 保持一致）
            for qa in privacy_qa_list:
                question = qa['question']
                true_answer = qa['answer']

                pred_answer = self.query_backend.query_model(masked_images, question)

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

            # 正常任务评估（同样在遮挡图上）
            for qa in normal_qa_list:
                question = qa['question']
                true_answer = qa['answer']
                pred_answer = self.query_backend.query_model(masked_images, question)
                # 规则判定
                is_rule = self._is_keyword_matched_rule(pred_answer, true_answer)
                # GPT 判定（可选）
                is_gpt = False
                if self.normal_judge in ("gpt", "both"):
                    is_gpt = self._judge_keyword_with_gpt(pred_answer, true_answer)
                is_correct = (
                    is_gpt if self.normal_judge == "gpt" else (
                        (is_rule or is_gpt) if self.normal_judge == "both" else is_rule
                    )
                )

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

            # 增量保存（与 eval.py 保持一致的格式）
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
                
                mode_str = f"baseline ({'attention' if self.use_attention_mask else 'full-image'} masked: {self.mask_style})"
                intermediate_results = {
                    'status': 'evaluating',
                    'mode': mode_str,
                    'progress': idx + 1,
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

        # 汇总（与 eval.py 保持一致的格式）
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

        mode_str = f"baseline ({'attention' if self.use_attention_mask else 'full-image'} masked: {self.mask_style})"
        results = {
            'status': 'completed',
            'mode': mode_str,
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


def main():
    import argparse

    parser = argparse.ArgumentParser(description="基于注意力黑块遮挡的Baseline评估")
    parser.add_argument('--checkpoint', type=str, default=None, help='训练时的检查点路径（用于加载训练配置以保持一致）')
    parser.add_argument('--output', type=str, default='./eval_results/mask_baseline_results.json', help='结果保存路径')
    parser.add_argument('--app', type=str, default=None, help='仅评估指定应用（如: tiktok, meituan_waimai）')
    parser.add_argument('--llm-model', type=str, default='gpt-5-mini', help='用于字段抽取的LLM模型名称')

    # API 相关
    parser.add_argument('--use-api', action='store_true', help='使用API进行评估')
    parser.add_argument('--api-type', type=str, choices=['openai', 'claude', 'gemini'], default='openai', help='API类型')
    parser.add_argument('--api-key', type=str, default=None, help='API密钥')
    parser.add_argument('--api-model', type=str, default=None, help='API模型名称')

    # 注意力与掩膜相关（可覆盖 Config 或 checkpoint 中的配置）
    parser.add_argument('--attn-method', type=str, default=None, help='注意力方法: pixel_grad | xattn_grad | clip_text_match | contrast_* (不建议对比)')
    parser.add_argument('--attn-topk', type=float, default=None, help='Top-K 百分比 (0~100)')
    parser.add_argument('--attn-threshold', type=float, default=None, help='二值化阈值 [0,1]')
    parser.add_argument('--attn-gamma', type=float, default=None, help='注意力幂次增强 gamma')
    parser.add_argument('--attn-dilate', type=int, default=None, help='膨胀核大小(1/3/5)')
    # 遮挡样式
    parser.add_argument('--mask-style', type=str, default='black', choices=['black','mosaic','gaussian','noise','gaussian_noise','blur'], help='遮挡样式')
    parser.add_argument('--mosaic-block', type=int, default=None, help='马赛克块大小（越大越粗糙）')
    parser.add_argument('--noise-sigma', type=float, default=None, help='高斯噪声强度（如 0.15~0.3）')
    parser.add_argument('--blur-kernel', type=int, default=None, help='模糊核大小（奇数，如 9/11/15）')
    parser.add_argument('--blur-sigma', type=float, default=None, help='模糊sigma（如 2.0~4.0）')
    # Normal QA 判定方式
    parser.add_argument('--normal-judge', type=str, default='gpt', choices=['rule','gpt','both'], help='normal QA 判断方式')
    # 可视化导出
    parser.add_argument('--save-masks-dir', type=str, default=None, help='仅保存遮挡后的图像到该目录')
    # 是否使用注意力图
    parser.add_argument('--no-attention', action='store_true', help='不使用注意力图，直接对整个图像应用遮挡')

    args = parser.parse_args()

    config = Config()

    print("加载数据集...")
    dataset = PrivacyProtectionDataset(
        data_root=config.data_root,
        image_size=config.image_size,
        app_filter=args.app,
        split='eval',
        split_ratio=getattr(config, 'train_split_ratio', 0.8)
    )
    if len(dataset) == 0:
        print("错误: 数据集为空，请检查数据目录")
        return

    evaluator = MaskedAttentionBaseline(
        config=config,
        checkpoint_path=args.checkpoint,
        use_api=args.use_api,
        api_type=args.api_type,
        api_key=args.api_key,
        api_model_name=args.api_model,
        llm_model=args.llm_model,
        attn_method=args.attn_method,
        normal_judge=args.normal_judge,
        attn_topk_percent=args.attn_topk,
        attn_threshold=args.attn_threshold,
        attn_gamma=args.attn_gamma,
        attn_dilate_kernel=args.attn_dilate,
        mask_style=args.mask_style,
        mosaic_block=args.mosaic_block,
        noise_sigma=args.noise_sigma,
        blur_kernel=args.blur_kernel,
        blur_sigma=args.blur_sigma,
        save_masks_dir=args.save_masks_dir,
        use_attention_mask=not args.no_attention,  # 添加注意力掩膜开关
    )

    print(f"结果将实时保存至: {args.output}")
    results = evaluator.evaluate(dataset, output_path=args.output)

    # 简要打印（与 eval.py 保持一致）
    print("\n" + "="*50)
    print("Mask Baseline 评估结果（黑块遮挡）")
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


if __name__ == "__main__":
    main()


