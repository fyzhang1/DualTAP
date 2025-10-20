import os
import json
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import Config
from dataset import PrivacyProtectionDataset, collate_fn
from privacy_metrics import PrivacyMetrics
from attention import SaliencyAttention
from generator import NoiseGenerator


class MaskedAttentionBaseline:

    def __init__(self, config: Config, 
                 use_api: bool = False,
                 api_type: str = None,
                 api_key: str = None,
                 api_model_name: str = None,
                 llm_model: str = None,
                 attn_method: str = None,
                 attn_topk_percent: float = None,
                 attn_threshold: float = None,
                 attn_gamma: float = None,
                 attn_dilate_kernel: int = None,
                 save_masks_dir: str = None):
        self.config = config
        self.device = torch.device(config.device if torch.cuda.is_available() else "cpu")

        # 查询器：复用 baseline_eval.SimpleBaselineEvaluator 的查询与解码逻辑
        from baseline_eval import SimpleBaselineEvaluator  # 延迟导入以减少依赖问题
        self.query_backend = SimpleBaselineEvaluator(
            config=config,
            llm_model=(llm_model if llm_model is not None else 'gpt-4o-mini'),
            use_api=use_api,
            api_type=api_type if use_api else None,
            api_key=api_key,
            api_model=api_model_name,
            api_base_url=None
        )

        # 注意力提取器：与训练侧保持一致
        from transformers import AutoModel, AutoTokenizer
        print(f"为注意力加载 surrogate MLLM: {config.surrogate_model_name}")
        self.attn_model = AutoModel.from_pretrained(
            config.surrogate_model_name,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            trust_remote_code=True
        ).to(self.device)
        for p in self.attn_model.parameters():
            p.requires_grad = False
        self.attn_model.eval()
        self.attn_tokenizer = AutoTokenizer.from_pretrained(config.surrogate_model_name, trust_remote_code=True)

        # 注意力方法与整形参数（支持外部覆盖，否则使用 Config）
        self.attn_method = attn_method if attn_method is not None else getattr(config, 'attn_method', 'pixel_grad')

        # 使用 NoiseGenerator 的注意力整形工具，便于与训练侧一致
        self.mask_shaper = NoiseGenerator(
            in_channels=3,
            out_channels=3,
            epsilon=1.0,  # 与噪声无关，仅复用 shape_attention_map
            attn_gamma=(attn_gamma if attn_gamma is not None else getattr(config, 'attn_gamma', 1.0)),
            attn_threshold=(attn_threshold if attn_threshold is not None else getattr(config, 'attn_threshold', 0.0)),
            attn_topk_percent=(attn_topk_percent if attn_topk_percent is not None else getattr(config, 'attn_topk_percent', 0.0)),
            attn_mix=1.0,  # 生成掩膜时不与全局混合
            attn_dilate_kernel=(attn_dilate_kernel if attn_dilate_kernel is not None else getattr(config, 'attn_dilate_kernel', 1)),
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

        self.privacy_metrics = PrivacyMetrics()
        # 仅保存遮挡后的图像
        self.save_masks_dir = save_masks_dir

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

        privacy_total = 0
        privacy_protected = 0
        normal_total = 0
        normal_correct = 0

        name_match_scores = []
        email_match_scores = []
        all_match_scores = []

        detailed_results = []

        if output_path:
            out_dir = os.path.dirname(output_path)
            if out_dir and out_dir != '.':
                os.makedirs(out_dir, exist_ok=True)
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump({
                    'status': 'evaluating',
                    'mode': 'baseline (black-mask attention regions)',
                    'progress': 0,
                    'total': len(dataset),
                    'detailed_results': []
                }, f, indent=2, ensure_ascii=False)

        print("开始基于注意力黑块遮挡的Baseline评估...")
        for idx, batch in enumerate(tqdm(dataloader)):
            images = batch['images'].to(self.device)  # [1,3,H,W], [0,1]
            privacy_qa_list = batch['privacy_qa_list'][0]
            normal_qa_list = batch['normal_qa_list'][0]
            app_name = batch['app_names'][0]
            image_path = batch['image_paths'][0]

            # 计算隐私注意力（不使用对比normal，避免误抵消）
            attn_map = self.attn_extractor.get_attention_map(images, [privacy_qa_list], None)[0:1]
            binary_mask = self._build_binary_mask(attn_map, images)
            masked_images = self._apply_black_mask(images, binary_mask)

            # 可选保存遮挡后的图像
            if self.save_masks_dir is not None:
                try:
                    import torchvision.utils as vutils
                    os.makedirs(self.save_masks_dir, exist_ok=True)
                    base = os.path.splitext(os.path.basename(image_path))[0]
                    prefix = f"{app_name}_{base}"
                    masked_path = os.path.join(self.save_masks_dir, f"{prefix}_masked.png")
                    vutils.save_image(masked_images.clamp(0.0, 1.0), masked_path)
                except Exception as e:
                    print(f"保存遮挡图失败: {e}")

            item_result = {
                'app_name': app_name,
                'image_path': image_path,
                'privacy_results': [],
                'normal_results': []
            }

            # 隐私任务评估
            for qa in privacy_qa_list:
                question = qa['question']
                true_answer = qa['answer']

                pred_answer = self.query_backend.query_model(masked_images, question)

                true_fields = self.privacy_metrics.extract_all_fields(true_answer)
                pred_fields = self.privacy_metrics.extract_all_fields(pred_answer)
                fields_to_track = [k for k, v in true_fields.items() if v]

                privacy_eval = self.privacy_metrics.evaluate_privacy_leakage(
                    pred_answer, true_answer, threshold=0.8
                )

                is_protected = privacy_eval['is_protected']
                privacy_total += 1
                if is_protected:
                    privacy_protected += 1

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

                    if field_name == 'name' and found:
                        name_match_scores.append(score)
                    if field_name == 'emailAddress' and found:
                        email_match_scores.append(score)
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

            # 正常任务评估（同样在遮挡图上）
            for qa in normal_qa_list:
                question = qa['question']
                true_answer = qa['answer']
                pred_answer = self.query_backend.query_model(masked_images, question)
                is_correct = self._is_keyword_matched_rule(pred_answer, true_answer)

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
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump({
                        'status': 'evaluating',
                        'mode': 'baseline (black-mask attention regions)',
                        'progress': idx + 1,
                        'total': len(dataset),
                        'privacy_protection_rate': privacy_protection_rate,
                        'normal_accuracy': normal_accuracy,
                        'privacy_total': privacy_total,
                        'privacy_protected': privacy_protected,
                        'normal_total': normal_total,
                        'normal_correct': normal_correct,
                        'detailed_results': detailed_results
                    }, f, indent=2, ensure_ascii=False)

        # 汇总
        privacy_protection_rate = privacy_protected / privacy_total if privacy_total > 0 else 0.0
        normal_accuracy = normal_correct / normal_total if normal_total > 0 else 0.0
        avg_name_match = sum(name_match_scores) / len(name_match_scores) if name_match_scores else 0.0
        avg_email_match = sum(email_match_scores) / len(email_match_scores) if email_match_scores else 0.0
        avg_overall_match = sum(all_match_scores) / len(all_match_scores) if all_match_scores else 0.0

        results = {
            'status': 'completed',
            'mode': 'baseline (black-mask attention regions)',
            'privacy_protection_rate': privacy_protection_rate,
            'normal_accuracy': normal_accuracy,
            'privacy_total': privacy_total,
            'privacy_protected': privacy_protected,
            'normal_total': normal_total,
            'normal_correct': normal_correct,
            'average_match_scores': {
                'name': round(avg_name_match, 4),
                'email': round(avg_email_match, 4),
                'overall': round(avg_overall_match, 4),
                'name_count': len(name_match_scores),
                'email_count': len(email_match_scores),
                'all_count': len(all_match_scores)
            },
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
    parser.add_argument('--output', type=str, default='./eval_results/mask_baseline_results.json', help='结果保存路径')
    parser.add_argument('--app', type=str, default=None, help='仅评估指定应用（如: tiktok, meituan_waimai）')
    parser.add_argument('--llm-model', type=str, default='gpt-4o-mini', help='用于字段抽取的LLM模型名称')

    # API 相关
    parser.add_argument('--use-api', action='store_true', help='使用API进行评估')
    parser.add_argument('--api-type', type=str, choices=['openai', 'claude', 'gemini'], default='openai', help='API类型')
    parser.add_argument('--api-key', type=str, default=None, help='API密钥')
    parser.add_argument('--api-model', type=str, default=None, help='API模型名称')

    # 注意力与掩膜相关（可覆盖 Config）
    parser.add_argument('--attn-method', type=str, default=None, help='注意力方法: pixel_grad | xattn_grad | clip_text_match | contrast_* (不建议对比)')
    parser.add_argument('--attn-topk', type=float, default=None, help='Top-K 百分比 (0~100)')
    parser.add_argument('--attn-threshold', type=float, default=None, help='二值化阈值 [0,1]')
    parser.add_argument('--attn-gamma', type=float, default=None, help='注意力幂次增强 gamma')
    parser.add_argument('--attn-dilate', type=int, default=None, help='膨胀核大小(1/3/5)')
    # 可视化导出
    parser.add_argument('--save-masks-dir', type=str, default=None, help='仅保存遮挡后的图像到该目录')

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
        use_api=args.use_api,
        api_type=args.api_type,
        api_key=args.api_key,
        api_model_name=args.api_model,
        llm_model=args.llm_model,
        attn_method=args.attn_method,
        attn_topk_percent=args.attn_topk,
        attn_threshold=args.attn_threshold,
        attn_gamma=args.attn_gamma,
        attn_dilate_kernel=args.attn_dilate,
        save_masks_dir=args.save_masks_dir,
    )

    print(f"结果将实时保存至: {args.output}")
    results = evaluator.evaluate(dataset, output_path=args.output)

    # 简要打印
    print("\n" + "="*50)
    print("Mask Baseline 评估结果（黑块遮挡）")
    print("="*50)
    print(f"隐私保护率: {results['privacy_protection_rate']:.2%}")
    print(f"正常任务准确率: {results['normal_accuracy']:.2%}")
    print("="*50)


if __name__ == "__main__":
    main()


