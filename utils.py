"""
工具函数
"""

import os
import time
import torch
import numpy as np
from PIL import Image
from typing import Tuple, Dict, Optional

_bert_scorer = None
_st_model = None
_rouge = None
_sacrebleu = None


_psutil = None

def _lazy_import_bert_scorer():
    global _bert_scorer
    if _bert_scorer is None:
        try:
            from bert_score import BERTScorer 
            _bert_scorer = BERTScorer(lang="en", model_type="microsoft/deberta-base-mnli", rescale_with_baseline=True)
        except Exception:
            _bert_scorer = False
    return _bert_scorer

def _lazy_import_st_model():
    global _st_model
    if _st_model is None:
        try:
            from sentence_transformers import SentenceTransformer
            _st_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        except Exception:
            _st_model = False
    return _st_model

def _lazy_import_rouge():
    global _rouge
    if _rouge is None:
        try:
            from rouge_score import rouge_scorer  # type: ignore
            _rouge = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
        except Exception:
            _rouge = False
    return _rouge

def _lazy_import_sacrebleu():
    global _sacrebleu
    if _sacrebleu is None:
        try:
            import sacrebleu  # type: ignore
            _sacrebleu = sacrebleu
        except Exception:
            _sacrebleu = False
    return _sacrebleu


def _lazy_import_psutil():
    global _psutil
    if _psutil is None:
        try:
            import psutil  # type: ignore
            _psutil = psutil
        except Exception:
            _psutil = False
    return _psutil


def tensor_to_numpy(tensor):
    if tensor.dim() == 4:
        tensor = tensor.squeeze(0)
    return tensor.permute(1, 2, 0).cpu().numpy()


def numpy_to_tensor(array, device='cpu'):
    tensor = torch.from_numpy(array).permute(2, 0, 1).float()
    return tensor.to(device)


def calculate_psnr(img1, img2):
    mse = torch.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    
    max_pixel = 1.0
    psnr = 20 * torch.log10(max_pixel / torch.sqrt(mse))
    return psnr.item()


def calculate_linf_norm(tensor):
    return torch.max(torch.abs(tensor)).item()


def visualize_noise(noise, scale=10.0):
    noise_vis = (noise * scale + 0.5).clamp(0, 1)
    from torchvision.transforms import ToPILImage
    to_pil = ToPILImage()
    return to_pil(noise_vis)


def compute_text_metrics(pred_text: str, true_text: str) -> Dict[str, Optional[float]]:
    pred = (pred_text or "").strip()
    ref = (true_text or "").strip()
    if not pred or not ref:
        return {
            "bertscore_f1": 0.0 if pred or ref else None,
            "cosine_sim": 0.0 if pred or ref else None,
            "bleu": 0.0 if pred or ref else None,
            "rouge_l": 0.0 if pred or ref else None,
        }

    # BERTScore
    bert_f1: Optional[float]
    scorer = _lazy_import_bert_scorer()
    if scorer is False:
        bert_f1 = None
    else:
        try:
            P, R, F1 = scorer.score([pred], [ref])  # tensors
            bert_f1 = float(F1.mean().item())
        except Exception:
            bert_f1 = None

    # Cosine similarity via sentence-transformers
    cosine_sim: Optional[float]
    st = _lazy_import_st_model()
    if st is False:
        cosine_sim = None
    else:
        try:
            import numpy as _np  # local alias
            pred_emb = st.encode([pred], normalize_embeddings=True)
            ref_emb = st.encode([ref], normalize_embeddings=True)
            cosine_sim = float((_np.asarray(pred_emb) @ _np.asarray(ref_emb).T).item())
        except Exception:
            cosine_sim = None

    # BLEU via sacrebleu (sentence-level)
    bleu: Optional[float]
    sb = _lazy_import_sacrebleu()
    if sb is False:
        bleu = None
    else:
        try:
            bleu = float(sb.sentence_bleu(pred, [ref]).score / 100.0)
        except Exception:
            bleu = None

    # ROUGE-L via rouge-score
    rouge_l: Optional[float]
    rs = _lazy_import_rouge()
    if rs is False:
        rouge_l = None
    else:
        try:
            scores = rs.score(ref, pred)
            rouge_l = float(scores["rougeL"].fmeasure)
        except Exception:
            rouge_l = None

    return {
        "bertscore_f1": bert_f1,
        "cosine_sim": cosine_sim,
        "bleu": bleu,
        "rouge_l": rouge_l,
    }


class Timer:
    def __enter__(self):
        self._start = time.perf_counter()
        self.ms = 0.0
        return self

    def __exit__(self, exc_type, exc, tb):
        self.ms = (time.perf_counter() - self._start) * 1000.0


def get_cpu_memory_mb() -> Optional[float]:
    psutil = _lazy_import_psutil()
    try:
        if psutil and psutil is not False:
            process = psutil.Process(os.getpid())
            rss = float(process.memory_info().rss) / (1024.0 * 1024.0)
            return rss
    except Exception:
        pass
    try:
        import resource  # type: ignore
        usage = resource.getrusage(resource.RUSAGE_SELF)
        # On Linux, ru_maxrss is in kilobytes
        rss_mb = float(getattr(usage, "ru_maxrss", 0.0)) / 1024.0
        return rss_mb
    except Exception:
        return None


def get_cuda_memory_stats(device=None) -> Dict[str, Optional[float]]:
    if not torch.cuda.is_available():
        return {
            "current_allocated_mb": None,
            "current_reserved_mb": None,
            "max_allocated_mb": None,
            "max_reserved_mb": None,
        }
    try:
        dev = device if device is not None else torch.device("cuda")
        allocated = torch.cuda.memory_allocated(dev) / (1024.0 * 1024.0)
        reserved = torch.cuda.memory_reserved(dev) / (1024.0 * 1024.0)
        max_alloc = torch.cuda.max_memory_allocated(dev) / (1024.0 * 1024.0)
        max_res = torch.cuda.max_memory_reserved(dev) / (1024.0 * 1024.0)
        return {
            "current_allocated_mb": float(allocated),
            "current_reserved_mb": float(reserved),
            "max_allocated_mb": float(max_alloc),
            "max_reserved_mb": float(max_res),
        }
    except Exception:
        return {
            "current_allocated_mb": None,
            "current_reserved_mb": None,
            "max_allocated_mb": None,
            "max_reserved_mb": None,
        }


def create_data_template():
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
        

if __name__ == "__main__":
    # 创建数据目录模板
    create_data_template()

