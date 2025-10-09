"""
设置随机种子以保证训练可复现的示例代码
可以将这段代码添加到 train_new.py 的 main() 函数开头
"""

import torch
import numpy as np
import random


def set_seed(seed=42):
    """
    设置所有随机种子以保证实验可复现性
    
    Args:
        seed: 随机种子值，默认42
    """
    # Python 内置的 random 模块
    random.seed(seed)
    
    # NumPy
    np.random.seed(seed)
    
    # PyTorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # 如果使用多GPU
    
    # 确保 CUDA 操作的确定性
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    print(f"已设置随机种子为: {seed}")


# 在 train_new.py 的 main() 函数中使用示例：
"""
def main():
    # 首先设置随机种子
    set_seed(42)  # 或者使用任何你想要的种子值
    
    # 加载配置
    config = Config()
    
    # ... 其余训练代码 ...
"""

