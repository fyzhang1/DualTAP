"""
配置文件：包含所有训练和评估的超参数
"""

class Config:
    # 数据相关配置
    data_root = "./data"  # 数据根目录
    image_size = 224  # 图像大小
    
    # 噪声约束
    epsilon = 6.0 / 255.0  # L-infinity 范数约束
    
    # 训练相关配置
    batch_size = 2  # 降低以适应显存
    num_epochs = 50
    learning_rate = 1e-4
    
    # 损失函数权重
    alpha = 3.0  # 正常任务损失权重（增大以保护正常功能）
    beta = 1.0   # 隐私任务损失权重
    
    # 代理 MLLM 配置
    # surrogate_model_name = "llava-hf/llava-1.5-7b-hf"  # 或其他MLLM llava-1.5-7b-hf的效果差的跟屎一样
    # lmms-lab/LLaVA-OneVision-1.5-8B-Instruct；Qwen/Qwen2.5-VL-7B-Instruct
    surrogate_model_name = "OpenGVLab/InternVL2-1B"  
    
    # 保存相关
    checkpoint_dir = "./checkpoints"
    save_interval = 50  # 每N个epoch保存一次
    
    # 设备
    device = "cuda"
    
    # 日志
    log_dir = "./logs"
    
    # 评估
    eval_interval = 1  # 每N个epoch评估一次
    
    # 测试单个app
    test_single_app = "email"  # 设置为None则使用所有app
