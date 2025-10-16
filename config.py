"""
配置文件：包含所有训练和评估的超参数
"""

class Config:
    # 数据相关配置
    data_root = "./data"  # 数据根目录
    image_size = 448  # 图像大小
    
    # 噪声约束
    epsilon = 128.0 / 255.0
    
    # 训练相关配置
    batch_size = 4  # 降低以适应显存
    num_epochs = 10
    learning_rate = 1e-4
    
    # 损失函数权重
    alpha = 1.0  # 正常任务损失权重
    beta = 1.0   # 隐私任务损失权重
    
    # 代理 MLLM 配置
    # surrogate_model_name = "llava-hf/llava-1.5-7b-hf"  # 或其他MLLM llava-1.5-7b-hf的效果差的跟屎一样
    # openbmb/MiniCPM-V-2_6；Qwen/Qwen2.5-VL-7B-Instruct "OpenGVLab/InternVL3_5-2B"   openbmb/MiniCPM-V-4_5 "llava-hf/llava-onevision-qwen2-7b-ov-hf"
    surrogate_model_name = "Qwen/Qwen2.5-VL-7B-Instruct"
    
    # 保存相关
    checkpoint_dir = "./checkpoints"
    save_interval = 50  # 每N个epoch保存一次
    # 注意力可视化
    save_attention = True
    attention_dir = "./logs_eot/attn"
    # 注意力提取方法：'pixel_grad' | 'xattn_grad' | 'clip_text_match' | 'contrast_*'
    # 对比注意力用法示例：'contrast_clip_text_match' / 'contrast_xattn_grad' / 'contrast_pixel_grad'
    attn_method = "contrast_pixel_grad"
    # 注意力整形参数（用于生成器侧）
    attn_gamma = 2.0           # >1 提升热点对比度，增大以提高区分度
    attn_threshold = 0.75      # >0 时按阈值二值化，范围 [0,1] - 提高阈值
    attn_topk_percent = 60    # >0 时按比例保留前 k%（0~100），优先于阈值 - 大幅降低
    attn_mix = 0.9             # 与全局权重混合：m*attn + (1-m)
    # 注意力扩张与重归一
    attn_dilate_kernel = 3     # 3/5 进行邻域扩张；1 表示不扩张 - 禁用膨胀
    attn_renorm = True        # 乘注意力后是否重归一到 ±epsilon
    # 注意力用作"每像素 ε 预算"，高置信度处更强噪声
    attn_as_epsilon = True
    # 训练时对非注意力区域的噪声惩罚与 TV 正则
    noise_outside_weight = 0.05
    tv_weight = 0.01
    
    # 设备
    device = "cuda"
    
    # 日志
    log_dir = "./logs"
    
    # 评估
    eval_interval = 1  # 每N个epoch评估一次
    
    # 测试单个app
    test_single_app = None # 设置为None则使用所有app
