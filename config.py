"""
配置文件：包含所有训练和评估的超参数
"""

class Config:
    # 数据相关配置
    data_root = "./data"  # 数据根目录
    image_size = 448  # 图像大小
    # 按每个 app 进行数据划分比例（4:1 -> 0.8）
    train_split_ratio = 0.0
    
    # 噪声约束
    epsilon = 128.0 / 255.0
    
    # 训练相关配置
    batch_size = 4  # 降低以适应显存
    num_epochs =8
    learning_rate = 1e-4
    
    # 损失函数权重
    alpha = 1.0  # 正常任务损失权重
    beta = 1.0   # 隐私任务损失权重
    
    # 代理 MLLM 配置
    # xlangai/OpenCUA-7B；ByteDance-Seed/UI-TARS-7B-SFT"; Qwen/Qwen2.5-VL-7B-Instruct "OpenGVLab/InternVL3_5-2B"   openbmb/MiniCPM-V-4_5 "llava-hf/llava-onevision-qwen2-7b-ov-hf"
    surrogate_model_name = "OpenGVLab/InternVL3_5-2B"
    # 保存相关
    checkpoint_dir = "./checkpoints"
    save_interval = 60  # 每N个epoch保存一次
    # 注意力可视化
    # 是否在训练中使用注意力图；设为 False 即完全不使用注意力，仅训练纯噪声生成
    use_attention = True
    save_attention = True
    attention_dir = "./logs_eot/attn"
    # 注意力提取方法：'pixel_grad' | 'xattn_grad' | 'clip_text_match' | 'contrast_*'
    # 对比注意力用法示例：'contrast_clip_text_match' / 'contrast_xattn_grad' / 'contrast_pixel_grad'
    attn_method = "contrast_pixel_grad"
    # 注意力整形参数（用于生成器侧）
    attn_gamma = 4.0           # >1 提升热点对比度，增大以提高区分度（从2.0提升至4.0）
    attn_threshold = 0.85      # >0 时按阈值二值化，范围 [0,1]（从0.75提升至0.85）
    attn_topk_percent = 60      # >0 时按比例保留前 k%（0~100）（从10降至5）
    attn_mix = 0.95             # 【关键修复】与全局权重混合：m*attn + (1-m)（从0.9改为1.0，消除全局baseline）
    # 注意力扩张与重归一
    attn_dilate_kernel = 3     # 3/5 进行邻域扩张；1 表示不扩张 - 禁用膨胀
    attn_renorm = True        # 乘注意力后是否重归一到 ±epsilon
    # 注意力融合方式：'film' | 'mutile'（film 为推荐：不再使用后乘法）
    attn_integration = 'film'
    # FiLM 模块宽度与强度
    film_hidden = 32
    film_strength = 1.0
    # 旧模式：注意力用作"每像素 ε 预算"（仅在 attn_integration!='film' 时生效）
    attn_as_epsilon = False
    
    # 设备
    device = "cuda"
    
    # 日志
    log_dir = "./logs"
    
    # 评估
    eval_interval = 1  # 每N个epoch评估一次
    
    # 可视化
    vis_interval = 10  # 每N个batch保存一次训练可视化图像（0表示不保存）
    vis_noise_only = False  # 仅保存噪声图（灰度网格），不含原图/注意力/差异
    
    # 测试单个app
    test_single_app = "real" # 设置为None则使用所有app
