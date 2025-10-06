# 快速开始指南

本指南帮助您快速开始使用对抗性噪声生成器项目。

## 步骤 1：环境配置

### 1.1 安装依赖

```bash
pip install -r requirements.txt
```

**注意**：
- 需要 CUDA 11.0+ 用于 GPU 加速
- 需要至少 16GB 显存
- 首次运行会下载大型 MLLM 模型（约 13GB）

### 1.2 初始化项目

```bash
python setup.py
```

这会创建以下目录：
- `data/` - 存放训练数据
- `checkpoints/` - 存放模型检查点
- `logs/` - 存放训练日志

## 步骤 2：准备数据

### 2.1 创建数据模板

```bash
python utils.py
```

这会在 `data/` 目录下创建示例应用目录结构：
```
data/
├── wechat/
│   ├── images/
│   ├── privacy_qa.json
│   └── normal_qa.json
├── alipay/
│   └── ...
└── taobao/
    └── ...
```

### 2.2 添加图像

将您的手机截图放入对应应用的 `images/` 目录：

```bash
# 示例
data/
└── wechat/
    └── images/
        ├── screenshot_001.jpg
        ├── screenshot_002.jpg
        └── screenshot_003.jpg
```

### 2.3 编辑 QA 文件

#### privacy_qa.json（隐私任务）

包含需要保护的隐私信息问答对：

```json
{
  "screenshot_001.jpg": [
    {
      "question": "截图中的人名是什么？",
      "answer": "张三"
    },
    {
      "question": "电话号码是多少？",
      "answer": "13800138000"
    },
    {
      "question": "地址在哪里？",
      "answer": "北京市朝阳区xx路xx号"
    }
  ],
  "screenshot_002.jpg": [
    {
      "question": "这是谁的微信？",
      "answer": "李四"
    }
  ]
}
```

**注意事项**：
- 问题应该直接针对图像中的隐私信息
- 答案应该是图像中实际出现的内容
- 每张图像至少要有 1 个隐私 QA 对

#### normal_qa.json（正常任务）

包含与隐私无关的正常问答对：

```json
{
  "screenshot_001.jpg": [
    {
      "question": "这是什么应用？",
      "answer": "微信"
    },
    {
      "question": "屏幕上有几个对话框？",
      "answer": "1个"
    },
    {
      "question": "界面颜色是什么？",
      "answer": "白色"
    }
  ],
  "screenshot_002.jpg": [
    {
      "question": "这是聊天界面还是通讯录界面？",
      "answer": "聊天界面"
    }
  ]
}
```

**注意事项**：
- 问题应该与隐私信息无关
- 关注界面元素、布局、功能等
- 每张图像至少要有 1 个正常 QA 对

## 步骤 3：配置参数

编辑 `config.py` 文件，根据您的需求调整参数：

```python
class Config:
    # 数据配置
    data_root = "./data"        # 数据目录
    image_size = 224            # 图像尺寸
    
    # 噪声配置
    epsilon = 8.0 / 255.0       # 噪声强度（建议 8/255）
    
    # 训练配置
    batch_size = 4              # 批大小（显存不足可减小）
    num_epochs = 50             # 训练轮数
    learning_rate = 1e-4        # 学习率
    
    # 损失权重
    alpha = 1.0                 # 正常任务权重
    beta = 1.0                  # 隐私任务权重
    
    # 模型配置
    surrogate_model_name = "llava-hf/llava-1.5-7b-hf"
    
    # 其他
    checkpoint_dir = "./checkpoints"
    log_dir = "./logs"
    save_interval = 5           # 每N个epoch保存一次
    device = "cuda"
```

**参数说明**：

- **epsilon**：控制噪声强度
  - 较小（4/255）：噪声更隐蔽，但效果可能较弱
  - 中等（8/255）：平衡性能和隐蔽性（推荐）
  - 较大（16/255）：效果更好，但可能可见

- **alpha 和 beta**：控制两个任务的权重
  - alpha > beta：优先保持正常功能
  - alpha < beta：优先保护隐私
  - alpha = beta：平衡两者（推荐初始值）

- **batch_size**：根据显存调整
  - 24GB 显存：batch_size = 8
  - 16GB 显存：batch_size = 4
  - 12GB 显存：batch_size = 2

## 步骤 4：开始训练

### 4.1 运行训练脚本

```bash
python train.py
```

训练过程中会显示：
```
加载数据集...
成功加载 10 个数据项
初始化噪声生成器...
加载代理 MLLM: llava-hf/llava-1.5-7b-hf
开始训练，共 50 个 epoch

===== Epoch 1/50 =====
Epoch 1 [████████████████████] 100%
  L_total:   2.3456
  L_normal:  1.2345
  L_privacy: 1.1111

===== Epoch 2/50 =====
...
```

### 4.2 监控训练

查看训练历史：

```bash
# 查看日志文件
cat logs/train_history.json
```

**正常训练的特征**：
- L_normal 逐渐下降并稳定
- L_privacy 逐渐下降（注意是负值）
- L_total 整体呈下降趋势

### 4.3 中断和恢复

训练可以随时中断（Ctrl+C），检查点已保存在 `checkpoints/` 目录。

要恢复训练，需要修改 `train.py` 添加加载检查点的代码（或重新开始）。

## 步骤 5：评估模型

训练完成后，评估模型效果：

```bash
python evaluate.py \
  --checkpoint /home/ecs-user/Agent_VLM/checkpoints/generator_epoch_50.pth \
  --output ./eval_results/eval_results_qwen.json
```

评估结果示例：
```
加载数据集...
成功加载 10 个数据项
加载噪声生成器...
开始评估...
[████████████████████] 100%

==================================================
评估结果
==================================================
隐私保护率: 85.50%
  - 隐私问题总数: 20
  - 成功保护（回答错误）: 17

正常任务准确率: 92.00%
  - 正常问题总数: 25
  - 回答正确: 23
==================================================

详细结果已保存至: eval_results.json
```

**理想结果**：
- 隐私保护率 > 80%（越高越好）
- 正常任务准确率 > 90%（越高越好）

## 步骤 6：生成对抗样本

使用训练好的模型保护新图像：

```bash
python inference.py \
  --image /home/ecs-user/Agent_VLM/data/amazon/images/amazon_0.png \
  --checkpoint /home/ecs-user/Agent_VLM/checkpoints/generator_epoch_50.pth \
  --output protected_image.jpg
```

输出：
```
加载噪声生成器...
处理图像: data/wechat/images/screenshot_001.jpg
对抗样本已保存至: protected_image.jpg

噪声统计:
  最大值: 0.031373
  平均值: 0.012456
  epsilon: 0.031373
```

现在 `protected_image.jpg` 包含对抗噪声，可以用于隐私保护。

## 常见问题排查

### 问题 1：数据集为空

**错误信息**：
```
成功加载 0 个数据项
错误: 数据集为空，请检查数据目录
```

**解决方法**：
1. 检查 `data/` 目录结构是否正确
2. 确保每个应用目录下有 `images/`、`privacy_qa.json` 和 `normal_qa.json`
3. 确保 JSON 文件格式正确
4. 确保图像文件存在

### 问题 2：CUDA 内存不足

**错误信息**：
```
RuntimeError: CUDA out of memory
```

**解决方法**：
1. 减小 `batch_size`（config.py）
2. 减小 `image_size`（config.py）
3. 使用 CPU 训练：`device = "cpu"`（很慢）

### 问题 3：模型下载失败

**错误信息**：
```
HTTPError: 403 Client Error: Forbidden
```

**解决方法**：
1. 设置 HuggingFace 镜像：
```bash
export HF_ENDPOINT=https://hf-mirror.com
```
2. 或手动下载模型到本地，修改 `surrogate_model_name` 为本地路径

### 问题 4：训练损失不下降

**可能原因**：
1. 学习率不合适
2. alpha 和 beta 权重不平衡
3. 数据质量问题

**解决方法**：
1. 调整 `learning_rate`（尝试 1e-3 或 1e-5）
2. 调整 `alpha` 和 `beta`
3. 检查数据标注是否准确

### 问题 5：隐私保护率低

**解决方法**：
1. 增大 `beta`（提高隐私损失权重）
2. 增大 `epsilon`（允许更大扰动）
3. 增加训练轮数
4. 检查隐私 QA 是否准确

### 问题 6：正常任务受影响

**解决方法**：
1. 增大 `alpha`（提高正常任务权重）
2. 减小 `epsilon`（减小扰动）
3. 增加正常任务 QA 数据量

## 最佳实践

### 数据准备
1. **多样性**：每个应用收集多种场景的截图
2. **质量**：确保图像清晰，QA 标注准确
3. **平衡**：隐私 QA 和正常 QA 数量相当
4. **覆盖**：涵盖所有需要保护的隐私类型

### 训练策略
1. **初始训练**：使用 alpha = beta = 1.0
2. **观察结果**：根据评估调整权重
3. **渐进训练**：可以先用较小 epsilon 训练，再增大
4. **定期评估**：每 5-10 个 epoch 评估一次

### 参数调整
1. **epsilon**：从 8/255 开始，根据效果调整
2. **alpha/beta**：根据实际需求倾向调整
3. **learning_rate**：通常 1e-4 效果较好
4. **num_epochs**：至少 30-50 轮

## 下一步

完成基础训练后，您可以：

1. **测试迁移性**：在其他 MLLM 上测试效果
2. **扩展数据**：添加更多应用和场景
3. **优化模型**：尝试不同的生成器架构
4. **实际部署**：集成到应用中使用

## 获取帮助

如果遇到问题，可以：

1. 查看 `PROJECT_OVERVIEW.md` 了解详细技术细节
2. 查看 `README.md` 了解项目原理
3. 检查代码注释
4. 查看训练日志和评估结果

祝您使用顺利！

