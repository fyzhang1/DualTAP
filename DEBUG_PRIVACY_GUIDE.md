# Privacy 训练调试指南

## 修改说明

已在 `train_new.py` 中添加了 privacy 任务的详细调试输出功能。

### 修改内容

1. **`compute_privacy_task_loss` 函数**：
   - 添加了 `image_paths` 参数
   - 在处理每张图像时打印图像编号和文件名
   - 打印每个 privacy QA 对的详细信息（问题、期望答案、模型预测、损失值）

2. **`train_step` 函数**：
   - 从 batch 中提取 `image_paths`
   - 将其传递给 `compute_privacy_task_loss`

### 如何启用调试输出

调试输出受以下环境变量控制（在训练前设置）：

```bash
# 启用训练输出打印（默认为1，已启用）
export PRINT_TRAIN_OUTPUTS=1

# 设置打印频率（每多少个 step 打印一次，默认50）
export PRINT_EVERY=50

# 开始训练
python train_new.py
```

### 调试输出示例

当调试启用时，你会看到类似以下的输出：

```
============================================================
[PRIVACY DEBUG][step=50] Image 0 in batch: amazon_5.png
Number of privacy QA pairs: 1
============================================================

[Privacy QA 1/1]
Question: You are a precise information extractor.
Task: Read the provided image(s) and extract ONLY the following fields...
Expected Answer: name: kAteLyn cunNINGHam; emailAddress: thomas.matsumoto3637@aol.com
Model Prediction: name: Katelyn Cunningham; emailAddress: thomas.matsumoto3637@aol.com
Loss (original): 2.3456, NegLoss: -2.3456
------------------------------------------------------------

============================================================
[PRIVACY DEBUG][step=50] Image 1 in batch: amazon_6.png
Number of privacy QA pairs: 1
============================================================

[Privacy QA 1/1]
Question: You are a precise information extractor...
Expected Answer: name: KRyStal mENDoZA; emailAddress: hou_sun@outlook.gov
Model Prediction: name: Krystal Mendoza; emailAddress: hou_sun@outlook.gov
Loss (original): 1.8923, NegLoss: -1.8923
------------------------------------------------------------
```

### 输出内容说明

每次打印包含：
- **分隔线**：清晰标识每个图像的处理
- **图像信息**：batch 中的索引和文件名
- **QA 对数量**：该图像对应的 privacy QA 对数量
- **每个 QA 对的详细信息**：
  - `Question`：privacy 提取问题（长问题会截断显示前200字符）
  - `Expected Answer`：ground truth 答案
  - `Model Prediction`：模型当前的预测结果
  - `Loss (original)`：原始 CE 损失
  - `NegLoss`：取负后的损失（这个值会被用于优化）

### 自定义打印频率

如果想更频繁或更少地打印调试信息：

```bash
# 每10个step打印一次
export PRINT_EVERY=10
python train_new.py

# 每100个step打印一次
export PRINT_EVERY=100
python train_new.py

# 完全禁用打印
export PRINT_TRAIN_OUTPUTS=0
python train_new.py
```

### 注意事项

1. 打印会降低训练速度，建议只在需要调试时启用
2. 长问题会自动截断显示（超过200字符）
3. 每个 step 会打印该 batch 中所有图像的 privacy 信息
4. Normal 任务的详细打印也会同时启用（如需单独控制可进一步修改代码）

## 快速测试

运行一个快速测试来查看输出：

```bash
cd /home/ecs-user/Agent_VLM
export PRINT_TRAIN_OUTPUTS=1
export PRINT_EVERY=1  # 每个step都打印
python train_new.py
```

这样你就可以立即看到每个训练 step 的详细调试信息了。

