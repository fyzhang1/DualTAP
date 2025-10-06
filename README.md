# 对抗性噪声生成器：隐私保护系统

本项目旨在开发一种对抗性噪声生成器，用于保护图像中的个人隐私信息，防止被多模态大模型（MLLM）恶意或无意地提取。

## 项目特点

生成的对抗噪声具有以下特点：

1. **高隐蔽性**：噪声扰动微小，人眼难以察觉。
2. **任务选择性攻击**：能有效干扰 MLLM 对图像中特定隐私信息（如姓名、电话、地址）的识别任务。
3. **主任务无害性**：不影响 MLLM 执行正常的 QA 任务。

## 核心原理

### 系统架构

我们通过训练一个**噪声生成器**网络 $G_\theta$，该网络接收一张原始图像 $X$ 作为输入，并输出一个微小的扰动（噪声）$\delta$：

$$X_{adv} = X + \delta \quad \text{where} \quad \delta = G_\theta(X)$$

扰动受到 L-infinity 范数约束：

$$\|\delta\|_{\infty} = \max(| \delta_{i,j,k} |) \le \epsilon$$

其中 $\epsilon$ 是一个很小的常数（默认 8/255）。

### 双重损失函数

我们设计了一个**双重损失函数**来训练生成器：

$$L_{total} = \alpha \cdot L_{normal} + \beta \cdot L_{privacy}$$

其中：

- **正常任务损失** $L_{normal}$：保持 MLLM 在正常任务上的性能
  
  $$L_{normal} = \mathbb{E}_{(X_p, Q_n, A_n) \sim D_{normal}} \left[ \mathcal{L}_{CE} \left( M_{surrogate}(X_{adv}, Q_n), A_n \right) \right]$$

- **隐私任务损失** $L_{privacy}$：使 MLLM 在隐私任务上出错
  
  $$L_{privacy} = - \mathbb{E}_{(X_p, Q_p, A_p) \sim D_{privacy}} \left[ \mathcal{L}_{CE} \left( M_{surrogate}(X_{adv}, Q_p), A_p \right) \right]$$

通过最小化负交叉熵，我们迫使模型的预测远离真实答案。

## 项目结构

```
.
├── config.py              # 配置文件
├── generator.py           # 噪声生成器网络（U-Net）
├── dataset.py             # 数据集加载器
├── train.py               # 训练脚本
├── evaluate.py            # 评估脚本
├── requirements.txt       # 依赖包
├── data/                  # 数据目录
│   ├── app1/              # 应用1（如微信）
│   │   ├── images/        # 图像文件
│   │   │   ├── img1.jpg
│   │   │   └── img2.jpg
│   │   ├── privacy_qa.json    # 隐私任务QA对
│   │   └── normal_qa.json     # 正常任务QA对
│   ├── app2/              # 应用2（如支付宝）
│   │   └── ...
│   └── ...
├── checkpoints/           # 模型检查点
└── logs/                  # 训练日志
```

## 数据格式

### 目录结构

数据按照应用（app）进行组织，每个应用对应一个子目录：

```
data/
├── wechat/           # 微信截图
├── alipay/           # 支付宝截图
├── taobao/           # 淘宝截图
└── ...
```

### QA 数据格式

每个应用目录下包含三个部分：

1. **images/** 目录：存放图像文件
2. **privacy_qa.json**：隐私任务 QA 对
3. **normal_qa.json**：正常任务 QA 对

#### privacy_qa.json 示例

```json
{
  "img1.jpg": [
    {
      "question": "截图中的人名是什么？",
      "answer": "Tom"
    },
    {
      "question": "电话号码是多少？",
      "answer": "138xxxx1234"
    }
  ],
  "img2.jpg": [
    {
      "question": "地址是什么？",
      "answer": "北京市朝阳区"
    }
  ]
}
```

#### normal_qa.json 示例

```json
{
  "img1.jpg": [
    {
      "question": "这是什么应用？",
      "answer": "微信"
    },
    {
      "question": "界面上有几个按钮？",
      "answer": "3个"
    }
  ],
  "img2.jpg": [
    {
      "question": "屏幕上显示的是什么页面？",
      "answer": "聊天页面"
    }
  ]
}
```

## 安装

### 环境要求

- Python 3.8+
- CUDA 11.0+（用于 GPU 加速）

### 安装依赖

```bash
pip install -r requirements.txt
```

## 使用方法

### 1. 准备数据

按照上述格式准备数据，将其放置在 `data/` 目录下。

### 2. 配置参数

编辑 `config.py` 文件，设置相关参数：

```python
class Config:
    # 数据相关
    data_root = "./data"
    image_size = 224
    
    # 噪声约束
    epsilon = 8.0 / 255.0
    
    # 训练参数
    batch_size = 4
    num_epochs = 50
    learning_rate = 1e-4
    
    # 损失权重
    alpha = 1.0  # 正常任务损失权重
    beta = 1.0   # 隐私任务损失权重
    
    # 代理 MLLM
    surrogate_model_name = "llava-hf/llava-1.5-7b-hf"
    
    # 其他
    checkpoint_dir = "./checkpoints"
    log_dir = "./logs"
    device = "cuda"
```

### 3. 训练模型

运行训练脚本：

```bash
python train.py
```

训练过程中会：
- 在每个 epoch 结束时打印损失信息
- 定期保存检查点到 `checkpoints/` 目录
- 保存训练历史到 `logs/train_history.json`

### 4. 评估模型

使用训练好的模型进行评估：

```bash
python evaluate.py --checkpoint checkpoints/generator_epoch_50.pth --output eval_results.json
```

评估指标包括：
- **隐私保护率**：MLLM 在对抗样本上回答隐私问题的错误率（越高越好）
- **正常任务准确率**：MLLM 在对抗样本上回答正常问题的准确率（越高越好）

## 核心模块说明

### 1. 噪声生成器（generator.py）

基于 U-Net 架构的深度神经网络，用于生成对抗性噪声。

**关键方法**：
- `forward(x)`：生成扰动 $\delta$
- `generate_adversarial(x)`：生成对抗样本 $X_{adv} = X + \delta$

### 2. 数据集加载器（dataset.py）

加载按应用分类的隐私数据和正常任务数据。

**关键类**：
- `PrivacyProtectionDataset`：数据集类
- `collate_fn`：自定义批处理函数

### 3. 训练器（train.py）

实现双重损失函数训练。

**关键类**：
- `DualLossTrainer`：训练器类
  - `compute_mllm_loss()`：计算 MLLM 交叉熵损失
  - `train_epoch()`：训练一个 epoch
  - `train()`：完整训练流程

### 4. 评估器（evaluate.py）

评估模型性能。

**关键类**：
- `Evaluator`：评估器类
  - `query_mllm()`：查询 MLLM
  - `check_answer_correctness()`：检查答案正确性
  - `evaluate()`：完整评估流程

## 注意事项

1. **显存需求**：由于需要加载大型 MLLM，建议使用至少 16GB 显存的 GPU。
2. **代理模型选择**：可以在 `config.py` 中更换其他开源 MLLM，如 InstructBLIP、MiniGPT-4 等。
3. **损失权重调整**：根据实际需求调整 `alpha` 和 `beta`，平衡隐私保护和正常功能。
4. **epsilon 调整**：较大的 $\epsilon$ 会产生更明显的扰动，但隐私保护效果更好。

## 预期效果

训练成功后，对抗样本应具有：
- **隐私保护率** > 80%（MLLM 无法正确识别隐私信息）
- **正常任务准确率** > 90%（MLLM 正常功能基本不受影响）
- **视觉质量**：人眼难以察觉扰动

## 扩展方向

1. **迁移性测试**：在其他未见过的 MLLM 上测试攻击效果
2. **鲁棒性增强**：对图像变换（缩放、旋转、压缩）保持鲁棒
3. **通用性提升**：训练一个通用生成器，适用于多种类型的图像

## 引用

如果您使用了本项目，请引用：

```
@misc{privacy-protection-adversarial-noise,
  title={Adversarial Noise Generator for Privacy Protection against MLLMs},
  author={Your Name},
  year={2025}
}
```

## 许可证

MIT License

