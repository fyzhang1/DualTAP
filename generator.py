import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = nn.functional.pad(x1, [diffX // 2, diffX - diffX // 2,
                                     diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class NoiseGenerator(nn.Module):
    """
    基于 U-Net 的噪声生成器
    输入：原始图像 X (C, H, W)
    输出：扰动 δ (C, H, W)，其中 ||δ||_∞ ≤ ε
    """
    def __init__(self, in_channels=3, out_channels=3, epsilon=8.0/255.0,
                 attn_gamma=1.0, attn_threshold=0.0, attn_topk_percent=0.0, attn_mix=1.0,
                 attn_dilate_kernel=1, attn_renorm=False, attn_as_epsilon=False):
        super(NoiseGenerator, self).__init__()
        self.epsilon = epsilon
        self.attn_gamma = attn_gamma
        self.attn_threshold = attn_threshold
        self.attn_topk_percent = attn_topk_percent
        self.attn_mix = attn_mix
        self.attn_dilate_kernel = attn_dilate_kernel
        self.attn_renorm = attn_renorm
        self.attn_as_epsilon = attn_as_epsilon
        
        # 编码器
        self.inc = DoubleConv(in_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 1024)
        
        # 解码器
        self.up1 = Up(1024, 512)
        self.up2 = Up(512, 256)
        self.up3 = Up(256, 128)
        self.up4 = Up(128, 64)
        
        # 输出层
        self.outc = nn.Conv2d(64, out_channels, kernel_size=1)
        self.tanh = nn.Tanh()

    def shape_attention_map(self, attention_map, target_size, out_channels):
        """
        对外暴露的注意力整形函数：
        - clamp 到 [0,1]
        - 插值到目标尺寸
        - gamma 增强
        - 可选膨胀（max-pool）
        - Top-k 或阈值二值化（Top-k 优先）
        - 与全局混合：m*attn + (1-m)
        - 通道广播或汇聚到 out_channels
        返回形状为 [B, out_channels, H, W] 的张量，范围 [0,1]
        """
        attn = attention_map.clamp(0.0, 1.0)
        attn = nn.functional.interpolate(attn, size=target_size, mode='bilinear', align_corners=False)
        if self.attn_gamma != 1.0:
            attn = attn.pow(self.attn_gamma)
        if self.attn_dilate_kernel and self.attn_dilate_kernel > 1:
            k = int(self.attn_dilate_kernel)
            pad = k // 2
            attn = nn.functional.max_pool2d(attn, kernel_size=k, stride=1, padding=pad)
        if self.attn_topk_percent and self.attn_topk_percent > 0.0:
            b, c1, h, w = attn.shape
            flat = attn.view(b, -1)
            k = (flat.shape[1] * min(100.0, max(0.0, self.attn_topk_percent)) / 100.0)
            k = max(1, int(k))
            topk_vals, _ = flat.topk(k, dim=1)
            thr = topk_vals[:, -1].view(b, 1, 1, 1)
            attn = (attn >= thr).float()
        elif self.attn_threshold and self.attn_threshold > 0.0:
            attn = (attn >= self.attn_threshold).float()
        if self.attn_mix != 1.0:
            attn = self.attn_mix * attn + (1.0 - self.attn_mix)
            attn = attn.clamp(0.0, 1.0)
        # 调整通道数
        if out_channels == 1 and attn.shape[1] > 1:
            attn = attn.mean(dim=1, keepdim=True)
        if out_channels > 1 and attn.shape[1] == 1:
            attn = attn.repeat(1, out_channels, 1, 1)
        # 最终保障尺寸
        attn = nn.functional.interpolate(attn, size=target_size, mode='bilinear', align_corners=False)
        return attn

    def forward(self, x, attention_map=None):
        """
        前向传播
        Args:
            x: 输入图像，形状为 (B, C, H, W)，值域 [0, 1]
            attention_map: 可选注意力图 (B, 1, H, W) 或 (B, 3, H, W)，值域 [0,1]
        Returns:
            delta: 生成的扰动，形状为 (B, C, H, W)，值域 [-ε, ε]
        """
        # 编码器
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        
        # 解码器
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        
        # 输出：先用 tanh 将输出限制在 [-1, 1]
        delta_raw = self.outc(x)
        delta_raw = self.tanh(delta_raw)

        if attention_map is None:
            # 无注意力：全局统一 ε 预算
            delta = delta_raw * self.epsilon
            return delta

        # 使用注意力进行空间调制
        if self.attn_as_epsilon:
            # 将注意力作为“每像素 ε 预算”（更强的高置信度噪声）
            attn1 = self.shape_attention_map(attention_map, target_size=delta_raw.shape[-2:], out_channels=1)
            attn_c = attn1.repeat(1, delta_raw.shape[1], 1, 1)
            epsilon_map = self.epsilon * attn_c
            delta = delta_raw * epsilon_map
            return delta
        else:
            # 传统门控：先缩放到 ε，再按注意力乘法
            delta = delta_raw * self.epsilon
            mask = self.shape_attention_map(attention_map, target_size=delta.shape[-2:], out_channels=delta.shape[1])
            delta = delta * mask
            if self.attn_renorm:
                denom = mask.abs().amax(dim=(1,2,3), keepdim=True) + 1e-8
                delta = delta / denom
                delta = delta.clamp(-self.epsilon, self.epsilon)
            return delta

    def generate_adversarial(self, x):
        """
        生成对抗样本
        Args:
            x: 原始图像，形状为 (B, C, H, W)，值域 [0, 1]
        Returns:
            x_adv: 对抗样本，形状为 (B, C, H, W)，值域 [0, 1]
        """
        delta = self.forward(x)
        x_adv = x + delta
        # 确保对抗样本在 [0, 1] 范围内
        x_adv = torch.clamp(x_adv, 0.0, 1.0)
        return x_adv
