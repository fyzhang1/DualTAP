"""
噪声生成器网络
使用 U-Net 架构生成对抗性噪声
"""

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
    def __init__(self, in_channels=3, out_channels=3, epsilon=8.0/255.0):
        super(NoiseGenerator, self).__init__()
        self.epsilon = epsilon
        
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

    def forward(self, x):
        """
        前向传播
        Args:
            x: 输入图像，形状为 (B, C, H, W)，值域 [0, 1]
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
        
        # 输出：使用 tanh 将输出限制在 [-1, 1]，然后缩放到 [-ε, ε]
        delta = self.outc(x)
        delta = self.tanh(delta) * self.epsilon
        
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
