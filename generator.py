import torch
import torch.nn as nn
import torch.nn.functional as F


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
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.reduce = nn.Conv2d(in_channels, in_channels // 2, kernel_size=1)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x1 = self.reduce(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = nn.functional.pad(x1, [diffX // 2, diffX - diffX // 2,
                                     diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class SpatialFiLM(nn.Module):
    def __init__(self, out_channels: int, hidden_channels: int = 32):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, hidden_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, 2 * out_channels, kernel_size=3, padding=1)
        )

        nn.init.zeros_(self.encoder[-1].weight)
        nn.init.zeros_(self.encoder[-1].bias)

    def forward(self, attn_map: torch.Tensor, target_hw) -> (torch.Tensor, torch.Tensor):

        attn_resized = nn.functional.interpolate(attn_map, size=target_hw, mode='bilinear', align_corners=False)
        film_params = self.encoder(attn_resized) 
        gamma_delta, beta = torch.chunk(film_params, chunks=2, dim=1)
        return gamma_delta, beta


def apply_film(features: torch.Tensor,
               gamma_delta: torch.Tensor,
               beta: torch.Tensor,
               strength: float = 1.0) -> torch.Tensor:

    scaled_gamma = 1.0 + strength * torch.tanh(gamma_delta)
    scaled_beta = strength * torch.tanh(beta)
    return scaled_gamma * features + scaled_beta


class NoiseGenerator(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, epsilon=8.0/255.0,
                 attn_gamma=1.0, attn_threshold=0.0, attn_topk_percent=0.0, attn_mix=1.0,
                 attn_dilate_kernel=1, attn_renorm=False, attn_as_epsilon=False,
                 attn_integration: str = 'film', film_hidden: int = 32, film_strength: float = 1.0,
                 noise_center_dc: bool = True, noise_balance_by_std: bool = True,
                 output_gate_with_attention: bool = True):
        super(NoiseGenerator, self).__init__()
        self.epsilon = epsilon
        self.attn_gamma = attn_gamma
        self.attn_threshold = attn_threshold
        self.attn_topk_percent = attn_topk_percent
        self.attn_mix = attn_mix
        self.attn_dilate_kernel = attn_dilate_kernel
        self.attn_renorm = attn_renorm
        self.attn_as_epsilon = attn_as_epsilon
        self.attn_integration = attn_integration
        self.film_strength = film_strength

        self.output_gate_with_attention = output_gate_with_attention
        self.noise_center_dc = noise_center_dc
        self.noise_balance_by_std = noise_balance_by_std

        self.register_buffer('imagenet_std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        

        self.inc = DoubleConv(in_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 1024)
        

        self.up1 = Up(1024, 512)
        self.up2 = Up(512, 256)
        self.up3 = Up(256, 128)
        self.up4 = Up(128, 64)
        

        self.outc = nn.Conv2d(64, out_channels, kernel_size=1)
        self.tanh = nn.Tanh()


        if self.attn_integration == 'film':
            self.film_inc = SpatialFiLM(out_channels=64, hidden_channels=film_hidden)
            self.film_d1 = SpatialFiLM(out_channels=128, hidden_channels=film_hidden)
            self.film_d2 = SpatialFiLM(out_channels=256, hidden_channels=film_hidden)
            self.film_d3 = SpatialFiLM(out_channels=512, hidden_channels=film_hidden)
            self.film_d4 = SpatialFiLM(out_channels=1024, hidden_channels=film_hidden)
            self.film_u1 = SpatialFiLM(out_channels=512, hidden_channels=film_hidden)
            self.film_u2 = SpatialFiLM(out_channels=256, hidden_channels=film_hidden)
            self.film_u3 = SpatialFiLM(out_channels=128, hidden_channels=film_hidden)
            self.film_u4 = SpatialFiLM(out_channels=64, hidden_channels=film_hidden)

    def shape_attention_map(self, attention_map, target_size, out_channels):
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

        if out_channels == 1 and attn.shape[1] > 1:
            attn = attn.mean(dim=1, keepdim=True)
        if out_channels > 1 and attn.shape[1] == 1:
            attn = attn.repeat(1, out_channels, 1, 1)

        attn = nn.functional.interpolate(attn, size=target_size, mode='bilinear', align_corners=False)
        return attn

    def forward(self, x, attention_map=None):

        x1 = self.inc(x)
        if attention_map is not None and self.attn_integration == 'film':
            gd, bt = self.film_inc(attention_map, x1.shape[-2:])
            x1 = apply_film(x1, gd, bt, strength=self.film_strength)

        x2 = self.down1(x1)
        if attention_map is not None and self.attn_integration == 'film':
            gd, bt = self.film_d1(attention_map, x2.shape[-2:])
            x2 = apply_film(x2, gd, bt, strength=self.film_strength)

        x3 = self.down2(x2)
        if attention_map is not None and self.attn_integration == 'film':
            gd, bt = self.film_d2(attention_map, x3.shape[-2:])
            x3 = apply_film(x3, gd, bt, strength=self.film_strength)

        x4 = self.down3(x3)
        if attention_map is not None and self.attn_integration == 'film':
            gd, bt = self.film_d3(attention_map, x4.shape[-2:])
            x4 = apply_film(x4, gd, bt, strength=self.film_strength)

        x5 = self.down4(x4)
        if attention_map is not None and self.attn_integration == 'film':
            gd, bt = self.film_d4(attention_map, x5.shape[-2:])
            x5 = apply_film(x5, gd, bt, strength=self.film_strength)


        x = self.up1(x5, x4)
        if attention_map is not None and self.attn_integration == 'film':
            gd, bt = self.film_u1(attention_map, x.shape[-2:])
            x = apply_film(x, gd, bt, strength=self.film_strength)

        x = self.up2(x, x3)
        if attention_map is not None and self.attn_integration == 'film':
            gd, bt = self.film_u2(attention_map, x.shape[-2:])
            x = apply_film(x, gd, bt, strength=self.film_strength)

        x = self.up3(x, x2)
        if attention_map is not None and self.attn_integration == 'film':
            gd, bt = self.film_u3(attention_map, x.shape[-2:])
            x = apply_film(x, gd, bt, strength=self.film_strength)

        x = self.up4(x, x1)
        if attention_map is not None and self.attn_integration == 'film':
            gd, bt = self.film_u4(attention_map, x.shape[-2:])
            x = apply_film(x, gd, bt, strength=self.film_strength)
        

        delta_raw = self.outc(x)
        delta_raw = self.tanh(delta_raw)


        if attention_map is None:
            return delta_raw * self.epsilon


        if self.attn_integration == 'film':
            delta = delta_raw * self.epsilon
        if self.attn_as_epsilon:
            attn1 = self.shape_attention_map(attention_map, target_size=delta_raw.shape[-2:], out_channels=1)
            attn_c = attn1.repeat(1, delta_raw.shape[1], 1, 1)
            epsilon_map = self.epsilon * attn_c
            return delta_raw * epsilon_map
        else:
            delta = delta_raw * self.epsilon
            mask = self.shape_attention_map(attention_map, target_size=delta.shape[-2:], out_channels=delta.shape[1])
            delta = delta * mask
            if self.attn_renorm:
                denom = mask.abs().amax(dim=(1,2,3), keepdim=True) + 1e-8
                delta = delta / denom
                delta = delta.clamp(-self.epsilon, self.epsilon)
            return delta

    def generate_adversarial(self, x):

        delta = self.forward(x)
        x_adv = x + delta

        x_adv = torch.clamp(x_adv, 0.0, 1.0)
        return x_adv
