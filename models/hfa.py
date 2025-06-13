import torch.nn as nn
import torch
import pywt

import torch.nn.functional as F
from functools import partial
from torch.nn import init
from models.dpcd_parts import (Conv_BN_ReLU, CGSU, Encoder_Block, DPFA, Decoder_Block,
                               Changer_channel_exchange, log_feature, ABconv)


def compute_entropy(tensor):
    """
    计算归一化张量的平均信息熵
    参数:
        tensor: shape = [B, C, H, W]，已进行softmax归一化
    返回:
        entropies: shape = [B]，每个batch样本的平均信息熵
    """
    # 避免log(0)
    eps = 1e-12
    entropy_map = -tensor * torch.log(tensor + eps)  # shape: [B, C, H, W]
    entropy_per_pixel = torch.sum(entropy_map, dim=1)  # shape: [B, H, W]
    entropy_mean = torch.mean(entropy_per_pixel.view(tensor.size(0), -1), dim=1)  # shape: [B]
    return entropy_mean  # 每个batch样本的平均信息熵

def HFA_select(A, B):

    # Softmax 归一化（沿通道维度 C=1）
    A_soft = F.softmax(A, dim=1)
    B_soft = F.softmax(B, dim=1)

    # 计算每个样本的信息熵
    entropy_A = compute_entropy(A_soft)  # shape: [16]
    entropy_B = compute_entropy(B_soft)  # shape: [16]

    # 比较每个样本的熵差
    delta_entropy = entropy_A - entropy_B  # shape: [16]

    return delta_entropy

class SAFCB(nn.Module):
    def __init__(self, in_chans, out_chans, embed_dims, device):
        super().__init__()
        self.device = device
        self.fc = in_chans // 2
        self.sc = in_chans

        self.embed_dims = embed_dims

        self.out_chans = out_chans

        self.fft_dim = (-2, -1)
        self.fft_norm = "ortho"

        self.fft_atten = nn.Sequential(
            nn.Conv2d(in_chans, embed_dims, 1, 1, 0),
            nn.BatchNorm2d(embed_dims),
            nn.ReLU(),
            nn.Conv2d(embed_dims, out_chans, 1, 1, 0),
            nn.BatchNorm2d(out_chans),
        )


    def forward(self, x1, x2):
        b, c, h, w = x2.shape
        # lpf = torch.zeros((c, h, w)).to(self.device)
        hpf = torch.ones((c, h, w)).to(self.device)
        R = (h + w) // 32  # 或其他
        for x in range(w):
            for y in range(h):
                if ((x - (w - 1) / 2) ** 2 + (y - (h - 1) / 2) ** 2) < (R ** 2):
                    # lpf[:, y, x] = 1
                    hpf[:, y, x] = 0
        # hpf = 1 - lpf
        if self.select:
            if HFA_select(x1, x2) > 0:
                x2_freq = torch.fft.fftn(x2, dim=(2, 3), norm='ortho')
                x2_freq = torch.fft.fftshift(x2_freq)
            else:
                x2_freq = torch.fft.fftn(x1, dim=(2, 3), norm='ortho')
                x2_freq = torch.fft.fftshift(x2_freq)
        else:
            x2_freq = torch.fft.fftn(x2, dim=(2, 3), norm='ortho')
            x2_freq = torch.fft.fftshift(x2_freq)

        hf = x2_freq * hpf  # high freq

        #高频分量做注意力
        real = hf.real + self.fft_atten(hf.real)
        imag = hf.imag + self.fft_atten(hf.imag)

        hx2_ffted = torch.complex(real, imag)
        ifft_shape_slice = x1.shape[-2:]
        hx2_ffted = torch.abs(torch.fft.ifftn(hx2_ffted,s=ifft_shape_slice, dim=(2, 3),norm='ortho'))


        x1 = torch.mul(x1, hx2_ffted) + x1
        x2 = torch.mul(x2, hx2_ffted) + x2

        # del hpf
        return x1, x2, hx2_ffted
