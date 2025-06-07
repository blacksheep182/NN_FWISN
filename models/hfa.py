import torch.nn as nn
import torch
import pywt

import torch.nn.functional as F
from functools import partial
from torch.nn import init
from models.dpcd_parts import (Conv_BN_ReLU, CGSU, Encoder_Block, DPFA, Decoder_Block,
                               Changer_channel_exchange, log_feature, ABconv)

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
        # self.hconv_layer1 = torch.nn.Conv2d(in_chans, embed_dims, 1, 1, 0)
        # self.hconv_layer2 = torch.nn.Conv2d(embed_dims, embed_dims // 2, 1, 1, 0)
        # self.hconv_layer3 = torch.nn.Conv2d(embed_dims // 2, embed_dims, 1, 1, 0)
        # self.hconv_layer4 = torch.nn.Conv2d(embed_dims, out_chans, 1, 1, 0)
        # self.hrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        # self.lconv_layer1 = torch.nn.Conv2d(in_chans, embed_dims, 5, 1, 2)
        # self.lconv_layer2 = torch.nn.Conv2d(embed_dims, embed_dims // 2, 5, 1, 2)
        # self.lconv_layer3 = torch.nn.Conv2d(embed_dims // 2, embed_dims, 5, 1, 2)
        # self.lconv_layer4 = torch.nn.Conv2d(embed_dims, out_chans, 5, 1, 2)

        # self.lpf = torch.zeros((4, 3, 512, 512))

        # self.conv1 = nn.Conv2d()

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

        x2_freq = torch.fft.fftn(x2, dim=(2, 3),norm='ortho')
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
