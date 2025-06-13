import torch.nn as nn
import torch
import pywt

import torch.nn.functional as F
from functools import partial
from torch.nn import init
from models.dpcd_parts import (Conv_BN_ReLU, CGSU, Encoder_Block, DPFA, Decoder_Block,
                               Changer_channel_exchange, log_feature, ABconv)
class WTAB1(nn.Module):
    def __init__(self, in_chans, out_chans,kernel_size, device, wavelet ='db2'):
        super().__init__()
        self.in_chans = in_chans
        self.out_chans = out_chans
        self.device = device
        self.kernel_size = kernel_size

        self.wt_filter, self.iwt_filter = self.create_wavelet_filter(wavelet, in_chans, out_chans, torch.float32)
        self.wt_filter = nn.Parameter(self.wt_filter, requires_grad=False)
        self.iwt_filter = nn.Parameter(self.iwt_filter, requires_grad=False)

        self.wt_function = partial(self.wavelet_transform, filters = self.wt_filter)
        self.iwt_function = partial(self.inverse_wavelet_transform, filters = self.iwt_filter)

        self.wavelet = wavelet

        self.WTCH_conv = nn.Conv1d(self.in_chans, self.out_chans, self.kernel_size, padding=kernel_size//2)
        self.WTCV_conv = nn.Conv1d(self.in_chans, self.out_chans, self.kernel_size, padding=kernel_size//2)
        self.WTCD_conv = nn.Conv2d(self.in_chans, self.out_chans, self.kernel_size, padding=kernel_size//2)

        self.hwt_conv = nn.Sequential(
            nn.Conv2d(in_chans * 3 ,in_chans, 1, 1, 0),
            nn.BatchNorm2d(in_chans),
            nn.ReLU(),
            nn.Conv2d(in_chans, out_chans, 1, 1, 0),
            nn.BatchNorm2d(out_chans),
        )

        self.iwt_conv = nn.Sequential(
            nn.Conv2d(in_chans * 2 ,in_chans, 1, 1, 0),
            nn.BatchNorm2d(in_chans),
            nn.ReLU(),
            nn.Conv2d(in_chans, out_chans, 1, 1, 0),
            nn.BatchNorm2d(out_chans),
        )

        from pytorch_wavelets import DWTForward, DWTInverse  # (or import DWT, IDWT)
        self.xfm = DWTForward(J=1, mode='zero', wave=wavelet) # Accepts all wave types available to PyWavelets
        self.ifm = DWTInverse(mode='zero', wave=wavelet)

        self.up_sample2 = nn.UpsamplingBilinear2d(scale_factor=2)
        self.l_fuse = DPFA(in_channel=in_chans)
        self.h_fuse_ch = nn.Conv2d(self.in_chans*2, self.out_chans, self.kernel_size, padding=kernel_size // 2)
        self.h_fuse_cv = nn.Conv2d(self.in_chans*2, self.out_chans, self.kernel_size, padding=kernel_size // 2)
        self.h_fuse_cd = nn.Conv2d(self.in_chans*2, self.out_chans, self.kernel_size, padding=kernel_size // 2)

    def create_wavelet_filter(self,wave, in_size, out_size, type=torch.float):
        w = pywt.Wavelet(wave)
        dec_hi = torch.tensor(w.dec_hi[::-1], dtype=type)
        dec_lo = torch.tensor(w.dec_lo[::-1], dtype=type)
        dec_filters = torch.stack([dec_lo.unsqueeze(0) * dec_lo.unsqueeze(1),
                                   dec_lo.unsqueeze(0) * dec_hi.unsqueeze(1),
                                   dec_hi.unsqueeze(0) * dec_lo.unsqueeze(1),
                                   dec_hi.unsqueeze(0) * dec_hi.unsqueeze(1)], dim=0)

        dec_filters = dec_filters[:, None].repeat(in_size, 1, 1, 1)

        rec_hi = torch.tensor(w.rec_hi[::-1], dtype=type).flip(dims=[0])
        rec_lo = torch.tensor(w.rec_lo[::-1], dtype=type).flip(dims=[0])
        rec_filters = torch.stack([rec_lo.unsqueeze(0) * rec_lo.unsqueeze(1),
                                   rec_lo.unsqueeze(0) * rec_hi.unsqueeze(1),
                                   rec_hi.unsqueeze(0) * rec_lo.unsqueeze(1),
                                   rec_hi.unsqueeze(0) * rec_hi.unsqueeze(1)], dim=0)

        rec_filters = rec_filters[:, None].repeat(out_size, 1, 1, 1)

        return dec_filters, rec_filters

    def wavelet_transform(self, x, filters):
        b, c, h, w = x.shape
        pad = (filters.shape[2] // 2 - 1, filters.shape[3] // 2 - 1)
        x = F.conv2d(x, filters, stride=2, groups=c, padding=pad)
        x = x.reshape(b, c, 4, h // 2, w // 2)
        return x

    def inverse_wavelet_transform(self,x, filters):
        b, c, _, h_half, w_half = x.shape
        pad = (filters.shape[2] // 2 - 1, filters.shape[3] // 2 - 1)
        x = x.reshape(b, c * 4, h_half, w_half)
        x = F.conv_transpose2d(x, filters, stride=2, groups=c, padding=pad)
        return x

    def wt_forward(self, CH, CV, CD, lwt):
        b, c, h, w = lwt.shape

        CH = self.WTCH_conv(CH.reshape(b, c, -1)).reshape(b, c, h, w)
        CV = self.WTCV_conv(CV.permute(0, 1, 3, 2).reshape(b, c, -1)).reshape(b, c, h, w)
        CD = self.WTCD_conv(CD)

        coeffs = torch.cat((lwt.reshape(b, c, 1, h, w ),CH.reshape(b, c, 1, h, w),
                            CH.reshape(b, c, 1, h, w), CH.reshape(b, c, 1, h, w)), dim=2)

        iwt = self.iwt_function(coeffs)

        hwt_c = self.hwt_conv(torch.cat((CH, CV, CD), dim=1))
        hwt_c = self.up_sample2(hwt_c)
        wtx = self.iwt_conv(torch.cat((iwt, hwt_c), dim=1))
        return wtx


    def forward(self, x1, x2):

        b, c, h, w = x1.shape
        coeffs1 = self.wt_function(x1)
        lwt1 = coeffs1[:,:,0,:,:]
        hwt_list1 = []
        for i in range(3):
            hwt_list1.append(coeffs1[:, :, i+1, :, :])
        CH1, CV1, CD1 = hwt_list1

        coeffs2 = self.wt_function(x2)
        lwt2 = coeffs2[:,:,0,:,:]
        hwt_list2 = []
        for i in range(3):
            hwt_list2.append(coeffs2[:, :, i+1, :, :])
        CH2, CV2, CD2 = hwt_list2

 

        wtx2 = self.wt_forward(CH1, CV1, CD1, lwt2)
        wtx1 = self.wt_forward(CH2, CV2, CD2, lwt1)

        wtx = self.l_fuse(wtx1, wtx2)


        return wtx

class WTAB(nn.Module):
    def __init__(self, in_chans, out_chans,kernel_size, device, wavelet ='db2'):
        super().__init__()
        self.in_chans = in_chans
        self.out_chans = out_chans
        self.device = device
        self.kernel_size = kernel_size

        self.wt_filter, self.iwt_filter = self.create_wavelet_filter(wavelet, in_chans, out_chans, torch.float32)
        self.wt_filter = nn.Parameter(self.wt_filter, requires_grad=False)
        self.iwt_filter = nn.Parameter(self.iwt_filter, requires_grad=False)

        self.wt_function = partial(self.wavelet_transform, filters = self.wt_filter)
        self.iwt_function = partial(self.inverse_wavelet_transform, filters = self.iwt_filter)

        self.wavelet = wavelet

        self.WTCH_conv = nn.Conv1d(self.in_chans, self.out_chans, self.kernel_size, padding=kernel_size//2)
        self.WTCV_conv = nn.Conv1d(self.in_chans, self.out_chans, self.kernel_size, padding=kernel_size//2)
        self.WTCD_conv = nn.Conv2d(self.in_chans, self.out_chans, self.kernel_size, padding=kernel_size//2)

        self.hwt_conv = nn.Sequential(
            nn.Conv2d(in_chans * 3 ,in_chans, 1, 1, 0),
            nn.BatchNorm2d(in_chans),
            nn.ReLU(),
            nn.Conv2d(in_chans, out_chans, 1, 1, 0),
            nn.BatchNorm2d(out_chans),
        )

        self.iwt_conv = nn.Sequential(
            nn.Conv2d(in_chans * 2 ,in_chans, 1, 1, 0),
            nn.BatchNorm2d(in_chans),
            nn.ReLU(),
            nn.Conv2d(in_chans, out_chans, 1, 1, 0),
            nn.BatchNorm2d(out_chans),
        )

        from pytorch_wavelets import DWTForward, DWTInverse  # (or import DWT, IDWT)
        self.xfm = DWTForward(J=1, mode='zero', wave=wavelet) # Accepts all wave types available to PyWavelets
        self.ifm = DWTInverse(mode='zero', wave=wavelet)

        self.up_sample2 = nn.UpsamplingBilinear2d(scale_factor=2)

    def create_wavelet_filter(self,wave, in_size, out_size, type=torch.float):
        w = pywt.Wavelet(wave)
        dec_hi = torch.tensor(w.dec_hi[::-1], dtype=type)
        dec_lo = torch.tensor(w.dec_lo[::-1], dtype=type)
        dec_filters = torch.stack([dec_lo.unsqueeze(0) * dec_lo.unsqueeze(1),
                                   dec_lo.unsqueeze(0) * dec_hi.unsqueeze(1),
                                   dec_hi.unsqueeze(0) * dec_lo.unsqueeze(1),
                                   dec_hi.unsqueeze(0) * dec_hi.unsqueeze(1)], dim=0)

        dec_filters = dec_filters[:, None].repeat(in_size, 1, 1, 1)

        rec_hi = torch.tensor(w.rec_hi[::-1], dtype=type).flip(dims=[0])
        rec_lo = torch.tensor(w.rec_lo[::-1], dtype=type).flip(dims=[0])
        rec_filters = torch.stack([rec_lo.unsqueeze(0) * rec_lo.unsqueeze(1),
                                   rec_lo.unsqueeze(0) * rec_hi.unsqueeze(1),
                                   rec_hi.unsqueeze(0) * rec_lo.unsqueeze(1),
                                   rec_hi.unsqueeze(0) * rec_hi.unsqueeze(1)], dim=0)

        rec_filters = rec_filters[:, None].repeat(out_size, 1, 1, 1)

        return dec_filters, rec_filters

    def wavelet_transform(self, x, filters):
        b, c, h, w = x.shape
        pad = (filters.shape[2] // 2 - 1, filters.shape[3] // 2 - 1)
        x = F.conv2d(x, filters, stride=2, groups=c, padding=pad)
        x = x.reshape(b, c, 4, h // 2, w // 2)
        return x

    def inverse_wavelet_transform(self,x, filters):
        b, c, _, h_half, w_half = x.shape
        pad = (filters.shape[2] // 2 - 1, filters.shape[3] // 2 - 1)
        x = x.reshape(b, c * 4, h_half, w_half)
        x = F.conv_transpose2d(x, filters, stride=2, groups=c, padding=pad)
        return x

    def wt_forward(self, x):
        b, c, h, w = x.shape
        coeffs = self.wt_function(x)

        lwt = coeffs[:,:,0,:,:]

        hwt_list = []
        for i in range(3):
            hwt_list.append(coeffs[:, :, i+1, :, :])
        CH, CV, CD = hwt_list

        CH = self.WTCH_conv(CH.reshape(b, c, -1)).reshape(b, c, h // 2, w // 2)
        CV = self.WTCV_conv(CV.permute(0, 1, 3, 2).reshape(b, c, -1)).reshape(b, c, h // 2, w // 2)
        CD = self.WTCD_conv(CD)

        coeffs = torch.cat((lwt.reshape(b, c, 1, h // 2, w // 2),CH.reshape(b, c, 1, h // 2, w // 2),
                            CH.reshape(b, c, 1, h // 2, w // 2), CH.reshape(b, c, 1, h // 2, w // 2)), dim=2)


        iwt = self.iwt_function(coeffs)

        hwt_c = self.hwt_conv(torch.cat((CH, CV, CD), dim=1))
        hwt_c = self.up_sample2(hwt_c)
        wtx = self.iwt_conv(torch.cat((iwt, hwt_c), dim=1))
        return wtx + x


    def forward(self, x1, x2):
        wtx1 = self.wt_forward(x1)
        wtx2 = self.wt_forward(x2)

        return wtx1, wtx2


