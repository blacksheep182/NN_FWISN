import torch
import torch.nn as nn
import math
import copy
from torch.nn import init
from models.dpcd_parts import (Conv_BN_ReLU, CGSU, Encoder_Block, DPFA, Decoder_Block,
                               Changer_channel_exchange, log_feature, ABconv)
from torchvision import models
from collections import OrderedDict
from functools import partial
from typing import Optional, Callable, Any
from timm.models.layers import DropPath, trunc_normal_
from einops import rearrange, repeat

DropPath.__repr__ = lambda self: f"timm.DropPath({self.drop_prob})"
# import selective scan ==============================
try:
    import selective_scan_cuda_oflex
except Exception as e:
    ...
    # print(f"WARNING: can not import selective_scan_cuda_oflex.", flush=True)
    # print(e, flush=True)

try:
    import selective_scan_cuda_core
except Exception as e:
    ...
    # print(f"WARNING: can not import selective_scan_cuda_core.", flush=True)
    # print(e, flush=True)

try:
    import selective_scan_cuda
except Exception as e:
    ...


def make_maskembed_layer(in_chans, out_chans):
    return nn.Sequential(
        nn.Conv2d(in_channels=in_chans, out_channels=out_chans, kernel_size=2, stride=2),
        nn.BatchNorm2d(out_chans),
        nn.GELU(),
    )


class Permute(nn.Module):
    def __init__(self, *args):
        super().__init__()
        self.args = args

    def forward(self, x: torch.Tensor):
        return x.permute(*self.args)


class SelectiveScanMamba(torch.autograd.Function):
    # comment all checks if inside cross_selective_scan
    @staticmethod
    @torch.cuda.amp.custom_fwd
    def forward(ctx, u, delta, A, B, C, D=None, delta_bias=None, delta_softplus=False, nrows=1, backnrows=1,
                oflex=True):
        # assert nrows in [1, 2, 3, 4], f"{nrows}" # 8+ is too slow to compile
        # assert u.shape[1] % (B.shape[1] * nrows) == 0, f"{nrows}, {u.shape}, {B.shape}"
        ctx.delta_softplus = delta_softplus
        # all in float
        # if u.stride(-1) != 1:
        #     u = u.contiguous()
        # if delta.stride(-1) != 1:
        #     delta = delta.contiguous()
        # if D is not None and D.stride(-1) != 1:
        #     D = D.contiguous()
        # if B.stride(-1) != 1:
        #     B = B.contiguous()
        # if C.stride(-1) != 1:
        #     C = C.contiguous()
        # if B.dim() == 3:
        #     B = B.unsqueeze(dim=1)
        #     ctx.squeeze_B = True
        # if C.dim() == 3:
        #     C = C.unsqueeze(dim=1)
        #     ctx.squeeze_C = True

        out, x, *rest = selective_scan_cuda.fwd(u, delta, A, B, C, D, None, delta_bias, delta_softplus)
        ctx.save_for_backward(u, delta, A, B, C, D, delta_bias, x)
        return out

    @staticmethod
    @torch.cuda.amp.custom_bwd
    def backward(ctx, dout, *args):
        u, delta, A, B, C, D, delta_bias, x = ctx.saved_tensors
        if dout.stride(-1) != 1:
            dout = dout.contiguous()

        du, ddelta, dA, dB, dC, dD, ddelta_bias, *rest = selective_scan_cuda.bwd(
            u, delta, A, B, C, D, None, delta_bias, dout, x, None, None, ctx.delta_softplus,
            False
        )
        # dB = dB.squeeze(1) if getattr(ctx, "squeeze_B", False) else dB
        # dC = dC.squeeze(1) if getattr(ctx, "squeeze_C", False) else dC
        return (du, ddelta, dA, dB, dC, dD, ddelta_bias, None, None, None, None)


class SelectiveScanCore(torch.autograd.Function):
    # comment all checks if inside cross_selective_scan
    @staticmethod
    @torch.cuda.amp.custom_fwd
    def forward(ctx, u, delta, A, B, C, D=None, delta_bias=None, delta_softplus=False, nrows=1, backnrows=1,
                oflex=True):
        ctx.delta_softplus = delta_softplus
        out, x, *rest = selective_scan_cuda_core.fwd(u, delta, A, B, C, D, delta_bias, delta_softplus, 1)
        ctx.save_for_backward(u, delta, A, B, C, D, delta_bias, x)
        return out

    @staticmethod
    @torch.cuda.amp.custom_bwd
    def backward(ctx, dout, *args):
        u, delta, A, B, C, D, delta_bias, x = ctx.saved_tensors
        if dout.stride(-1) != 1:
            dout = dout.contiguous()
        du, ddelta, dA, dB, dC, dD, ddelta_bias, *rest = selective_scan_cuda_core.bwd(
            u, delta, A, B, C, D, delta_bias, dout, x, ctx.delta_softplus, 1
        )
        return (du, ddelta, dA, dB, dC, dD, ddelta_bias, None, None, None, None)


class SelectiveScanOflex(torch.autograd.Function):
    # comment all checks if inside cross_selective_scan
    @staticmethod
    @torch.cuda.amp.custom_fwd
    def forward(ctx, u, delta, A, B, C, D=None, delta_bias=None, delta_softplus=False, nrows=1, backnrows=1,
                oflex=True):
        ctx.delta_softplus = delta_softplus
        out, x, *rest = selective_scan_cuda_oflex.fwd(u, delta, A, B, C, D, delta_bias, delta_softplus, 1, oflex)
        ctx.save_for_backward(u, delta, A, B, C, D, delta_bias, x)
        return out

    @staticmethod
    @torch.cuda.amp.custom_bwd
    def backward(ctx, dout, *args):
        u, delta, A, B, C, D, delta_bias, x = ctx.saved_tensors
        if dout.stride(-1) != 1:
            dout = dout.contiguous()
        du, ddelta, dA, dB, dC, dD, ddelta_bias, *rest = selective_scan_cuda_oflex.bwd(
            u, delta, A, B, C, D, delta_bias, dout, x, ctx.delta_softplus, 1
        )
        return (du, ddelta, dA, dB, dC, dD, ddelta_bias, None, None, None, None)


class SelectiveScanFake(torch.autograd.Function):
    # comment all checks if inside cross_selective_scan
    @staticmethod
    @torch.cuda.amp.custom_fwd
    def forward(ctx, u, delta, A, B, C, D=None, delta_bias=None, delta_softplus=False, nrows=1, backnrows=1,
                oflex=True):
        ctx.delta_softplus = delta_softplus
        ctx.backnrows = backnrows
        x = delta
        out = u
        ctx.save_for_backward(u, delta, A, B, C, D, delta_bias, x)
        return out

    @staticmethod
    @torch.cuda.amp.custom_bwd
    def backward(ctx, dout, *args):
        u, delta, A, B, C, D, delta_bias, x = ctx.saved_tensors
        if dout.stride(-1) != 1:
            dout = dout.contiguous()
        du, ddelta, dA, dB, dC, dD, ddelta_bias = u * 0, delta * 0, A * 0, B * 0, C * 0, C * 0, (
            D * 0 if D else None), (delta_bias * 0 if delta_bias else None)
        return (du, ddelta, dA, dB, dC, dD, ddelta_bias, None, None, None, None)


def antidiagonal_gather(tensor):
    # 取出矩阵所有反斜向的元素并拼接
    B, C, H, W = tensor.size()
    shift = torch.arange(H, device=tensor.device).unsqueeze(1)  # 创建一个列向量[H, 1]
    index = (torch.arange(W, device=tensor.device) - shift) % W  # 利用广播创建索引矩阵[H, W]
    # 扩展索引以适应B和C维度
    expanded_index = index.unsqueeze(0).unsqueeze(0).expand(B, C, -1, -1)
    # 使用gather进行索引选择
    return tensor.gather(3, expanded_index).transpose(-1,-2).reshape(B, C, H*W)

def diagonal_gather(tensor):
    # 取出矩阵所有反斜向的元素并拼接
    B, C, H, W = tensor.size()
    shift = torch.arange(H, device=tensor.device).unsqueeze(1)
    shift1 = torch.arange(W, device=tensor.device)# 创建一个列向量[H, 1]
    index = (shift + shift1) % W  # 利用广播创建索引矩阵[H, W]
    # 扩展索引以适应B和C维度
    expanded_index = index.unsqueeze(0).unsqueeze(0).expand(B, C, -1, -1)
    # 使用gather进行索引选择
    return tensor.gather(3, expanded_index).transpose(-1,-2).reshape(B, C, H*W)

def diagonal_scatter(tensor_flat, original_shape):
    # 把斜向元素拼接起来的一维向量还原为最初的矩阵形式
    B, C, H, W = original_shape
    shift = torch.arange(H, device=tensor_flat.device).unsqueeze(1)  # 创建一个列向量[H, 1]
    index = (shift + torch.arange(W, device=tensor_flat.device)) % W  # 利用广播创建索引矩阵[H, W]
    # 扩展索引以适应B和C维度
    expanded_index = index.unsqueeze(0).unsqueeze(0).expand(B, C, -1, -1)
    # 创建一个空的张量来存储反向散布的结果
    result_tensor = torch.zeros(B, C, H, W, device=tensor_flat.device, dtype=tensor_flat.dtype)
    # 将平铺的张量重新变形为[B, C, H, W]，考虑到需要使用transpose将H和W调换
    tensor_reshaped = tensor_flat.reshape(B, C, W, H).transpose(-1, -2)
    # 使用scatter_根据expanded_index将元素放回原位
    result_tensor.scatter_(3, expanded_index, tensor_reshaped)
    return result_tensor


def antidiagonal_scatter(tensor_flat, original_shape):
    # 把反斜向元素拼接起来的一维向量还原为最初的矩阵形式
    B, C, H, W = original_shape
    shift = torch.arange(H, device=tensor_flat.device).unsqueeze(1)  # 创建一个列向量[H, 1]
    index = (torch.arange(W, device=tensor_flat.device) - shift) % W  # 利用广播创建索引矩阵[H, W]
    expanded_index = index.unsqueeze(0).unsqueeze(0).expand(B, C, -1, -1)
    # 初始化一个与原始张量形状相同、元素全为0的张量
    result_tensor = torch.zeros(B, C, H, W, device=tensor_flat.device, dtype=tensor_flat.dtype)
    # 将平铺的张量重新变形为[B, C, W, H]，因为操作是沿最后一个维度收集的，需要调整形状并交换维度
    tensor_reshaped = tensor_flat.reshape(B, C, W, H).transpose(-1, -2)
    # 使用scatter_将元素根据索引放回原位
    result_tensor.scatter_(3, expanded_index, tensor_reshaped)
    return result_tensor


class CrossScan(torch.autograd.Function):
    # ZSJ 这里是把图像按照特定方向展平的地方，改变扫描方向可以在这里修改
    @staticmethod
    def forward(ctx, x: torch.Tensor):
        B, C, H, W = x.shape
        ctx.shape = (B, C, H, W)
        # xs = x.new_empty((B, 4, C, H * W))
        xs = x.new_empty((B, 8, C, H * W))
        # 添加横向和竖向的扫描
        xs[:, 0] = x.flatten(2, 3)
        xs[:, 1] = x.transpose(dim0=2, dim1=3).flatten(2, 3)
        xs[:, 2:4] = torch.flip(xs[:, 0:2], dims=[-1])
        #
        # 提供斜向和反斜向的扫描
        xs[:, 4] = diagonal_gather(x)
        xs[:, 5] = antidiagonal_gather(x)
        xs[:, 6:8] = torch.flip(xs[:, 4:6], dims=[-1])

        return xs

    @staticmethod
    def backward(ctx, ys: torch.Tensor):
        # out: (b, k, d, l)
        B, C, H, W = ctx.shape
        L = H * W
        # 把横向和竖向的反向部分再反向回来，并和原来的横向和竖向相加
        # ys = ys[:, 0:2] + ys[:, 2:4].flip(dims=[-1]).view(B, 2, -1, L)
        y_rb = ys[:, 0:2] + ys[:, 2:4].flip(dims=[-1]).view(B, 2, -1, L)
        # 把竖向的部分转成横向，然后再相加,再转回最初是的矩阵形式
        # y = ys[:, 0] + ys[:, 1].view(B, -1, W, H).transpose(dim0=2, dim1=3).contiguous().view(B, -1, L)
        y_rb = y_rb[:, 0] + y_rb[:, 1].view(B, -1, W, H).transpose(dim0=2, dim1=3).contiguous().view(B, -1, L)
        y_rb = y_rb.view(B, -1, H, W)

        # 把斜向和反斜向的反向部分再反向回来，并和原来的斜向和反斜向相加
        y_da = ys[:, 4:6] + ys[:, 6:8].flip(dims=[-1]).view(B, 2, -1, L)
        # 把斜向和反斜向的部分都转成原来的最初的矩阵形式，再相加
        y_da = diagonal_scatter(y_da[:, 0], (B, C, H, W)) + antidiagonal_scatter(y_da[:, 1], (B, C, H, W))

        # y_res = y_rb
        y_res = y_rb + y_da
        # return y.view(B, -1, H, W)
        return y_res


class CrossMerge(torch.autograd.Function):
    @staticmethod
    def forward(ctx, ys: torch.Tensor):
        B, K, D, H, W = ys.shape
        ctx.shape = (H, W)
        ys = ys.view(B, K, D, -1)
        # ys = ys[:, 0:2] + ys[:, 2:4].flip(dims=[-1]).view(B, 2, D, -1)
        # y = ys[:, 0] + ys[:, 1].view(B, -1, W, H).transpose(dim0=2, dim1=3).contiguous().view(B, D, -1)

        y_rb = ys[:, 0:2] + ys[:, 2:4].flip(dims=[-1]).view(B, 2, D, -1)
        # 把竖向的部分转成横向，然后再相加,再转回最初是的矩阵形式
        y_rb = y_rb[:, 0] + y_rb[:, 1].view(B, -1, W, H).transpose(dim0=2, dim1=3).contiguous().view(B, D, -1)
        y_rb = y_rb.view(B, -1, H, W)

        # 把斜向和反斜向的反向部分再反向回来，并和原来的斜向和反斜向相加
        y_da = ys[:, 4:6] + ys[:, 6:8].flip(dims=[-1]).view(B, 2, D, -1)
        # 把斜向和反斜向的部分都转成原来的最初的矩阵形式，再相加
        y_da = diagonal_scatter(y_da[:, 0], (B, D, H, W)) + antidiagonal_scatter(y_da[:, 1], (B, D, H, W))

        # y_res = y_rb
        y_res = y_rb + y_da
        return y_res.view(B, D, -1)
        # return y

    @staticmethod
    def backward(ctx, x: torch.Tensor):
        # B, D, L = x.shape
        # out: (b, k, d, l)
        H, W = ctx.shape
        B, C, L = x.shape
        # xs = x.new_empty((B, 4, C, L))
        xs = x.new_empty((B, 8, C, L))

        # 横向和竖向扫描
        xs[:, 0] = x
        xs[:, 1] = x.view(B, C, H, W).transpose(dim0=2, dim1=3).flatten(2, 3)
        xs[:, 2:4] = torch.flip(xs[:, 0:2], dims=[-1])
        # xs = xs.view(B, 4, C, H, W)

        # # 提供斜向和反斜向的扫描
        xs[:, 4] = diagonal_gather(x.view(B, C, H, W))
        xs[:, 5] = antidiagonal_gather(x.view(B, C, H, W))
        xs[:, 6:8] = torch.flip(xs[:, 4:6], dims=[-1])

        # return xs
        # return xs.view(B, 4, C, H, W)
        return xs.view(B, 8, C, H, W)


# =============
# ZSJ 这里是mamba的具体内容，要增加扫描方向就在这里改
def cross_selective_scan(
        x: torch.Tensor = None,
        x_proj_weight: torch.Tensor = None,
        x_proj_bias: torch.Tensor = None,
        dt_projs_weight: torch.Tensor = None,
        dt_projs_bias: torch.Tensor = None,
        A_logs: torch.Tensor = None,
        Ds: torch.Tensor = None,
        delta_softplus=True,
        out_norm: torch.nn.Module = None,
        out_norm_shape="v0",
        # ==============================
        to_dtype=True,  # True: final out to dtype
        force_fp32=False,  # True: input fp32
        # ==============================
        nrows=-1,  # for SelectiveScanNRow; 0: auto; -1: disable;
        backnrows=-1,  # for SelectiveScanNRow; 0: auto; -1: disable;
        ssoflex=True,  # True: out fp32 in SSOflex; else, SSOflex is the same as SSCore
        # ==============================
        SelectiveScan=None,
        CrossScan=CrossScan,
        CrossMerge=CrossMerge,
):
    # out_norm: whatever fits (B, L, C); LayerNorm; Sigmoid; Softmax(dim=1);...

    B, D, H, W = x.shape
    D, N = A_logs.shape
    K, D, R = dt_projs_weight.shape
    L = H * W

    if nrows == 0:
        if D % 4 == 0:
            nrows = 4
        elif D % 3 == 0:
            nrows = 3
        elif D % 2 == 0:
            nrows = 2
        else:
            nrows = 1

    if backnrows == 0:
        if D % 4 == 0:
            backnrows = 4
        elif D % 3 == 0:
            backnrows = 3
        elif D % 2 == 0:
            backnrows = 2
        else:
            backnrows = 1

    def selective_scan(u, delta, A, B, C, D=None, delta_bias=None, delta_softplus=True):
        return SelectiveScan.apply(u, delta, A, B, C, D, delta_bias, delta_softplus, nrows, backnrows, ssoflex)

    # 义各个方向的序列 x[:, 0:8]
    xs = CrossScan.apply(x)

    x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs, x_proj_weight)
    if x_proj_bias is not None:
        x_dbl = x_dbl + x_proj_bias.view(1, K, -1, 1)
    dts, Bs, Cs = torch.split(x_dbl, [R, N, N], dim=2)
    dts = torch.einsum("b k r l, k d r -> b k d l", dts, dt_projs_weight)
    xs = xs.view(B, -1, L)
    dts = dts.contiguous().view(B, -1, L)
    As = -torch.exp(A_logs.to(torch.float))  # (k * c, d_state)
    Bs = Bs.contiguous()
    Cs = Cs.contiguous()
    Ds = Ds.to(torch.float)  # (K * c)
    delta_bias = dt_projs_bias.view(-1).to(torch.float)

    if force_fp32:
        xs = xs.to(torch.float)
        dts = dts.to(torch.float)
        Bs = Bs.to(torch.float)
        Cs = Cs.to(torch.float)
    # ZSJ 这里把矩阵拆分成不同方向的序列，并进行扫描
    ys: torch.Tensor = selective_scan(
        xs, dts, As, Bs, Cs, Ds, delta_bias, delta_softplus
    ).view(B, K, -1, H, W)
    # ZSJ 这里把处理之后的序列融合起来，并还原回原来的矩阵形式
    y: torch.Tensor = CrossMerge.apply(ys)

    if out_norm_shape in ["v1"]:  # (B, C, H, W)
        y = out_norm(y.view(B, -1, H, W)).permute(0, 2, 3, 1)  # (B, H, W, C)
    else:  # (B, L, C)
        y = y.transpose(dim0=1, dim1=2).contiguous()  # (B, L, C)
        y = out_norm(y).view(B, H, W, -1)

    return (y.to(x.dtype) if to_dtype else y)


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.,channels_first=False):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        Linear = partial(nn.Conv2d, kernel_size=1, padding=0) if channels_first else nn.Linear
        self.fc1 = Linear(in_features, hidden_features)
        # self.act = act_layer()
        self.act = nn.SiLU()
        self.fc2 = Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class OSSM(nn.Module):
    def __init__(
            self,
            # basic dims ===========
            d_model=96,
            d_state=16,
            ssm_ratio=2.0,
            dt_rank="auto",
            act_layer=nn.SiLU,
            # dwconv ===============
            d_conv=3,  # < 2 means no conv
            conv_bias=True,
            # ======================
            dropout=0.0,
            bias=False,
            # dt init ==============
            dt_min=0.001,
            dt_max=0.1,
            dt_init="random",
            dt_scale=1.0,
            dt_init_floor=1e-4,
            initialize="v0",
            # ======================
            forward_type="v2",
            # ======================
            **kwargs,
    ):
        factory_kwargs = {"device": None, "dtype": None}
        super().__init__()
        d_inner = int(ssm_ratio * d_model)
        dt_rank = math.ceil(d_model / 16) if dt_rank == "auto" else dt_rank
        self.d_conv = d_conv

        # tags for forward_type ==============================
        def checkpostfix(tag, value):
            ret = value[-len(tag):] == tag
            if ret:
                value = value[:-len(tag)]
            return ret, value

        self.disable_force32, forward_type = checkpostfix("no32", forward_type)
        self.disable_z, forward_type = checkpostfix("noz", forward_type)
        self.disable_z_act, forward_type = checkpostfix("nozact", forward_type)

        # softmax | sigmoid | dwconv | norm ===========================
        if forward_type[-len("none"):] == "none":
            forward_type = forward_type[:-len("none")]
            self.out_norm = nn.Identity()
        elif forward_type[-len("dwconv3"):] == "dwconv3":
            forward_type = forward_type[:-len("dwconv3")]
            self.out_norm = nn.Conv2d(d_inner, d_inner, kernel_size=3, padding=1, groups=d_inner, bias=False)
            self.out_norm_shape = "v1"
        elif forward_type[-len("softmax"):] == "softmax":
            forward_type = forward_type[:-len("softmax")]
            self.out_norm = nn.Softmax(dim=1)
        elif forward_type[-len("sigmoid"):] == "sigmoid":
            forward_type = forward_type[:-len("sigmoid")]
            self.out_norm = nn.Sigmoid()
        else:
            self.out_norm = nn.LayerNorm(d_inner)

        # forward_type debug =======================================
        FORWARD_TYPES = dict(
            v0=self.forward_corev0,
            # v2=partial(self.forward_corev2, force_fp32=(not self.disable_force32), SelectiveScan=SelectiveScanCore),
            v2=partial(self.forward_corev2, force_fp32=True, SelectiveScan=SelectiveScanCore),
            v3=partial(self.forward_corev2, force_fp32=False, SelectiveScan=SelectiveScanOflex),
            # v31d=partial(self.forward_corev2, force_fp32=False, SelectiveScan=SelectiveScanOflex,
            #              cross_selective_scan=partial(
            #                  cross_selective_scan, CrossScan=CrossScan_Ab_1direction,
            #                  CrossMerge=CrossMerge_Ab_1direction,
            #              )),
            # v32d=partial(self.forward_corev2, force_fp32=False, SelectiveScan=SelectiveScanOflex,
            #              cross_selective_scan=partial(
            #                  cross_selective_scan, CrossScan=CrossScan_Ab_2direction,
            #                  CrossMerge=CrossMerge_Ab_2direction,
            #              )),
            # ===============================
            fake=partial(self.forward_corev2, force_fp32=(not self.disable_force32), SelectiveScan=SelectiveScanFake),
            v1=partial(self.forward_corev2, force_fp32=True, SelectiveScan=SelectiveScanOflex),
            v01=partial(self.forward_corev2, force_fp32=(not self.disable_force32), SelectiveScan=SelectiveScanMamba),
        )
        if forward_type.startswith("debug"):
            from .ss2d_ablations import SS2D_ForwardCoreSpeedAblations, SS2D_ForwardCoreModeAblations, \
                cross_selective_scanv2
            FORWARD_TYPES.update(dict(
                debugforward_core_mambassm_seq=partial(SS2D_ForwardCoreSpeedAblations.forward_core_mambassm_seq, self),
                debugforward_core_mambassm=partial(SS2D_ForwardCoreSpeedAblations.forward_core_mambassm, self),
                debugforward_core_mambassm_fp16=partial(SS2D_ForwardCoreSpeedAblations.forward_core_mambassm_fp16,
                                                        self),
                debugforward_core_mambassm_fusecs=partial(SS2D_ForwardCoreSpeedAblations.forward_core_mambassm_fusecs,
                                                          self),
                debugforward_core_mambassm_fusecscm=partial(
                    SS2D_ForwardCoreSpeedAblations.forward_core_mambassm_fusecscm, self),
                debugforward_core_sscore_fusecscm=partial(SS2D_ForwardCoreSpeedAblations.forward_core_sscore_fusecscm,
                                                          self),
                debugforward_core_sscore_fusecscm_fwdnrow=partial(
                    SS2D_ForwardCoreSpeedAblations.forward_core_ssnrow_fusecscm_fwdnrow, self),
                debugforward_core_sscore_fusecscm_bwdnrow=partial(
                    SS2D_ForwardCoreSpeedAblations.forward_core_ssnrow_fusecscm_bwdnrow, self),
                debugforward_core_sscore_fusecscm_fbnrow=partial(
                    SS2D_ForwardCoreSpeedAblations.forward_core_ssnrow_fusecscm_fbnrow, self),
                debugforward_core_ssoflex_fusecscm=partial(SS2D_ForwardCoreSpeedAblations.forward_core_ssoflex_fusecscm,
                                                           self),
                debugforward_core_ssoflex_fusecscm_i16o32=partial(
                    SS2D_ForwardCoreSpeedAblations.forward_core_ssoflex_fusecscm_i16o32, self),
                debugscan_sharessm=partial(self.forward_corev2, force_fp32=False, SelectiveScan=SelectiveScanOflex,
                                           cross_selective_scan=cross_selective_scanv2),
            ))
        self.forward_core = FORWARD_TYPES.get(forward_type, None)
        # ZSJ k_group 指的是扫描的方向
        # k_group = 4 if forward_type not in ["debugscan_sharessm"] else 1
        k_group = 8 if forward_type not in ["debugscan_sharessm"] else 1

        # in proj =======================================
        d_proj = d_inner if self.disable_z else (d_inner * 2)
        self.in_proj = nn.Linear(d_model, d_proj, bias=bias, **factory_kwargs)
        self.act = nn.SiLU()

        # conv =======================================
        if d_conv > 1:
            self.conv2d = nn.Conv2d(
                in_channels=d_inner,
                out_channels=d_inner,
                groups=d_inner,
                bias=conv_bias,
                kernel_size=d_conv,
                padding=(d_conv - 1) // 2,
                **factory_kwargs,
            )

        # x proj ============================
        self.x_proj = [
            nn.Linear(d_inner, (dt_rank + d_state * 2), bias=False, **factory_kwargs)
            for _ in range(k_group)
        ]
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0))  # (K, N, inner)
        del self.x_proj

        # out proj =======================================
        self.out_proj = nn.Linear(d_inner, d_model, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else nn.Identity()

        if initialize in ["v0"]:
            # dt proj ============================
            self.dt_projs = [
                self.dt_init(dt_rank, d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs)
                for _ in range(k_group)
            ]
            self.dt_projs_weight = nn.Parameter(
                torch.stack([t.weight for t in self.dt_projs], dim=0))  # (K, inner, rank)
            self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0))  # (K, inner)
            del self.dt_projs

            # A, D =======================================
            self.A_logs = self.A_log_init(d_state, d_inner, copies=k_group, merge=True)  # (K * D, N)
            self.Ds = self.D_init(d_inner, copies=k_group, merge=True)  # (K * D)
        elif initialize in ["v1"]:
            # simple init dt_projs, A_logs, Ds
            self.Ds = nn.Parameter(torch.ones((k_group * d_inner)))
            self.A_logs = nn.Parameter(
                torch.randn((k_group * d_inner, d_state)))  # A == -A_logs.exp() < 0; # 0 < exp(A * dt) < 1
            self.dt_projs_weight = nn.Parameter(torch.randn((k_group, d_inner, dt_rank)))
            self.dt_projs_bias = nn.Parameter(torch.randn((k_group, d_inner)))
        elif initialize in ["v2"]:
            # simple init dt_projs, A_logs, Ds
            self.Ds = nn.Parameter(torch.ones((k_group * d_inner)))
            self.A_logs = nn.Parameter(
                torch.zeros((k_group * d_inner, d_state)))  # A == -A_logs.exp() < 0; # 0 < exp(A * dt) < 1
            self.dt_projs_weight = nn.Parameter(torch.randn((k_group, d_inner, dt_rank)))
            self.dt_projs_bias = nn.Parameter(torch.randn((k_group, d_inner)))

    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4,
                **factory_kwargs):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)

        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        # dt_proj.bias._no_reinit = True

        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=-1, device=None, merge=True):
        # S4D real initialization
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=d_inner,
        ).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        if copies > 0:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=-1, device=None, merge=True):
        # D "skip" parameter
        D = torch.ones(d_inner, device=device)
        if copies > 0:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)  # Keep in fp32
        D._no_weight_decay = True
        return D

    # only used to run previous version
    def forward_corev0(self, x: torch.Tensor, to_dtype=False, channel_first=False):
        def selective_scan(u, delta, A, B, C, D=None, delta_bias=None, delta_softplus=True, nrows=1):
            return SelectiveScanCore.apply(u, delta, A, B, C, D, delta_bias, delta_softplus, nrows, False)

        if not channel_first:
            x = x.permute(0, 3, 1, 2).contiguous()
        B, D, H, W = x.shape
        D, N = self.A_logs.shape
        K, D, R = self.dt_projs_weight.shape
        L = H * W

        # ZSJ 这里进行data expand操作，也就是把相同的数据在不同方向展开成一维，并拼接起来,但是这个函数只用在旧版本
        # 把横向和竖向拼接在K维度
        x_hwwh = torch.stack([x.view(B, -1, L), torch.transpose(x, dim0=2, dim1=3).contiguous().view(B, -1, L)],
                             dim=1).view(B, 2, -1, L)
        # torch.flip把横向和竖向两个方向都进行反向操作
        xs = torch.cat([x_hwwh, torch.flip(x_hwwh, dims=[-1])], dim=1)  # (b, k, d, l)

        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs, self.x_proj_weight)
        # x_dbl = x_dbl + self.x_proj_bias.view(1, K, -1, 1)
        dts, Bs, Cs = torch.split(x_dbl, [R, N, N], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts, self.dt_projs_weight)

        xs = xs.float().view(B, -1, L)  # (b, k * d, l)
        dts = dts.contiguous().float().view(B, -1, L)  # (b, k * d, l)
        Bs = Bs.float()  # (b, k, d_state, l)
        Cs = Cs.float()  # (b, k, d_state, l)

        As = -torch.exp(self.A_logs.float())  # (k * d, d_state)
        Ds = self.Ds.float()  # (k * d)
        dt_projs_bias = self.dt_projs_bias.float().view(-1)  # (k * d)

        # assert len(xs.shape) == 3 and len(dts.shape) == 3 and len(Bs.shape) == 4 and len(Cs.shape) == 4
        # assert len(As.shape) == 2 and len(Ds.shape) == 1 and len(dt_projs_bias.shape) == 1

        out_y = selective_scan(
            xs, dts,
            As, Bs, Cs, Ds,
            delta_bias=dt_projs_bias,
            delta_softplus=True,
        ).view(B, K, -1, L)
        # assert out_y.dtype == torch.float

        inv_y = torch.flip(out_y[:, 2:4], dims=[-1]).view(B, 2, -1, L)
        wh_y = torch.transpose(out_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
        invwh_y = torch.transpose(inv_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
        y = out_y[:, 0] + inv_y[:, 0] + wh_y + invwh_y
        y = y.transpose(dim0=1, dim1=2).contiguous()  # (B, L, C)
        y = self.out_norm(y).view(B, H, W, -1)

        return (y.to(x.dtype) if to_dtype else y)

    def forward_corev2(self, x: torch.Tensor, channel_first=False, SelectiveScan=SelectiveScanOflex,
                       cross_selective_scan=cross_selective_scan, force_fp32=None):
        if not channel_first:
            x = x.permute(0, 3, 1, 2).contiguous()
        # ZSJ V2版本使用的mamba，要改扫描方向在这里改
        x = cross_selective_scan(
            x, self.x_proj_weight, None, self.dt_projs_weight, self.dt_projs_bias,
            self.A_logs, self.Ds, delta_softplus=True,
            out_norm=getattr(self, "out_norm", None),
            out_norm_shape=getattr(self, "out_norm_shape", "v0"),
            force_fp32=force_fp32,
            SelectiveScan=SelectiveScan,
        )
        return x

    def forward(self, x: torch.Tensor, **kwargs):
        with_dconv = (self.d_conv > 1)
        x = self.in_proj(x)
        if not self.disable_z:
            x, z = x.chunk(2, dim=-1)  # (b, h, w, d)
            if not self.disable_z_act:
                z = self.act(z)
        if with_dconv:
            x = x.permute(0, 3, 1, 2).contiguous()
            x = self.conv2d(x)  # (b, d, h, w)
        x = self.act(x)
        y = self.forward_core(x, channel_first=with_dconv)
        if not self.disable_z:
            y = y * z
        out = self.dropout(self.out_proj(y))
        return out

class VSSBlock(nn.Module):
    def __init__(
            self,
            hidden_dim: int = 0,
            drop_path: float = 0,
            norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
            # =============================
            ssm_d_state: int = 16,
            ssm_ratio=2.0,
            ssm_dt_rank: Any = "auto",
            ssm_act_layer=nn.SiLU,
            ssm_conv: int = 3,
            ssm_conv_bias=True,
            ssm_drop_rate: float = 0,
            ssm_init="v0",
            forward_type="v2",
            # =============================
            mlp_ratio=4.0,
            mlp_act_layer=nn.GELU,
            mlp_drop_rate: float = 0.0,
            # =============================
            use_checkpoint: bool = False,
            post_norm: bool = False,
            **kwargs,
    ):
        super().__init__()
        self.ssm_branch = ssm_ratio > 0
        self.mlp_branch = mlp_ratio > 0
        self.use_checkpoint = use_checkpoint
        self.post_norm = post_norm

        try:
            from ss2d_ablations import SS2DDev
            _OSSM = SS2DDev if forward_type.startswith("dev") else OSSM
        except:
            _OSSM = OSSM

        if self.ssm_branch:
            self.norm = nn.LayerNorm(hidden_dim)
            self.op = _OSSM(
                d_model=hidden_dim,
                d_state=ssm_d_state,
                ssm_ratio=ssm_ratio,
                dt_rank=ssm_dt_rank,
                act_layer=ssm_act_layer,
                # ==========================
                d_conv=ssm_conv,
                conv_bias=ssm_conv_bias,
                # ==========================
                dropout=ssm_drop_rate,
                # bias=False,
                # ==========================
                # dt_min=0.001,
                # dt_max=0.1,
                # dt_init="random",
                # dt_scale="random",
                # dt_init_floor=1e-4,
                initialize=ssm_init,
                # ==========================
                forward_type=forward_type,
            )

        self.drop_path = DropPath(drop_path)

        if self.mlp_branch:
            self.norm2 = nn.LayerNorm(hidden_dim)
            mlp_hidden_dim = int(hidden_dim * mlp_ratio)
            self.mlp = Mlp(in_features=hidden_dim, hidden_features=mlp_hidden_dim, act_layer=mlp_act_layer,
                           drop=mlp_drop_rate, channels_first=False)

    def _forward(self, input: torch.Tensor):
        if self.ssm_branch:
            if self.post_norm:
                x = input + self.drop_path(self.norm(self.op(input)))
            else:
                x = input + self.drop_path(self.op(self.norm(input)))
        if self.mlp_branch:
            if self.post_norm:
                x = x + self.drop_path(self.norm2(self.mlp(x)))  # FFN
            else:
                x = x + self.drop_path(self.mlp(self.norm2(x)))  # FFN
        return x

    def forward(self, input: torch.Tensor):
        if self.use_checkpoint:
            return checkpoint.checkpoint(self._forward, input)
        else:
            return self._forward(input)

class DPCD(nn.Module):
    """ Change detection model

    Input :obj:`t1_img` and :obj:`t2_img`, extract encoder feature by :obj:`en_block1-4`,
    then exchange channel feature of :obj:`t1_feature` and :obj:`t2_feature`, and extract
    encoder feature by :obj:`en_block5`.

    Upsample to get decoder feature by :obj:`de_block1-3`, get :obj:`seg_feature1` and :obj:`seg_feature2`
    by :obj:`seg_out1` and :obj:`seg_out2`.

    Fuse t1 and t2 corresponding feature to get change feature by :obj:`dpfa` and :obj:`change_blcok`.

    Notice that output of module and model could be log in this model.

    Attribute:
        en_block(class): encoder feature extractor.
        channel_exchange(class): exchange t1 and t2 feature.
        de_block(class): decoder feature upsampler and extractor.
        dpfa(class): fuse t1 and t2 feature to get change feature
            using both spatial and channel attention.
        change_block(class): change feature upsampler and extracor.
        seg_out(class): get decoder feature seg out result.
        upsample_x2(class): upsample change feature by 2.
        conv_out_change(class): conv out change feature out result.
    """

    def __init__(self,
                 patch_size=4,
                 in_chans=3,
                 num_classes=1000,
                 depths=[2, 2, 9, 2],
                 dims=[96, 192, 384, 768],
                 ssm_d_state=16,

                 ssm_ratio=2.0,
                 ssm_dt_rank="auto",
                 ssm_act_layer="silu",
                 ssm_conv=3,
                 ssm_conv_bias=True,
                 ssm_drop_rate=0.0,
                 ssm_init="v0",
                 forward_type="v2",

                 mlp_ratio=4.0,
                 mlp_act_layer="gelu",
                 mlp_drop_rate=0.0,

                 drop_path_rate=0.1,
                 patch_norm=True,
                 norm_layer="LN",
                 downsample_version: str = "v2",  # "v1", "v2", "v3"
                 patchembed_version: str = "v1",  # "v1", "v2"
                 use_checkpoint=False,

                 ):
        super().__init__()
        self.num_classes = num_classes
        self.num_layers = len(depths)
        if isinstance(dims, int):
            dims = [int(dims * 2 ** i_layer) for i_layer in range(self.num_layers)]
        self.num_features = dims[-1]
        self.dims = dims

        mask_in_chans  = 16
        channel_list = [32, 64, 128, 256, 512]
        # encoder
        self.en_block1 = nn.Sequential(Conv_BN_ReLU(in_channel=3, out_channel=channel_list[0], kernel=3, stride=1),
                                       CGSU(in_channel=channel_list[0]),
                                       CGSU(in_channel=channel_list[0]),
                                       )
        self.en_block2 = Encoder_Block(in_channel=channel_list[0], out_channel=channel_list[1])
        self.en_block3 = Encoder_Block(in_channel=channel_list[1], out_channel=channel_list[2])
        self.en_block4 = Encoder_Block(in_channel=channel_list[2], out_channel=channel_list[3])
        self.en_block5 = Encoder_Block(in_channel=channel_list[3], out_channel=channel_list[4])

        self.channel_exchange4 = Changer_channel_exchange()

        # decoder
        # self.de_block1 = Decoder_Block(in_channel=channel_list[4], out_channel=channel_list[3])
        # self.de_block2 = Decoder_Block(in_channel=channel_list[3], out_channel=channel_list[2])
        # self.de_block3 = Decoder_Block(in_channel=channel_list[2], out_channel=channel_list[1])

        self.de_block1 = Decoder_Block(in_channel=self.dims[3], out_channel=self.dims[2])
        self.de_block2 = Decoder_Block(in_channel=self.dims[2], out_channel=self.dims[1])
        self.de_block3 = Decoder_Block(in_channel=self.dims[1], out_channel=self.dims[0])

        # dpfa
        # self.conv_replace_DPFA1 = nn.Conv2d(in_channels=channel_list[4]*2, out_channels=channel_list[4], kernel_size=3, stride=1, padding=1)
        # self.conv_replace_DPFA2 = nn.Conv2d(in_channels=channel_list[3]*2, out_channels=channel_list[3], kernel_size=3, stride=1, padding=1)
        # self.conv_replace_DPFA3 = nn.Conv2d(in_channels=channel_list[2]*2, out_channels=channel_list[2], kernel_size=3, stride=1, padding=1)
        # self.conv_replace_DPFA4 = nn.Conv2d(in_channels=channel_list[1]*2, out_channels=channel_list[1], kernel_size=3, stride=1, padding=1)


        # self.dpfa1 = DPFA(in_channel=channel_list[4])
        # self.dpfa2 = DPFA(in_channel=channel_list[3])
        # self.dpfa3 = DPFA(in_channel=channel_list[2])
        # self.dpfa4 = DPFA(in_channel=channel_list[1])

        self.dpfa1 = DPFA(in_channel=self.dims[3])
        self.dpfa2 = DPFA(in_channel=self.dims[2])
        self.dpfa3 = DPFA(in_channel=self.dims[1])
        self.dpfa4 = DPFA(in_channel=self.dims[0])

        # change path
        # the change block is the same as decoder block
        # the change block is used to fuse former and latter change features
        # self.change_block4 = Decoder_Block(in_channel=channel_list[4], out_channel=channel_list[3])
        # self.change_block3 = Decoder_Block(in_channel=channel_list[3], out_channel=channel_list[2])
        # self.change_block2 = Decoder_Block(in_channel=channel_list[2], out_channel=channel_list[1])

        self.change_block4 = Decoder_Block(in_channel=self.dims[3], out_channel=self.dims[2])
        self.change_block3 = Decoder_Block(in_channel=self.dims[2], out_channel=self.dims[1])
        self.change_block2 = Decoder_Block(in_channel=self.dims[1], out_channel=self.dims[0])

        # self.seg_out1 = nn.Conv2d(channel_list[1], 1, kernel_size=3, stride=1, padding=1)
        # self.seg_out2 = nn.Conv2d(channel_list[1], 1, kernel_size=3, stride=1, padding=1)

        self.seg_out1 = nn.Sequential(
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(self.dims[0], 1, kernel_size=3, stride=1, padding=1),
        )
        self.seg_out2 = nn.Sequential(
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(self.dims[0], 1, kernel_size=3, stride=1, padding=1),
        )

        self.upsample_x4 = nn.Sequential(
            nn.Conv2d(self.dims[0], 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 8, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),
            nn.UpsamplingBilinear2d(scale_factor=4)
        )
        self.conv_out_change = nn.Conv2d(8, 1, kernel_size=7, stride=1, padding=3)
        #
        # self.alpha = nn.Parameter(torch.zeros(1))
        # self.Sigmoid = nn.Sigmoid()
        #
        # self.ex_en_block1 = nn.Sequential(Conv_BN_ReLU(in_channel=3, out_channel=channel_list[0], kernel=3, stride=1),
        #                                CGSU(in_channel=channel_list[0]),
        #                                CGSU(in_channel=channel_list[0]),
        #                                )
        # self.ex_en_block2 = Encoder_Block(in_channel=channel_list[0], out_channel=channel_list[1])
        # self.ex_en_block3 = Encoder_Block(in_channel=channel_list[1], out_channel=channel_list[2])
        # self.ex_en_block4 = Encoder_Block(in_channel=channel_list[2], out_channel=channel_list[3])
        # self.ex_en_block5 = Encoder_Block(in_channel=channel_list[3], out_channel=channel_list[4])
        #
        # self.mask_downscaling1 = nn.Sequential(
        #     nn.Conv2d(1, mask_in_chans // 4, kernel_size=3, stride=1, padding=1),
        #     nn.BatchNorm2d(mask_in_chans // 4),
        #     nn.GELU(),
        #     nn.Conv2d(mask_in_chans // 4, mask_in_chans, kernel_size=3, stride=1, padding=1),
        #     nn.BatchNorm2d(mask_in_chans),
        #     nn.GELU(),
        #     nn.Conv2d(mask_in_chans, channel_list[0], kernel_size=1),
        # )
        #
        # self.mask_conv1 = make_maskembed_layer(in_chans=channel_list[0], out_chans=channel_list[1])
        # self.mask_conv2 = make_maskembed_layer(in_chans=channel_list[1], out_chans=channel_list[2])
        # self.mask_conv3 = make_maskembed_layer(in_chans=channel_list[2], out_chans=channel_list[3])
        # self.mask_conv4 = make_maskembed_layer(in_chans=channel_list[3], out_chans=channel_list[4])
        # # #
        # self.fuse_conv = nn.ModuleList()
        # for i in range(5):
        #     conv = nn.Sequential(
        #         nn.Conv2d(in_channels=channel_list[i] * 2, out_channels=channel_list[i], kernel_size=3, stride=1, padding=1)
        #     )
        #     self.fuse_conv.append(conv)
        # self.fuse_conv1  = nn.Conv2d(in_channels=channel_list[0] * 2, out_channels=channel_list[0], kernel_size=3, stride=1, padding=1)
        # self.fuse_conv2  = nn.Conv2d(in_channels=channel_list[1] * 2, out_channels=channel_list[1], kernel_size=3, stride=1, padding=1)
        # self.fuse_conv3  = nn.Conv2d(in_channels=channel_list[2] * 2, out_channels=channel_list[3], kernel_size=3, stride=1, padding=1)
        # self.fuse_conv4  = nn.Conv2d(in_channels=channel_list[3] * 2, out_channels=channel_list[3], kernel_size=3, stride=1, padding=1)
        # self.fuse_conv5  = nn.Conv2d(in_channels=channel_list[4] * 2, out_channels=channel_list[4], kernel_size=3, stride=1, padding=1)

        """mamba block"""


        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        _make_patch_embed = dict(
            v1=self._make_patch_embed,
            v2=self._make_patch_embed_v2,
        ).get(patchembed_version, None)
        #v2
        self.patch_embed = _make_patch_embed(in_chans, dims[0], patch_size, True)


        self.encoder_layers = []
        self.decoder_layers = []
        _make_downsample = dict(
            v2=self._make_downsample,
            v3=self._make_downsample_v3,
            none=(lambda *_, **_k: None),
        ).get(downsample_version, None)

        for i_layer in range(self.num_layers):
            downsample = _make_downsample(
                self.dims[i_layer - 1],
                self.dims[i_layer],
                norm_layer=norm_layer,
            ) if (i_layer != 0) else nn.Identity()
            self.encoder_layers.append(self._make_layer(
                dim=self.dims[i_layer],
                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                use_checkpoint=use_checkpoint,
                norm_layer=norm_layer,
                downsample=downsample,
                # =================
                ssm_d_state=ssm_d_state,
                ssm_ratio=ssm_ratio,
                ssm_dt_rank=ssm_dt_rank,
                ssm_act_layer=ssm_act_layer,
                ssm_conv=ssm_conv,
                ssm_conv_bias=ssm_conv_bias,
                ssm_drop_rate=ssm_drop_rate,
                ssm_init=ssm_init,
                forward_type=forward_type,
                # =================
                mlp_ratio=mlp_ratio,
                mlp_act_layer=mlp_act_layer,
                mlp_drop_rate=mlp_drop_rate,
            ))
        #     if i_layer != 0:
        #         self.decoder_layers.append(
        #             Decoder_Block(in_channel=self.dims[i_layer], out_channel=self.dims[i_layer - 1]))
        #
        # self.decoder_layers = []

        self.encoder_block1, self.encoder_block2, self.encoder_block3, self.encoder_block4 = self.encoder_layers
        # self.deocder_block1, self.deocder_block2, self.deocder_block3 = self.decoder_layers


        # init parameters
        # using pytorch default init is enough
        # self.init_params()

    @staticmethod
    def _make_patch_embed(in_chans=3, embed_dim=96, patch_size=4, patch_norm=True):
        return nn.Sequential(
            nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size, bias=True),
            Permute(0, 2, 3, 1),
            (nn.LayerNorm(embed_dim) if patch_norm else nn.Identity()),
        )
    @staticmethod
    def _make_patch_embed_v2(in_chans=3, embed_dim=96, patch_size=4, patch_norm=True):
        assert patch_size == 4
        return nn.Sequential(
            nn.Conv2d(in_chans, embed_dim // 2, kernel_size=3, stride=2, padding=1),
            (Permute(0, 2, 3, 1) if patch_norm else nn.Identity()),
            (nn.LayerNorm(embed_dim // 2) if patch_norm else nn.Identity()),
            (Permute(0, 3, 1, 2) if patch_norm else nn.Identity()),
            nn.GELU(),
            nn.Conv2d(embed_dim // 2, embed_dim, kernel_size=3, stride=2, padding=1),
            Permute(0, 2, 3, 1),
            (nn.LayerNorm(embed_dim) if patch_norm else nn.Identity()),
        )
    @staticmethod
    def _make_layer(
            dim=96,
            drop_path=[0.1, 0.1],
            use_checkpoint=False,
            norm_layer=nn.LayerNorm,
            downsample=nn.Identity(),
            # ===========================
            ssm_d_state=16,
            ssm_ratio=2.0,
            ssm_dt_rank="auto",
            ssm_act_layer=nn.SiLU,
            ssm_conv=3,
            ssm_conv_bias=True,
            ssm_drop_rate=0.0,
            ssm_init="v0",
            forward_type="v2",
            # ===========================
            mlp_ratio=4.0,
            mlp_act_layer=nn.GELU,
            mlp_drop_rate=0.0,
            **kwargs,
    ):
        depth = len(drop_path)
        blocks = []
        for d in range(depth):
            blocks.append(VSSBlock(
                hidden_dim=dim,
                drop_path=drop_path[d],
                norm_layer=norm_layer,
                ssm_d_state=ssm_d_state,
                ssm_ratio=ssm_ratio,
                ssm_dt_rank=ssm_dt_rank,
                ssm_act_layer=ssm_act_layer,
                ssm_conv=ssm_conv,
                ssm_conv_bias=ssm_conv_bias,
                ssm_drop_rate=ssm_drop_rate,
                ssm_init=ssm_init,
                forward_type=forward_type,
                mlp_ratio=mlp_ratio,
                mlp_act_layer=mlp_act_layer,
                mlp_drop_rate=mlp_drop_rate,
                use_checkpoint=use_checkpoint,
            ))

        return nn.Sequential(OrderedDict(
            # ZSJ 把downsample放到前面来，方便我取出encoder中每个尺度处理好的图像，而不是刚刚下采样完的图像
            downsample=downsample,
            blocks=nn.Sequential(*blocks, ),
        ))

    @staticmethod
    def _make_downsample(dim=96, out_dim=192, norm_layer=nn.LayerNorm):
        return nn.Sequential(
            Permute(0, 3, 1, 2),
            nn.Conv2d(dim, out_dim, kernel_size=2, stride=2),
            Permute(0, 2, 3, 1),
            nn.LayerNorm(out_dim),
        )

    @staticmethod
    def _make_downsample_v3(dim=96, out_dim=192, norm_layer=nn.LayerNorm):
        return nn.Sequential(
            Permute(0, 3, 1, 2),
            nn.Conv2d(dim, out_dim, kernel_size=3, stride=2, padding=1),
            Permute(0, 2, 3, 1),
            nn.LayerNorm(out_dim),
        )


    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv1d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def image_random_exchange(self, image1, image2):
        b, h, w, c = image1.shape
        l1 = []
        l2 = []
        for i in range(b):
            temp1 = image1[i].cuda()
            temp2 = image2[i].cuda()

            mask = torch.zeros(h, w).cuda()
            mask.bernoulli_(0.5)
            mask = torch.reshape(mask, (h, w, 1))
            mask = mask.bool()

            out1 = torch.add(torch.mul(temp1, mask), torch.mul(temp2, ~mask))
            out2 = torch.add(torch.mul(temp2, mask), torch.mul(temp1, ~mask))

            l1.append(out1)
            l2.append(out2)

        out1 = l1[-1].unsqueeze(dim=0)
        out2 = l2[-1].unsqueeze(dim=0)
        for i in range(len(l1) - 1):
            nn = l1[i].unsqueeze(dim=0)
            out1 = torch.cat((out1, nn), dim=0)
            nn = l2[i].unsqueeze(dim=0)
            out2 = torch.cat((out2, nn), dim=0)

        return out1, out2

    def forward(self, t1, t2, maskA, maskB, log=False, img_name=None):
        """ Model forward and log feature if :obj:`log: is True.

        If :obj:`log` is True, some module output and model output will be saved.

        To be more specific, module output will be saved in folder named
        `:obj:`module_input_feature_name`_:obj:`module_name`-
        :obj:`module_input_feature_name`_:obj:`module_name`- ...
        _:obj:`log_feature_name``. For example, module output saved folder could be named
        `t1_1_en_block2-x_cbam-spatial_weight`.

        Module output in saved folder will have the same name as corresponding input image.

        Model output saved folder will be simply named `model_:obj:`log_feature_name``. For example,
        it could be `model_seg_out_1`.

        Model output in saved folder will have the same name as corresponding input image.

        :obj:`seg_out1` and :obj:`seg_out2` could be used in loss function to train model better,
        and :obj:`change_out` is the prediction of model.

        Parameter:
            t1(tensor): input t1 image.
            t2(tensor): input t2 image.
            log(bool): if True, log output of module and model.
            img_name(tensor): name of input image.

        Return:
            change_out(tensor): change prediction of model.
            seg_out1(tensor): auxiliary change prediction through t1 decoder branch.
            seg_out2(tensor): auxiliary change prediction through t2 decoder branch.
        """

        """mamba block"""
        t1 = self.patch_embed(t1)
        t2 = self.patch_embed(t2)

        t1_1 = self.encoder_block1(t1)
        t2_1 = self.encoder_block1(t2)

        t1_2 = self.encoder_block2(t1_1)
        t2_2 = self.encoder_block2(t2_1)

        t1_3 = self.encoder_block3(t1_2)
        t2_3 = self.encoder_block3(t2_2)

        t1_4 = self.encoder_block4(t1_3)
        t2_4 = self.encoder_block4(t2_3)

        # maskA = maskA.unsqueeze(1)
        # maskB = maskB.unsqueeze(1)
        # fm1_0 = self.mask_downscaling1(maskA)
        # fm2_0 = self.mask_downscaling1(maskB)
        #
        # fm1_1 = self.mask_conv1(fm1_0)
        # fm1_2 = self.mask_conv2(fm1_1)
        # fm1_3 = self.mask_conv3(fm1_2)
        # fm1_4 = self.mask_conv4(fm1_3)
        #
        # fm2_1 = self.mask_conv1(fm2_0)
        # fm2_2 = self.mask_conv2(fm2_1)
        # fm2_3 = self.mask_conv3(fm2_2)
        # fm2_4 = self.mask_conv4(fm2_3)
        #
        # # encoder
        # ex_1, ex_2 = self.image_random_exchange(t1, t2)
        # t1_1 = self.en_block1(t1)
        # t2_1 = self.en_block1(t2)

        # if log:
        #     t1_2 = self.en_block2(t1_1, log=log, module_name='t1_1_en_block2', img_name=img_name)
        #     t2_2 = self.en_block2(t2_1, log=log, module_name='t2_1_en_block2', img_name=img_name)
        #
        #     t1_3 = self.en_block3(t1_2, log=log, module_name='t1_2_en_block3', img_name=img_name)
        #     t2_3 = self.en_block3(t2_2, log=log, module_name='t2_2en_block3', img_name=img_name)
        #
        #     t1_4 = self.en_block4(t1_3, log=log, module_name='t1_3_en_block4', img_name=img_name)
        #     t2_4 = self.en_block4(t2_3, log=log, module_name='t2_3_en_block4', img_name=img_name)
        #     t1_4, t2_4 = self.channel_exchange4(t1_4, t2_4)
        #
        #     t1_5 = self.en_block5(t1_4, log=log, module_name='t1_4_en_block5', img_name=img_name)
        #     t2_5 = self.en_block5(t2_4, log=log, module_name='t2_4_en_block5', img_name=img_name)
        # else:

        """encoder"""
        # t1_2 = self.en_block2(t1_1)
        # t2_2 = self.en_block2(t2_1)
        #
        # t1_3 = self.en_block3(t1_2)
        # t2_3 = self.en_block3(t2_2)
        #
        # t1_4 = self.en_block4(t1_3)
        # t2_4 = self.en_block4(t2_3)
        # t1_4, t2_4 = self.channel_exchange4(t1_4, t2_4)
        #
        # t1_5 = self.en_block5(t1_4)
        # t2_5 = self.en_block5(t2_4)

        # ext1_1 = self.ex_en_block1(ex_1)
        # ext2_1 = self.ex_en_block1(ex_2)
        #
        # ext1_2 = self.ex_en_block2(ext1_1)
        # ext2_2 = self.ex_en_block2(ext2_1)
        #
        # ext1_3 = self.ex_en_block3(ext1_2)
        # ext2_3 = self.ex_en_block3(ext2_2)
        #
        # ext1_4 = self.ex_en_block4(ext1_3)
        # ext2_4 = self.ex_en_block4(ext2_3)
        #
        # ext1_5 = self.en_block5(ext1_4)
        # ext2_5 = self.en_block5(ext2_4)
        #
        # ft1 = [t1_1, t1_2, t1_3, t1_4, t1_5]
        # ft2 = [t2_1, t2_2, t2_3, t2_4, t2_5]
        # #
        # fm1 = [fm1_0, fm1_1, fm1_2, fm1_3, fm1_4]
        # fm2 = [fm2_0, fm2_1, fm2_2, fm2_3, fm2_4]
        #
        # exft1 = [ext1_1, ext1_2, ext1_3, ext1_4, ext1_5]
        # exft2 = [ext2_1, ext2_2, ext2_3, ext2_4, ext2_5]
        #
        # self.gate = self.Sigmoid(self.alpha)
        # ft1 = [self.gate * x + (1 - self.gate) * y for x, y in zip(ft1, exft1)]
        # ft2 = [self.gate * x + (1 - self.gate) * y for x, y in zip(ft2, exft2)]
        # #
        # ft1 = [conv(torch.cat((x, y), dim=1)) for x, y, conv in zip(ft1, fm1, self.fuse_conv)]
        # ft2 = [conv(torch.cat((x, y), dim=1)) for x, y, conv in zip(ft2, fm2, self.fuse_conv)]

        #
        # t1_1, t1_2, t1_3, t1_4, t1_5 = ft1
        # t2_1, t2_2, t2_3, t2_4, t2_5 = ft2


        """decoder"""
        # de1_5 = t1_5
        # de2_5 = t2_5
        #
        # de1_4 = self.de_block1(de1_5, t1_4)
        # de2_4 = self.de_block1(de2_5, t2_4)
        #
        # de1_3 = self.de_block2(de1_4, t1_3)
        # de2_3 = self.de_block2(de2_4, t2_3)
        #
        # de1_2 = self.de_block3(de1_3, t1_2)
        # de2_2 = self.de_block3(de2_3, t2_2)
        #
        # seg_out1 = self.seg_out1(de1_2)
        # seg_out2 = self.seg_out2(de2_2)

        de1_4 = t1_4.permute(0,3,1,2)
        de2_4 = t2_4.permute(0,3,1,2)

        de1_3 = self.de_block1(de1_4, t1_3.permute(0,3,1,2))
        de2_3 = self.de_block1(de2_4, t2_3.permute(0,3,1,2))

        de1_2 = self.de_block2(de1_3, t1_2.permute(0,3,1,2))
        de2_2 = self.de_block2(de2_3, t2_2.permute(0,3,1,2))

        de1_1 = self.de_block3(de1_2, t1_1.permute(0,3,1,2))
        de2_1 = self.de_block3(de2_2, t2_1.permute(0,3,1,2))


        seg_out1 = self.seg_out1(de1_1)
        seg_out2 = self.seg_out2(de2_1)
        #
        # de1_2 = self.de_block3(de1_3, t1_2)
        # de2_2 = self.de_block3(de2_3, t2_2)
        # if log:
        #     change_5 = self.dpfa1(de1_5, de2_5, log=log, module_name='de1_5_de2_5_dpfa1',
        #                           img_name=img_name)
        #
        #     change_4 = self.change_block4(change_5, self.dpfa2(de1_4, de2_4, log=log, module_name='de1_4_de2_4_dpfa2',
        #                                                        img_name=img_name))
        #
        #     change_3 = self.change_block3(change_4, self.dpfa3(de1_3, de2_3, log=log, module_name='de1_3_de2_3_dpfa3',
        #                                                        img_name=img_name))
        #
        #     change_2 = self.change_block2(change_3, self.dpfa4(de1_2, de2_2, log=log, module_name='de1_2_de2_2_dpfa4',
        #                                                        img_name=img_name))
        # else:
        # change_5 = self.dpfa1(de1_5, de2_5)
        #
        # change_4 = self.change_block4(change_5, self.dpfa2(de1_4, de2_4))
        #
        # change_3 = self.change_block3(change_4, self.dpfa3(de1_3, de2_3))
        #
        # change_2 = self.change_block2(change_3,  self.dpfa4(de1_2, de2_2))

        change_4 = self.dpfa1(de1_4, de2_4)

        change_3 = self.change_block4(change_4, self.dpfa2(de1_3, de2_3))

        change_2 = self.change_block3(change_3, self.dpfa3(de1_2, de2_2))

        change_1 = self.change_block2(change_2, self.dpfa4(de1_1, de2_1))
        #
        # change_2 = self.change_block2(change_3, self.dpfa4(de1_2, de2_2))

        change = self.upsample_x4(change_1)
        change_out = self.conv_out_change(change)

        if log:
            log_feature(log_list=[change_out, seg_out1, seg_out2], module_name='model',
                        feature_name_list=['change_out', 'seg_out1', 'seg_out2'],
                        img_name=img_name, module_output=False)

        return change_out, seg_out1, seg_out2
