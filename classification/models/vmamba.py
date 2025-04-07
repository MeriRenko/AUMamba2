import os
import time
import math
import copy
from functools import partial
from typing import Optional, Callable, Any
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from timm.models.layers import DropPath, trunc_normal_
from fvcore.nn import FlopCountAnalysis, flop_count_str, flop_count, parameter_count

DropPath.__repr__ = lambda self: f"timm.DropPath({self.drop_prob})"
# train speed is slower after enabling this opts.
# torch.backends.cudnn.enabled = True
# torch.backends.cudnn.benchmark = True
# torch.backends.cudnn.deterministic = True

try:
    from .csm_triton import scan_fn, merge_fn
except:
    from csm_triton import scan_fn, merge_fn

try:
    from .csms6s import selective_scan_fn, selective_scan_flop_jit
except:
    from csms6s import selective_scan_fn, selective_scan_flop_jit

# FLOPs counter not prepared fro mamba2
try:
    from .mamba2.ssd_minimal import selective_scan_chunk_fn
except:
    from mamba2.ssd_minimal import selective_scan_chunk_fn


# =====================================================
# we have this class as linear and conv init differ from each other
# this function enable loading from both conv2d or linear
# Linear2d 继承自 torch.nn.Linear，但它 用 conv2d 实现了全连接层
class Linear2d(nn.Linear):
    def forward(self, x: torch.Tensor):
        # B, C, H, W = x.shape
        # 使用 conv2d 代替 linear 运算
        # self.weight 原本是 nn.Linear 里的 2D 权重矩阵，形状 (out_features, in_features)
        # [:, :, None, None]：增加两个维度，使 weight 变成 (out_features, in_features, 1, 1)，即 一个 1×1 卷积核
        # 1×1 卷积和 linear 层本质上等价，当卷积核大小为 1×1 时，相当于对每个像素位置单独执行 linear 计算
        return F.conv2d(x, self.weight[:, :, None, None], self.bias)
        # 在加载模型时，确保 weight 形状正确
    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        # PyTorch 在存储 nn.Linear 的 state_dict 时，权重是 2D 矩阵 (out_features, in_features)。
        # 但是 Linear2d 期望的 weight 是 (out_features, in_features, 1, 1)（由于 conv2d）。
        # view(self.weight.shape) 确保 weight 在加载时转换成 (out_features, in_features, 1, 1) 的形状，以适配 conv2d
        # state_dict[prefix + "weight"] = state_dict[prefix + "weight"].view(self.weight.shape)
        return super()._load_from_state_dict(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)


class LayerNorm2d(nn.LayerNorm):
    def forward(self, x: torch.Tensor):
        x = x.permute(0, 2, 3, 1)
        x = nn.functional.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        x = x.permute(0, 3, 1, 2)
        return x


class PatchMerging2D(nn.Module):
    def __init__(self, dim, out_dim=-1, norm_layer=nn.LayerNorm, channel_first=False):
        super().__init__()
        self.dim = dim
        Linear = Linear2d if channel_first else nn.Linear
        self._patch_merging_pad = self._patch_merging_pad_channel_first if channel_first else self._patch_merging_pad_channel_last
        self.reduction = Linear(4 * dim, (2 * dim) if out_dim < 0 else out_dim, bias=False)
        self.norm = norm_layer(4 * dim)

    @staticmethod
    def _patch_merging_pad_channel_last(x: torch.Tensor):
        H, W, _ = x.shape[-3:]
        if (W % 2 != 0) or (H % 2 != 0):
            x = F.pad(x, (0, 0, 0, W % 2, 0, H % 2))
        x0 = x[..., 0::2, 0::2, :]  # ... H/2 W/2 C
        x1 = x[..., 1::2, 0::2, :]  # ... H/2 W/2 C
        x2 = x[..., 0::2, 1::2, :]  # ... H/2 W/2 C
        x3 = x[..., 1::2, 1::2, :]  # ... H/2 W/2 C
        x = torch.cat([x0, x1, x2, x3], -1)  # ... H/2 W/2 4*C
        return x

    @staticmethod
    def _patch_merging_pad_channel_first(x: torch.Tensor):
        H, W = x.shape[-2:]
        if (W % 2 != 0) or (H % 2 != 0):
            x = F.pad(x, (0, 0, 0, W % 2, 0, H % 2))
        x0 = x[..., 0::2, 0::2]  # ... H/2 W/2
        x1 = x[..., 1::2, 0::2]  # ... H/2 W/2
        x2 = x[..., 0::2, 1::2]  # ... H/2 W/2
        x3 = x[..., 1::2, 1::2]  # ... H/2 W/2
        x = torch.cat([x0, x1, x2, x3], 1)  # ... H/2 W/2 4*C
        return x

    def forward(self, x):
        x = self._patch_merging_pad(x)
        x = self.norm(x)
        x = self.reduction(x)

        return x


class Permute(nn.Module):
    def __init__(self, *args):
        super().__init__()
        self.args = args

    def forward(self, x: torch.Tensor):
        return x.permute(*self.args)


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.,channels_first=False):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        Linear = Linear2d if channels_first else nn.Linear
        self.fc1 = Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class gMlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.,channels_first=False):
        super().__init__()
        self.channel_first = channels_first
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        Linear = Linear2d if channels_first else nn.Linear
        self.fc1 = Linear(in_features, 2 * hidden_features)
        self.act = act_layer()
        self.fc2 = Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor):
        x = self.fc1(x)
        x, z = x.chunk(2, dim=(1 if self.channel_first else -1))
        x = self.fc2(x * self.act(z))
        x = self.drop(x)
        return x


class SoftmaxSpatial(nn.Softmax):
    def forward(self, x: torch.Tensor):
        if self.dim == -1:
            B, C, H, W = x.shape
            return super().forward(x.view(B, C, -1)).view(B, C, H, W)
        elif self.dim == 1:
            B, H, W, C = x.shape
            return super().forward(x.view(B, -1, C)).view(B, H, W, C)
        else:
            raise NotImplementedError


# =====================================================
class mamba_init:
    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4):
        # dt_rank 输入特征的维度
        # d_inner 输出特征的维度
        # dt_scale: 影响初始化标准差的缩放因子（默认 1.0）
        # dt_init: 控制权重初始化方式，支持 "constant" 和 "random"
        # dt_min, dt_max: 控制 dt_bias 的范围（经过 softplus 变换后）
        # dt_init_floor: 确保 dt 不小于 1e-4，防止数值太小导致梯度消失

        # dt_proj 是一个线性变换层（dt_rank × d_inner），会学习一个权重矩阵 W 和一个偏置 b
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True)

        # Initialize special dt projection to preserve variance at initialization
        # 这个初始化策略 类似于 Kaiming/Xavier 初始化，用来控制权重的方差，确保前向传播时信号不会被放大或缩小
        dt_init_std = dt_rank**-0.5 * dt_scale
        # 如果 dt_init == "constant"，则所有权重都初始化为 dt_init_std
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        # 如果 dt_init == "random"，则使用 均匀分布 U(-dt_init_std, dt_init_std) 初始化权重
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        # 偏置 dt_proj.bias 的初始化
        # 让 dt 是位于 dt_min 到 dt_max 之间，且是对数均匀分布
        dt = torch.exp(
            torch.rand(d_inner) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        # 计算 dt_proj.bias
        # torch.expm1(x) = exp(x) - 1
        # 这个公式是 softplus 的逆函数，用于确保 F.softplus(dt_bias) = dt
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        # 赋值到 dt_proj.bias
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        # dt_proj.bias._no_reinit = True
        # 返回这个特殊初始化的 Linear 层，可以直接用于神经网络
        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=-1, device=None, merge=True):
        # d_state：状态维度（State Size）。
        # d_inner：隐藏层维度（Inner Dimension）。
        # copies（默认 -1）：用于重复 A_log 以支持多个 copies（如不同头的共享参数）。
        # device：指定计算设备（如 cuda 或 cpu）。
        # merge（默认 True）：如果 copies > 0，是否展平（flatten）A_log 以合并多个 copies。

        # S4D real initialization
        # torch.arange(1, d_state + 1): 生成从 1 到 d_state（包含 d_state）的序列
        # .view(1, -1): 变成形状 (1, d_state)，即 行向量
        # .repeat(d_inner, 1): 在第 0 维重复 d_inner 次，最终形状变为 (d_inner, d_state)
        A = torch.arange(1, d_state + 1, dtype=torch.float32, device=device).view(1, -1).repeat(d_inner, 1).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        if copies > 0:
            # A_log[None]：在 dim=0 维度上增加一维，形状变为 (1, d_inner, d_state)
            # .repeat(copies, 1, 1): 复制 copies 份，变成 (copies, d_inner, d_state)
            A_log = A_log[None].repeat(copies, 1, 1).contiguous()
            # 如果 merge=True，则 合并 copies 维度，.flatten(0, 1) 是 PyTorch 中的一个张量操作，它的作用是 将第 0 维和第 1 维合并
            if merge:
                A_log = A_log.flatten(0, 1)     # 形状变为 (copies * d_inner, d_state)
        # 让 A_log 成为 可训练参数，用于梯度更新
        A_log = nn.Parameter(A_log)
        # 禁止权重衰减
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=-1, device=None, merge=True):
        # D "skip" parameter
        D = torch.ones(d_inner, device=device)
        if copies > 0:
            D = D[None].repeat(copies, 1).contiguous()  # (copies, d_inner)
            if merge:
                D = D.flatten(0, 1)                     # (copies*d_inner)
        D = nn.Parameter(D)  # Keep in fp32
        D._no_weight_decay = True
        return D

    @classmethod
    def init_dt_A_D(cls, d_state, dt_rank, d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, k_group=4):
        # dt proj ============================
        dt_projs = [
            cls.dt_init(dt_rank, d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor)
            for _ in range(k_group)
        ]
        dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in dt_projs], dim=0)) # (K, inner, rank)
        dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in dt_projs], dim=0)) # (K, inner)
        del dt_projs
            
        # A, D =======================================
        A_logs = cls.A_log_init(d_state, d_inner, copies=k_group, merge=True) # (K * d_state, d_inner)
        Ds = cls.D_init(d_inner, copies=k_group, merge=True) # (K * d_state)  
        return A_logs, Ds, dt_projs_weight, dt_projs_bias


# support: v0, v0seq
class SS2Dv0:
    def __initv0__(
        self,
        # basic dims ===========
        d_model=96,
        d_state=16,
        ssm_ratio=2.0,
        dt_rank="auto",
        # ======================
        dropout=0.0,
        # ======================
        seq=False,
        force_fp32=True,
        **kwargs,
    ):
        if "channel_first" in kwargs:
            assert not kwargs["channel_first"]
        act_layer = nn.SiLU
        dt_min = 0.001
        dt_max = 0.1
        dt_init = "random"
        dt_scale = 1.0
        dt_init_floor = 1e-4
        bias = False
        conv_bias = True
        d_conv = 3
        k_group = 4
        factory_kwargs = {"device": None, "dtype": None}
        super().__init__()
        d_inner = int(ssm_ratio * d_model)
        dt_rank = math.ceil(d_model / 16) if dt_rank == "auto" else dt_rank

        self.forward = self.forwardv0 
        if seq:
            self.forward = partial(self.forwardv0, seq=True)
        if not force_fp32:
            self.forward = partial(self.forwardv0, force_fp32=False)

        # in proj ============================
        self.in_proj = nn.Linear(d_model, d_inner * 2, bias=bias)
        self.act: nn.Module = act_layer()
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
            nn.Linear(d_inner, (dt_rank + d_state * 2), bias=False)
            for _ in range(k_group)
        ]
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0)) # (K, N, inner)
        del self.x_proj

        # dt proj, A, D ============================
        self.A_logs, self.Ds, self.dt_projs_weight, self.dt_projs_bias = mamba_init.init_dt_A_D(
            d_state, dt_rank, d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, k_group=4,
        )

        # out proj =======================================
        self.out_norm = nn.LayerNorm(d_inner)
        self.out_proj = nn.Linear(d_inner, d_model, bias=bias)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else nn.Identity()

    def forwardv0(self, x: torch.Tensor, seq=False, force_fp32=True, **kwargs):
        x = self.in_proj(x)
        x, z = x.chunk(2, dim=-1) # (b, h, w, d)
        z = self.act(z)
        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.conv2d(x) # (b, d, h, w)
        x = self.act(x)
        selective_scan = partial(selective_scan_fn, backend="mamba")
        
        B, D, H, W = x.shape
        D, N = self.A_logs.shape
        K, D, R = self.dt_projs_weight.shape
        L = H * W

        x_hwwh = torch.stack([x.view(B, -1, L), torch.transpose(x, dim0=2, dim1=3).contiguous().view(B, -1, L)], dim=1).view(B, 2, -1, L)
        xs = torch.cat([x_hwwh, torch.flip(x_hwwh, dims=[-1])], dim=1) # (b, k, d, l)

        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs, self.x_proj_weight)
        if hasattr(self, "x_proj_bias"):
            x_dbl = x_dbl + self.x_proj_bias.view(1, K, -1, 1)
        dts, Bs, Cs = torch.split(x_dbl, [R, N, N], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts, self.dt_projs_weight)

        xs = xs.view(B, -1, L) # (b, k * d, l)
        dts = dts.contiguous().view(B, -1, L) # (b, k * d, l)
        Bs = Bs.contiguous() # (b, k, d_state, l)
        Cs = Cs.contiguous() # (b, k, d_state, l)
        
        As = -self.A_logs.float().exp() # (k * d, d_state)
        Ds = self.Ds.float() # (k * d)
        dt_projs_bias = self.dt_projs_bias.float().view(-1) # (k * d)

        # assert len(xs.shape) == 3 and len(dts.shape) == 3 and len(Bs.shape) == 4 and len(Cs.shape) == 4
        # assert len(As.shape) == 2 and len(Ds.shape) == 1 and len(dt_projs_bias.shape) == 1
        to_fp32 = lambda *args: (_a.to(torch.float32) for _a in args)
        
        if force_fp32:
            xs, dts, Bs, Cs = to_fp32(xs, dts, Bs, Cs)

        if seq:
            out_y = []
            for i in range(4):
                yi = selective_scan(
                    xs.view(B, K, -1, L)[:, i], dts.view(B, K, -1, L)[:, i], 
                    As.view(K, -1, N)[i], Bs[:, i].unsqueeze(1), Cs[:, i].unsqueeze(1), Ds.view(K, -1)[i],
                    delta_bias=dt_projs_bias.view(K, -1)[i],
                    delta_softplus=True,
                ).view(B, -1, L)
                out_y.append(yi)
            out_y = torch.stack(out_y, dim=1)
        else:
            out_y = selective_scan(
                xs, dts, 
                As, Bs, Cs, Ds,
                delta_bias=dt_projs_bias,
                delta_softplus=True,
            ).view(B, K, -1, L)
        assert out_y.dtype == torch.float

        inv_y = torch.flip(out_y[:, 2:4], dims=[-1]).view(B, 2, -1, L)
        wh_y = torch.transpose(out_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
        invwh_y = torch.transpose(inv_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
        y = out_y[:, 0] + inv_y[:, 0] + wh_y + invwh_y
        
        y = y.transpose(dim0=1, dim1=2).contiguous() # (B, L, C)
        y = self.out_norm(y).view(B, H, W, -1)

        y = y * z
        out = self.dropout(self.out_proj(y))
        return out


# support: v01-v05; v051d,v052d,v052dc; 
# postfix: _onsigmoid,_onsoftmax,_ondwconv3,_onnone;_nozact,_noz;_oact;_no32;
# history support: v2,v3;v31d,v32d,v32dc;
class SS2Dv2:
    def __initv2__(
        self,
        # basic dims ===========
        d_model=96,
        d_state=16,
        ssm_ratio=2.0,
        dt_rank="auto",
        act_layer=nn.SiLU,
        # dwconv ===============
        d_conv=3, # < 2 means no conv 
        conv_bias=True,             # 配置中设置为False
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
        forward_type="v2",          # 配置中设置为v05_noz
        channel_first=False,
        scan="hilbert",
        # ======================
        **kwargs,    
    ):
        factory_kwargs = {"device": None, "dtype": None}
        super().__init__()
        self.k_group = 4
        self.d_model = int(d_model)
        self.d_state = int(d_state)             # 状态空间模型的隐状态维度  d_state=1
        self.d_inner = int(ssm_ratio * d_model) # 计算 d_inner 作为 d_model 的扩展维度 d_inner=96
        self.dt_rank = int(math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank)  # dt_rank=6    # math.ceil(...)：向上取整，确保 dt_rank 仍然是整数，避免小数值
        self.channel_first = channel_first
        self.with_dconv = d_conv > 1            #d_conv=3
        Linear = Linear2d if channel_first else nn.Linear #Linear2D channel_first=True
        self.forward = self.forwardv2

        # tags for forward_type ==============================
        # 这行代码将 self.checkpostfix 绑定到局部变量 checkpostfix，这样在后续调用时可以少写 self.，提高可读性
        checkpostfix = self.checkpostfix
        # 这个调用 checkpostfix("_no32", forward_type)，检查 forward_type 是否包含 _no32 后缀
        self.disable_force32, forward_type = checkpostfix("_no32", forward_type)
        # self.oact=False
        self.oact, forward_type = checkpostfix("_oact", forward_type)
        # v05_noz   disable_z=True
        self.disable_z, forward_type = checkpostfix("_noz", forward_type)
        self.disable_z_act, forward_type = checkpostfix("_nozact", forward_type)
        # 直接使用LayerNorm
        self.out_norm, forward_type = self.get_outnorm(forward_type, self.d_inner, channel_first)

        # forward_type debug =======================================
        # FORWARD_TYPES 字典定义了不同的前向传播方式（v01、v02、v03...）
        # 主要使用 partial(self.forward_corev2, ...) 指定不同的计算模式
        FORWARD_TYPES = dict(
            v01=partial(self.forward_corev2, force_fp32=(not self.disable_force32), selective_scan_backend="mamba", scan_force_torch=True),
            v02=partial(self.forward_corev2, force_fp32=(not self.disable_force32), selective_scan_backend="mamba"),
            v03=partial(self.forward_corev2, force_fp32=(not self.disable_force32), selective_scan_backend="oflex"),
            v04=partial(self.forward_corev2, force_fp32=False), # selective_scan_backend="oflex", scan_mode="cross2d"
            v05=partial(self.forward_corev2, force_fp32=False, no_einsum=True,scan_mode=scan),  # selective_scan_backend="oflex", scan_mode="cross2d"
            # ===============================
            v051d=partial(self.forward_corev2, force_fp32=False, no_einsum=True, scan_mode="unidi"),
            v052d=partial(self.forward_corev2, force_fp32=False, no_einsum=True, scan_mode="bidi"),
            v052dc=partial(self.forward_corev2, force_fp32=False, no_einsum=True, scan_mode="cascade2d"),
            v052d3=partial(self.forward_corev2, force_fp32=False, no_einsum=True, scan_mode=3), # debug
            # ===============================
            v2=partial(self.forward_corev2, force_fp32=(not self.disable_force32), selective_scan_backend="core"),
            v3=partial(self.forward_corev2, force_fp32=False, selective_scan_backend="oflex"),
        )
        self.forward_core = FORWARD_TYPES.get(forward_type, None)   # 选择v05

        # in proj =======================================
        d_proj = self.d_inner if self.disable_z else (self.d_inner * 2)
        # in_proj 是一个 Linear（全连接层），输出维度：d_proj（由 disable_z 控制），
        self.in_proj = Linear(self.d_model, d_proj, bias=bias) 
        # 通过 act_layer() 创建，可变的激活函数（例如 ReLU()、GELU()等）
        self.act: nn.Module = act_layer()
        
        # conv =======================================
        if self.with_dconv:
            # 该卷积层是深度可分离卷积 Conv2d(96, 96, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=96, bias=False)
            self.conv2d = nn.Conv2d(
                # 设置输入和输出的通道数为 d_inner
                in_channels=self.d_inner,
                out_channels=self.d_inner,
                # 深度可分离卷积的关键，将 groups 设置为 in_channels，即每个输入通道都使用一个独立的卷积核,避免了不同通道间的卷积操作融合。
                groups=self.d_inner,
                bias=conv_bias,
                # 卷积核的大小由 d_conv 给出，通常为奇数，如3等。
                kernel_size=d_conv,
                # 常见的 对称填充 计算方式，适用于卷积核为奇数时,确保输出的空间尺寸与输入保持一致
                padding=(d_conv - 1) // 2,
                # 用于扩展 Conv2d 构造函数的额外参数，如激活函数、正则化等
                **factory_kwargs,
            )
        # x proj ============================
        # 这里创建了 k_group 个 Linear 变换，每个变换输入维度：self.d_inner，输出维度：self.dt_rank + self.d_state * 2
        self.x_proj = [
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False)
            for _ in range(self.k_group)
        ]
        # 将 k_group 个 Linear 层的权重合并成一个 Tensor，然后存入 x_proj_weight
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0)) # (K, self.dt_rank + self.d_state * 2, d_inner)
        # del self.x_proj 删除了 self.x_proj，节省内存
        del self.x_proj
        
        # out proj =======================================
        # 输出投影 (out_proj)
        self.out_act = nn.GELU() if self.oact else nn.Identity()        # Identity
        self.out_proj = Linear(self.d_inner, self.d_model, bias=bias)
        # nn.Dropout 是 PyTorch 中的一个 丢弃层（Dropout Layer），用于防止过拟合。在训练时，随机地将一部分神经元的输出设为零
        # dropout 是一个概率值，表示 每个神经元被丢弃的概率，通常在训练过程中使用，值范围是 (0, 1)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else nn.Identity()
        
        # initialize=v0
        if initialize in ["v0"]:
            self.A_logs, self.Ds, self.dt_projs_weight, self.dt_projs_bias = mamba_init.init_dt_A_D(
                self.d_state, self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, k_group=self.k_group,
            )
        elif initialize in ["v1"]:
            # simple init dt_projs, A_logs, Ds
            self.Ds = nn.Parameter(torch.ones((self.k_group * self.d_inner)))
            self.A_logs = nn.Parameter(torch.randn((self.k_group * self.d_inner, self.d_state))) # A == -A_logs.exp() < 0; # 0 < exp(A * dt) < 1
            self.dt_projs_weight = nn.Parameter(0.1 * torch.randn((self.k_group, self.d_inner, self.dt_rank))) # 0.1 is added in 0430
            self.dt_projs_bias = nn.Parameter(0.1 * torch.randn((self.k_group, self.d_inner))) # 0.1 is added in 0430
        elif initialize in ["v2"]:
            # simple init dt_projs, A_logs, Ds
            self.Ds = nn.Parameter(torch.ones((self.k_group * self.d_inner)))
            self.A_logs = nn.Parameter(torch.zeros((self.k_group * self.d_inner, self.d_state))) # A == -A_logs.exp() < 0; # 0 < exp(A * dt) < 1
            self.dt_projs_weight = nn.Parameter(0.1 * torch.rand((self.k_group, self.d_inner, self.dt_rank)))
            self.dt_projs_bias = nn.Parameter(0.1 * torch.rand((self.k_group, self.d_inner)))

    def forward_corev2(
        self,
        x: torch.Tensor=None, 
        # ==============================
        force_fp32=False, # True: input fp32                                        # 当其为 True 时，强制将输入数据转换为 float32 类型。
        # ==============================
        ssoflex=True, # True: input 16 or 32 output 32 False: output dtype as input
        no_einsum=False, # replace einsum with linear or conv1d to raise throughput # 表示是否禁用 einsum（爱因斯坦求和约定）。
        # ==============================
        selective_scan_backend = None,
        # ==============================
        scan_mode = "continuous",
        scan_force_torch = False,   # 决定是否强制使用 PyTorch 来执行扫描操作。
        # ==============================
        **kwargs,
    ):
        # 使用 assert 语句检查 selective_scan_backend 是否为合法的选项
        assert selective_scan_backend in [None, "oflex", "mamba", "torch"]
        # 解析 scan_mode 为整数值 _scan_mode，并且确保它是有效的
        _scan_mode = dict(cross2d=0, unidi=1, bidi=2, hilbert=3,diagonal=4,continuous=5,cascade2d=-1).get(scan_mode, None) if isinstance(scan_mode, str) else scan_mode # for debug
        assert isinstance(_scan_mode, int)
        delta_softplus = True
        out_norm = self.out_norm # LayerNorm2d((96,), eps=1e-05, elementwise_affine=True)
        channel_first = self.channel_first
        # lambda 函数 to_fp32，它的作用是将输入的多个张量转换为 torch.float32 数据类型
        # for _a in args 会遍历 args 中的每个元素（即每个张量）
        # _a.to(torch.float32) 会将张量 _a 转换为 torch.float32 类型
        to_fp32 = lambda *args: (_a.to(torch.float32) for _a in args)

        B, D, H, W = x.shape
        N = self.d_state
        K, D, R = self.k_group, self.d_inner, self.dt_rank
        L = H * W

        def selective_scan(u, delta, A, B, C, D=None, delta_bias=None, delta_softplus=True):
            return selective_scan_fn(u, delta, A, B, C, D, delta_bias, delta_softplus, ssoflex, backend=selective_scan_backend)
        
        if _scan_mode == -1:
            # 该行代码尝试从当前对象（通常是神经网络的模型）中获取 x_proj_bias 属性。如果该属性不存在，则返回 None。x_proj_bias 通常是一个偏置项，用于项目权重的计算
            x_proj_bias = getattr(self, "x_proj_bias", None)
            def scan_rowcol(
                x: torch.Tensor, 
                proj_weight: torch.Tensor, 
                proj_bias: torch.Tensor, 
                dt_weight: torch.Tensor, 
                dt_bias: torch.Tensor, # (2*c)
                _As: torch.Tensor, # As = -torch.exp(A_logs.to(torch.float))[:2,] # (2*c, d_state)
                _Ds: torch.Tensor,
                width = True,
            ):
                # x: (B, D, H, W)
                # proj_weight: (2 * D, (R+N+N))
                XB, XD, XH, XW = x.shape
                if width:
                    _B, _D, _L = XB * XH, XD, XW
                    xs = x.permute(0, 2, 1, 3).contiguous()
                else:
                    _B, _D, _L = XB * XW, XD, XH
                    xs = x.permute(0, 3, 1, 2).contiguous()
                xs = torch.stack([xs, xs.flip(dims=[-1])], dim=2) # (B, H, 2, D, W)
                if no_einsum:
                    x_dbl = F.conv1d(xs.view(_B, -1, _L), proj_weight.view(-1, _D, 1), bias=(proj_bias.view(-1) if proj_bias is not None else None), groups=2)
                    dts, Bs, Cs = torch.split(x_dbl.view(_B, 2, -1, _L), [R, N, N], dim=2)
                    dts = F.conv1d(dts.contiguous().view(_B, -1, _L), dt_weight.view(2 * _D, -1, 1), groups=2)
                else:
                    x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs, proj_weight)
                    if x_proj_bias is not None:
                        x_dbl = x_dbl + x_proj_bias.view(1, 2, -1, 1)
                    dts, Bs, Cs = torch.split(x_dbl, [R, N, N], dim=2)
                    dts = torch.einsum("b k r l, k d r -> b k d l", dts, dt_weight)

                xs = xs.view(_B, -1, _L)
                dts = dts.contiguous().view(_B, -1, _L)
                As = _As.view(-1, N).to(torch.float)
                Bs = Bs.contiguous().view(_B, 2, N, _L)
                Cs = Cs.contiguous().view(_B, 2, N, _L)
                Ds = _Ds.view(-1)
                delta_bias = dt_bias.view(-1).to(torch.float)

                if force_fp32:
                    xs = xs.to(torch.float)
                dts = dts.to(xs.dtype)
                Bs = Bs.to(xs.dtype)
                Cs = Cs.to(xs.dtype)

                ys: torch.Tensor = selective_scan(
                    xs, dts, As, Bs, Cs, Ds, delta_bias, delta_softplus
                ).view(_B, 2, -1, _L)
                return ys
            
            As = -self.A_logs.to(torch.float).exp().view(4, -1, N)
            x = F.layer_norm(x.permute(0, 2, 3, 1), normalized_shape=(int(x.shape[1]),)).permute(0, 3, 1, 2).contiguous() # added0510 to avoid nan
            y_row = scan_rowcol(
                x,
                proj_weight = self.x_proj_weight.view(4, -1, D)[:2].contiguous(), 
                proj_bias = (x_proj_bias.view(4, -1)[:2].contiguous() if x_proj_bias is not None else None),
                dt_weight = self.dt_projs_weight.view(4, D, -1)[:2].contiguous(),
                dt_bias = (self.dt_projs_bias.view(4, -1)[:2].contiguous() if self.dt_projs_bias is not None else None),
                _As = As[:2].contiguous().view(-1, N),
                _Ds = self.Ds.view(4, -1)[:2].contiguous().view(-1),
                width=True,
            ).view(B, H, 2, -1, W).sum(dim=2).permute(0, 2, 1, 3) # (B,C,H,W)
            y_row = F.layer_norm(y_row.permute(0, 2, 3, 1), normalized_shape=(int(y_row.shape[1]),)).permute(0, 3, 1, 2).contiguous() # added0510 to avoid nan
            y_col = scan_rowcol(
                y_row,
                proj_weight = self.x_proj_weight.view(4, -1, D)[2:].contiguous().to(y_row.dtype), 
                proj_bias = (x_proj_bias.view(4, -1)[2:].contiguous().to(y_row.dtype) if x_proj_bias is not None else None),
                dt_weight = self.dt_projs_weight.view(4, D, -1)[2:].contiguous().to(y_row.dtype),
                dt_bias = (self.dt_projs_bias.view(4, -1)[2:].contiguous().to(y_row.dtype) if self.dt_projs_bias is not None else None),
                _As = As[2:].contiguous().view(-1, N),
                _Ds = self.Ds.view(4, -1)[2:].contiguous().view(-1),
                width=False,
            ).view(B, W, 2, -1, H).sum(dim=2).permute(0, 2, 3, 1)
            y = y_col
        else:
            x_proj_bias = getattr(self, "x_proj_bias", None)    # None
            # 输入x ([1, 96, 56, 56]-->输出 xs ([1, 4, 96, 3136])
            xs = scan_fn(x, in_channel_first=True, out_channel_first=True, scans=_scan_mode, force_torch=scan_force_torch)
            if no_einsum:  # no_einsum=True
                # xs.view(B, -1, L)结果： (B, 4*C, L) -> (1, 4*96, 3136)
                # x_proj_weight (4 ,8, 96).view(-1, D, 1)-->(32 ,96, 1)
                # xs.view(1, 384, 3136) 拆成 4 组，每组 96 通道
                # x_proj_weight.view(32, 96, 1) 也拆成 4 组，每组 8 个卷积核，大小 (96, 1)
                # F.conv1d input：输入张量，形状为 (batch_size, in_channels, length)，表示输入数据。
                # F.conv1d weight：卷积核，形状为 (out_channels, in_channels/groups, kernel_size)，即滤波器。
                test=self.x_proj_weight.view(-1, D, 1)
                x_dbl = F.conv1d(xs.view(B, -1, L), self.x_proj_weight.view(-1, D, 1), bias=(x_proj_bias.view(-1) if x_proj_bias is not None else None), groups=K)
                # x_dbl (1, 32, 3136)
                # dts torch.Size([1, 4, 6, 3136])
                # Bs  torch.Size([1, 4, 1, 3136])
                # Cs  torch.Size([1, 4, 1, 3136])
                dts, Bs, Cs = torch.split(x_dbl.view(B, K, -1, L), [R, N, N], dim=2)
                # dt_projs_weight torch.Size([4, 96, 6])
                if hasattr(self, "dt_projs_weight"):
                    dts = F.conv1d(dts.contiguous().view(B, -1, L), self.dt_projs_weight.view(K * D, -1, 1), groups=K)
                # dts torch.Size([1, 384, 3136])
            else:
                x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs, self.x_proj_weight)
                if x_proj_bias is not None:
                    x_dbl = x_dbl + x_proj_bias.view(1, K, -1, 1)
                dts, Bs, Cs = torch.split(x_dbl, [R, N, N], dim=2)
                if hasattr(self, "dt_projs_weight"):
                    dts = torch.einsum("b k r l, k d r -> b k d l", dts, self.dt_projs_weight)

            xs = xs.view(B, -1, L)                                      # torch.Size([1, 384, 3136])
            dts = dts.contiguous().view(B, -1, L)                       # torch.Size([1, 384, 3136])
            As = -self.A_logs.to(torch.float).exp() # (k * c, d_state)  # torch.Size([384, 1])
            Ds = self.Ds.to(torch.float) # (K * c)                      # torch.Size([384])
            Bs = Bs.contiguous().view(B, K, N, L)                       # torch.Size([1, 4, 1, 3136])
            Cs = Cs.contiguous().view(B, K, N, L)                       # torch.Size([1, 4, 1, 3136])
            # dt_projs_bias torch.Size([4, 96])
            delta_bias = self.dt_projs_bias.view(-1).to(torch.float)    # torch.Size([384])

            # force_fp32=False
            if force_fp32:
                xs, dts, Bs, Cs = to_fp32(xs, dts, Bs, Cs)
            # ys=torch.Size([1, 384, 3136]).view(B, K, -1, H, W)=(1,4,96,56,56)
            ys: torch.Tensor = selective_scan(
                xs, dts, As, Bs, Cs, Ds, delta_bias, delta_softplus
            ).view(B, K, -1, H, W)
            
            y: torch.Tensor = merge_fn(ys, in_channel_first=True, out_channel_first=True, scans=_scan_mode, force_torch=scan_force_torch)

            if getattr(self, "__DEBUG__", False):
                setattr(self, "__data__", dict(
                    A_logs=self.A_logs, Bs=Bs, Cs=Cs, Ds=Ds,
                    us=xs, dts=dts, delta_bias=delta_bias,
                    ys=ys, y=y, H=H, W=W,
                ))

        y = y.view(B, -1, H, W)
        if not channel_first:
            y = y.view(B, -1, H * W).transpose(dim0=1, dim1=2).contiguous().view(B, H, W, -1) # (B, L, C)
        # y = out_norm(y)

        return y.to(x.dtype)

    def forwardv2(self, x: torch.Tensor, **kwargs):

        # 通道由d_model变成d_inner，其他大小不变，Linear2D
        # x[1, 96, 56, 56])
        x = self.in_proj(x)             # (b, d_inner, w, d)   
        if not self.disable_z:
            x, z = x.chunk(2, dim=(1 if self.channel_first else -1)) # (b, h, w, d)
            if not self.disable_z_act:
                z = self.act(z)
        if not self.channel_first:
            x = x.permute(0, 3, 1, 2).contiguous()
        if self.with_dconv:
            x = self.conv2d(x) # (b, d_inner, h, w)
        x = self.act(x)        # SiLU
        y = self.forward_core(x)
        y = self.out_act(y)    # nn.Identity()
        y = self.out_norm(y)   # LN
        if not self.disable_z:
            y = y * z
        # 通道由d_inner变成d_model，其他大小不变，Linear2D
        out = self.dropout(self.out_proj(y))
        return out
    
    # 根据 forward_type 选择不同的归一化层
    @staticmethod
    def get_outnorm(forward_type="", d_inner=192, channel_first=True):
        def checkpostfix(tag, value):
            ret = value[-len(tag):] == tag
            if ret:
                value = value[:-len(tag)]
            return ret, value

        LayerNorm = LayerNorm2d if channel_first else nn.LayerNorm
        # 解析 forward_type 以确定归一化方式
        out_norm_none, forward_type = checkpostfix("_onnone", forward_type)
        out_norm_dwconv3, forward_type = checkpostfix("_ondwconv3", forward_type)
        out_norm_cnorm, forward_type = checkpostfix("_oncnorm", forward_type)
        out_norm_softmax, forward_type = checkpostfix("_onsoftmax", forward_type)
        out_norm_sigmoid, forward_type = checkpostfix("_onsigmoid", forward_type)

        out_norm = nn.Identity()
        # 直接使用 nn.Identity()，即不做归一化
        if out_norm_none:
            out_norm = nn.Identity()
        # 使用 LayerNorm，然后应用 3x3 深度可分离卷积 (groups=d_inner)
        elif out_norm_cnorm:
            out_norm = nn.Sequential(
                LayerNorm(d_inner),
                (nn.Identity() if channel_first else Permute(0, 3, 1, 2)),
                nn.Conv2d(d_inner, d_inner, kernel_size=3, padding=1, groups=d_inner, bias=False),
                (nn.Identity() if channel_first else Permute(0, 2, 3, 1)),
            )
        # 仅应用 3x3 深度可分离卷积
        elif out_norm_dwconv3:
            out_norm = nn.Sequential(
                (nn.Identity() if channel_first else Permute(0, 3, 1, 2)),
                nn.Conv2d(d_inner, d_inner, kernel_size=3, padding=1, groups=d_inner, bias=False),
                (nn.Identity() if channel_first else Permute(0, 2, 3, 1)),
            )
        # 归一化方式为 SoftmaxSpatial
        elif out_norm_softmax:
            out_norm = SoftmaxSpatial(dim=(-1 if channel_first else 1))
        # 归一化方式为 Sigmoid
        elif out_norm_sigmoid:
            out_norm = nn.Sigmoid()
        # 默认情况 → 直接使用 LayerNorm(d_inner)
        else:
            out_norm = LayerNorm(d_inner)

        return out_norm, forward_type

    # 检查字符串 value 是否以 tag 结尾，如果是，就去掉这个后缀，并返回一个布尔值和修改后的字符串
    # 这个方法是 静态方法，不依赖于实例（self），可以直接通过类调用，例如 ClassName.checkpostfix(tag, value)
    @staticmethod
    def checkpostfix(tag, value):
        # 取出 value 最后 len(tag) 个字符，即 value 的后缀
        # 判断 value 是否以 tag 结尾,如果是则为True
        ret = value[-len(tag):] == tag
        if ret:
            # value[:-len(tag)] 去掉这个后缀
            value = value[:-len(tag)]
        return ret, value


# support: xv1a,xv2a,xv3a; 
# postfix: _cpos;_ocov;_ocov2;_ca,_ca1;_act;_mul;_onsigmoid,_onsoftmax,_ondwconv3,_onnone;
class SS2Dv3:
    def __initxv__(
        self,
        # basic dims ===========
        d_model=96,
        d_state=16,
        ssm_ratio=2.0,
        dt_rank="auto",
        # dwconv ===============
        d_conv=3, # < 2 means no conv 
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
        channel_first=False,
        # ======================
        **kwargs,
    ):
        super().__init__()
        d_inner = int(ssm_ratio * d_model)
        dt_rank = math.ceil(d_model / 16) if dt_rank == "auto" else dt_rank
        self.channel_first = channel_first
        self.d_state = d_state
        self.dt_rank = dt_rank
        self.d_inner = d_inner
        k_group = 4
        self.with_dconv = d_conv > 1
        Linear = Linear2d if channel_first else nn.Linear
        self.forward = self.forwardxv

        # tags for forward_type ==============================
        checkpostfix = SS2Dv2.checkpostfix
        self.out_norm, forward_type = SS2Dv2.get_outnorm(forward_type, d_inner, channel_first)
        self.omul, forward_type = checkpostfix("_mul", forward_type)
        self.oact, forward_type = checkpostfix("_act", forward_type)
        self.f_omul = nn.Identity() if self.omul else None
        self.out_act = nn.GELU() if self.oact else nn.Identity()

        mode = forward_type[:4]
        assert mode in ["xv1a", "xv2a", "xv3a"]

        self.forward = partial(self.forwardxv, mode=mode)
        self.dts_dim = dict(xv1a=self.dt_rank, xv2a=self.d_inner, xv3a=4 * self.dt_rank)[mode]
        d_inner_all = d_inner + self.dts_dim + 8 * d_state
        self.in_proj = Linear(d_model, d_inner_all, bias=bias)
        
        # conv =======================================
        self.cpos = False
        self.iconv = False
        self.oconv = False
        self.oconv2 = False
        if self.with_dconv:
            cact, forward_type = checkpostfix("_ca", forward_type)
            cact1, forward_type = checkpostfix("_ca1", forward_type)
            self.cact = nn.SiLU() if cact else nn.Identity()
            self.cact = nn.GELU() if cact1 else self.cact
                
            self.oconv2, forward_type = checkpostfix("_ocov2", forward_type)
            self.oconv, forward_type = checkpostfix("_ocov", forward_type)
            self.cpos, forward_type = checkpostfix("_cpos", forward_type)
            self.iconv = (not self.oconv) and (not self.oconv2)

            if self.iconv:
                self.conv2d = nn.Conv2d(
                    in_channels=d_model,
                    out_channels=d_model,
                    groups=d_model,
                    bias=conv_bias,
                    kernel_size=d_conv,
                    padding=(d_conv - 1) // 2,
                )
            if self.oconv:
                self.oconv2d = nn.Conv2d(
                    in_channels=d_inner,
                    out_channels=d_inner,
                    groups=d_inner,
                    bias=conv_bias,
                    kernel_size=d_conv,
                    padding=(d_conv - 1) // 2,
                )
            if self.oconv2:
                self.conv2d = nn.Conv2d(
                    in_channels=d_inner_all,
                    out_channels=d_inner_all,
                    groups=d_inner_all,
                    bias=conv_bias,
                    kernel_size=d_conv,
                    padding=(d_conv - 1) // 2,
                )

        # out proj =======================================
        self.out_proj = Linear(d_inner, d_model, bias=bias)
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

        if initialize in ["v0"]:
            self.A_logs, self.Ds, self.dt_projs_weight, self.dt_projs_bias = mamba_init.init_dt_A_D(
                d_state, dt_rank, d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, k_group=4,
            )
        elif initialize in ["v1"]:
            # simple init dt_projs, A_logs, Ds
            self.Ds = nn.Parameter(torch.ones((k_group * d_inner)))
            self.A_logs = nn.Parameter(torch.randn((k_group * d_inner, d_state))) # A == -A_logs.exp() < 0; # 0 < exp(A * dt) < 1
            self.dt_projs_weight = nn.Parameter(torch.randn((k_group, d_inner, dt_rank)))
            self.dt_projs_bias = nn.Parameter(torch.randn((k_group, d_inner))) 
        elif initialize in ["v2"]:
            # simple init dt_projs, A_logs, Ds
            self.Ds = nn.Parameter(torch.ones((k_group * d_inner)))
            self.A_logs = nn.Parameter(torch.zeros((k_group * d_inner, d_state))) # A == -A_logs.exp() < 0; # 0 < exp(A * dt) < 1
            self.dt_projs_weight = nn.Parameter(0.1 * torch.rand((k_group, d_inner, dt_rank)))
            self.dt_projs_bias = nn.Parameter(0.1 * torch.rand((k_group, d_inner)))


        if forward_type.startswith("xv2"):
            del self.dt_projs_weight
            self.dt_projs_weight = None

    def forwardxv(self, x: torch.Tensor, **kwargs):
        B, (H, W) = x.shape[0], (x.shape[2:4] if self.channel_first else x.shape[1:3])
        L = H * W
        force_fp32 = False
        delta_softplus = True
        out_norm = self.out_norm
        to_dtype = True

        to_fp32 = lambda *args: (_a.to(torch.float32) for _a in args)

        def selective_scan(u, delta, A, B, C, D, delta_bias, delta_softplus):
            return selective_scan_fn(u, delta, A, B, C, D, delta_bias, delta_softplus, oflex=True, backend=None)

        if self.iconv:
            x = self.cact(self.conv2d(x)) # (b, d, h, w)
        elif self.cpos:
            x = x + self.conv2d(x) # (b, d, h, w)

        x = self.in_proj(x)
        
        if self.oconv2:
            x = self.conv2d(x) # (b, d, h, w)

        us, dts, Bs, Cs = x.split([self.d_inner, self.dts_dim, 4 * self.d_state, 4 * self.d_state], dim=(1 if self.channel_first else -1))

        _us = us
        # Bs, Cs = Bs.view(B, H, W, 4, -1), Cs.view(B, H, W, 4, -1)
        # Bs, Cs = Bs.view(B, 4, -1, H, W), Cs.view(B, 4, -1, H, W)
        us = scan_fn(us.contiguous(), in_channel_first=self.channel_first, out_channel_first=True).view(B, -1, L)
        Bs = scan_fn(Bs.contiguous(), in_channel_first=self.channel_first, out_channel_first=True, one_by_one=True).view(B, 4, -1, L)
        Cs = scan_fn(Cs.contiguous(), in_channel_first=self.channel_first, out_channel_first=True, one_by_one=True).view(B, 4, -1, L)
        dts = scan_fn(dts.contiguous(), in_channel_first=self.channel_first, out_channel_first=True, one_by_one=(self.dts_dim == 4 * self.dt_rank)).view(B, L, -1)
        if self.dts_dim == self.dt_rank:
            dts = F.conv1d(dts, self.dt_projs_weight.view(4 * self.d_inner, self.dt_rank, 1), None, groups=4)
        elif self.dts_dim == 4 * self.dt_rank:
            dts = F.conv1d(dts, self.dt_projs_weight.view(4 * self.d_inner, self.dt_rank, 1), None, groups=4)

        As = -self.A_logs.to(torch.float).exp() # (k * c, d_state)
        Ds = self.Ds.to(torch.float) # (K * c)
        delta_bias = self.dt_projs_bias.view(-1).to(torch.float) # (K * c)

        if force_fp32:
            us, dts, Bs, Cs = to_fp32(us, dts, Bs, Cs)

        ys: torch.Tensor = selective_scan(
            us, dts, As, Bs, Cs, Ds, delta_bias, delta_softplus
        ).view(B, 4, -1, H, W)
        y: torch.Tensor = merge_fn(ys.contiguous(), in_channel_first=self.channel_first, out_channel_first=True)
        y = y.view(B, -1, H, W) if self.channel_first else y.view(B, H, W, -1)
        y = out_norm(y)
        
        if getattr(self, "__DEBUG__", False):
            setattr(self, "__data__", dict(
                A_logs=self.A_logs, Bs=Bs, Cs=Cs, Ds=Ds,
                us=us, dts=dts, delta_bias=delta_bias,
                ys=ys, y=y,
            ))

        y = (y.to(x.dtype) if to_dtype else y)
        
        y = self.out_act(y)
        
        if self.omul:
            y = y * _us

        if self.oconv:
            y = y + self.cact(self.oconv2d(_us))

        out = self.dropout(self.out_proj(y))
        return out


# mamba2 support ================================
class SS2Dm0:
    def __initm0__(
        self,
        # basic dims ===========
        d_model=96,
        d_state=16, # now with mamba2, dstate should be bigger...
        ssm_ratio=2.0,
        dt_rank="auto",
        act_layer=nn.GELU,
        # dwconv ===============
        d_conv=3, # < 2 means no conv 
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
        initialize="v2",
        # ======================
        forward_type="m0",
        # ======================
        with_initial_state=False,
        # ======================
        **kwargs,    
    ):
        factory_kwargs = {"device": None, "dtype": None}
        super().__init__()
        d_inner = int(ssm_ratio * d_model)
        dt_rank = math.ceil(d_model / 16) if dt_rank == "auto" else dt_rank
        assert d_inner % dt_rank == 0
        self.with_dconv = d_conv > 1
        Linear = nn.Linear
        self.forward = self.forwardm0

        # tags for forward_type ==============================
        checkpostfix = SS2Dv2.checkpostfix
        self.disable_force32, forward_type = checkpostfix("_no32", forward_type)
        self.oact, forward_type = checkpostfix("_oact", forward_type)
        self.disable_z, forward_type = checkpostfix("_noz", forward_type)
        self.disable_z_act, forward_type = checkpostfix("_nozact", forward_type)
        self.out_norm, forward_type = SS2Dv2.get_outnorm(forward_type, d_inner, False)

        # forward_type debug =======================================
        FORWARD_TYPES = dict(
            m0=partial(self.forward_corem0, force_fp32=False, dstate=d_state),
        )
        self.forward_core = FORWARD_TYPES.get(forward_type, None)
        k_group = 4

        # in proj =======================================
        d_proj = d_inner if self.disable_z else (d_inner * 2)
        self.in_proj = Linear(d_model, d_proj, bias=bias)
        self.act: nn.Module = act_layer()
        
        # conv =======================================
        if self.with_dconv:
            self.conv2d = nn.Sequential(
                Permute(0, 3, 1, 2),
                nn.Conv2d(
                    in_channels=d_inner,
                    out_channels=d_inner,
                    groups=d_inner,
                    bias=conv_bias,
                    kernel_size=d_conv,
                    padding=(d_conv - 1) // 2,
                    **factory_kwargs,
                ),
                Permute(0, 2, 3, 1),
            ) 
        
        # x proj ============================
        self.x_proj = [
            nn.Linear(d_inner, (dt_rank + d_state * 2), bias=False)
            for _ in range(k_group)
        ]
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0)) # (K, N, inner)
        del self.x_proj
        
        # out proj =======================================
        self.out_act = nn.GELU() if self.oact else nn.Identity()
        self.out_proj = Linear(d_inner, d_model, bias=bias)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else nn.Identity()

        if initialize in ["v1"]:
            # simple init dt_projs, A_logs, Ds
            self.Ds = nn.Parameter(torch.ones((k_group, dt_rank, int(d_inner // dt_rank))))
            self.A_logs = nn.Parameter(torch.randn((k_group, dt_rank))) # A == -A_logs.exp() < 0; # 0 < exp(A * dt) < 1
            self.dt_projs_bias = nn.Parameter(0.1 * torch.randn((k_group, dt_rank))) # 0.1 is added in 0430
        elif initialize in ["v2"]:
            # simple init dt_projs, A_logs, Ds
            self.Ds = nn.Parameter(torch.ones((k_group, dt_rank, int(d_inner // dt_rank))))
            self.A_logs = nn.Parameter(torch.zeros((k_group, dt_rank))) # A == -A_logs.exp() < 0; # 0 < exp(A * dt) < 1
            self.dt_projs_bias = nn.Parameter(0.1 * torch.rand((k_group, dt_rank)))

        # init state ============================
        self.initial_state = None
        if with_initial_state:
            self.initial_state = nn.Parameter(torch.zeros((1, k_group * dt_rank, int(d_inner // dt_rank), d_state)), requires_grad=False)

    def forward_corem0(
        self,
        x: torch.Tensor=None, 
        # ==============================
        force_fp32=False, # True: input fp32
        chunk_size = 64,
        dstate = 64,        
        # ==============================
        selective_scan_backend = None,
        scan_mode = "cross2d",
        scan_force_torch = False,
        # ==============================
        **kwargs,
    ):
        assert scan_mode in ["unidi", "bidi", "cross2d"]
        assert selective_scan_backend in [None, "triton", "torch"]
        x_proj_bias = getattr(self, "x_proj_bias", None)
        to_fp32 = lambda *args: (_a.to(torch.float32) for _a in args)

        N = dstate
        B, H, W, RD = x.shape
        K, R = self.A_logs.shape
        K, R, D = self.Ds.shape
        assert RD == R * D
        L = H * W
        KR = K * R
        _scan_mode = dict(cross2d=0, unidi=1, bidi=2, cascade2d=3)[scan_mode]

        initial_state = None
        if self.initial_state is not None:
            assert self.initial_state.shape[-1] == dstate
            initial_state = self.initial_state.detach().repeat(B, 1, 1, 1)
        xs = scan_fn(x.view(B, H, W, RD), in_channel_first=False, out_channel_first=False, scans=_scan_mode, force_torch=scan_force_torch) # (B, H, W, 4, D)
        x_dbl = torch.einsum("b l k d, k c d -> b l k c", xs, self.x_proj_weight)
        if x_proj_bias is not None:
            x_dbl = x_dbl + x_proj_bias.view(1, -1, K, 1)
        dts, Bs, Cs = torch.split(x_dbl, [R, N, N], dim=3)
        xs = xs.contiguous().view(B, L, KR, D)
        dts = dts.contiguous().view(B, L, KR)
        Bs = Bs.contiguous().view(B, L, K, N)
        Cs = Cs.contiguous().view(B, L, K, N)
        if force_fp32:
            xs, dts, Bs, Cs = to_fp32(xs, dts, Bs, Cs)

        As = -self.A_logs.to(torch.float).exp().view(KR)
        Ds = self.Ds.to(torch.float).view(KR, D)
        dt_bias = self.dt_projs_bias.view(KR)

        if force_fp32:
            xs, dts, Bs, Cs = to_fp32(xs, dts, Bs, Cs)

        ys, final_state = selective_scan_chunk_fn(
            xs, dts, As, Bs, Cs, chunk_size=chunk_size, D=Ds, dt_bias=dt_bias, 
            initial_states=initial_state, dt_softplus=True, return_final_states=True,
            backend=selective_scan_backend,
        )
        y: torch.Tensor = merge_fn(ys.view(B, H, W, K, RD), in_channel_first=False, out_channel_first=False, scans=_scan_mode, force_torch=scan_force_torch)

        if getattr(self, "__DEBUG__", False):
            setattr(self, "__data__", dict(
                A_logs=self.A_logs, Bs=Bs, Cs=Cs, Ds=self.Ds,
                us=xs, dts=dts, delta_bias=self.dt_projs_bias, 
                initial_state=self.initial_state, final_satte=final_state,
                ys=ys, y=y, H=H, W=W,
            ))
        if self.initial_state is not None:
            self.initial_state = nn.Parameter(final_state.detach().sum(0, keepdim=True), requires_grad=False)

        y = self.out_norm(y.view(B, H, W, -1))

        return y.to(x.dtype)

    def forwardm0(self, x: torch.Tensor, **kwargs):
        x = self.in_proj(x)
        if not self.disable_z:
            x, z = x.chunk(2, dim=(1 if self.channel_first else -1)) # (b, h, w, d)
            if not self.disable_z_act:
                z = self.act(z)
        if self.with_dconv:
            x = self.conv2d(x) # (b, d, h, w)
        x = self.act(x)
        y = self.forward_core(x)
        y = self.out_act(y)
        if not self.disable_z:
            y = y * z
        out = self.dropout(self.out_proj(y))
        return out


class SS2D(nn.Module, SS2Dv0, SS2Dv2, SS2Dv3, SS2Dm0):
    def __init__(
        self,
        # basic dims ===========
        d_model=96,
        d_state=16,                     # SSM 内部的状态大小
        ssm_ratio=2.0,                  # 控制 SSM 分支的计算规模
        dt_rank="auto",
        act_layer=nn.SiLU,
        # dwconv ===============
        d_conv=3, # < 2 means no conv 
        conv_bias=True,
        # ======================
        dropout=0.0,
        bias=False,
        # dt init ==============
        dt_min=0.001,                   # 最小时间步长
        dt_max=0.1,                     # 最大时间步长
        dt_init="random",               # 时间步长的初始化方式
        dt_scale=1.0,                   # 对 Δt 进行缩放
        dt_init_floor=1e-4,             # 确保 dt 不会太小，防止数值不稳定
        initialize="v0",                # SSM 初始化方式（"v0", "v2", "m0" 等）
        # ======================
        forward_type="v2",
        channel_first=False,
        scan="hilbert",
        # ======================
        **kwargs,
    ):
        # nn.Module.__init__(self)：初始化 torch.nn.Module，确保 SS2D 作为 PyTorch 模块可以正常使用。
        nn.Module.__init__(self)
        #  kwargs用于批量存储参数,避免手动传递
        kwargs.update(
            d_model=d_model, d_state=d_state, ssm_ratio=ssm_ratio, dt_rank=dt_rank,
            act_layer=act_layer, d_conv=d_conv, conv_bias=conv_bias, dropout=dropout, bias=bias,
            dt_min=dt_min, dt_max=dt_max, dt_init=dt_init, dt_scale=dt_scale, dt_init_floor=dt_init_floor,
            initialize=initialize, forward_type=forward_type, channel_first=channel_first,scan=scan,
        )
        if forward_type in ["v0", "v0seq"]:
            self.__initv0__(seq=("seq" in forward_type), **kwargs)
        elif forward_type.startswith("xv"):
            self.__initxv__(**kwargs)
        elif forward_type.startswith("m"):
            self.__initm0__(**kwargs)
        else:
            self.__initv2__(**kwargs)


# =====================================================
class VSSBlock(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 0,
        drop_path: float = 0,
        norm_layer: nn.Module = nn.LayerNorm,
        channel_first=False,
        # =============================
        ssm_d_state: int = 16,
        ssm_ratio=2.0,
        ssm_dt_rank: Any = "auto",          # 控制 SSM 的时间步长 rank，可以是 "auto" 或一个整数
        ssm_act_layer=nn.SiLU,
        ssm_conv: int = 3,                  # SSM 分支中的卷积核大小
        ssm_conv_bias=True,
        ssm_drop_rate: float = 0,
        ssm_init="v0",
        forward_type="v2",
        scan='hilbert',
        # =============================
        mlp_ratio=4.0,
        mlp_act_layer=nn.GELU,
        mlp_drop_rate: float = 0.0,
        gmlp=False,
        # =============================
        use_checkpoint: bool = False,       # 进行梯度检查点计算，以节省显存
        post_norm: bool = False,
        # =============================
        _SS2D: type = SS2D,
        **kwargs,
    ):
        super().__init__()
        # 是否启用 SSM 分支
        self.ssm_branch = ssm_ratio > 0
        # 是否启用 MLP 分支
        self.mlp_branch = mlp_ratio > 0
        self.use_checkpoint = use_checkpoint
        self.post_norm = post_norm          # 是否在计算后再进行归一化

        if self.ssm_branch:
            self.norm = norm_layer(hidden_dim)
            # 状态空间模型的计算模块
            self.op = _SS2D(
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
                channel_first=channel_first,
                scan=scan,
            )
        
        self.drop_path = DropPath(drop_path)
        
        if self.mlp_branch:
            _MLP = Mlp if not gmlp else gMlp
            self.norm2 = norm_layer(hidden_dim)
            mlp_hidden_dim = int(hidden_dim * mlp_ratio)
            # 初始化 self.mlp（MLP 层）
            self.mlp = _MLP(in_features=hidden_dim, hidden_features=mlp_hidden_dim, act_layer=mlp_act_layer, drop=mlp_drop_rate, channels_first=channel_first)

    def _forward(self, input: torch.Tensor):
        x = input
        if self.ssm_branch:
            if self.post_norm:
                x = x + self.drop_path(self.norm(self.op(x)))
            else:
                x = x + self.drop_path(self.op(self.norm(x)))
        if self.mlp_branch:
            if self.post_norm:
                x = x + self.drop_path(self.norm2(self.mlp(x))) # FFN
            else:
                x = x + self.drop_path(self.mlp(self.norm2(x))) # FFN
        return x

    def forward(self, input: torch.Tensor):
        if self.use_checkpoint:
            return checkpoint.checkpoint(self._forward, input)
        else:
            return self._forward(input)


class VSSM(nn.Module):
    def __init__(
        self, 
        patch_size=4,             # 图像切片的大小
        in_chans=3,               # 输入通道数（RGB 三通道）
        num_classes=1000,         # 分类类别数
        depths=[2, 2, 9, 2],      # 不同层的 VSSM block 数量
        dims=[96, 192, 384, 768], # 不同层的特征维度
        # =========================
        ssm_d_state=16,           # 状态空间维度
        ssm_ratio=2.0,            # SSM 计算的比例
        ssm_dt_rank="auto",       # 时间步长的秩
        ssm_act_layer="silu",     # SSM 激活函数   
        ssm_conv=3,               # 卷积核大小
        ssm_conv_bias=True,       # 是否使用偏置
        ssm_drop_rate=0.0,        # SSM 层的 Dropout 率
        ssm_init="v0",            # SSM 的初始化方式
        forward_type="v2",
        scan="hilbert",
        # =========================
        mlp_ratio=4.0,            # MLP 隐藏层扩展比
        mlp_act_layer="gelu",     # MLP 激活函数
        mlp_drop_rate=0.0,        # MLP Dropout 率
        gmlp=False,               # 是否使用 gMLP
        # =========================
        drop_path_rate=0.1,       # 随机深度（DropPath）衰减率
        patch_norm=True,          # 是否对 patch 进行归一化
        norm_layer="LN", # "BN", "LN2D"                         # 归一化方式（LayerNorm）
        downsample_version: str = "v2", # "v1", "v2", "v3"      # 下采样方式（v1, v2, v3）
        patchembed_version: str = "v1", # "v1", "v2"            # Patch embedding 版本（v1, v2）
        use_checkpoint=False,     # Patch embedding 版本（v1, v2）
        # =========================
        posembed=False,
        imgsize=224,
        _SS2D=SS2D,
        # =========================
        **kwargs,
    ):
        # super().__init__() 允许子类调用父类的 __init__() 方法，避免代码重复
        super().__init__()
        # 如果 norm_layer 是 "bn" 或 "ln2d"，表达式返回 True
        self.channel_first = (norm_layer.lower() in ["bn", "ln2d"])
        self.num_classes = num_classes      # 记录分类类别数
        self.num_layers = len(depths)       # 计算层数
        # 如果 dims 是整数（比如 96），int(dims * 2 ** i_layer) 让特征维度在每一层加倍，变成[96, 192, 384, 768]
        if isinstance(dims, int):
            dims = [int(dims * 2 ** i_layer) for i_layer in range(self.num_layers)]
        # 取 dims 的最后一个值，代表最后一层的特征维度
        self.num_features = dims[-1]
        # 把计算后的 dims 赋值给 self.dims，后面模型的其他部分可以用这个值
        self.dims = dims


        # drop_path_rate代表 随机深度（Stochastic Depth）衰减率，控制训练时随机丢弃一些路径，以提升模型泛化能力。
        # 生成从 0 到 drop_path_rate 之间的 sum(depths) 个等间距数值，作为不同 Block 的 Drop Path 值。
        # x.item() 只是把 tensor 变成 float。
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
        
        # 归一化 & 激活层映射
        _NORMLAYERS = dict(
            ln=nn.LayerNorm,
            ln2d=LayerNorm2d,
            bn=nn.BatchNorm2d,
        )

        _ACTLAYERS = dict(
            silu=nn.SiLU, 
            gelu=nn.GELU, 
            relu=nn.ReLU, 
            sigmoid=nn.Sigmoid,
        )

        norm_layer: nn.Module = _NORMLAYERS.get(norm_layer.lower(), None)      # LN
        ssm_act_layer: nn.Module = _ACTLAYERS.get(ssm_act_layer.lower(), None) # silu
        mlp_act_layer: nn.Module = _ACTLAYERS.get(mlp_act_layer.lower(), None) # GELU

        # posembed 决定是否使用位置编码。
        self.pos_embed = self._pos_embed(dims[0], patch_size, imgsize) if posembed else None
        
        # Patch Embedding 选择
        _make_patch_embed = dict(
            v1=self._make_patch_embed, 
            v2=self._make_patch_embed_v2,
        ).get(patchembed_version, None)
        self.patch_embed = _make_patch_embed(in_chans, dims[0], patch_size, patch_norm, norm_layer, channel_first=self.channel_first)
       
        # 下采样层选择
        _make_downsample = dict(
            v1=PatchMerging2D, 
            v2=self._make_downsample, 
            v3=self._make_downsample_v3, 
            none=(lambda *_, **_k: None),
        ).get(downsample_version, None)

        # 构建层
        self.layers = nn.ModuleList()
        # 下采样层只有0，1，2，三层
        for i_layer in range(self.num_layers):
            downsample = _make_downsample(
                self.dims[i_layer],         #  例如dims[0]=96
                self.dims[i_layer + 1],     #  例如dims[1]=192
                norm_layer=norm_layer,
                channel_first=self.channel_first,
            ) if (i_layer < self.num_layers - 1) else nn.Identity()

            self.layers.append(self._make_layer(
                dim = self.dims[i_layer],   #  例如dims[0]=96
                # sum(depths[:i_layer]) 就是当前层之前所有层的 VSSBlock 总数。对于 i_layer = 2，sum(depths[:2]) 会返回 4，i_layer = 3，sum(depths[:2]) 会返回 13
                drop_path = dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                use_checkpoint=use_checkpoint,
                norm_layer=norm_layer,
                downsample=downsample,
                channel_first=self.channel_first,
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
                scan=scan,
                # =================
                mlp_ratio=mlp_ratio,
                mlp_act_layer=mlp_act_layer,
                mlp_drop_rate=mlp_drop_rate,
                gmlp=gmlp,
                # =================
                _SS2D=_SS2D,
            ))
        # 分类头
        self.classifier = nn.Sequential(OrderedDict(
            # 归一化层
            norm=norm_layer(self.num_features), 
            permute=(Permute(0, 3, 1, 2) if not self.channel_first else nn.Identity()),
            # 将输入的特征图的空间尺寸（高度和宽度）池化到指定的大小。
            # 这里的 1 表示池化后的输出尺寸是 (batch_size, channels, 1, 1)，即每个通道的输出变成一个单一的标量（全局平均池化）。
            avgpool=nn.AdaptiveAvgPool2d(1),   
            # 1 表示从第二个维度开始展平（通常是通道维度）。展平后，张量的形状变为 (batch_size, num_features)
            flatten=nn.Flatten(1),              
            # 这是一个全连接层（线性层），其输入大小是 self.num_features，输出大小是 num_classes
            head=nn.Linear(self.num_features, num_classes),
        ))

        # 使用 _init_weights 初始化模型权重
        self.apply(self._init_weights)

    @staticmethod
    def _pos_embed(embed_dims, patch_size, img_size):
        # embed_dims 每个 patch 处理后的特征维度
        # patch_size 图像被切割成的小块尺寸，例如 4×4

        # 计算 patch 的个数
        patch_height, patch_width = (img_size // patch_size, img_size // patch_size)
        # 创建位置编码pos_embed，例如 (1, 96, 56, 56)
        pos_embed = nn.Parameter(torch.zeros(1, embed_dims, patch_height, patch_width))
        # 用 trunc_normal_() 进行初始化
        # trunc_normal_() 是截断正态分布初始化（均值 0，标准差 0.02）
        # 这样初始化能防止数值过大，保持梯度稳定
        trunc_normal_(pos_embed, std=0.02)
        return pos_embed

    def _init_weights(self, m: nn.Module):
        # 初始化Linear层
        if isinstance(m, nn.Linear):
            # m.weight用截断正态分布 (std=0.02) 进行初始化，避免梯度爆炸/消失
            trunc_normal_(m.weight, std=.02)
            # m.bias（如果存在）全部初始化为 0，防止偏移过大
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        # 初始化LayerNorm层
        elif isinstance(m, nn.LayerNorm):
            # nn.init.constant_ 是 PyTorch 提供的参数初始化函数，用于将某个张量的所有元素赋值为指定的常数。
            # bias=0，weight=1.0，确保初始状态不会影响特征分布
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    # used in building optimizer
    @torch.jit.ignore
    def no_weight_decay(self):
        return {"pos_embed"}

    # used in building optimizer
    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {}
    
    # 补丁嵌入（Patch Embedding）
    # _make_patch_embed 和 _make_patch_embed_v2 将输入图像转化为固定大小的 patch，并通过卷积降维。
    @staticmethod
    def _make_patch_embed(in_chans=3, embed_dim=96, patch_size=4, patch_norm=True, norm_layer=nn.LayerNorm, channel_first=False):
        # if channel first, then Norm and Output are both channel_first
        return nn.Sequential(
            # 使用 Conv2d 进行 patch 切片（patch embedding）
            # 输入通道数：in_chans，通常是 3（RGB 图像）
            # 输出通道数：embed_dim，每个 patch 的特征维度（比如 96）。
            # 卷积核大小 (kernel_size=patch_size)：假设 patch_size=4，那么相当于 4x4 的卷积。
            # 步长 (stride=patch_size)：确保不重叠地切片，比如 4x4 切片时，步长 4 确保每个 patch 不重叠。
            nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size, bias=True),
            # 保持 [B, C, H, W] 格式 或是 [B, H, W, C] 
            # nn.Identity() 不改变输入，起到占位作用。
            (nn.Identity() if channel_first else Permute(0, 2, 3, 1)),
            # 如果 patch_norm=True，使用 LayerNorm 归一化嵌入特征（推荐）
            (norm_layer(embed_dim) if patch_norm else nn.Identity()),
        )

    # 相比 v1 版本，_make_patch_embed_v2 采用了两层卷积
    # 使用 GELU 激活函数 增强非线性表达能力。
    @staticmethod
    def _make_patch_embed_v2(in_chans=3, embed_dim=96, patch_size=4, patch_norm=True, norm_layer=nn.LayerNorm, channel_first=False):
        # if channel first, then Norm and Output are both channel_first
        # 采用 stride=patch_size//2，让卷积更细粒度地提取特征
        # 例如 patch_size=4，那么 stride=2，这样会提取更丰富的局部信息。
        stride = patch_size // 2
        kernel_size = stride + 1
        padding = 1
        return nn.Sequential(
            # 第一层卷积：降维提取特征
            # 例如输入尺寸为224*224*3，输出便是（224-3+2*1)/2+1=112.5=112,112*112*48
            nn.Conv2d(in_chans, embed_dim // 2, kernel_size=kernel_size, stride=stride, padding=padding),
            (nn.Identity() if (channel_first or (not patch_norm)) else Permute(0, 2, 3, 1)),
            # 如果 patch_norm=True，对降维后的 48 维特征进行 LayerNorm 归一化
            (norm_layer(embed_dim // 2) if patch_norm else nn.Identity()),
            (nn.Identity() if (channel_first or (not patch_norm)) else Permute(0, 3, 1, 2)),
            # 使用 GELU（Gaussian Error Linear Unit）增加非线性能力。
            nn.GELU(),
            # 第二层卷积：恢复维度,输出是（112-3+2*1)/2+1=56.5=56,56*56*96
            nn.Conv2d(embed_dim // 2, embed_dim, kernel_size=kernel_size, stride=stride, padding=padding),
            (nn.Identity() if channel_first else Permute(0, 2, 3, 1)),
            # 如果 patch_norm=True，对最终 96 维特征归一化。
            (norm_layer(embed_dim) if patch_norm else nn.Identity()),
        )
    
    @staticmethod
    def _make_downsample(dim=96, out_dim=192, norm_layer=nn.LayerNorm, channel_first=False):
        # if channel first, then Norm and Output are both channel_first
        return nn.Sequential(
            (nn.Identity() if channel_first else Permute(0, 3, 1, 2)),
            # 卷积操作 输入是56*56*96，那么输出是（56-2)/2+1=28,28*28*192
            nn.Conv2d(dim, out_dim, kernel_size=2, stride=2),
            (nn.Identity() if channel_first else Permute(0, 2, 3, 1)),
            # 归一化层
            norm_layer(out_dim),
        )

    # 唯一的区别在于卷积层的配置（3x3卷积，步长为2，填充为1）
    @staticmethod
    def _make_downsample_v3(dim=96, out_dim=192, norm_layer=nn.LayerNorm, channel_first=False):
        # if channel first, then Norm and Output are both channel_first
        return nn.Sequential(
            (nn.Identity() if channel_first else Permute(0, 3, 1, 2)),
            # 卷积操作 输入是56*56*96，那么输出是（56-3+2*1)/2+1=28.5=28,28*28*192
            nn.Conv2d(dim, out_dim, kernel_size=3, stride=2, padding=1),
            (nn.Identity() if channel_first else Permute(0, 2, 3, 1)),
            norm_layer(out_dim),
        )
    
    # 主干网络（Backbone）
    # _make_layer 里包含多个 VSSBlock，并支持 SSM 计算。
    @staticmethod
    def _make_layer(
        dim=96,                             # 该层的输入和输出的特征维度。即每个 VSSBlock 的隐层维度。
        drop_path=[0.1, 0.1],               # 控制了每个 VSSBlock 中的随机深度（drop path）的比率
        use_checkpoint=False, 
        norm_layer=nn.LayerNorm,            # 使用的归一化层，默认为 LayerNorm
        downsample=nn.Identity(),           # 如果设置为 nn.Identity()，则表示不进行下采样
        channel_first=False,
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
        scan="hilbert",
        # ===========================
        mlp_ratio=4.0,
        mlp_act_layer=nn.GELU,
        mlp_drop_rate=0.0,
        gmlp=False,
        # ===========================
        _SS2D=SS2D,
        **kwargs,
    ):
        # if channel first, then Norm and Output are both channel_first
        depth = len(drop_path)
        blocks = []
        for d in range(depth):
            blocks.append(VSSBlock(
                hidden_dim=dim, 
                drop_path=drop_path[d],
                norm_layer=norm_layer,
                channel_first=channel_first,
                ssm_d_state=ssm_d_state,
                ssm_ratio=ssm_ratio,
                ssm_dt_rank=ssm_dt_rank,
                ssm_act_layer=ssm_act_layer,
                ssm_conv=ssm_conv,
                ssm_conv_bias=ssm_conv_bias,
                ssm_drop_rate=ssm_drop_rate,
                ssm_init=ssm_init,
                forward_type=forward_type,
                scan=scan,
                mlp_ratio=mlp_ratio,
                mlp_act_layer=mlp_act_layer,
                mlp_drop_rate=mlp_drop_rate,
                gmlp=gmlp,
                use_checkpoint=use_checkpoint,
                _SS2D=_SS2D,
            ))
        # 返回一个 nn.Sequential 对象，其中包含了多个 VSSBlock 和一个下采样操作
        # 默认是 nn.Identity()，表示不进行下采样。
        return nn.Sequential(OrderedDict(
            blocks=nn.Sequential(*blocks,),
            downsample=downsample,
        ))
    # 模型的前向传播逻辑
    def forward(self, x: torch.Tensor):
        x = self.patch_embed(x)             # 图像转为 token
        if self.pos_embed is not None:
            pos_embed = self.pos_embed.permute(0, 2, 3, 1) if not self.channel_first else self.pos_embed
            x = x + pos_embed               # 位置编码
        for layer in self.layers:
            x = layer(x)                    # 经过多个VSSM层
        x = self.classifier(x)              # 分类头 由 LayerNorm、全局平均池化、Flatten 和全连接层（nn.Linear）组成，最终输出分类结果。
        return x
    # flops 方法用于计算模型的浮点运算量（FLOPs），用于分析计算复杂度
    def flops(self, shape=(3, 256, 256), verbose=True):
        # shape = self.__input_shape__[1:]
        # supported_ops 字典
        supported_ops={
            "aten::silu": None, # as relu is in _IGNORED_OPS
            "aten::neg": None, # as relu is in _IGNORED_OPS
            "aten::exp": None, # as relu is in _IGNORED_OPS
            "aten::flip": None, # as permute is in _IGNORED_OPS
            # "prim::PythonOp.CrossScan": None,
            # "prim::PythonOp.CrossMerge": None,
            "prim::PythonOp.SelectiveScanCuda": partial(selective_scan_flop_jit, backend="prefixsum", verbose=verbose),# 是一个函数工厂，用于生成支持特定操作（如 SelectiveScan）的计算方法
        }
        # 创建当前模型的一个深拷贝，目的是为了避免对原始模型的修改，保持其不变。
        model = copy.deepcopy(self)
        # 将模型转移到 GPU（如果可用），并将其设置为评估模式（eval）
        model.cuda().eval()
        # 随机输入一个 batch 大小为 1 的 RGB 图像，尺寸为 224x224
        # next(model.parameters()).device 会获取模型的参数所在的设备（如 GPU 或 CPU），确保输入张量与模型在同一设备上
        input = torch.randn((1, *shape), device=next(model.parameters()).device)
        # 计算模型中的参数总数
        params = parameter_count(model)[""]
        # 计算 FLOPs 的函数，它会通过输入样本（这里是 input）进行前向传播，并计算模型的浮点运算量
        Gflops, unsupported = flop_count(model=model, inputs=(input,), supported_ops=supported_ops)
        # 释放 model 和 input 变量占用的内存，避免内存泄漏
        del model, input
        # 返回模型的总 FLOPs 值。Gflops 是一个字典，表示每种操作的 FLOPs，sum(Gflops.values()) 将所有操作的 FLOPs 求和
        # 1e9 用来将 GFLOPs 转换为 FLOPs，因为 1 GFLOP = 1e9 FLOPs
        return sum(Gflops.values()) * 1e9
        return f"params {params} GFLOPs {sum(Gflops.values())}"

    # used to load ckpt from previous training code
    # 从给定的 state_dict 中加载模型参数。
    # strict 一个布尔值，指示是否严格要求所有的键都必须匹配
    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        # 检查 state_dict 中是否包含某个参数名
        def check_name(src, state_dict: dict = state_dict, strict=False):
            # 严格检查是否存在完全匹配的键
            if strict:
                if prefix + src in list(state_dict.keys()):
                    return True
            else:
                key = prefix + src
                for k in list(state_dict.keys()):
                    if k.startswith(key):
                        return True
            return False
        # 用于在 state_dict 中修改参数的名称
        def change_name(src, dst, state_dict: dict = state_dict, strict=False):
            # 直接修改匹配的键
            if strict:
                if prefix + src in list(state_dict.keys()):
                    state_dict[prefix + dst] = state_dict[prefix + src]
                    state_dict.pop(prefix + src)
            else:
                key = prefix + src
                for k in list(state_dict.keys()):
                    if k.startswith(key):
                        new_k = prefix + dst + k[len(key):]
                        state_dict[new_k] = state_dict[k]
                        state_dict.pop(k)

        if check_name("pos_embed", strict=True):
            srcEmb: torch.Tensor = state_dict[prefix + "pos_embed"]
            state_dict[prefix + "pos_embed"] = F.interpolate(srcEmb.float(), size=self.pos_embed.shape[2:4], align_corners=False, mode="bicubic").to(srcEmb.device)

        change_name("patch_embed.proj", "patch_embed.0")
        change_name("patch_embed.norm", "patch_embed.2")
        # 对每一层中的 ln_1 和 self_attention 进行重命名，可能是为了与新模型的命名规范匹配
        for i in range(100):
            for j in range(100):
                change_name(f"layers.{i}.blocks.{j}.ln_1", f"layers.{i}.blocks.{j}.norm")
                change_name(f"layers.{i}.blocks.{j}.self_attention", f"layers.{i}.blocks.{j}.op")
        change_name("norm", "classifier.norm")
        change_name("head", "classifier.head")
        # 调用父类的 _load_from_state_dict 方法，完成剩余的加载操作
        return super()._load_from_state_dict(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)


# compatible with openmmlab
class Backbone_VSSM(VSSM):
    # 初始化函数
    # pretrained: 用于加载预训练权重。如果提供了该路径，网络会加载预训练的权重来初始化模型
    def __init__(self, out_indices=(0, 1, 2, 3), pretrained=None, norm_layer="ln", **kwargs):
        # 调用父类 VSSM 的初始化函数，初始化网络的其他部分（如嵌入层、卷积层等）
        kwargs.update(norm_layer=norm_layer)
        super().__init__(**kwargs)
        self.channel_first = (norm_layer.lower() in ["bn", "ln2d"])
        _NORMLAYERS = dict(
            ln=nn.LayerNorm,
            ln2d=LayerNorm2d,
            bn=nn.BatchNorm2d,
        )
        norm_layer: nn.Module = _NORMLAYERS.get(norm_layer.lower(), None)        
        
        self.out_indices = out_indices
        for i in out_indices:
            layer = norm_layer(self.dims[i])
            layer_name = f'outnorm{i}'
            self.add_module(layer_name, layer)
        # 通常这意味着此类只用于特征提取，而不用于分类任务
        del self.classifier
        self.load_pretrained(pretrained)
    # 从指定路径 ckpt 加载预训练的模型权重
    def load_pretrained(self, ckpt=None, key="model"):
        if ckpt is None:
            return
        # 它使用 torch.load 加载预训练模型，并通过 load_state_dict 加载模型的参数
        try:
            _ckpt = torch.load(open(ckpt, "rb"), map_location=torch.device("cpu"))
            print(f"Successfully load ckpt {ckpt}")
            incompatibleKeys = self.load_state_dict(_ckpt[key], strict=False)
            print(incompatibleKeys)        
        except Exception as e:
            print(f"Failed loading checkpoint form {ckpt}: {e}")

    def forward(self, x):
        # 用于将输入 x 传递给每一层的 blocks 和 downsample 操作，返回处理后的结果
        def layer_forward(l, x):
            x = l.blocks(x)
            y = l.downsample(x)
            return x, y

        x = self.patch_embed(x)
        outs = []
        for i, layer in enumerate(self.layers):
            o, x = layer_forward(layer, x) # (B, H, W, C)
            if i in self.out_indices:
                norm_layer = getattr(self, f'outnorm{i}')
                out = norm_layer(o)
                if not self.channel_first:
                    out = out.permute(0, 3, 1, 2)
                outs.append(out.contiguous())

        if len(self.out_indices) == 0:
            return x
        # 将每层网络提取的特征输出为一个列表
        return outs


# =====================================================
def vanilla_vmamba_tiny():
    return VSSM(
        depths=[2, 2, 9, 2], dims=96, drop_path_rate=0.2, 
        patch_size=4, in_chans=3, num_classes=1000, 
        ssm_d_state=16, ssm_ratio=2.0, ssm_dt_rank="auto", ssm_act_layer="silu",
        ssm_conv=3, ssm_conv_bias=True, ssm_drop_rate=0.0, 
        ssm_init="v0", forward_type="v0", 
        mlp_ratio=0.0, mlp_act_layer="gelu", mlp_drop_rate=0.0, gmlp=False,
        patch_norm=True, norm_layer="ln", 
        downsample_version="v1", patchembed_version="v1", 
        use_checkpoint=False, posembed=False, imgsize=224, 
    )


def vanilla_vmamba_small():
    return VSSM(
        depths=[2, 2, 27, 2], dims=96, drop_path_rate=0.3, 
        patch_size=4, in_chans=3, num_classes=1000, 
        ssm_d_state=16, ssm_ratio=2.0, ssm_dt_rank="auto", ssm_act_layer="silu",
        ssm_conv=3, ssm_conv_bias=True, ssm_drop_rate=0.0, 
        ssm_init="v0", forward_type="v0", 
        mlp_ratio=0.0, mlp_act_layer="gelu", mlp_drop_rate=0.0, gmlp=False,
        patch_norm=True, norm_layer="ln", 
        downsample_version="v1", patchembed_version="v1", 
        use_checkpoint=False, posembed=False, imgsize=224, 
    )


def vanilla_vmamba_base():
    return VSSM(
        depths=[2, 2, 27, 2], dims=128, drop_path_rate=0.6, 
        patch_size=4, in_chans=3, num_classes=1000, 
        ssm_d_state=16, ssm_ratio=2.0, ssm_dt_rank="auto", ssm_act_layer="silu",
        ssm_conv=3, ssm_conv_bias=True, ssm_drop_rate=0.0, 
        ssm_init="v0", forward_type="v0", 
        mlp_ratio=0.0, mlp_act_layer="gelu", mlp_drop_rate=0.0, gmlp=False,
        patch_norm=True, norm_layer="ln", 
        downsample_version="v1", patchembed_version="v1", 
        use_checkpoint=False, posembed=False, imgsize=224, 
    )


# =====================================================
def vmamba_tiny_s2l5(channel_first=True):
    return VSSM(
        depths=[2, 2, 5, 2], dims=96, drop_path_rate=0.2, 
        patch_size=4, in_chans=3, num_classes=1000, 
        ssm_d_state=1, ssm_ratio=2.0, ssm_dt_rank="auto", ssm_act_layer="silu",
        ssm_conv=3, ssm_conv_bias=False, ssm_drop_rate=0.0, 
        ssm_init="v0", forward_type="v05_noz", 
        mlp_ratio=4.0, mlp_act_layer="gelu", mlp_drop_rate=0.0, gmlp=False,
        patch_norm=True, norm_layer=("ln2d" if channel_first else "ln"), 
        downsample_version="v3", patchembed_version="v2", 
        use_checkpoint=False, posembed=False, imgsize=224, 
    )


def vmamba_small_s2l15(channel_first=True):
    return VSSM(
        depths=[2, 2, 15, 2], dims=96, drop_path_rate=0.3, 
        patch_size=4, in_chans=3, num_classes=1000, 
        ssm_d_state=1, ssm_ratio=2.0, ssm_dt_rank="auto", ssm_act_layer="silu",
        ssm_conv=3, ssm_conv_bias=False, ssm_drop_rate=0.0, 
        ssm_init="v0", forward_type="v05_noz", 
        mlp_ratio=4.0, mlp_act_layer="gelu", mlp_drop_rate=0.0, gmlp=False,
        patch_norm=True, norm_layer=("ln2d" if channel_first else "ln"), 
        downsample_version="v3", patchembed_version="v2", 
        use_checkpoint=False, posembed=False, imgsize=224, 
    )


def vmamba_base_s2l15(channel_first=True):
    return VSSM(
        depths=[2, 2, 15, 2], dims=128, drop_path_rate=0.6, 
        patch_size=4, in_chans=3, num_classes=1000, 
        ssm_d_state=1, ssm_ratio=2.0, ssm_dt_rank="auto", ssm_act_layer="silu",
        ssm_conv=3, ssm_conv_bias=False, ssm_drop_rate=0.0, 
        ssm_init="v0", forward_type="v05_noz", 
        mlp_ratio=4.0, mlp_act_layer="gelu", mlp_drop_rate=0.0, gmlp=False,
        patch_norm=True, norm_layer=("ln2d" if channel_first else "ln"), 
        downsample_version="v3", patchembed_version="v2", 
        use_checkpoint=False, posembed=False, imgsize=224, 
    )


# =====================================================
def vmamba_tiny_s1l8(channel_first=True):
    return VSSM(
        depths=[2, 2, 8, 2], dims=96, drop_path_rate=0.2, 
        patch_size=4, in_chans=3, num_classes=1000, 
        ssm_d_state=1, ssm_ratio=1.0, ssm_dt_rank="auto", ssm_act_layer="silu",
        ssm_conv=3, ssm_conv_bias=False, ssm_drop_rate=0.0, 
        ssm_init="v0", forward_type="v05_noz", 
        mlp_ratio=4.0, mlp_act_layer="gelu", mlp_drop_rate=0.0, gmlp=False,
        patch_norm=True, norm_layer=("ln2d" if channel_first else "ln"), 
        downsample_version="v3", patchembed_version="v2", 
        use_checkpoint=False, posembed=False, imgsize=224, 
    )


def vmamba_small_s1l20(channel_first=True):
    return VSSM(
        depths=[2, 2, 20, 2], dims=96, drop_path_rate=0.3, 
        patch_size=4, in_chans=3, num_classes=1000, 
        ssm_d_state=1, ssm_ratio=1.0, ssm_dt_rank="auto", ssm_act_layer="silu",
        ssm_conv=3, ssm_conv_bias=False, ssm_drop_rate=0.0, 
        ssm_init="v0", forward_type="v05_noz", 
        mlp_ratio=4.0, mlp_act_layer="gelu", mlp_drop_rate=0.0, gmlp=False,
        patch_norm=True, norm_layer=("ln2d" if channel_first else "ln"), 
        downsample_version="v3", patchembed_version="v2", 
        use_checkpoint=False, posembed=False, imgsize=224, 
    )


def vmamba_base_s1l20(channel_first=True):
    return VSSM(
        depths=[2, 2, 20, 2], dims=128, drop_path_rate=0.5, 
        patch_size=4, in_chans=3, num_classes=1000, 
        ssm_d_state=1, ssm_ratio=1.0, ssm_dt_rank="auto", ssm_act_layer="silu",
        ssm_conv=3, ssm_conv_bias=False, ssm_drop_rate=0.0, 
        ssm_init="v0", forward_type="v05_noz", 
        mlp_ratio=4.0, mlp_act_layer="gelu", mlp_drop_rate=0.0, gmlp=False,
        patch_norm=True, norm_layer=("ln2d" if channel_first else "ln"), 
        downsample_version="v3", patchembed_version="v2", 
        use_checkpoint=False, posembed=False, imgsize=224, 
    )


# mamba2 support =====================================================
# FLOPS count do not work now for mamba2!
def vmamba_tiny_m2():
    return VSSM(
        depths=[2, 2, 4, 2], dims=96, drop_path_rate=0.2, 
        patch_size=4, in_chans=3, num_classes=1000, 
        ssm_d_state=64, ssm_ratio=1.0, ssm_dt_rank="auto", ssm_act_layer="gelu",
        ssm_conv=3, ssm_conv_bias=False, ssm_drop_rate=0.0, 
        ssm_init="v2", forward_type="m0_noz", 
        mlp_ratio=4.0, mlp_act_layer="gelu", mlp_drop_rate=0.0, gmlp=False,
        patch_norm=True, norm_layer="ln",
        downsample_version="v3", patchembed_version="v2", 
        use_checkpoint=False, posembed=False, imgsize=224, 
    )


def vmamba_small_m2():
    return VSSM(
        depths=[2, 2, 12, 2], dims=96, drop_path_rate=0.3, 
        patch_size=4, in_chans=3, num_classes=1000, 
        ssm_d_state=64, ssm_ratio=1.0, ssm_dt_rank="auto", ssm_act_layer="gelu",
        ssm_conv=3, ssm_conv_bias=False, ssm_drop_rate=0.0, 
        ssm_init="v2", forward_type="m0_noz", 
        mlp_ratio=4.0, mlp_act_layer="gelu", mlp_drop_rate=0.0, gmlp=False,
        patch_norm=True, norm_layer="ln",
        downsample_version="v3", patchembed_version="v2", 
        use_checkpoint=False, posembed=False, imgsize=224, 
    )


def vmamba_base_m2():
    return VSSM(
        depths=[2, 2, 12, 2], dims=128, drop_path_rate=0.3, 
        patch_size=4, in_chans=3, num_classes=1000, 
        ssm_d_state=64, ssm_ratio=1.0, ssm_dt_rank="auto", ssm_act_layer="gelu",
        ssm_conv=3, ssm_conv_bias=False, ssm_drop_rate=0.0, 
        ssm_init="v2", forward_type="m0_noz", 
        mlp_ratio=4.0, mlp_act_layer="gelu", mlp_drop_rate=0.0, gmlp=False,
        patch_norm=True, norm_layer="ln",
        downsample_version="v3", patchembed_version="v2", 
        use_checkpoint=False, posembed=False, imgsize=224, 
    )


# 这行代码确保了只有在脚本作为主程序运行时，下面的代码才会执行。它可以防止在被其他脚本导入时自动执行这段代码
if __name__ == "__main__":
    # 模型初始化
    model_ref = vmamba_tiny_s1l8()

    model = VSSM(
        depths=[2, 2, 4, 2], dims=96, drop_path_rate=0.2, 
        patch_size=4, in_chans=3, num_classes=1000, 
        ssm_d_state=64, ssm_ratio=1.0, ssm_dt_rank="auto", ssm_act_layer="gelu",
        ssm_conv=3, ssm_conv_bias=False, ssm_drop_rate=0.0, 
        ssm_init="v2", forward_type="m0_noz", 
        mlp_ratio=4.0, mlp_act_layer="gelu", mlp_drop_rate=0.0, gmlp=False,
        patch_norm=True, norm_layer="ln",
        downsample_version="v3", patchembed_version="v2", 
        use_checkpoint=False, posembed=False, imgsize=224, 
    )
    # 模型参数输出
    print(parameter_count(model)[""])
    print(model.flops()) # wrong
    # 模型训练准备
    model.cuda().train()
    model_ref.cuda().train()
    # 性能基准测试
    def bench(model):
        import time
        # 创建一个大小为 (128, 3, 224, 224) 的随机输入张量，并将其移动到GPU
        inp = torch.randn((128, 3, 224, 224)).cuda()
        for _ in range(30):
            model(inp)
        # 确保在 GPU 上的所有操作完成后才继续执行后续的代码。
        torch.cuda.synchronize()
        tim = time.time()
        # 接着对模型进行 30 次前向传播，记录时间
        for _ in range(30):
            model(inp)
        torch.cuda.synchronize()
        tim1 = time.time() - tim

        for _ in range(30):
            model(inp).sum().backward()
        torch.cuda.synchronize()
        # 然后进行反向传播（每次反向传播后计算梯度），并记录时间
        tim = time.time()
        for _ in range(30):
            model(inp).sum().backward()
        torch.cuda.synchronize()
        tim2 = time.time() - tim
        # tim1 / 30 和 tim2 / 30 分别是前向传播和反向传播的平均时间
        return tim1 / 30, tim2 / 30
    
    print(bench(model_ref))
    print(bench(model))
    # 在执行到这一行时，程序会暂停并进入调试模式，可以用来检查当前代码的状态或变量
    breakpoint()


