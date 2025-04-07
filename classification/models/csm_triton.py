import torch
import warnings

from hilbertcurve.hilbertcurve import HilbertCurve

WITH_TRITON = True
# WITH_TRITON = False
try:
    import triton
    import triton.language as tl
except:
    WITH_TRITON = False
    warnings.warn("Triton not installed, fall back to pytorch implements.")


# to make sure cached_property can be loaded for triton
if WITH_TRITON:
    try:
        from functools import cached_property
    except:
        warnings.warn("if you are using py37, add this line to functools.py: "
            "cached_property = lambda func: property(lru_cache()(func))")
        

def hilbert_curve(order, size):
    """ 生成 Hilbert 曲线索引 """
    hilbert = HilbertCurve(order, 2)  # 2D Hilbert Curve
    points = [hilbert.point_from_distance(i) for i in range(size * size)]
    points = torch.tensor(points)
    # 转换为行优先索引
    indices = points[:, 0] * size + points[:, 1]
    return indices

def diagonal_indices(H, W,reverse=False):
    """
    获取斜向扫描索引，从左上到右下（或右上到左下）。
    Args:
        H, W: 高和宽
        reverse: 是否反向（右上到左下）
    Returns:
        indices: (H * W,) 形状的 1D 索引张量
    """
    indices = []
    if not reverse:
        for s in range(H + W - 1):  # 所有斜线编号
            temp = []
            for i in range(H):
                j = s - i
                if 0 <= j < W:
                    temp.append(i * W + j)
            if s % 2 == 0:      #连续式扫描，如果没有的话，便是z形状的
                 temp.reverse()
            indices.extend(temp)
    else:
        for s in range(H + W - 1):  # 所有斜线编号
            temp = []
            for i in range(H):
                j = s - (H - 1 - i)
                if 0 <= j < W:
                    temp.append(i * W + j)
            if s % 2 == 0:
                 temp.reverse()  # 锯齿式可选
            indices.extend(temp)
    return torch.tensor(indices, dtype=torch.long)

def continuous_indices(H, W):
    """
    获取斜向扫描索引，从左上到右下（或右上到左下）。
    Args:
        H, W: 高和宽
        reverse: 是否反向（右上到左下）
    Returns:
        indices: (H * W,) 形状的 1D 索引张量
    """
    indices = []
    for i in range(H):
        if i%2==0:
            for j in range(W):
                indices.append(i*W+j)
        else:
            for j in range(W - 1, -1, -1):
                indices.append(i*W+j)
    return torch.tensor(indices, dtype=torch.long)


# torch implementation ========================================
def scan_fwd(x: torch.Tensor, in_channel_first=True, out_channel_first=True, scans=0):

    B, C, H, W = x.shape
    if scans == 0:
        y = x.new_empty((B, 4, C, H * W))
        y[:, 0, :, :] = x.flatten(2, 3)
        y[:, 1, :, :] = x.transpose(dim0=2, dim1=3).flatten(2, 3)
        y[:, 2:4, :, :] = torch.flip(y[:, 0:2, :, :], dims=[-1])
    elif scans == 1:
        y = x.view(B, 1, C, H * W).repeat(1, 4, 1, 1)
    elif scans == 2:
        y = x.view(B, 1, C, H * W).repeat(1, 2, 1, 1)
        y = torch.cat([y, y.flip(dims=[-1])], dim=1)
    elif scans == 3:
        y = x.new_empty((B, 4, C, H * W))
        order = int(torch.log2(torch.tensor(H)))
          # 计算 Hilbert 曲线的阶数（假设 H=W 且为 2 的幂）
        indices = hilbert_curve(order, H)  # 生成 Hilbert 扫描索引
        x_flat = x.flatten(2, 3)  # 将 x 展平
        x_flat_trans = x.transpose(dim0=2, dim1=3).flatten(2, 3)  # 将 x 展平
        x_hilbert = x_flat[:, :, indices]  # 按照 Hilbert 索引重新排序
        x_hilbert_trans = x_flat_trans[:, :, indices]  # 按照 Hilbert 索引重新排序
        assert indices.max() < x_flat.shape[-1], "Hilbert 索引越界！"
        assert indices.min() >= 0, "Hilbert 索引小于 0！"
        
        # 创建输出张量
        y = x.new_empty((B, 4, C, H * W))
        # y[:, 0, :, :] = x_hilbert
        # y[:, 1, :, :] = y[:, 0, :, :]  # 复制原始 Hilbert 扫描
        # y[:, 2, :, :] = torch.flip(y[:, 0, :, :], dims=[-1])  # 反向 Hilbert 扫描
        # y[:, 3, :, :] = y[:, 2, :, :]  # 复制反向 Hilbert 扫描

        y[:, 0, :, :] = x_hilbert
        y[:, 1, :, :] = x_hilbert_trans # 复制原始 Hilbert 扫描
        y[:, 2, :, :] = torch.flip(y[:, 0, :, :], dims=[-1])  # 反向 Hilbert 扫描
        y[:, 3, :, :] = torch.flip(y[:, 1, :, :], dims=[-1])   # 复制反向 Hilbert 扫描

        # 释放无用变量
        del x_flat, x_flat_trans,x_hilbert, x_hilbert_trans,indices
        torch.cuda.empty_cache()  # 手动清理 CUDA 缓存
    elif scans == 4:
        # 假设输入 x: (B, C, H, W)
        x_flat = x.flatten(2, 3)  # 展平为 (B, C, H*W)
        diag_idx = diagonal_indices(H, W)  # 左上到右下扫描
        anti_diag_idx=diagonal_indices(H, W,reverse=True)

        y = x.new_empty((B, 4, C, H * W))
        y[:, 0, :, :] = x_flat[:, :, diag_idx]       # 斜向扫描
        y[:, 1, :, :] = x_flat[:, :, anti_diag_idx]  # 反斜向扫描
        y[:, 2, :, :] = torch.flip(y[:, 0, :, :], dims=[-1])  # 反转版本
        y[:, 3, :, :] = torch.flip(y[:, 1, :, :], dims=[-1])  # 反转版本
        # 释放无用变量
        del x_flat, diag_idx, anti_diag_idx
        torch.cuda.empty_cache()  # 手动清理 CUDA 缓存

    elif scans == 5:
        # 假设输入 x: (B, C, H, W)
        x_flat = x.flatten(2, 3)  # 展平为 (B, C, H*W)
        x_flat_trans = x.transpose(dim0=2, dim1=3).flatten(2, 3)  # 将 x 展平
        continuous_idx = continuous_indices(H, W) # 左上到右下扫描

        y = x.new_empty((B, 4, C, H * W))
        y[:, 0, :, :] = x_flat[:, :, continuous_idx]       # 斜向扫描
        y[:, 1, :, :] = x_flat_trans[:, :, continuous_idx]  # 反斜向扫描
        y[:, 2, :, :] = torch.flip(y[:, 0, :, :], dims=[-1])  # 反转版本
        y[:, 3, :, :] = torch.flip(y[:, 1, :, :], dims=[-1])  # 反转版本
        # 释放无用变量
        del x_flat, x_flat_trans, continuous_idx
        torch.cuda.empty_cache()  # 手动清理 CUDA 缓存

    if in_channel_first and (not out_channel_first):
        y = y.permute(0, 3, 1, 2).contiguous()
    elif (not in_channel_first) and out_channel_first:
        y = y.permute(0, 2, 3, 1).contiguous()
    # y torch.Size([1, 4, 96, 3136])
    return y


def merge_fwd(y: torch.Tensor, in_channel_first=True, out_channel_first=True, scans=0):

    B, K, D, H, W = y.shape
    y = y.view(B, K, D, -1)
    if scans == 0:
        # y[:, 0:2]：这里我们选择了 y 的第 0 到第 1 个 K 的部分，形状为 (B, 2, D, H * W)
        y = y[:, 0:2] + y[:, 2:4].flip(dims=[-1]).view(B, 2, D, -1)
        y = y[:, 0] + y[:, 1].view(B, -1, W, H).transpose(dim0=2, dim1=3).contiguous().view(B, D, -1)
    elif scans == 1:
        y = y.sum(1)
    elif scans == 2:
        y = y[:, 0:2] + y[:, 2:4].flip(dims=[-1]).view(B, 2, D, -1)
        y = y.sum(1)
    elif scans == 3:
        # y[:, 0:2]：这里我们选择了 y 的第 0 到第 1 个 K 的部分，形状为 (B, 2, D, H * W)
        # y = y[:, 0:2] + y[:, 2:4].flip(dims=[1]).view(B,2,D,-1)
        # y = y.sum(1)
        y = y[:, 0:2] + y[:, 2:4].flip(dims=[-1]).view(B, 2, D, -1)
        y = y[:, 0] + y[:, 1].view(B, -1, W, H).transpose(dim0=2, dim1=3).contiguous().view(B, D, -1)
    elif scans == 4:
        # y[:, 0:2]：这里我们选择了 y 的第 0 到第 1 个 K 的部分，形状为 (B, 2, D, H * W)
        y = y[:, 0:2] + y[:, 2:4].flip(dims=[-1]).view(B, 2, D, -1)
        y = y[:, 0] + y[:, 1].view(B, -1, W, H).transpose(dim0=2, dim1=3).flip(dims=[-1]).contiguous().view(B, D, -1)
    elif scans == 5:
        y = y[:, 0:2] + y[:, 2:4].flip(dims=[-1]).view(B, 2, D, -1)
        y = y[:, 0] + y[:, 1].view(B, -1, W, H).transpose(dim0=2, dim1=3).contiguous().view(B, D, -1)

    if in_channel_first and (not out_channel_first):
        y = y.permute(0, 2, 1).contiguous()
    elif (not in_channel_first) and out_channel_first:
        y = y.permute(0, 2, 1).contiguous()
    
    return y


def cross_scan1b1_fwd(x: torch.Tensor, in_channel_first=True, out_channel_first=True, scans=0):
    if in_channel_first:
        B, _, C, H, W = x.shape
        if scans == 0:
            y = torch.stack([
                x[:, 0].flatten(2, 3),
                x[:, 1].transpose(dim0=2, dim1=3).flatten(2, 3),
                torch.flip(x[:, 2].flatten(2, 3), dims=[-1]),
                torch.flip(x[:, 3].transpose(dim0=2, dim1=3).flatten(2, 3), dims=[-1]),
            ], dim=1)
        elif scans == 1:
            y = x.flatten(2, 3)
        elif scans == 2:
            y = torch.stack([
                x[:, 0].flatten(2, 3),
                x[:, 1].flatten(2, 3),
                torch.flip(x[:, 2].flatten(2, 3), dims=[-1]),
                torch.flip(x[:, 3].flatten(2, 3), dims=[-1]),
            ], dim=1)
    else:
        B, H, W, _, C = x.shape
        if scans == 0:
            y = torch.stack([
                x[:, :, :, 0].flatten(1, 2),
                x[:, :, :, 1].transpose(dim0=1, dim1=2).flatten(1, 2),
                torch.flip(x[:, :, :, 2].flatten(1, 2), dims=[1]),
                torch.flip(x[:, :, :, 3].transpose(dim0=1, dim1=2).flatten(1, 2), dims=[1]),
            ], dim=2)
        elif scans == 1:
            y = x.flatten(1, 2)
        elif scans == 2:
            y = torch.stack([
                x[:, 0].flatten(1, 2),
                x[:, 1].flatten(1, 2),
                torch.flip(x[:, 2].flatten(1, 2), dims=[-1]),
                torch.flip(x[:, 3].flatten(1, 2), dims=[-1]),
            ], dim=2)

    if in_channel_first and (not out_channel_first):
        y = y.permute(0, 3, 1, 2).contiguous()
    elif (not in_channel_first) and out_channel_first:
        y = y.permute(0, 2, 3, 1).contiguous()

    return y


def cross_merge1b1_fwd(y: torch.Tensor, in_channel_first=True, out_channel_first=True, scans=0):
    if out_channel_first:
        B, K, D, H, W = y.shape
        y = y.view(B, K, D, -1)
        if scans == 0:
            y = torch.stack([
                y[:, 0],
                y[:, 1].view(B, -1, W, H).transpose(dim0=2, dim1=3).flatten(2, 3),
                torch.flip(y[:, 2], dims=[-1]),
                torch.flip(y[:, 3].view(B, -1, W, H).transpose(dim0=2, dim1=3).flatten(2, 3), dims=[-1]),
            ], dim=1)
        elif scans == 1:
            y = y
        elif scans == 2:
            y = torch.stack([
                y[:, 0],
                y[:, 1],
                torch.flip(y[:, 2], dims=[-1]),
                torch.flip(y[:, 3], dims=[-1]),
            ], dim=1)
    else:
        B, H, W, K, D = y.shape
        y = y.view(B, -1, K, D)
        if scans == 0:
            y = torch.stack([
                y[:, :, 0],
                y[:, :, 1].view(B, W, H, -1).transpose(dim0=1, dim1=2).flatten(1, 2),
                torch.flip(y[:, :, 2], dims=[1]),
                torch.flip(y[:, :, 3].view(B, W, H, -1).transpose(dim0=1, dim1=2).flatten(1, 2), dims=[1]),
            ], dim=2)
        elif scans == 1:
            y = y
        elif scans == 2:
            y = torch.stack([
                y[:, :, 0],
                y[:, :, 1],
                torch.flip(y[:, :, 2], dims=[1]),
                torch.flip(y[:, :, 3], dims=[1]),
            ], dim=2)

    if out_channel_first and (not in_channel_first):
        y = y.permute(0, 3, 1, 2).contiguous()
    elif (not out_channel_first) and in_channel_first:
        y = y.permute(0, 2, 3, 1).contiguous()

    return y

# CrossScanF 是 torch.autograd.Function 的子类，实现了自定义的 前向传播 (forward) 和反向传播 (backward)
class ScanF(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, in_channel_first=True, out_channel_first=True, one_by_one=False, scans=0):
        # x: (B, C, H, W) | (B, H, W, C) | (B, 4, C, H, W) | (B, H, W, 4, C)
        # y: (B, 4, C, H * W) | (B, H * W, 4, C)
        # ctx 用于存储前向传播过程中需要在反向传播时使用的信息。
        ctx.in_channel_first = in_channel_first
        ctx.out_channel_first = out_channel_first
        ctx.one_by_one = one_by_one
        ctx.scans = scans

        if one_by_one:
            B, K, C, H, W = x.shape
            if not in_channel_first:
                B, H, W, K, C = x.shape
        else:
            B, C, H, W = x.shape
            if not in_channel_first:
                B, H, W, C = x.shape
        ctx.shape = (B, C, H, W)
        # B=1,C=96,H=56,W=56
        _fn = cross_scan1b1_fwd if one_by_one else scan_fwd
        y = _fn(x, in_channel_first, out_channel_first, scans)
        # y = torch.Size([1, 4, 96, 3136])
        test=y
        return y
    
    @staticmethod
    def backward(ctx, ys: torch.Tensor):
        # out: (b, k, d, l)
        in_channel_first = ctx.in_channel_first
        out_channel_first = ctx.out_channel_first
        one_by_one = ctx.one_by_one
        scans = ctx.scans
        B, C, H, W = ctx.shape

        ys = ys.view(B, -1, C, H, W) if out_channel_first else ys.view(B, H, W, -1, C)
        _fn = cross_merge1b1_fwd if one_by_one else merge_fwd
        y = _fn(ys, in_channel_first, out_channel_first, scans)
        
        if one_by_one:
            y = y.view(B, 4, -1, H, W) if in_channel_first else y.view(B, H, W, 4, -1)
        else:
            y = y.view(B, -1, H, W) if in_channel_first else y.view(B, H, W, -1)
        test=y
        return y, None, None, None, None


class MergeF(torch.autograd.Function):
    @staticmethod
    def forward(ctx, ys: torch.Tensor, in_channel_first=True, out_channel_first=True, one_by_one=False, scans=0):
        # x: (B, C, H, W) | (B, H, W, C) | (B, 4, C, H, W) | (B, H, W, 4, C)
        # y: (B, 4, C, H * W) | (B, H * W, 4, C)
        ctx.in_channel_first = in_channel_first
        ctx.out_channel_first = out_channel_first
        ctx.one_by_one = one_by_one
        ctx.scans = scans

        B, K, C, H, W = ys.shape
        if not out_channel_first:
            B, H, W, K, C = ys.shape
        ctx.shape = (B, C, H, W)
        
        _fn = cross_merge1b1_fwd if one_by_one else merge_fwd
        y = _fn(ys, in_channel_first, out_channel_first, scans)
        # y= torch.Size([1, 96, 3136])
        test=y
        return y
    
    @staticmethod
    def backward(ctx, x: torch.Tensor):
        # B, D, L = x.shape
        # out: (b, k, d, h, w)
        in_channel_first = ctx.in_channel_first
        out_channel_first = ctx.out_channel_first
        one_by_one = ctx.one_by_one
        scans = ctx.scans
        B, C, H, W = ctx.shape
    
        if not one_by_one:
            if in_channel_first:
                x = x.view(B, C, H, W)
            else:
                x = x.view(B, H, W, C)
        else:
            if in_channel_first:
                x = x.view(B, 4, C, H, W)
            else:
                x = x.view(B, H, W, 4, C)   
                     
        _fn = cross_scan1b1_fwd if one_by_one else scan_fwd
        x = _fn(x, in_channel_first, out_channel_first, scans)
        x = x.view(B, 4, C, H, W) if out_channel_first else x.view(B, H, W, 4, C)
        test=x
        return x, None, None, None, None


# triton implements ========================================

@triton.jit
def triton_cross_scan_flex(
    x: tl.tensor, # (B, C, H, W) | (B, H, W, C) | (B, 4, C, H, W) | (B, H, W, 4, C)
    y: tl.tensor, # (B, 4, C, H, W) | (B, H, W, 4, C)
    x_layout: tl.constexpr,
    y_layout: tl.constexpr,
    operation: tl.constexpr, 
    onebyone: tl.constexpr,
    scans: tl.constexpr, 
    BC: tl.constexpr,
    BH: tl.constexpr,
    BW: tl.constexpr,
    DC: tl.constexpr,
    DH: tl.constexpr,
    DW: tl.constexpr,
    NH: tl.constexpr,
    NW: tl.constexpr, 
):
    # x_layout = 0
    # y_layout = 1 # 0 BCHW, 1 BHWC
    # operation = 0 # 0 scan, 1 merge
    # onebyone = 0 # 0 false, 1 true
    # scans = 0 # 0 cross scan, 1 unidirectional, 2 bidirectional

    # BC, BH, BW = 1, 32, 32
    # NH=2，NW=2,NC=96
    # DC=96,DH=56,DW=56
    
    # 在 Triton 中，每个 GPU 线程都会被分配一个 program_id，用于在计算时标识自己的任务
    # program_id(0)：用于索引 H-W 维度的块（即 i_hw，表示图像在 H-W 方向上的块索引）
    # program_id(1)：用于索引 通道分块（i_c）
    # program_id(2)：用于索引 batch 维度（i_b）
    i_hw, i_c, i_b = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    # i_hw 表示当前线程在 H-W 方向的块索引，它需要拆分成具体的 高度块索引 i_h 和 宽度块索引 i_w
    # i_h = i_hw // NW：计算当前线程对应的 高度块索引（整数除法）
    # i_w = i_hw % NW：计算当前线程对应的 宽度块索引（取余数）
    i_h, i_w = (i_hw // NW), (i_hw % NW)
    # 创建 H 和 W 方向的掩码，避免访问超出图像大小的像素
    # i_h * BH：表示当前块的起始高度索引
    # i_h * BH + tl.arange(0, BH)：计算当前块内所有像素在原始图像中的高度索引
    # < DH：检查这些索引是否超出了原始图像的高度
    _mask_h = (i_h * BH + tl.arange(0, BH)) < DH
    _mask_w = (i_w * BW + tl.arange(0, BW)) < DW
    #_mask_h[:, None]：将 _mask_h 变成列向量（增加一个维度）。
    # _mask_w[None, :]：将 _mask_w 变成行向量（增加一个维度）。
    # &（按位与）：计算 H-W 方向的有效像素掩码，确保只有 H 和 W 都在有效范围内的像素才会被处理。
    # 最终 _mask_hw 是一个形状 (BH, BW) 的布尔掩码，表示当前块内哪些像素是有效的，哪些需要忽略
    _mask_hw = _mask_h[:, None] & _mask_w[None, :]
    # 计算当前块内实际剩余的通道数（如果 DC 不能被 BC 整除，最后一个块可能不满）
    # min(1, 16) = 1，说明最后一个通道块只包含 1 个通道，而前面的块都包含 16 个通道。
    _for_C = min(DC - i_c * BC, BC)
    # tl.arange(0, BH)[:, None]：生成一个形状 (BH, 1) 的向量，表示块内部的行索引（沿着高度方向）。
    pos_h = (i_h * BH + tl.arange(0, BH)[:, None])          # 计算当前块内部每个像素在 原图像高度方向上的实际位置索引
    pos_w = (i_w * BW + tl.arange(0, BW)[None, :])          # 当前块内的所有像素在整个图像中的 宽度索引
    neg_h = (DH - i_h * BH - 1 - tl.arange(0, BH)[:, None])
    # DW - 1：宽度方向的最大索引。
    # i_w * BW：当前块的起始宽度索引。
    # tl.arange(0, BW)[None, :]：块内的列索引

    neg_w = (DW - i_w * BW - 1 - tl.arange(0, BW)[None, :])
    if scans == 0:
        # none; trans; flip; trans + flip;
        HWRoute0 = pos_h * DW + pos_w
        HWRoute1 = pos_w * DH + pos_h # trans
        HWRoute2 = neg_h * DW + neg_w # flip
        HWRoute3 = neg_w * DH + neg_h # trans + flip
    elif scans == 1:
        # none; none; none; none;
        HWRoute0 = pos_h * DW + pos_w
        HWRoute1 = HWRoute0
        HWRoute2 = HWRoute0
        HWRoute3 = HWRoute0
    elif scans == 2:
        # none; none; flip; flip;
        HWRoute0 = pos_h * DW + pos_w
        HWRoute1 = HWRoute0
        HWRoute2 = neg_h * DW + neg_w # flip
        HWRoute3 = HWRoute2      

    _tmp1 = DC * DH * DW

    y_ptr_base = y + i_b * 4 * _tmp1 + (i_c * BC * DH * DW if y_layout == 0 else i_c * BC)
    if y_layout == 0:
        p_y1 = y_ptr_base + HWRoute0
        p_y2 = y_ptr_base + _tmp1 + HWRoute1
        p_y3 = y_ptr_base + 2 * _tmp1 + HWRoute2
        p_y4 = y_ptr_base + 3 * _tmp1 + HWRoute3
    else:
        p_y1 = y_ptr_base + HWRoute0 * 4 * DC
        p_y2 = y_ptr_base + DC + HWRoute1 * 4 * DC
        p_y3 = y_ptr_base + 2 * DC + HWRoute2 * 4 * DC
        p_y4 = y_ptr_base + 3 * DC + HWRoute3 * 4 * DC       
    
    if onebyone == 0:
        x_ptr_base = x + i_b * _tmp1 + (i_c * BC * DH * DW if x_layout == 0 else i_c * BC)
        if x_layout == 0:
            p_x = x_ptr_base + HWRoute0
        else:
            p_x = x_ptr_base + HWRoute0 * DC

        if operation == 0:
            for idxc in range(_for_C):
                _idx_x = idxc * DH * DW if x_layout == 0 else idxc
                _idx_y = idxc * DH * DW if y_layout == 0 else idxc
                _x = tl.load(p_x + _idx_x, mask=_mask_hw)
                tl.store(p_y1 + _idx_y, _x, mask=_mask_hw)
                tl.store(p_y2 + _idx_y, _x, mask=_mask_hw)
                tl.store(p_y3 + _idx_y, _x, mask=_mask_hw)
                tl.store(p_y4 + _idx_y, _x, mask=_mask_hw)
        elif operation == 1:
            for idxc in range(_for_C):
                _idx_x = idxc * DH * DW if x_layout == 0 else idxc
                _idx_y = idxc * DH * DW if y_layout == 0 else idxc
                _y1 = tl.load(p_y1 + _idx_y, mask=_mask_hw)
                _y2 = tl.load(p_y2 + _idx_y, mask=_mask_hw)
                _y3 = tl.load(p_y3 + _idx_y, mask=_mask_hw)
                _y4 = tl.load(p_y4 + _idx_y, mask=_mask_hw)
                tl.store(p_x + _idx_x, _y1 + _y2 + _y3 + _y4, mask=_mask_hw)

    else:
        x_ptr_base = x + i_b * 4 * _tmp1 + (i_c * BC * DH * DW if x_layout == 0 else i_c * BC)
        if x_layout == 0:
            p_x1 = x_ptr_base + HWRoute0
            p_x2 = p_x1 + _tmp1
            p_x3 = p_x2 + _tmp1
            p_x4 = p_x3 + _tmp1  
        else:
            p_x1 = x_ptr_base + HWRoute0 * 4 * DC
            p_x2 = p_x1 + DC
            p_x3 = p_x2 + DC
            p_x4 = p_x3 + DC        
    
        if operation == 0:
            for idxc in range(_for_C):
                _idx_x = idxc * DH * DW if x_layout == 0 else idxc
                _idx_y = idxc * DH * DW if y_layout == 0 else idxc
                tl.store(p_y1 + _idx_y, tl.load(p_x1 + _idx_x, mask=_mask_hw), mask=_mask_hw)
                tl.store(p_y2 + _idx_y, tl.load(p_x2 + _idx_x, mask=_mask_hw), mask=_mask_hw)
                tl.store(p_y3 + _idx_y, tl.load(p_x3 + _idx_x, mask=_mask_hw), mask=_mask_hw)
                tl.store(p_y4 + _idx_y, tl.load(p_x4 + _idx_x, mask=_mask_hw), mask=_mask_hw)
        else:
            for idxc in range(_for_C):
                _idx_x = idxc * DH * DW if x_layout == 0 else idxc
                _idx_y = idxc * DH * DW if y_layout == 0 else idxc
                tl.store(p_x1 + _idx_x, tl.load(p_y1 + _idx_y), mask=_mask_hw)
                tl.store(p_x2 + _idx_x, tl.load(p_y2 + _idx_y), mask=_mask_hw)
                tl.store(p_x3 + _idx_x, tl.load(p_y3 + _idx_y), mask=_mask_hw)
                tl.store(p_x4 + _idx_x, tl.load(p_y4 + _idx_y), mask=_mask_hw)


class CrossScanTritonF(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, in_channel_first=True, out_channel_first=True, one_by_one=False, scans=0):
        # one_by_one 指示是否是 1x1 的卷积，为 False
        if one_by_one:
            if in_channel_first:
                B, _, C, H, W = x.shape
            else:
                B, H, W, _, C = x.shape
        else:
            if in_channel_first:
                B, C, H, W = x.shape
            else:
                B, H, W, C = x.shape
        B, C, H, W = int(B), int(C), int(H), int(W)  #B=1,C=96,H=56,W=56
        # 划分块数
        BC, BH, BW = 1, 32, 32
        # triton.cdiv为Triton 中的一个函数，用于执行“向上整除”
        # triton.cdiv(5, 2) 结果是 3（因为 5 除以 2 等于 2.5，向上取整为 3） 意思是用3个块处理5个像素
        # 计算对应块处理的元素个数
        NH, NW, NC = triton.cdiv(H, BH), triton.cdiv(W, BW), triton.cdiv(C, BC) # NH=2，NW=2,NC=96
        
        ctx.in_channel_first = in_channel_first         # True
        ctx.out_channel_first = out_channel_first       # True
        ctx.one_by_one = one_by_one                     # False
        ctx.scans = scans                               # 0
        ctx.shape = (B, C, H, W)                        # B=1,C=96,H=56,W=56
        ctx.triton_shape = (BC, BH, BW, NC, NH, NW)     # 1, 32, 32,96,2,2

        y = x.new_empty((B, 4, C, H * W)) if out_channel_first else x.new_empty((B, H * W, 4, C))
        triton_cross_scan_flex[(NH * NW, NC, B)](
            x.contiguous(), y, 
            (0 if in_channel_first else 1), (0 if out_channel_first else 1), 0, (0 if not one_by_one else 1), scans, 
            BC, BH, BW, C, H, W, NH, NW
        )
        return y
        
    @staticmethod
    def backward(ctx, y: torch.Tensor):
        in_channel_first = ctx.in_channel_first
        out_channel_first = ctx.out_channel_first
        one_by_one = ctx.one_by_one
        scans = ctx.scans
        B, C, H, W = ctx.shape
        BC, BH, BW, NC, NH, NW = ctx.triton_shape
        if one_by_one:
            x = y.new_empty((B, 4, C, H, W)) if in_channel_first else y.new_empty((B, H, W, 4, C))
        else:
            x = y.new_empty((B, C, H, W)) if in_channel_first else y.new_empty((B, H, W, C))
        
        triton_cross_scan_flex[(NH * NW, NC, B)](
            x, y.contiguous(), 
            (0 if in_channel_first else 1), (0 if out_channel_first else 1), 1, (0 if not one_by_one else 1), scans,
            BC, BH, BW, C, H, W, NH, NW
        )
        return x, None, None, None, None


class CrossMergeTritonF(torch.autograd.Function):
    @staticmethod
    def forward(ctx, y: torch.Tensor, in_channel_first=True, out_channel_first=True, one_by_one=False, scans=0):
        if out_channel_first:
            B, _, C, H, W = y.shape
        else:
            B, H, W, _, C = y.shape
        B, C, H, W = int(B), int(C), int(H), int(W)
        BC, BH, BW = 1, 32, 32
        NH, NW, NC = triton.cdiv(H, BH), triton.cdiv(W, BW), triton.cdiv(C, BC)
        ctx.in_channel_first = in_channel_first
        ctx.out_channel_first = out_channel_first
        ctx.one_by_one = one_by_one
        ctx.scans = scans
        ctx.shape = (B, C, H, W)
        ctx.triton_shape = (BC, BH, BW, NC, NH, NW)
        if one_by_one:
            x = y.new_empty((B, 4, C, H * W)) if in_channel_first else y.new_empty((B, H * W, 4, C))
        else:
            x = y.new_empty((B, C, H * W)) if in_channel_first else y.new_empty((B, H * W, C))
        triton_cross_scan_flex[(NH * NW, NC, B)](
            x, y.contiguous(), 
            (0 if in_channel_first else 1), (0 if out_channel_first else 1), 1, (0 if not one_by_one else 1), scans,
            BC, BH, BW, C, H, W, NH, NW
        )
        return x
        
    @staticmethod
    def backward(ctx, x: torch.Tensor):
        in_channel_first = ctx.in_channel_first
        out_channel_first = ctx.out_channel_first
        one_by_one = ctx.one_by_one
        scans = ctx.scans
        B, C, H, W = ctx.shape
        BC, BH, BW, NC, NH, NW = ctx.triton_shape
        y = x.new_empty((B, 4, C, H, W)) if out_channel_first else x.new_empty((B, H, W, 4, C))
        triton_cross_scan_flex[(NH * NW, NC, B)](
            x.contiguous(), y, 
            (0 if in_channel_first else 1), (0 if out_channel_first else 1), 0, (0 if not one_by_one else 1), scans,
            BC, BH, BW, C, H, W, NH, NW
        )
        return y, None, None, None, None, None


# @torch.compile(options={"triton.cudagraphs": True}, fullgraph=True)
def scan_fn(x: torch.Tensor, in_channel_first=True, out_channel_first=True, one_by_one=False, scans=0, force_torch=False):
    # x: (B, C, H, W) | (B, H, W, C) | (B, 4, C, H, W) | (B, H, W, 4, C)
    # y: (B, 4, C, L) | (B, L, 4, C)
    # scans: 0: cross scan; 1 unidirectional; 2: bidirectional;
    # WITH_TRITON=False
    CSF = CrossScanTritonF if WITH_TRITON and x.is_cuda and (not force_torch) else ScanF  #CrossScanTritonF
    with torch.cuda.device(x.device):
        return CSF.apply(x, in_channel_first, out_channel_first, one_by_one, scans)


# @torch.compile(options={"triton.cudagraphs": True}, fullgraph=True)
def merge_fn(y: torch.Tensor, in_channel_first=True, out_channel_first=True, one_by_one=False, scans=0, force_torch=False):
    # y: (B, 4, C, L) | (B, L, 4, C)
    # x: (B, C, H * W) | (B, H * W, C) | (B, 4, C, H * W) | (B, H * W, 4, C)
    # scans: 0: cross scan; 1 unidirectional; 2: bidirectional;
    # WITH_TRITON=False
    CMF = CrossMergeTritonF if WITH_TRITON and y.is_cuda and (not force_torch) else MergeF
    with torch.cuda.device(y.device):
        return CMF.apply(y, in_channel_first, out_channel_first, one_by_one, scans)


# checks =================================================================

class CHECK:
    def check_csm_triton():
        B, C, H, W = 256, 192, 56, 57
        dtype=torch.float16
        dtype=torch.float32
        x = torch.randn((B, C, H, W), dtype=dtype, device=torch.device("cuda")).requires_grad_(True)
        y = torch.randn((B, 4, C, H, W), dtype=dtype, device=torch.device("cuda")).requires_grad_(True)
        x1 = x.clone().detach().requires_grad_(True)
        y1 = y.clone().detach().requires_grad_(True)

        def cross_scan(x: torch.Tensor):
            B, C, H, W = x.shape
            L = H * W
            xs = torch.stack([
                x.view(B, C, L),
                torch.transpose(x, dim0=2, dim1=3).contiguous().view(B, C, L),
                torch.flip(x.contiguous().view(B, C, L), dims=[-1]),
                torch.flip(torch.transpose(x, dim0=2, dim1=3).contiguous().view(B, C, L), dims=[-1]),
            ], dim=1).view(B, 4, C, L)
            return xs
        
        def cross_merge(out_y: torch.Tensor):
            B, K, D, H, W = out_y.shape
            L = H * W
            out_y = out_y.view(B, K, D, L)
            inv_y = torch.flip(out_y[:, 2:4], dims=[-1]).view(B, 2, -1, L)
            wh_y = torch.transpose(out_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
            invwh_y = torch.transpose(inv_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
            y = out_y[:, 0] + inv_y[:, 0] + wh_y + invwh_y
            return y

        def cross_scan_1b1(x: torch.Tensor):
            B, K, C, H, W = x.shape
            L = H * W
            xs = torch.stack([
                x[:, 0].view(B, C, L),
                torch.transpose(x[:, 1], dim0=2, dim1=3).contiguous().view(B, C, L),
                torch.flip(x[:, 2].contiguous().view(B, C, L), dims=[-1]),
                torch.flip(torch.transpose(x[:, 3], dim0=2, dim1=3).contiguous().view(B, C, L), dims=[-1]),
            ], dim=1).view(B, 4, C, L)
            return xs
        
        def unidi_scan(x):
            B, C, H, W = x.shape
            x = x.view(B, 1, C, H * W).repeat(1, 4, 1, 1)
            return x
        
        def unidi_merge(ys):
            B, K, C, H, W = ys.shape
            return ys.view(B, 4, -1, H * W).sum(1)

        def bidi_scan(x):
            B, C, H, W = x.shape
            x = x.view(B, 1, C, H * W).repeat(1, 2, 1, 1)
            x = torch.cat([x, x.flip(dims=[-1])], dim=1)
            return x
        
        def bidi_merge(ys):
            B, K, D, H, W = ys.shape
            ys = ys.view(B, K, D, -1)
            ys = ys[:, 0:2] + ys[:, 2:4].flip(dims=[-1]).view(B, 2, D, -1)
            return ys.contiguous().sum(1)

        if True:
            res0 = triton.testing.do_bench(lambda :cross_scan(x))
            res1 = triton.testing.do_bench(lambda :cross_scan_fn(x, True, True, False))
            # res2 = triton.testing.do_bench(lambda :CrossScanTriton.apply(x))
            res3 = triton.testing.do_bench(lambda :cross_merge(y))
            res4 = triton.testing.do_bench(lambda :cross_merge_fn(y, True, True, False))
            # res5 = triton.testing.do_bench(lambda :CrossMergeTriton.apply(y))
            # print(res0, res1, res2, res3, res4, res5)
            print(res0, res1, res3, res4)
            res0 = triton.testing.do_bench(lambda :cross_scan(x).sum().backward())
            res1 = triton.testing.do_bench(lambda :cross_scan_fn(x, True, True, False).sum().backward())
            # res2 = triton.testing.do_bench(lambda :CrossScanTriton.apply(x).sum().backward())
            res3 = triton.testing.do_bench(lambda :cross_merge(y).sum().backward())
            res4 = triton.testing.do_bench(lambda :cross_merge_fn(y, True, True, False).sum().backward())
            # res5 = triton.testing.do_bench(lambda :CrossMergeTriton.apply(y).sum().backward())
            # print(res0, res1, res2, res3, res4, res5)
            print(res0, res1, res3, res4)

        print("test cross scan")
        for (cs0, cm0, cs1, cm1) in [
            # channel_first -> channel_first
            (cross_scan, cross_merge, cross_scan_fn, cross_merge_fn),
            (unidi_scan, unidi_merge, lambda x: cross_scan_fn(x, scans=1), lambda x: cross_merge_fn(x, scans=1)),
            (bidi_scan, bidi_merge, lambda x: cross_scan_fn(x, scans=2), lambda x: cross_merge_fn(x, scans=2)),
            
            # flex: BLC->BCL; BCL->BLC; BLC->BLC;
            (cross_scan, cross_merge, lambda x: cross_scan_fn(x.permute(0, 2, 3, 1), in_channel_first=False), lambda x: cross_merge_fn(x, in_channel_first=False).permute(0, 2, 1)),
            (cross_scan, cross_merge, lambda x: cross_scan_fn(x, out_channel_first=False).permute(0, 2, 3, 1), lambda x: cross_merge_fn(x.permute(0, 3, 4, 1, 2), out_channel_first=False)),
            (cross_scan, cross_merge, lambda x: cross_scan_fn(x.permute(0, 2, 3, 1), in_channel_first=False, out_channel_first=False).permute(0, 2, 3, 1), lambda x: cross_merge_fn(x.permute(0, 3, 4, 1, 2), in_channel_first=False, out_channel_first=False).permute(0, 2, 1)),
            
            # previous
            # (cross_scan, cross_merge, lambda x: CrossScanTriton.apply(x), lambda x: CrossMergeTriton.apply(x)),
            # (unidi_scan, unidi_merge, lambda x: getCSM(1)[0].apply(x), lambda x: getCSM(1)[1].apply(x)),
            # (bidi_scan, bidi_merge, lambda x: getCSM(2)[0].apply(x), lambda x: getCSM(2)[1].apply(x)),
        ]:
            x.grad, x1.grad, y.grad, y1.grad = None, None, None, None
            o0 = cs0(x)
            o1 = cs1(x1)
            o0.backward(y.view(B, 4, C, H * W))
            o1.backward(y.view(B, 4, C, H * W))
            print((o0 - o1).abs().max())
            print((x.grad - x1.grad).abs().max())
            o0 = cm0(y)
            o1 = cm1(y1)
            o0.backward(x.view(B, C, H * W))
            o1.backward(x.view(B, C, H * W))
            print((o0 - o1).abs().max())
            print((y.grad - y1.grad).abs().max())
            x.grad, x1.grad, y.grad, y1.grad = None, None, None, None
            print("===============", flush=True)

        print("test cross scan one by one")
        for (cs0, cs1) in [
            (cross_scan_1b1, lambda x: cross_scan_fn(x, one_by_one=True)),
            # (cross_scan_1b1, lambda x: CrossScanTriton1b1.apply(x)),
        ]:
            o0 = cs0(y)
            o1 = cs1(y1)
            o0.backward(y.view(B, 4, C, H * W))
            o1.backward(y.view(B, 4, C, H * W))
            print((o0 - o1).abs().max())
            print((y.grad - y1.grad).abs().max())
            x.grad, x1.grad, y.grad, y1.grad = None, None, None, None
            print("===============", flush=True)

    def check_csm_scan3():
        if False:
            x = torch.arange(0, 16).view(1, 1, 4, 4).cuda()
            out1 = cross_scan_fn(x, scans=3, force_torch=True).view(1, 4, 1, 4, 4)
            out2 = cross_merge_fn(out1, scans=3, force_torch=True).view(1, 1, 4, 4)
            out4 = cross_merge_fn(out1, one_by_one=True, scans=3, force_torch=True).view(1, 4, 1, 4, 4)
            out3 = cross_scan_fn(out4, one_by_one=True, scans=3, force_torch=True).view(1, 4, 1, 4, 4)
            out5 = cross_scan_fn(x.view(1, 4, 4, 1), in_channel_first=False, out_channel_first=False, scans=3, force_torch=True).view(1, 4, 4, 4, 1)
            out6 = cross_merge_fn(out5, in_channel_first=False, out_channel_first=False, scans=3, force_torch=True).view(1, 4, 4, 1)
            out8 = cross_merge_fn(out5, in_channel_first=False, out_channel_first=False, one_by_one=True, scans=3, force_torch=True).view(1, 4, 4, 4, 1)
            out7 = cross_scan_fn(out8, in_channel_first=False, out_channel_first=False, one_by_one=True, scans=3, force_torch=True).view(1, 4, 4, 4, 1)
            print(out1.view(4, -1))
            print(out2.view(-1))
            print(out3.view(4, -1))
            print(out4.view(4, -1))
            print(out5.view(-1, 4).t())
            print(out6.view(-1))
            print(out7.view(-1, 4).t())
            print(out8.view(-1, 4).t())

        B, C, H, W = 27, 253, 57, 58
        x = torch.randn((B, C, H, W)).cuda()

        for scans in [0, 1, 2, 3]:
            o1 = cross_scan_fn(x, scans=scans, force_torch=True).view(B, 4, C, H, W)
            print((cross_scan_fn(x, scans=scans) == cross_scan_fn(x, scans=scans, force_torch=True)).all())
            print((cross_merge_fn(o1, scans=scans) == cross_merge_fn(o1, scans=scans, force_torch=True)).all())

            kwargs = dict(in_channel_first=False, out_channel_first=False)
            x2 = x.permute(0, 2, 3, 1).contiguous()
            o2 = o1.permute(0, 3, 4, 1, 2).contiguous()
            print((cross_scan_fn(x, scans=scans, **kwargs) == cross_scan_fn(x, scans=scans, force_torch=True, **kwargs)).all())
            print((cross_merge_fn(o2, scans=scans, **kwargs) == cross_merge_fn(o2, scans=scans, force_torch=True, **kwargs)).all())            

        breakpoint()


if __name__ == "__main__":
    CHECK.check_csm_scan3()
    CHECK.check_csm_triton()




