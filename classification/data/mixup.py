""" Mixup and Cutmix

Papers:
mixup: Beyond Empirical Risk Minimization (https://arxiv.org/abs/1710.09412)

CutMix: Regularization Strategy to Train Strong Classifiers with Localizable Features (https://arxiv.org/abs/1905.04899)

Code Reference:
CutMix: https://github.com/clovaai/CutMix-PyTorch

Hacked together by / Copyright 2020 Ross Wightman
"""
import numpy as np
import torch

# 生成混合后的标签
def mixup_target(target, num_classes, lam=1., smoothing=0.0, device='cuda'):
    #target: 原始目标标签
    #lam: 混合比例因子
    y1 = target
    y2 = target.flip(0) # 将批次内的标签顺序反转
    return y1 * lam + y2 * (1. - lam)

# 标准CutMix矩形区域生成
def rand_bbox(img_shape, lam, margin=0., count=None):
    """ Standard CutMix bounding-box
    Generates a random square bbox based on lambda value. This impl includes
    support for enforcing a border margin as percent of bbox dimensions.

    Args:
        img_shape (tuple): Image shape as tuple
        lam (float): Cutmix lambda value
        margin (float): Percentage of bbox dimension to enforce as margin (reduce amount of box outside image)
        count (int): Number of bbox to generate
    """
    # np.sqrt(1-lam) 控制矩形区域面积比例
    ratio = np.sqrt(1 - lam)
    img_h, img_w = img_shape[-2:]
    cut_h, cut_w = int(img_h * ratio), int(img_w * ratio)
    # 提供 margin 参数避免边缘重叠
    margin_y, margin_x = int(margin * cut_h), int(margin * cut_w)
    cy = np.random.randint(0 + margin_y, img_h - margin_y, size=count)
    cx = np.random.randint(0 + margin_x, img_w - margin_x, size=count)
    yl = np.clip(cy - cut_h // 2, 0, img_h)
    yh = np.clip(cy + cut_h // 2, 0, img_h)
    xl = np.clip(cx - cut_w // 2, 0, img_w)
    xh = np.clip(cx + cut_w // 2, 0, img_w)
    # 生成边界坐标
    return yl, yh, xl, xh

# 动态范围CutMix生成
def rand_bbox_minmax(img_shape, minmax, count=None):
    """ Min-Max CutMix bounding-box
    Inspired by Darknet cutmix impl, generates a random rectangular bbox
    based on min/max percent values applied to each dimension of the input image.

    Typical defaults for minmax are usually in the  .2-.3 for min and .8-.9 range for max.

    Args:
        img_shape (tuple): Image shape as tuple
        minmax (tuple or list): Min and max bbox ratios (as percent of image size)
        count (int): Number of bbox to generate
    """
    #使用比例范围 minmax 代替固定 lam
    assert len(minmax) == 2
    img_h, img_w = img_shape[-2:]
    cut_h = np.random.randint(int(img_h * minmax[0]), int(img_h * minmax[1]), size=count)
    cut_w = np.random.randint(int(img_w * minmax[0]), int(img_w * minmax[1]), size=count)
    yl = np.random.randint(0, img_h - cut_h, size=count)
    xl = np.random.randint(0, img_w - cut_w, size=count)
    yu = yl + cut_h
    xu = xl + cut_w
    return yl, yu, xl, xu

# 校准实际lam值：目的是通过数学映射，使超参数 lam 严格等于第二张图片的实际贡献比例
def cutmix_bbox_and_lam(img_shape, lam, ratio_minmax=None, correct_lam=True, count=None):
    """ Generate bbox and apply lambda correction.
    """
    if ratio_minmax is not None:
        yl, yu, xl, xu = rand_bbox_minmax(img_shape, ratio_minmax, count=count)
    else:
        yl, yu, xl, xu = rand_bbox(img_shape, lam, count=count)
    if correct_lam or ratio_minmax is not None:
        bbox_area = (yu - yl) * (xu - xl)
        lam = 1. - bbox_area / float(img_shape[-2] * img_shape[-1])
    return (yl, yu, xl, xu), lam

# 集成 Mixup 和 CutMix 数据增强策略的混合类，持三种混合模式 (batch, pair, elem)
class Mixup:
    """ Mixup/Cutmix that applies different params to each element or whole batch

    Args:
        mixup_alpha (float): mixup alpha value, mixup is active if > 0.
        cutmix_alpha (float): cutmix alpha value, cutmix is active if > 0.
        cutmix_minmax (List[float]): cutmix min/max image ratio, cutmix is active and uses this vs alpha if not None.
        prob (float): probability of applying mixup or cutmix per batch or element
        switch_prob (float): probability of switching to cutmix instead of mixup when both are active
        mode (str): how to apply mixup/cutmix params (per 'batch', 'pair' (pair of elements), 'elem' (element)
        correct_lam (bool): apply lambda correction when cutmix bbox clipped by image borders
        label_smoothing (float): apply label smoothing to the mixed target tensor
        num_classes (int): number of classes for target
    """
    # 参数初始化
    def __init__(self, mixup_alpha=1., cutmix_alpha=0., cutmix_minmax=None, prob=1.0, switch_prob=0.5,
                 mode='batch', correct_lam=True, label_smoothing=0.1, num_classes=1000):
        self.mixup_alpha = mixup_alpha          # Mixup 的 Beta 分布参数 α，α>0 时激活 Mixup
        self.cutmix_alpha = cutmix_alpha        # CutMix 的 Beta 分布参数 α，α>0 时激活 CutMix
        self.cutmix_minmax = cutmix_minmax      # 指定 CutMix 的最小/最大面积比例（覆盖 cutmix_alpha）
        if self.cutmix_minmax is not None:
            assert len(self.cutmix_minmax) == 2
            # force cutmix alpha == 1.0 when minmax active to keep logic simple & safe
            self.cutmix_alpha = 1.0
        self.mix_prob = prob                    # 每次应用混合操作的整体概率
        self.switch_prob = switch_prob          # 当 Mixup 和 CutMix 均激活时，切换为 CutMix 的概率
        self.label_smoothing = label_smoothing
        self.num_classes = num_classes
        self.mode = mode                        # 混合模式：batch（整个批次）、pair（成对混合）、elem（独立混合）
        self.correct_lam = correct_lam          # CutMix 中是否校正 λ 值以保证实际混合比例
        self.mixup_enabled = True  # set to false to disable mixing (intended tp be set by train loop)
    # 生成 Mixup/Cutmix参数:混合比例参数 lam, 是否使用 CutMix (use_cutmix)
    # 针对批次中的每个元素 (_params_per_elem)
    def _params_per_elem(self, batch_size):
        # 初始化
        lam = np.ones(batch_size, dtype=np.float32)         # 默认 λ=1，表示不混合
        use_cutmix = np.zeros(batch_size, dtype=np.bool)    # 默认不使用 CutMix
        if self.mixup_enabled:
            # mixup/cutmix 均启用
            if self.mixup_alpha > 0. and self.cutmix_alpha > 0.:
                # 每个元素独立决定使用哪种混合模式（通过 switch_prob）
                use_cutmix = np.random.rand(batch_size) < self.switch_prob
                lam_mix = np.where(
                    use_cutmix,
                    # 生成 CutMix 的 Beta(α_cutmix, α_cutmix)
                    np.random.beta(self.cutmix_alpha, self.cutmix_alpha, size=batch_size),
                    # 生成 Mixup 的 Beta(α_mixup, α_mixup)
                    np.random.beta(self.mixup_alpha, self.mixup_alpha, size=batch_size))
            # 所有元素仅用 Mixup
            elif self.mixup_alpha > 0.:
                lam_mix = np.random.beta(self.mixup_alpha, self.mixup_alpha, size=batch_size)
            # 所有元素仅用 Cutmix
            elif self.cutmix_alpha > 0.:
                use_cutmix = np.ones(batch_size, dtype=np.bool)
                lam_mix = np.random.beta(self.cutmix_alpha, self.cutmix_alpha, size=batch_size)
            else:
                assert False, "One of mixup_alpha > 0., cutmix_alpha > 0., cutmix_minmax not None should be true."
            lam = np.where(np.random.rand(batch_size) < self.mix_prob, lam_mix.astype(np.float32), lam)
        # lam = [0.45, 0.12, 0.89, ...]  # 每个元素一个 λ
        # use_cutmix = [True, False, False, ...]  # 部分元素使用 CutMix
        return lam, use_cutmix
    # 针对整个批次 (_params_per_batch)
    def _params_per_batch(self):
        lam = 1.
        use_cutmix = False
        if self.mixup_enabled and np.random.rand() < self.mix_prob:
            # mixup/cutmix 均启用
            if self.mixup_alpha > 0. and self.cutmix_alpha > 0.:
                use_cutmix = np.random.rand() < self.switch_prob
                lam_mix = np.random.beta(self.cutmix_alpha, self.cutmix_alpha) if use_cutmix else \
                    np.random.beta(self.mixup_alpha, self.mixup_alpha)
            # 所有批次仅用 Mixup
            elif self.mixup_alpha > 0.:
                lam_mix = np.random.beta(self.mixup_alpha, self.mixup_alpha)
            # 所有批次仅用 Cutmix
            elif self.cutmix_alpha > 0.:
                use_cutmix = True
                lam_mix = np.random.beta(self.cutmix_alpha, self.cutmix_alpha)
            else:
                assert False, "One of mixup_alpha > 0., cutmix_alpha > 0., cutmix_minmax not None should be true."
            lam = float(lam_mix)
        return lam, use_cutmix

    # Mixup 和 CutMix操作实现
    # elem模式：每个元素都不同
    def _mix_elem(self, x):
        # 参数准备
        batch_size = len(x)
        lam_batch, use_cutmix = self._params_per_elem(batch_size)
        x_orig = x.clone()  # 缓存原数据,保证Mixup 和 CutMix操作时数据来源不变
        # 遍历混合
        for i in range(batch_size):
            j = batch_size - i - 1 #反向配对
            lam = lam_batch[i]
            if lam != 1.:   # λ=1 时跳过混合
                # CutMix 操作流程
                if use_cutmix[i]:
                    # 区域生成：调用 cutmix_bbox_and_lam 计算裁剪区域坐标 (yl, yh, xl, xh) 并可能修正 λ
                    (yl, yh, xl, xh), lam = cutmix_bbox_and_lam(
                        x[i].shape, lam, ratio_minmax=self.cutmix_minmax, correct_lam=self.correct_lam)
                    # 像素替换：将当前元素 i 的指定区域替换为配对元素 j 的对应区域
                    x[i][:, yl:yh, xl:xh] = x_orig[j][:, yl:yh, xl:xh]
                    lam_batch[i] = lam
                # Mixup 操作流程
                else:
                    x[i] = x[i] * lam + x_orig[j] * (1 - lam)
        # 返回 λ 张量：将 λ 转换为与 x 相同设备/类型的张量，并增加维度适配后续标签混合
        return torch.tensor(lam_batch, device=x.device, dtype=x.dtype).unsqueeze(1)
    # pair模式：每个元素都对称相同（只独特化前半部分，后半部分对称）
    def _mix_pair(self, x):
        batch_size = len(x)
        lam_batch, use_cutmix = self._params_per_elem(batch_size // 2)
        x_orig = x.clone()  # need to keep an unmodified original for mixing source
        for i in range(batch_size // 2):
            j = batch_size - i - 1
            lam = lam_batch[i]
            if lam != 1.:
                if use_cutmix[i]:
                    (yl, yh, xl, xh), lam = cutmix_bbox_and_lam(
                        x[i].shape, lam, ratio_minmax=self.cutmix_minmax, correct_lam=self.correct_lam)
                    x[i][:, yl:yh, xl:xh] = x_orig[j][:, yl:yh, xl:xh]
                    x[j][:, yl:yh, xl:xh] = x_orig[i][:, yl:yh, xl:xh]
                    lam_batch[i] = lam
                else:
                    x[i] = x[i] * lam + x_orig[j] * (1 - lam)
                    x[j] = x[j] * lam + x_orig[i] * (1 - lam)
        lam_batch = np.concatenate((lam_batch, lam_batch[::-1]))
        return torch.tensor(lam_batch, device=x.device, dtype=x.dtype).unsqueeze(1)

    def _mix_batch(self, x):
        lam, use_cutmix = self._params_per_batch()
        if lam == 1.:
            return 1.
        if use_cutmix:
            (yl, yh, xl, xh), lam = cutmix_bbox_and_lam(
                x.shape, lam, ratio_minmax=self.cutmix_minmax, correct_lam=self.correct_lam)
            x[:, :, yl:yh, xl:xh] = x.flip(0)[:, :, yl:yh, xl:xh]
        else:
            # .add_是in-place操作，比如x.add_(y)，x+y的结果会存储到原来的x中
            x_flipped = x.flip(0).mul_(1. - lam)
            x.mul_(lam).add_(x_flipped)
        return lam
    
    def update_params(self, mixup_alpha=None, cutmix_alpha=None):
        """ Update mixup and cutmix alpha values dynamically. """
        if mixup_alpha is not None:
            self.mixup_alpha = mixup_alpha
        if cutmix_alpha is not None:
            self.cutmix_alpha = cutmix_alpha

    def __call__(self, x, target):
        # 批大小为偶数
        assert len(x) % 2 == 0, 'Batch size should be even when using this'
        if self.mode == 'elem':
            lam = self._mix_elem(x)
        elif self.mode == 'pair':
            lam = self._mix_pair(x)
        else:
            lam = self._mix_batch(x)
        target = mixup_target(target, self.num_classes, lam, self.label_smoothing)
        return x, target

# 继承自Mixup,对方法进行改写，使其更专注于collate阶段的优化，适合需要高效数据加载和预处理的大规模训练场景
class FastCollateMixup(Mixup):
    """ Fast Collate w/ Mixup/Cutmix that applies different params to each element or whole batch

    A Mixup impl that's performed while collating the batches.
    """
    # 作用时点：预处理阶段
    # 数据格式：处理原始数据（numpy 数组），最后转为 PyTorch 张量
    # 内存管理策略： 预分配输出张量
    # 混合类型：使用 numpy 计算 + uint8 优化（内存效率更高）
    # 目标应用场景：大规模数据集或需要加速混合操作的高吞吐场景
    def _mix_elem_collate(self, output, batch, half=False):
        batch_size = len(batch)
        num_elem = batch_size // 2 if half else batch_size
        assert len(output) == num_elem
        lam_batch, use_cutmix = self._params_per_elem(num_elem)
        for i in range(num_elem):
            j = batch_size - i - 1
            lam = lam_batch[i]
            mixed = batch[i][0] 
            if lam != 1.:
                if use_cutmix[i]:
                    if not half:
                        mixed = mixed.copy() # 非半批模式需复制数据避免修改原批
                    (yl, yh, xl, xh), lam = cutmix_bbox_and_lam(
                        output.shape, lam, ratio_minmax=self.cutmix_minmax, correct_lam=self.correct_lam)
                    mixed[:, yl:yh, xl:xh] = batch[j][0][:, yl:yh, xl:xh]   # # 零拷贝操作：使用 batch[i][0] 直接操作内存而非深拷贝
                    lam_batch[i] = lam
                else:
                    mixed = mixed.astype(np.float32) * lam + batch[j][0].astype(np.float32) * (1 - lam)  # 转为 float 计算
                    np.rint(mixed, out=mixed)                                                            # 精度截断避免溢出
            output[i] += torch.from_numpy(mixed.astype(np.uint8))                                        # uint8 节省内存                    
        if half:
            lam_batch = np.concatenate((lam_batch, np.ones(num_elem)))
        return torch.tensor(lam_batch).unsqueeze(1)

    def _mix_pair_collate(self, output, batch):
        batch_size = len(batch)
        lam_batch, use_cutmix = self._params_per_elem(batch_size // 2)
        for i in range(batch_size // 2):
            j = batch_size - i - 1
            lam = lam_batch[i]
            mixed_i = batch[i][0]
            mixed_j = batch[j][0]
            assert 0 <= lam <= 1.0
            if lam < 1.:
                if use_cutmix[i]:
                    (yl, yh, xl, xh), lam = cutmix_bbox_and_lam(
                        output.shape, lam, ratio_minmax=self.cutmix_minmax, correct_lam=self.correct_lam)
                    patch_i = mixed_i[:, yl:yh, xl:xh].copy()
                    mixed_i[:, yl:yh, xl:xh] = mixed_j[:, yl:yh, xl:xh]
                    mixed_j[:, yl:yh, xl:xh] = patch_i
                    lam_batch[i] = lam
                else:
                    mixed_temp = mixed_i.astype(np.float32) * lam + mixed_j.astype(np.float32) * (1 - lam)
                    mixed_j = mixed_j.astype(np.float32) * lam + mixed_i.astype(np.float32) * (1 - lam)
                    mixed_i = mixed_temp
                    np.rint(mixed_j, out=mixed_j)
                    np.rint(mixed_i, out=mixed_i)
            output[i] += torch.from_numpy(mixed_i.astype(np.uint8))
            output[j] += torch.from_numpy(mixed_j.astype(np.uint8))
        lam_batch = np.concatenate((lam_batch, lam_batch[::-1]))
        return torch.tensor(lam_batch).unsqueeze(1)

    def _mix_batch_collate(self, output, batch):
        batch_size = len(batch)
        lam, use_cutmix = self._params_per_batch()
        if use_cutmix:
            (yl, yh, xl, xh), lam = cutmix_bbox_and_lam(
                output.shape, lam, ratio_minmax=self.cutmix_minmax, correct_lam=self.correct_lam)
        for i in range(batch_size):
            j = batch_size - i - 1
            mixed = batch[i][0]
            if lam != 1.:
                if use_cutmix:
                    mixed = mixed.copy()  # don't want to modify the original while iterating
                    mixed[:, yl:yh, xl:xh] = batch[j][0][:, yl:yh, xl:xh]
                else:
                    mixed = mixed.astype(np.float32) * lam + batch[j][0].astype(np.float32) * (1 - lam)
                    np.rint(mixed, out=mixed)
            output[i] += torch.from_numpy(mixed.astype(np.uint8))
        return lam

    # 初始化和调用入口 (__call__)
    def __call__(self, batch, _=None):
        # 输入校验：检查 batch_size 是否为偶数
        batch_size = len(batch)
        assert batch_size % 2 == 0, 'Batch size should be even when using this'
        # 模式判定：根据 self.mode 确定混合策略
        half = 'half' in self.mode
        if half:
            batch_size //= 2
        # 预分配输出张量：创建全零的 uint8 类型输出张量
        output = torch.zeros((batch_size, *batch[0][0].shape), dtype=torch.uint8)
        # 混合模式路由：根据模式调用具体混合方法
        if self.mode == 'elem' or self.mode == 'half':
            lam = self._mix_elem_collate(output, batch, half=half)
        elif self.mode == 'pair':
            lam = self._mix_pair_collate(output, batch)
        else:
            lam = self._mix_batch_collate(output, batch)
        # 标签处理：用 mixup_target 生成混合标签
        target = torch.tensor([b[1] for b in batch], dtype=torch.int64)
        target = mixup_target(target, self.num_classes, lam, self.label_smoothing, device='cpu')
        target = target[:batch_size]
        return output, target
