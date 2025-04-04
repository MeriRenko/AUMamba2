import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from timm.loss import SoftTargetCrossEntropy

# gama_i：负类的权重因子（类似于 Focal Loss 的 gamma）。
# m：对预测值的截断（Margin），控制学习难度。
# 返回损失值，对负类 (0类) 进行了加权
def wa_loss(pred, target, gama_i, m):
    # 展平预测值
    iflat = pred.contiguous().view(-1)
    # 展平真实标签
    tflat = target.contiguous().view(-1)
    xs_pos = iflat
    p_m = torch.clamp((iflat-m), min=0)
    xs_neg = 1 - p_m

    # Basic CE calculation
    los_pos = tflat * torch.log(xs_pos.clamp(min=1e-8))
    los_neg = (1 - tflat) * torch.log(xs_neg.clamp(min=1e-8))

    neg_weight = 1 - xs_neg

    loss = -(los_pos + (neg_weight**gama_i) * los_neg)

    return loss.mean()

def dice_loss(pred, target, smooth = 1):
    """This definition generalize to real valued pred and target vector.
    This should be differentiable.
    pred: tensor with first dimension as batch
    target: tensor with first dimension as batch
    """

    #  由于 view() 可能来自其他操作，所以需要 contiguous();
    # .contiguous()：保证内存布局是连续的，以防 view() 操作出错。
    iflat = pred.contiguous().view(-1)
    tflat = target.contiguous().view(-1)
    # 计算预测值 和 真实值 的交集（按元素相乘）。
    # intersection 计算的是正确预测为 1 的样本数量。
    intersection = (iflat * tflat).sum()

    A_sum = torch.sum(iflat * iflat)
    B_sum = torch.sum(tflat * tflat)

    return 1 - ((2. * intersection + smooth) / (A_sum + B_sum + smooth)) / iflat.size(0)

# 定义 AU Custom Loss 类，继承自 SoftTargetCrossEntropy
class AUCustomLoss(SoftTargetCrossEntropy):
    def __init__(self, config):
        super(AUCustomLoss, self).__init__()
        self.loss_type = config.TRAIN.LOSS_TYPE
        self.weight = torch.from_numpy(np.loadtxt(config.DATA.TRAIN_PATH_PREFIX + '_weight.txt'))
        self.size_average = True
        self.m = config.TRAIN.MARGIN_WA_LOSS    # 0
        # gama如何产生的？
        self.gama = torch.from_numpy(np.loadtxt(config.DATA.TRAIN_PATH_PREFIX + '_gama' + config.TRAIN.GAMA_WA_LOSS + '.txt'))
        self.smooth = 1.0
    # 支持三种损失模式
    def forward(self, inputs, targets):
        if self.loss_type == "wa":
            return self.au_wa_loss(inputs, targets)
        elif self.loss_type == "dice":
            return self.au_dice_loss(inputs, targets)
        elif self.loss_type == "wa+dice":
            self.loss_lambda = 10
            return self.loss_lambda * self.au_wa_loss(inputs, targets) + self.au_dice_loss(inputs, targets)
        else:
            raise ValueError("Invalid loss_type. Choose between 'wa' and 'dice'.")
    # 遍历每个 AU 任务，计算WA Loss。
    # self.weight[i] 让不同 AU 任务的权重不同。
    def au_wa_loss(self, inputs, targets):
        # inputs 128*12
        inputs = torch.sigmoid(inputs)     # 将 logits 转换为概率
        for i in range(inputs.size(1)):    # 遍历每个 AU 类别
            # input is log_softmax, t_input is probability
            t_input = inputs[:, i]
            t_target = targets[:, i].float()
            t_loss = wa_loss(t_input, t_target, self.gama[i], self.m)
            if self.weight is not None:
                t_loss = t_loss * self.weight[i]    # 乘以类别权重
            t_loss = torch.unsqueeze(t_loss, 0)
            if i == 0:
                loss = t_loss
            else:
                loss = torch.cat((loss, t_loss), 0)
        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()
    # 遍历每个 AU 任务，计算 Dice Loss。
    # inputs[:, i].exp() 可能是避免数值精度问题（虽然一般不需要）。
    def au_dice_loss(self, inputs, targets):
        inputs = torch.sigmoid(inputs)
        for i in range(inputs.size(1)):
            # input is log_softmax, t_input is probability
            # 输入为什么是log_softmax？
            t_input = inputs[:, i].exp()
            t_target = targets[:, i].float()
            t_loss = dice_loss(t_input, t_target, self.smooth)
            if self.weight is not None:
                t_loss = t_loss * self.weight[i]
            t_loss = torch.unsqueeze(t_loss, 0)
            if i == 0:
                loss = t_loss
            else:
                loss = torch.cat((loss, t_loss), 0)

        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()