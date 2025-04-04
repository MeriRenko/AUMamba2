import torch
from timm.utils import accuracy, AverageMeter
import time
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
import logging
#from data_list import heatmap2au
# 理论上总共50176(128*392)张图片作为测试集，实际上有50166张，最后一个batch不全。
@torch.no_grad()
def AU_detection_evalv2(config, data_loader, model):
    model.eval()  #将模型设置为评估模式（禁用 dropout 和 batch norm）
    logger = logging.getLogger(f"{config.MODEL.NAME}") #记录模型名称

    batch_time = AverageMeter()     # 用于统计批处理的计算时间
    end = time.time()

    missing_label = 9               # 丢失的标签值（数据集中可能存在未标注的 AU）
    for idx, (images, land, biocular, target) in enumerate(data_loader):
        images = images.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)

        # compute output
        with torch.cuda.amp.autocast(enabled=config.AMP_ENABLE):
            output = model(images)                  # 前向传播，获取 AU 预测结果

        output = torch.sigmoid(output)              # 将输出转换到 [0,1] 之间，适用于 二分类任务（是否激活某个 AU）
        # heatmap_pred = heatmap2au(heatmap_pred)

        # 第一批次
        if idx == 0:
            all_output = output.data.cpu().float()
            # all_heatmap_pred = heatmap_pred.data.cpu()
            all_au = target.data.cpu().float()
        # 后续批次：使用 torch.cat 逐步拼接预测结果和真实值
        else:
            all_output = torch.cat((all_output, output.data.cpu().float()), 0)
            # all_heatmap_pred = torch.cat((all_heatmap_pred, heatmap_pred.data.cpu()), 0)
            all_au = torch.cat((all_au, target.data.cpu().float()), 0)


        batch_time.update(time.time() - end)  # 计算当前 batch 的推理时间
        end = time.time()
        # config.PRINT_FREQ: Frequency to logging info
        if idx % config.PRINT_FREQ == 0:
            memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
            logger.info(
                f'Test: [{idx}/{len(data_loader)}]\t'
                # batch_time.val： 当前 batch 计算时间。
                # batch_time.avg： 平均计算时间。
                f'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                # 记录 GPU 显存占用情况（单位 MB）。
                f'Mem {memory_used:.0f}MB')
    # 将所有 AU 预测结果转换为 numpy 数组
    AUoccur_pred_prob = all_output.data.numpy()#.data：获取模型输出的原始数据，不包含梯度信息，适用于推理阶段（不需要反向传播）
    AUoccur_actual = all_au.data.numpy()
    # AUoccur_all_heatmap_pred = all_heatmap_pred.data.numpy()

    # AUs
    # 应用 0.5 阈值：预测概率 < 0.5 视为 未激活（0）;预测概率 >= 0.5 视为 激活（1）。
    AUoccur_pred = np.zeros(AUoccur_pred_prob.shape)
    AUoccur_pred[AUoccur_pred_prob < 0.5] = 0
    AUoccur_pred[AUoccur_pred_prob >= 0.5] = 1
    # AUoccur_pred = np.logical_or(AUoccur_pred, AUoccur_all_heatmap_pred).astype(int)

    # 转置后12*50166
    AUoccur_actual = AUoccur_actual.transpose((1, 0))
    AUoccur_pred = AUoccur_pred.transpose((1, 0))

    # 计算 F1-score 和 Accuracy
    f1score_arr = np.zeros(AUoccur_actual.shape[0])
    acc_arr = np.zeros(AUoccur_actual.shape[0])
    for i in range(AUoccur_actual.shape[0]):
        # """
        # BP4D: 0, 1, 2, 3, 6
        # DISFA: 0, 1, 2, 3, 5
        # """
        # 50166*1
        curr_actual = AUoccur_actual[i]
        curr_pred = AUoccur_pred[i]
        # 50166*1
        # 明明值在0到1之间，为什么要来missing_label条件
        new_curr_actual = curr_actual[curr_actual != missing_label]
        new_curr_pred = curr_pred[curr_actual != missing_label]

        # 对50166张图片求每个AU单元的F1与Accuracy
        f1score_arr[i] = f1_score(new_curr_actual, new_curr_pred)
        acc_arr[i] = accuracy_score(new_curr_actual, new_curr_pred)
    
    # 记录最终评估结果
    logger.info("Final Evaluation Metrics:")
    #  将 f1score_arr 数组中的每个数值格式化为小数点后三位的字符串，并用空格 " " 连接成一个字符串。
    #  f"{f1_meter:.3f}"：将 f1_meter 格式化为 小数点后 3 位 的字符串。
    #  join() 作用是 将列表中的所有字符串用空格拼接，形成一个新的字符串。
    logger.info(f'Final F1 Scores for each AU: {" ".join([f"{f1_meter:.3f}" for f1_meter in f1score_arr])}')
    logger.info(f'Final F1 Scores for all AU: {f1score_arr.mean()}')
    logger.info(f'Final Accuracy for all AU: {acc_arr.mean()}')
    
    # 返回评估结果
    f1score_arr = f1score_arr.mean() * 100
    acc_arr = acc_arr.mean() * 100

    return f1score_arr, acc_arr, None