import torch
import torch.nn as nn
import numpy as np
import sklearn
from sklearn.metrics import accuracy_score, f1_score
import matplotlib.pyplot as plt
from PIL import Image
import torch.nn.functional as F
import math
import pdb

def str2bool(v):
    return v.lower() in ('true')

def tensor2img(img):
    img = img.data.numpy()
    if img.shape[0] == 1:
        img = np.tile(img, (3, 1, 1))
    img = (np.transpose(img, (1, 2, 0))+ 1) / 2.0 * 255.0
    return img.astype(np.uint8)

def save_img(img, name, path):
    img = tensor2img(img)
    img = Image.fromarray(img)
    img.save(path + name + '.png')
    return img


def AU_detection_evalv2(loader, vit_model, use_gpu=True, fail_threshold = 0.1):
    missing_label = 9
    # total_loss_test = []
    for i, batch in enumerate(loader):
        input, au  = batch

        if use_gpu:
            input, au = input.cuda(), au.cuda()

        vit_feat, au_mlp = vit_model(input)
        aus_output = torch.sigmoid(vit_feat)

        adapter_output = torch.cat(au_mlp, dim=1)
        adapter_output = torch.sigmoid(adapter_output)
        

        if i == 0:
            all_output = aus_output.data.cpu().float()
            all_adapter_output = adapter_output.data.cpu().float()
            all_au = au.data.cpu().float()
        else:
            all_output = torch.cat((all_output, aus_output.data.cpu().float()), 0)
            all_adapter_output = torch.cat((all_adapter_output, adapter_output.data.cpu().float()), 0)
            all_au = torch.cat((all_au, au.data.cpu().float()), 0)

    AUoccur_pred_prob = all_output.data.numpy()
    AUoccur_adapter_pred_prob = all_adapter_output.data.numpy()
    AUoccur_actual = all_au.data.numpy()

    # AUs
    AUoccur_pred = np.zeros(AUoccur_pred_prob.shape)
    AUoccur_pred[AUoccur_pred_prob < 0.5] = 0
    AUoccur_pred[AUoccur_pred_prob >= 0.5] = 1

    AUoccur_adapter_pred = np.zeros(AUoccur_adapter_pred_prob.shape)
    AUoccur_adapter_pred[AUoccur_adapter_pred_prob < 0.5] = 0
    AUoccur_adapter_pred[AUoccur_adapter_pred_prob >= 0.5] = 1

    AUoccur_actual = AUoccur_actual.transpose((1, 0))
    AUoccur_pred = AUoccur_pred.transpose((1, 0))
    AUoccur_adapter_pred = AUoccur_adapter_pred.transpose((1, 0))

    f1score_arr = np.zeros(AUoccur_actual.shape[0])
    acc_arr = np.zeros(AUoccur_actual.shape[0])
    f1score_adapter_arr = np.zeros(AUoccur_actual.shape[0])
    acc_adapter_arr = np.zeros(AUoccur_actual.shape[0])
    for i in range(AUoccur_actual.shape[0]):
        curr_actual = AUoccur_actual[i]
        curr_pred = AUoccur_pred[i]
        curr_adapter_pred = AUoccur_adapter_pred[i]

        new_curr_actual = curr_actual[curr_actual != missing_label]
        new_curr_pred = curr_pred[curr_actual != missing_label]
        new_curr_adapter_pred = curr_adapter_pred[curr_actual != missing_label]

        f1score_arr[i] = f1_score(new_curr_actual, new_curr_pred)
        acc_arr[i] = accuracy_score(new_curr_actual, new_curr_pred)
        f1score_adapter_arr[i] = f1_score(new_curr_actual, new_curr_adapter_pred)
        acc_adapter_arr[i] = accuracy_score(new_curr_actual, new_curr_adapter_pred)
        

    return f1score_arr, acc_arr, f1score_adapter_arr, acc_adapter_arr

def AU_detection_eval_train(aus_output, au, use_gpu=True, fail_threshold = 0.1):
    missing_label = 9

    all_output = aus_output.data.cpu().float()
    all_au = au.data.cpu().float()
    
    AUoccur_pred_prob = all_output.data.numpy()
    AUoccur_actual = all_au.data.numpy()
    # np.savetxt('BP4D_part1_pred_land_49.txt', pred_land, fmt='%.4f', delimiter='\t')
    # np.savetxt('B3D_val_predAUprob-2_all_.txt', AUoccur_pred_prob, fmt='%f',
    #            delimiter='\t')

    # AUs
    AUoccur_pred = np.zeros(AUoccur_pred_prob.shape)
    AUoccur_pred[AUoccur_pred_prob < 0.5] = 0
    AUoccur_pred[AUoccur_pred_prob >= 0.5] = 1

    AUoccur_actual = AUoccur_actual.transpose((1, 0))
    AUoccur_pred = AUoccur_pred.transpose((1, 0))

    f1score_arr = np.zeros(AUoccur_actual.shape[0])
    acc_arr = np.zeros(AUoccur_actual.shape[0])
    for i in range(AUoccur_actual.shape[0]):
        curr_actual = AUoccur_actual[i]
        curr_pred = AUoccur_pred[i]

        new_curr_actual = curr_actual[curr_actual != missing_label]
        new_curr_pred = curr_pred[curr_actual != missing_label]

        f1score_arr[i] = f1_score(new_curr_actual, new_curr_pred)
        acc_arr[i] = accuracy_score(new_curr_actual, new_curr_pred)

    return f1score_arr, acc_arr

# def AU_detection_train_evalv2(batch, rest_learning, au_classify, use_gpu=True, fail_threshold = 0.1):
#     missing_label = 9
    
#     input, land, biocular, au  = batch
#     input = F.interpolate(input, size=(224,224), mode='bilinear', align_corners=True)

#     if use_gpu:
#         input, land, au = input.cuda(), land.cuda(), au.cuda()

#     rest_feat, rest_feat_concat = rest_learning(input)
#     au_classify_output = au_classify(rest_feat_concat)
#     aus_output = au_classify_output.view(au_classify_output.size(0), 2, int(au_classify_output.size(1)/2))
#     aus_output = F.log_softmax(aus_output, dim=1)
#     aus_output = (aus_output[:,1,:]).exp()

#     all_output = aus_output.data.cpu().float()
#     all_au = au.data.cpu().float()
    
#     AUoccur_pred_prob = all_output.data.numpy()
#     AUoccur_actual = all_au.data.numpy()
#     # np.savetxt('BP4D_part1_pred_land_49.txt', pred_land, fmt='%.4f', delimiter='\t')
    
    
#     # np.savetxt('B3D_val_predAUprob-2_all_.txt', AUoccur_pred_prob, fmt='%f',
#     #            delimiter='\t')

    
#     # AUs
#     AUoccur_pred = np.zeros(AUoccur_pred_prob.shape)
#     AUoccur_pred[AUoccur_pred_prob < 0.5] = 0
#     AUoccur_pred[AUoccur_pred_prob >= 0.5] = 1

#     AUoccur_actual = AUoccur_actual.transpose((1, 0))
#     AUoccur_pred = AUoccur_pred.transpose((1, 0))

#     f1score_arr = np.zeros(AUoccur_actual.shape[0])
#     acc_arr = np.zeros(AUoccur_actual.shape[0])
#     for i in range(AUoccur_actual.shape[0]):
#         curr_actual = AUoccur_actual[i]
#         curr_pred = AUoccur_pred[i]

#         new_curr_actual = curr_actual[curr_actual != missing_label]
#         new_curr_pred = curr_pred[curr_actual != missing_label]

#         f1score_arr[i] = f1_score(new_curr_actual, new_curr_pred)
#         acc_arr[i] = accuracy_score(new_curr_actual, new_curr_pred)

#     return f1score_arr, acc_arr

def vis_attention(loader, region_learning, align_net, local_attention_refine, write_path_prefix, net_name, epoch, alpha = 0.5, use_gpu=True):
    for i, batch in enumerate(loader):
        input, land, biocular, au = batch
        # if i > 1:
        #     break
        if use_gpu:
            input = input.cuda()
        region_feat = region_learning(input)
        align_feat, align_output, aus_map = align_net(region_feat)
        if use_gpu:
            aus_map = aus_map.cuda()
        output_aus_map = local_attention_refine(aus_map.detach())

        # aus_map is predefined, and output_aus_map is refined
        spatial_attention = output_aus_map #aus_map
        if i == 0:
            all_input = input.data.cpu().float()
            all_spatial_attention = spatial_attention.data.cpu().float()
        else:
            all_input = torch.cat((all_input, input.data.cpu().float()), 0)
            all_spatial_attention = torch.cat((all_spatial_attention, spatial_attention.data.cpu().float()), 0)
    pdb.set_trace()
    for i in range(all_spatial_attention.shape[0]):
        background = save_img(all_input[i], 'input', write_path_prefix + net_name + '/vis_map/' + str(epoch) +
                        '/' + str(i) + '_')
        for j in range(all_spatial_attention.shape[1]):
            fig, ax = plt.subplots()
            # print(all_spatial_attention[i,j].max(), all_spatial_attention[i,j].min())
            # cax = ax.imshow(all_spatial_attention[i,j], cmap='jet', interpolation='bicubic')
            cax = ax.imshow(all_spatial_attention[i, j], cmap='jet', interpolation='bicubic', vmin=0, vmax=1)
            ax.axis('off')
            ax.axes.get_xaxis().set_visible(False)
            ax.axes.get_yaxis().set_visible(False)
            #        cbar = fig.colorbar(cax)
            fig.savefig(write_path_prefix + net_name + '/vis_map/' + str(epoch) +
                        '/' + str(i) + '_au_' + str(j) + '.png', bbox_inches='tight', pad_inches=0)

        for j in range(all_spatial_attention.shape[1]):
            overlay = Image.open(write_path_prefix + net_name + '/vis_map/' + str(epoch) +
                        '/' + str(i) + '_au_' + str(j) + '.png')
            overlay = overlay.resize(background.size, Image.ANTIALIAS)
            background = background.convert('RGBA')
            overlay = overlay.convert('RGBA')
            new_img = Image.blend(background, overlay, alpha)
            new_img.save(write_path_prefix + net_name + '/overlay_vis_map/' + str(epoch) +
                        '/' + str(i) + '_au_' + str(j) + '.png', 'PNG')


def dice_loss(pred, target, smooth = 1):
    """This definition generalize to real valued pred and target vector.
    This should be differentiable.
    pred: tensor with first dimension as batch
    target: tensor with first dimension as batch
    """

    # have to use contiguous since they may from a torch.view op
    iflat = pred.contiguous().view(-1)
    tflat = target.contiguous().view(-1)
    intersection = (iflat * tflat).sum()

    A_sum = torch.sum(iflat * iflat)
    B_sum = torch.sum(tflat * tflat)

    return 1 - ((2. * intersection + smooth) / (A_sum + B_sum + smooth)) / iflat.size(0)


def cf_loss(pred, target, epoch, epochs, gamma_pos=2.0, gamma_neg=2.0, gamma_hc=3.0, factor=4):
    # Cyclical
    if factor*epoch < epochs:
        eta = 1 - factor * epoch/(epochs-1)
    else:
        eta = (factor*epoch/(epochs-1) - 1.0)/(factor - 1.0)

    # print(eta)
    
    # ASL weights
    iflat = pred.contiguous().view(-1) # p
    tflat = target.contiguous().view(-1) # y

    xs_pos = iflat  # p
    xs_neg = 1 - xs_pos  # 1-p
    
    asymmetric_w_pos = torch.pow(1 - xs_pos, gamma_pos)

    asymmetric_w_neg = torch.pow(1 - xs_neg, gamma_neg)

    positive_w_pos = torch.pow(1 + xs_pos, gamma_hc)

    positive_w_neg = torch.pow(1 + xs_neg, gamma_hc)

    positive_loss = -(tflat * positive_w_pos * torch.log(xs_pos.clamp(min=1e-8)) + (1 - tflat) * positive_w_neg * torch.log(xs_neg.clamp(min=1e-8)))
    asymmetric_loss = -(tflat * asymmetric_w_pos * torch.log(xs_pos.clamp(min=1e-8)) + (1 - tflat) * asymmetric_w_neg * torch.log(xs_neg.clamp(min=1e-8)))

    loss = (1 - eta)*asymmetric_loss + eta*positive_loss

    return loss.sum()

def au_cf_loss(input, target, epoch, epochs, weight=None, size_average=True):
    for i in range(input.size(1)):
        # input is log_softmax, t_input is probability
        t_input = (input[:, i])
        t_target = (target[:, i]).float()
        # t_loss = 1 - float(2*torch.dot(t_input, t_target) + smooth)/\
        #          (torch.dot(t_input, t_input)+torch.dot(t_target, t_target)+smooth)/t_input.size(0)
        t_loss = cf_loss(t_input, t_target, epoch, epochs, gamma_pos=2.0, gamma_neg=2.0, gamma_hc=3.0, factor=4)
        if weight is not None:
            t_loss = t_loss * weight[i]
        t_loss = torch.unsqueeze(t_loss, 0)
        if i == 0:
            loss = t_loss
        else:
            loss = torch.cat((loss, t_loss), 0)

    if size_average:
        return loss.mean()
    else:
        return loss.sum()


def wa_loss(pred, target, gama_i, m):
    iflat = pred.contiguous().view(-1)
    tflat = target.contiguous().view(-1)
    xs_pos = iflat
    p_m = torch.clamp((iflat-m), min=0)
    xs_neg = 1 - p_m

    # Basic CE calculation
    los_pos = tflat * torch.log(xs_pos.clamp(min=1e-8))
    los_neg = (1 - tflat) * torch.log(xs_neg.clamp(min=1e-8))

    neg_weight = 1 - xs_neg

    loss = -(los_pos + (neg_weight**gama_i) * los_neg)

    return loss.sum()


def au_wa_loss(input, target, weight=None, size_average=True, m=0.1, gama=None):
    for i in range(input.size(1)):
        # input is log_softmax, t_input is probability
        t_input = (input[:, i])
        t_target = (target[:, i]).float()
        # t_loss = 1 - float(2*torch.dot(t_input, t_target) + smooth)/\
        #          (torch.dot(t_input, t_input)+torch.dot(t_target, t_target)+smooth)/t_input.size(0)
        t_loss = wa_loss(t_input, t_target, gama[i], m)
        if weight is not None:
            t_loss = t_loss * weight[i]
        t_loss = torch.unsqueeze(t_loss, 0)
        if i == 0:
            loss = t_loss
        else:
            loss = torch.cat((loss, t_loss), 0)

    if size_average:
        return loss.mean()
    else:
        return loss.sum()


def wa_loss_level(pred, target, level):
    iflat = pred.contiguous().view(-1)
    tflat = target.contiguous().view(-1)
    xs_pos = iflat
    xs_neg = 1 - iflat

    # Basic CE calculation
    los_pos = tflat * torch.log(xs_pos.clamp(min=1e-8))
    los_neg = (1 - tflat) * torch.log(xs_neg.clamp(min=1e-8))

    neg_weight = 1 - xs_neg

    loss = -(los_pos + (neg_weight ** level) * los_neg)

    return loss.sum()



def au_wa_loss_level(input, target, weight=None, size_average=True):
    hard_list = [0, 1, 8, 10, 11]
    middle_list = [2, 7, 9]
    simple_list = [3, 4, 5, 6]
    for i in range(input.size(1)):
        # input is log_softmax, t_input is probability
        t_input = (input[:, i])
        t_target = (target[:, i]).float()
        # t_loss = 1 - float(2*torch.dot(t_input, t_target) + smooth)/\
        #          (torch.dot(t_input, t_input)+torch.dot(t_target, t_target)+smooth)/t_input.size(0)
        if i in hard_list:
            t_loss = wa_loss_level(t_input, t_target, 0.75)
        elif i in middle_list:
            t_loss = wa_loss_level(t_input, t_target, 1.0)
        else:
            t_loss = wa_loss_level(t_input, t_target, 1.25)
        
        if weight is not None:
            t_loss = t_loss * weight[i]
        t_loss = torch.unsqueeze(t_loss, 0)
        if i == 0:
            loss = t_loss
        else:
            loss = torch.cat((loss, t_loss), 0)

    if size_average:
        return loss.mean()
    else:
        return loss.sum()

def au_softmax_loss(input, target, weight=None, size_average=True, reduce=True):
    classify_loss = nn.NLLLoss(size_average=size_average, reduce=reduce)

    for i in range(input.size(2)):
        t_input = input[:, :, i]
        t_target = target[:, i]
        t_loss = classify_loss(t_input, t_target)
        if weight is not None:
            t_loss = t_loss * weight[i]
        t_loss = torch.unsqueeze(t_loss, 0)
        if i == 0:
            loss = t_loss
        else:
            loss = torch.cat((loss, t_loss), 0)

    if size_average:
        return loss.mean()
    else:
        return loss.sum()


def au_dice_loss(input, target, weight=None, smooth = 1, size_average=True):
    for i in range(input.size(1)):
        # input is log_softmax, t_input is probability
        t_input = (input[:, i]).exp()
        t_target = (target[:, i]).float()
        # t_loss = 1 - float(2*torch.dot(t_input, t_target) + smooth)/\
        #          (torch.dot(t_input, t_input)+torch.dot(t_target, t_target)+smooth)/t_input.size(0)
        t_loss = dice_loss(t_input, t_target, smooth)
        if weight is not None:
            t_loss = t_loss * weight[i]
        t_loss = torch.unsqueeze(t_loss, 0)
        if i == 0:
            loss = t_loss
        else:
            loss = torch.cat((loss, t_loss), 0)

    if size_average:
        return loss.mean()
    else:
        return loss.sum()


def landmark_loss(input, target, biocular, size_average=True):
    for i in range(input.size(0)):
        t_input = input[i,:]
        t_target = target[i,:]
        t_loss = torch.sum((t_input - t_target) ** 2) / (2.0*biocular[i])
        t_loss = torch.unsqueeze(t_loss, 0)
        if i == 0:
            loss = t_loss
        else:
            loss = torch.cat((loss, t_loss), 0)

    if size_average:
        return loss.mean()
    else:
        return loss.sum()


def attention_refine_loss(input, target, size_average=True, reduce=True):
    # loss is averaged over each point in the attention map,
    # note that Eq.(4) in our ECCV paper is to sum all the points,
    # change the value of lambda_refine can remove this difference.
    classify_loss = nn.BCELoss(size_average=size_average, reduce=reduce)

    input = input.view(input.size(0), input.size(1), -1)
    target = target.view(target.size(0), target.size(1), -1)
    for i in range(input.size(1)):
        t_input = input[:, i, :]
        t_target = target[:, i, :]
        t_loss = classify_loss(t_input, t_target)
        t_loss = torch.unsqueeze(t_loss, 0)
        if i == 0:
            loss = t_loss
        else:
            loss = torch.cat((loss, t_loss), 0)
    # sum losses of all AUs
    return loss.sum()


