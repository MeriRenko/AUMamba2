# --------------------------------------------------------
# Modified By $@#Anonymous#@$
# --------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# --------------------------------------------------------

import os
from math import inf
import torch
import torch.distributed as dist
from timm.utils import ModelEma as ModelEma


def load_checkpoint_ema(config, model, optimizer, lr_scheduler, loss_scaler, logger, model_ema: ModelEma=None):
    # 打印正在恢复模型的路径。
    logger.info(f"==============> Resuming form {config.MODEL.RESUME}....................")
    # 根据 checkpoint 路径是本地文件还是网络 URL，选择不同方式加载 checkpoint。
    if config.MODEL.RESUME.startswith('https'):
        checkpoint = torch.hub.load_state_dict_from_url(
            config.MODEL.RESUME, map_location='cpu', check_hash=True)
    else:
        checkpoint = torch.load(config.MODEL.RESUME, map_location='cpu',weights_only=False)
    # 如果 checkpoint 中包含 model 键（即模型权重），就加载进当前模型。使用 strict=False 表示即使有部分参数对不上，也不会报错。
    if 'model' in checkpoint:
        msg = model.load_state_dict(checkpoint['model'], strict=False)
        logger.info(f"resuming model: {msg}")
    else:
        logger.warning(f"No 'model' found in {config.MODEL.RESUME}! ")
    # 如果传入了 model_ema，并且 checkpoint 里面有 model_ema 的权重，则恢复。
    if model_ema is not None:
        if 'model_ema' in checkpoint:
            msg = model_ema.ema.load_state_dict(checkpoint['model_ema'], strict=False)
            logger.info(f"resuming model_ema: {msg}")
        else:
            logger.warning(f"No 'model_ema' found in {config.MODEL.RESUME}! ")
    # 恢复训练器状态（优化器、调度器、epoch 等）
    max_accuracy = 0.0
    max_accuracy_ema = 0.0
    if not config.EVAL_MODE and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        config.defrost()
        config.TRAIN.START_EPOCH = checkpoint['epoch'] + 1
        config.freeze()
        if 'scaler' in checkpoint:
            loss_scaler.load_state_dict(checkpoint['scaler'])
        logger.info(f"=> loaded successfully '{config.MODEL.RESUME}' (epoch {checkpoint['epoch']})")
        if 'max_accuracy' in checkpoint:
            max_accuracy = checkpoint['max_accuracy']
        if 'max_accuracy_ema' in checkpoint:
            max_accuracy_ema = checkpoint['max_accuracy_ema']
    # 清理内存并返回精度信息
    del checkpoint
    torch.cuda.empty_cache()
    return max_accuracy, max_accuracy_ema


def load_pretrained_ema(config, model, logger, model_ema: ModelEma=None):
    logger.info(f"==============> Loading weight {config.MODEL.PRETRAINED} for fine-tuning......")
    checkpoint = torch.load(config.MODEL.PRETRAINED, map_location='cpu')
    
    if 'model' in checkpoint:
        pretrained_dict = checkpoint['model']
        model_dict = model.state_dict()
        
        # 过滤掉名称中包含 "head" 的参数，或那些形状不匹配的参数
        filtered_dict = {k: v for k, v in pretrained_dict.items() 
                        if k in model_dict and v.shape == model_dict[k].shape}

        msg = model.load_state_dict(filtered_dict, strict=False)
        logger.warning(msg)
        logger.info(f"=> loaded 'model' successfully from '{config.MODEL.PRETRAINED}'")
    else:
        logger.warning(f"No 'model' found in {config.MODEL.PRETRAINED}! ")

    if model_ema is not None:
        if "model_ema" in checkpoint:
            logger.info(f"=> loading 'model_ema' separately...")
        key = "model_ema" if ("model_ema" in checkpoint) else "model"
        if key in checkpoint:
            pretrained_dict = checkpoint[key]
            model_dict = model.state_dict()
            # 过滤掉名称中包含 "head" 的参数，或那些形状不匹配的参数
            filtered_dict = {k: v for k, v in pretrained_dict.items() 
                            if k in model_dict and v.shape == model_dict[k].shape}
            msg = model_ema.ema.load_state_dict(filtered_dict, strict=False)
            logger.warning(msg)
            logger.info(f"=> loaded '{key}' successfully from '{config.MODEL.PRETRAINED}' for model_ema")
        else:
            logger.warning(f"No '{key}' found in {config.MODEL.PRETRAINED}! ")

    del checkpoint
    torch.cuda.empty_cache()


def save_checkpoint_ema(config, epoch, model, max_accuracy, optimizer, lr_scheduler, loss_scaler, logger, model_ema: ModelEma=None, max_accuracy_ema=None):
    save_state = {'model': model.state_dict(),
                  'optimizer': optimizer.state_dict(),
                  'lr_scheduler': lr_scheduler.state_dict(),
                  'max_accuracy': max_accuracy,
                  'scaler': loss_scaler.state_dict(),
                  'epoch': epoch,
                  'config': config}
    
    if model_ema is not None:
        save_state.update({'model_ema': model_ema.ema.state_dict(),
            'max_accuray_ema': max_accuracy_ema})

    save_path = os.path.join(config.OUTPUT, f'ckpt_epoch_{epoch}.pth')
    logger.info(f"{save_path} saving......")
    torch.save(save_state, save_path)
    logger.info(f"{save_path} saved !!!")


def get_grad_norm(parameters, norm_type=2):
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = list(filter(lambda p: p.grad is not None, parameters))
    norm_type = float(norm_type)
    total_norm = 0
    for p in parameters:
        param_norm = p.grad.data.norm(norm_type)
        total_norm += param_norm.item() ** norm_type
    total_norm = total_norm ** (1. / norm_type)
    return total_norm


def auto_resume_helper(output_dir):
    # 参数是 output_dir，表示检查点保存的目录。
    # 获取 output_dir 目录下的所有文件和文件夹名称，返回一个列表。
    checkpoints = os.listdir(output_dir)
    # 筛选出以 .pth 结尾的文件（通常是 PyTorch 模型的保存文件），构成一个新的列表。
    checkpoints = [ckpt for ckpt in checkpoints if ckpt.endswith('pth')]
    print(f"All checkpoints founded in {output_dir}: {checkpoints}")
    if len(checkpoints) > 0:
        # 在这些检查点中，找出最近修改的那个（根据文件的最后修改时间 os.path.getmtime）。os.path.join 用来拼接成完整路径。
        latest_checkpoint = max([os.path.join(output_dir, d) for d in checkpoints], key=os.path.getmtime)
        # 打印出最新的检查点文件路径。
        print(f"The latest checkpoint founded: {latest_checkpoint}")
        # 把这个路径赋值给 resume_file，准备返回。
        resume_file = latest_checkpoint
    else:
        resume_file = None
    return resume_file


def reduce_tensor(tensor):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= dist.get_world_size()
    return rt


def ampscaler_get_grad_norm(parameters, norm_type: float = 2.0) -> torch.Tensor:
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = [p for p in parameters if p.grad is not None]
    norm_type = float(norm_type)
    if len(parameters) == 0:
        return torch.tensor(0.)
    device = parameters[0].grad.device
    if norm_type == inf:
        total_norm = max(p.grad.detach().abs().max().to(device) for p in parameters)
    else:
        total_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(),
                                                        norm_type).to(device) for p in parameters]), norm_type)
    return total_norm


class NativeScalerWithGradNormCount:
    state_dict_key = "amp_scaler"

    def __init__(self):
        self._scaler = torch.cuda.amp.GradScaler()

    def __call__(self, loss, optimizer, clip_grad=None, parameters=None, create_graph=False, update_grad=True):
        self._scaler.scale(loss).backward(create_graph=create_graph)
        if update_grad:
            if clip_grad is not None:
                assert parameters is not None
                self._scaler.unscale_(optimizer)  # unscale the gradients of optimizer's assigned params in-place
                norm = torch.nn.utils.clip_grad_norm_(parameters, clip_grad)
            else:
                self._scaler.unscale_(optimizer)
                norm = ampscaler_get_grad_norm(parameters)
            self._scaler.step(optimizer)
            self._scaler.update()
        else:
            norm = None
        return norm

    def state_dict(self):
        return self._scaler.state_dict()

    def load_state_dict(self, state_dict):
        self._scaler.load_state_dict(state_dict)

