import os
import time
import json
import random
import argparse
import datetime
import tqdm
import sys
import numpy as np

import torch
import torch.utils.data as util_data
from timm.scheduler.cosine_lr import CosineLRScheduler
from torch.optim import AdamW
import torch.backends.cudnn as cudnn
import torch.distributed as dist

import warnings

import pre_process as prep
from data_list import ImageList

from util import *
# from vit import vit_base_patch16_224

from utils.logger import create_logger
from utils.optimizer import build_optimizer
from utils.utils import  NativeScalerWithGradNormCount, auto_resume_helper, reduce_tensor
from utils.utils import load_checkpoint_ema, load_pretrained_ema, save_checkpoint_ema

warnings.filterwarnings("ignore")

import wandb

# from vmamba.models import build_model
# from vttt import build_model
from models import build_model
# from qu_vttt import build_model
# from vittt import build_model
from config import get_config
from data import build_loader
from utils.lr_scheduler import build_scheduler

from fvcore.nn import FlopCountAnalysis, flop_count_str, flop_count

from timm.utils import accuracy, AverageMeter
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.utils import ModelEma as ModelEma

from au_custom_loss import AUCustomLoss
from au_validate import AU_detection_evalv2

def parse_option():
    parser = argparse.ArgumentParser('AUMamba', add_help=False)
    parser.add_argument('--cfg', type=str, metavar="FILE", default="", help='path to config file', )
    parser.add_argument(
        "--opts",
        help="Modify config options by adding 'KEY VALUE' pairs. ",
        default=None,
        nargs='+',
    )

    # easy config modification
    parser.add_argument('--batch-size', type=int, help="batch size for single GPU")
    parser.add_argument('--data-path', type=str, default="BP4D", help='path to dataset')
    parser.add_argument('--zip', action='store_true', help='use zipped dataset instead of folder dataset')
    parser.add_argument('--cache-mode', type=str, default='part', choices=['no', 'full', 'part'],
                        help='no: no cache, '
                             'full: cache all data, '
                             'part: sharding the dataset into nonoverlapping pieces and only cache one piece')
    parser.add_argument('--pretrained',
                        help='pretrained weight from checkpoint, could be imagenet22k pretrained weight')
    parser.add_argument('--resume', help='resume from checkpoint')
    parser.add_argument('--accumulation-steps', type=int, help="gradient accumulation steps")
    parser.add_argument('--use-checkpoint', action='store_true',
                        help="whether to use gradient checkpointing to save memory")
    parser.add_argument('--disable_amp', action='store_true', help='Disable pytorch amp')
    parser.add_argument('--output', default='output', type=str, metavar='PATH',
                        help='root of output folder, the full path is <output>/<model_name>/<tag> (default: output)')
    parser.add_argument('--tag', default=time.strftime("%Y%m%d%H%M%S", time.localtime()), help='tag of experiment')
    parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
    parser.add_argument('--throughput', action='store_true', help='Test throughput only')

    parser.add_argument('--fused_layernorm', action='store_true', help='Use fused layernorm.')
    parser.add_argument('--optim', type=str, help='overwrite optimizer if provided, can be adamw/sgd.')

    # EMA related parameters
    parser.add_argument('--model_ema', type=str2bool, default=True)
    parser.add_argument('--model_ema_decay', type=float, default=0.9999, help='')
    parser.add_argument('--model_ema_force_cpu', type=str2bool, default=False, help='')

    parser.add_argument('--memory_limit_rate', type=float, default=-1, help='limitation of gpu memory use')

    parser.add_argument('--use_wandb', action='store_true')

    args, unparsed = parser.parse_known_args()
    config = get_config(args)

    return args, config


def main(config, args):
    if dist.get_rank() == 0:
        if args.use_wandb:
            wandb.init(
                project="AUStudy",          # 项目名称
                entity="ly153496-tianjin-university",
                name=f"{config.OUTPUT}",               # 实验名称
                config=config                         # 记录超参数配置
            )
    dataset_train, dataset_val, data_loader_train, data_loader_val, mixup_fn = build_loader(config)

    logger.info(f"Creating model:{config.MODEL.TYPE}/{config.MODEL.NAME}")
    model = build_model(config)


    # 计算模型 FLOPs 和参数量（仅在 rank=0 的主进程执行）,用于分析模型大小和计算成本。
    if dist.get_rank() == 0:
        # 如果 model 有 flops() 方法：
        if hasattr(model, 'flops'):
            logger.info(str(model))
            n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
            logger.info(f"number of params: {n_parameters}")
            # 输入一张3*224*224图片，进行一次前后向传播
            flops = model.flops()
            logger.info(f"number of GFLOPs: {flops / 1e9}")
        # 否则：使用 FlopCountAnalysis() 计算 FLOPs，并转换为字符串格式打印。
        else:
            logger.info(flop_count_str(FlopCountAnalysis(model, (dataset_val[0][0][None],))))
    # 释放 GPU 未使用的显存，减少显存碎片化问题。
    torch.cuda.empty_cache()
    dist.barrier(device_ids=[local_rank])

    # 确保模型在 GPU 上运行
    model = model.cuda()
    # model.cuda()
    model_without_ddp = model

    model_ema = None
    # 创建指数移动平均模型（EMA），适用于半监督学习、目标检测等任务，能提升模型泛化能力。
    if args.model_ema:
        # Important to create EMA model after cuda(), DP wrapper, and AMP but before SyncBN and DDP wrapper
        # 控制 EMA 的衰减率（一般 0.99 ~ 0.999）
        # 根据条件，EMA 计算会在 CPU 上进行，减少 GPU 负担。
        model_ema = ModelEma(
            model,
            decay=args.model_ema_decay,
            device='cpu' if args.model_ema_force_cpu else '',
            resume='')
        print("Using EMA with decay = %.8f" % args.model_ema_decay)
    # 初始化优化器，准备训练
    optimizer = build_optimizer(config, model, logger, args=args)
    # model = torch.nn.parallel.DistributedDataParallel(model, broadcast_buffers=False, find_unused_parameters=True)
    # broadcast_buffers=False 避免在不同 GPU 之间同步无用的 buffer，提高效率。
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank],output_device=local_rank,broadcast_buffers=False,find_unused_parameters=True)
    
    # NativeScalerWithGradNormCount() 是 AMP（混合精度训练） 的梯度缩放器，能防止浮点溢出，提高训练稳定性。
    loss_scaler = NativeScalerWithGradNormCount()

    # 根据梯度累积调整学习率调度步长，保证训练稳定性
    if config.TRAIN.ACCUMULATION_STEPS > 1:
        lr_scheduler = build_scheduler(config, optimizer, len(data_loader_train) // config.TRAIN.ACCUMULATION_STEPS)
    else:
        lr_scheduler = build_scheduler(config, optimizer, len(data_loader_train))

    # 根据数据集和数据增强方式选择合适的损失函数，提高训练效果
    if config.DATA.DATASET == "BP4D" or config.DATA.DATASET=="DISFA":
        criterion = AUCustomLoss(config)
        criterion_mse = nn.MSELoss()
    else:
        if config.AUG.MIXUP > 0.:
            # smoothing is handled with mixup label transform
            criterion = SoftTargetCrossEntropy()
        elif config.MODEL.LABEL_SMOOTHING > 0.:
            criterion = LabelSmoothingCrossEntropy(smoothing=config.MODEL.LABEL_SMOOTHING)
        else:
            criterion = torch.nn.CrossEntropyLoss()

    max_accuracy = 0.0
    max_accuracy_ema = 0.0

    # 自动恢复训练，防止因意外中断导致训练数据丢失。
    if config.TRAIN.AUTO_RESUME:
        # 检查是否存在 checkpoint 文件
        resume_file = auto_resume_helper(config.OUTPUT)
        if resume_file:
            # config.MODEL.RESUME 设定为找到的 checkpoint 路径，防止 RESUME 为空导致重头训练。
            if config.MODEL.RESUME:
                logger.warning(f"auto-resume changing resume file from {config.MODEL.RESUME} to {resume_file}")
            config.defrost()
            config.MODEL.RESUME = resume_file
            config.freeze()
            logger.info(f'auto resuming from {resume_file}')
        # 如果没有 checkpoint，日志会提示没有找到检查点，跳过自动恢复。
        else:
            logger.info(f'no checkpoint found in {config.OUTPUT}, ignoring auto resume')
    # 加载 checkpoint
    if config.MODEL.RESUME:
        # 调用 load_checkpoint_ema() 加载模型、优化器、学习率调度器等状态
        max_accuracy, max_accuracy_ema = load_checkpoint_ema(config, model_without_ddp, optimizer, lr_scheduler, loss_scaler, logger, model_ema)
        # 在 data_loader_val 上进行验证，计算模型准确率 (acc1, acc5)。
        acc1, acc5, loss = validate(config, data_loader_val, model)
        logger.info(f"Accuracy of the network on the {len(dataset_val)} test images: {acc1:.1f}%")
        if model_ema is not None:
            acc1_ema, acc5_ema, loss_ema = validate(config, data_loader_val, model_ema.ema)
            logger.info(f"Accuracy of the network ema on the {len(dataset_val)} test images: {acc1_ema:.1f}%")
        # 如果 EVAL_MODE 开启，表示只做验证，直接 return 结束代码执行。
        if config.EVAL_MODE:
            return
    # 如果使用预训练模型（PRETRAINED=True），但没有 checkpoint
    if config.MODEL.PRETRAINED and (not config.MODEL.RESUME):
        load_pretrained_ema(config, model_without_ddp, logger, model_ema)
        # skip first validate
        # acc1, acc5, loss = validate(config, data_loader_val, model)
        # logger.info(f"Accuracy of the network on the {len(dataset_val)} test images: {acc1:.1f}%")
        # 如果 model_ema 可用，进行评估
        if model_ema is not None:
            acc1_ema, acc5_ema, loss_ema = validate(config, data_loader_val, model_ema.ema)
            logger.info(f"Accuracy of the network ema on the {len(dataset_val)} test images: {acc1_ema:.1f}%")
        # 如果 EVAL_MODE=True，表示只做评估，不训练，直接返回
        if config.EVAL_MODE:
            return


    # 如果 THROUGHPUT_MODE=True，表示不训练，而是测试模型吞吐量（每秒处理多少张图片）。
    if config.THROUGHPUT_MODE and (dist.get_rank() == 0):
        logger.info(f"throughput mode ==============================")
        throughput(data_loader_val, model, logger)
        # 如果 model_ema 存在，也对 model_ema 进行吞吐量测试
        if model_ema is not None:
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            
            throughput(data_loader_val, model_ema.ema, logger)
        return
    
    logger.info("Start training")
    start_time = time.time()


    for epoch in range(config.TRAIN.START_EPOCH, config.TRAIN.EPOCHS):
        data_loader_train.sampler.set_epoch(epoch)

        # new_mixup_alpha = config.AUG.MIXUP * (1  - (epoch+1) / (config.TRAIN.EPOCHS-config.TRAIN.START_EPOCH))
        # mixup_fn.update_params(mixup_alpha=new_mixup_alpha)

        train_one_epoch(config, model, criterion, data_loader_train, optimizer, epoch, mixup_fn, lr_scheduler, loss_scaler, model_ema, criterion_mse=criterion_mse)

        acc1, acc5, loss = validate(config, data_loader_val, model)

        if dist.get_rank() == 0 and (epoch % config.SAVE_FREQ == 0 or epoch == (config.TRAIN.EPOCHS - 1)):
            save_checkpoint_ema(config, epoch, model_without_ddp, max_accuracy, optimizer, lr_scheduler, loss_scaler, logger, model_ema, max_accuracy_ema)

        logger.info(f"Accuracy of the network on the {len(dataset_val)} test images: {acc1:.1f}%")
        if args.use_wandb and dist.get_rank() == 0:
            wandb.log({"epoch": epoch, "val_F1_Score": acc1, "val_Acc": acc5})

        max_accuracy = max(max_accuracy, acc1)
        logger.info(f'Max accuracy: {max_accuracy:.2f}%')
        if model_ema is not None:
            acc1_ema, acc5_ema, loss_ema = validate(config, data_loader_val, model_ema.ema)
            logger.info(f"Accuracy of the network on the {len(dataset_val)} test images: {acc1_ema:.1f}%")
            if args.use_wandb and dist.get_rank() == 0:
                wandb.log({"ema_val_F1_Score": acc1_ema, "ema_val_Acc": acc5_ema})
            max_accuracy_ema = max(max_accuracy_ema, acc1_ema)
            logger.info(f'Max accuracy ema: {max_accuracy_ema:.2f}%')


    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logger.info('Training time {}'.format(total_time_str))

def train_one_epoch(config, model, criterion, data_loader, optimizer, epoch, mixup_fn, lr_scheduler, loss_scaler, model_ema=None, model_time_warmup=50, criterion_mse=None):
    model.train()
    optimizer.zero_grad()

    num_steps = len(data_loader)
    batch_time = AverageMeter()
    model_time = AverageMeter()
    data_time = AverageMeter()
    loss_meter = AverageMeter()
    norm_meter = AverageMeter()
    scaler_meter = AverageMeter()

    start = time.time()
    end = time.time()
    for idx, (samples, land, biocular, targets) in enumerate(data_loader):
        torch.cuda.reset_peak_memory_stats()
        samples = samples.cuda(non_blocking=True)
        land = land.to(samples.device, dtype=samples.dtype, non_blocking=True)
        biocular = biocular.to(samples.device, dtype=samples.dtype, non_blocking=True)
        targets = targets.cuda(non_blocking=True)

        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)

        data_time.update(time.time() - end)
        with torch.cuda.amp.autocast(enabled=config.AMP_ENABLE):
            outputs = model(samples)

        loss = criterion(outputs, targets)
        # loss是一个值
        loss = loss / config.TRAIN.ACCUMULATION_STEPS

        # this attribute is added by timm on one optimizer (adahessian)
        is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
        grad_norm = loss_scaler(loss, optimizer, clip_grad=config.TRAIN.CLIP_GRAD,
                                parameters=model.parameters(), create_graph=is_second_order,
                                update_grad=(idx + 1) % config.TRAIN.ACCUMULATION_STEPS == 0)
        
        # for name, param in model.named_parameters():
        #     if param.grad is not None:
        #         # print(f"Gradient for {name}: Correct!")
        #         pass
        #     else:
        #         print(f"No gradient for {name}")
        # exit()
        if (idx + 1) % config.TRAIN.ACCUMULATION_STEPS == 0:
            optimizer.zero_grad()
            lr_scheduler.step_update((epoch * num_steps + idx) // config.TRAIN.ACCUMULATION_STEPS)
            if model_ema is not None:
                model_ema.update(model)
            
            # Only log metrics after accumulation steps
            if dist.get_rank() == 0:
                if args.use_wandb:
                    wandb.log({
                        "loss": loss_meter.avg * config.TRAIN.ACCUMULATION_STEPS,  # Log the average loss for the accumulation step
                        "lr": optimizer.param_groups[0]['lr'],
                        "weight_decay": optimizer.param_groups[0]['weight_decay'],
                        "grad_norm": norm_meter.avg,  # Use the average grad norm
                        "loss_scale": scaler_meter.avg,  # Use the average loss scale
                    })
        loss_scale_value = loss_scaler.state_dict()["scale"]

        torch.cuda.synchronize()

        loss_meter.update(loss.item(), targets.size(0))
        if grad_norm is not None:  # loss_scaler return None if not update
            norm_meter.update(grad_norm)
        scaler_meter.update(loss_scale_value)
        batch_time.update(time.time() - end)
        end = time.time()

        if idx > model_time_warmup:
            model_time.update(batch_time.val - data_time.val)

        if idx % config.PRINT_FREQ == 0:
            lr = optimizer.param_groups[0]['lr']
            wd = optimizer.param_groups[0]['weight_decay']
            memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
            etas = batch_time.avg * (num_steps - idx)
            logger.info(
                f'Train: [{epoch}/{config.TRAIN.EPOCHS}][{idx}/{num_steps}]\t'
                f'eta {datetime.timedelta(seconds=int(etas))} lr {lr:.6f}\t wd {wd:.4f}\t'
                f'time {batch_time.val:.4f} ({batch_time.avg:.4f})\t'
                f'data time {data_time.val:.4f} ({data_time.avg:.4f})\t'
                f'model time {model_time.val:.4f} ({model_time.avg:.4f})\t'
                f'loss {loss_meter.val:.4f} ({loss_meter.avg:.4f})\t'
                f'grad_norm {norm_meter.val:.4f} ({norm_meter.avg:.4f})\t'
                f'loss_scale {scaler_meter.val:.4f} ({scaler_meter.avg:.4f})\t'
                f'mem {memory_used:.0f}MB')
    epoch_time = time.time() - start
    logger.info(f"EPOCH {epoch} training takes {datetime.timedelta(seconds=int(epoch_time))}")

@torch.no_grad()
def validate(config, data_loader, model):
    if config.DATA.DATASET=="BP4D" or config.DATA.DATASET=="DISFA":
        return AU_detection_evalv2(config, data_loader, model)

    criterion = torch.nn.CrossEntropyLoss()
    model.eval()

    batch_time = AverageMeter()
    loss_meter = AverageMeter()
    acc1_meter = AverageMeter()
    acc5_meter = AverageMeter()

    end = time.time()
    for idx, (images, target) in enumerate(data_loader):
        images = images.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)

        # compute output
        with torch.cuda.amp.autocast(enabled=config.AMP_ENABLE):
            output = model(images)

        # measure accuracy and record loss
        loss = criterion(output, target)
        acc1, acc5 = accuracy(output, target, topk=(1, 5))

        acc1 = reduce_tensor(acc1)
        acc5 = reduce_tensor(acc5)
        loss = reduce_tensor(loss)

        loss_meter.update(loss.item(), target.size(0))
        acc1_meter.update(acc1.item(), target.size(0))
        acc5_meter.update(acc5.item(), target.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # config.PRINT_FREQ: Frequency to logging info
        if idx % config.PRINT_FREQ == 0:
            memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
            logger.info(
                f'Test: [{idx}/{len(data_loader)}]\t'
                f'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                f'Loss {loss_meter.val:.4f} ({loss_meter.avg:.4f})\t'
                f'Acc@1 {acc1_meter.val:.3f} ({acc1_meter.avg:.3f})\t'
                f'Acc@5 {acc5_meter.val:.3f} ({acc5_meter.avg:.3f})\t'
                f'Mem {memory_used:.0f}MB')
    logger.info(f' * Acc@1 {acc1_meter.avg:.3f} Acc@5 {acc5_meter.avg:.3f}')
    return acc1_meter.avg, acc5_meter.avg, loss_meter.avg

@torch.no_grad()
def throughput(data_loader, model, logger):
    model.eval()

    for idx, (images, _) in enumerate(data_loader):
        images = images.cuda(non_blocking=True)
        batch_size = images.shape[0]
        for i in range(50):
            model(images)
        torch.cuda.synchronize()
        logger.info(f"throughput averaged with 30 times")
        tic1 = time.time()
        for i in range(30):
            model(images)
        torch.cuda.synchronize()
        tic2 = time.time()
        logger.info(f"batch_size {batch_size} throughput {30 * batch_size / (tic2 - tic1)}")
        return


if __name__ == '__main__':
    args, config = parse_option()
    # 由于 Apex AMP 已被废弃，所以建议改用 PyTorch 内置的 AMP
    if config.AMP_OPT_LEVEL:
        print("[warning] Apex amp has been deprecated, please use pytorch amp instead!")
        
    # RANK：当前进程的编号（0 代表主进程）;ORLD_SIZE：总共有多少个进程（通常等于 GPU 数量）
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ['WORLD_SIZE'])
        print(f"RANK and WORLD_SIZE in environ: {rank}/{world_size}")
    else:
        rank = -1
        world_size = -1
    local_rank = int(os.environ["LOCAL_RANK"])
    # 确保每个进程绑定到特定 GPU
    torch.cuda.set_device(local_rank)
    # NCCL（NVIDIA Collective Communications Library）用于加速 GPU 之间的通信。
    dist.init_process_group(backend='nccl', init_method='env://', world_size=world_size, rank=rank)
    # 所有进程同步，确保所有进程都准备好，再继续执行
    dist.barrier(device_ids=[local_rank])

    seed = config.SEED + dist.get_rank()
    torch.manual_seed(seed)         # 控制 CPU 上的 PyTorch 随机数生成器
    torch.cuda.manual_seed(seed)    # 控制 GPU 上的 PyTorch 随机数生成器
    np.random.seed(seed) # 控制 NumPy 的随机数生成器
    random.seed(seed)    # 控制 Python 自带的 random 模块
    # cuDNN 自动寻找最佳的卷积算法
    cudnn.benchmark = True

    if True: 
        torch.backends.cudnn.enabled = True         # 开启 cuDNN 加速，提高深度学习运算性能
        torch.backends.cudnn.benchmark = True       # 让 cuDNN 自动选择最优的卷积算法，适用于输入尺寸固定的情况，可以提高速度
        torch.backends.cudnn.deterministic = True   # 强制 cuDNN 以确定性方式计算，保证每次运行结果一致，但可能会降低性能

    # 确保在不同 GPU 数量和 batch size 下自动调整学习率，保持训练稳定性。
    # 新学习率=原学习率×总 batch size/512
    # linear scale the learning rate according to total batch size, may not be optimal
    linear_scaled_lr = config.TRAIN.BASE_LR * config.DATA.BATCH_SIZE * dist.get_world_size() / 512.0
    linear_scaled_warmup_lr = config.TRAIN.WARMUP_LR * config.DATA.BATCH_SIZE * dist.get_world_size() / 512.0
    linear_scaled_min_lr = config.TRAIN.MIN_LR * config.DATA.BATCH_SIZE * dist.get_world_size() / 512.0

    # 梯度累加，顾名思义，就是将多次计算得到的梯度值进行累加，然后一次性进行参数更新。此时需要进一步放大学习率
    # gradient accumulation also need to scale the learning rate
    if config.TRAIN.ACCUMULATION_STEPS > 1:
        linear_scaled_lr = linear_scaled_lr * config.TRAIN.ACCUMULATION_STEPS
        linear_scaled_warmup_lr = linear_scaled_warmup_lr * config.TRAIN.ACCUMULATION_STEPS
        linear_scaled_min_lr = linear_scaled_min_lr * config.TRAIN.ACCUMULATION_STEPS

    # 动态更新学习率
    config.defrost()
    config.TRAIN.BASE_LR = linear_scaled_lr
    config.TRAIN.WARMUP_LR = linear_scaled_warmup_lr
    config.TRAIN.MIN_LR = linear_scaled_min_lr
    config.freeze()

    # 只有 rank 0 进程（主进程）设置 config.OUTPUT，然后广播给其他进程。确保所有进程的 config.OUTPUT 一致
    config.defrost()
    if dist.get_rank() == 0:
        obj = [config.OUTPUT]
        # obj = [str(random.randint(0, 100))] # for test
    else:
        obj = [None]
    dist.broadcast_object_list(obj)
    dist.barrier(device_ids=[local_rank])
    config.OUTPUT = obj[0]
    print(config.OUTPUT, flush=True)
    config.freeze()

    # 创建日志目录，如果已存在不会报错。
    os.makedirs(config.OUTPUT, exist_ok=True)
    # 初始化日志系统，create_logger() 会创建 logger，用于记录训练信息。
    logger = create_logger(output_dir=config.OUTPUT, dist_rank=dist.get_rank(), name=f"{config.MODEL.NAME}")

    # 只有 rank 0 进程会保存 config.json，避免多个进程重复写入。
    if dist.get_rank() == 0:
        path = os.path.join(config.OUTPUT, "config.json")
        with open(path, "w") as f:
            f.write(config.dump())
        logger.info(f"Full config saved to {path}")

    # 记录完整配置，方便之后复现实验。vars(args) 转换参数为 JSON 记录。
    logger.info(config.dump())
    logger.info(json.dumps(vars(args)))
    # 如果 memory_limit_rate 在 (0,1) 之间，限制 GPU 进程最大可用内存
    if args.memory_limit_rate > 0 and args.memory_limit_rate < 1:
        torch.cuda.set_per_process_memory_fraction(args.memory_limit_rate)
        usable_memory = torch.cuda.get_device_properties(0).total_memory * args.memory_limit_rate / 1e6
        print(f"===========> GPU memory is limited to {usable_memory}MB", flush=True)

    main(config, args)
