# --------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# --------------------------------------------------------

import os
import sys
import torch
import numpy as np
import torch.distributed as dist
import torch.utils.data.distributed
from torchvision import datasets, transforms
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.data import Mixup
from .mixup import Mixup as AUMixup
from timm.data import create_transform

from .cached_image_folder import CachedImageFolder
from .imagenet22k_dataset import IN22KDATASET
from .samplers import SubsetRandomSampler
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from data_list import ImageList as BP4DDATASET
from pre_process import image_train, image_test, land_transform
try:
    from torchvision.transforms import InterpolationMode


    def _pil_interp(method):
        if method == 'bicubic':
            return InterpolationMode.BICUBIC
        elif method == 'lanczos':
            return InterpolationMode.LANCZOS
        elif method == 'hamming':
            return InterpolationMode.HAMMING
        else:
            # default bilinear, do we want to allow nearest?
            return InterpolationMode.BILINEAR


    import timm.data.transforms as timm_transforms

    timm_transforms._pil_interp = _pil_interp
except:
    from timm.data.transforms import _pil_interp




# 代码整体结构
# 目标：为分布式训练准备数据集、数据转换和数据加载器，同时支持数据增强技术（如Mixup/Cutmix）
# 流程：数据转换--→构建数据集--→数据加载器

# build_loader()：构建训练和验证数据加载器
def build_loader(config):
    #解冻配置文件以允许修改 config.MODEL.NUM_CLASSES，数据集创建后重新冻结配置，确保配置安全
    config.defrost()
    dataset_train, config.MODEL.NUM_CLASSES = build_dataset(is_train=True, config=config)
    config.freeze()
    print(f"rank {dist.get_rank()} successfully build train dataset")
    dataset_val, _ = build_dataset(is_train=False, config=config)
    print(f"rank {dist.get_rank()} successfully build val dataset")

    num_tasks = dist.get_world_size() 
    global_rank = dist.get_rank() 

    # torch.utils.data.DistributedSampler与torch.utils.data.distributed.DistributedSampler 这两个其实是同一个类，只是导入路径不同，功能完全一样，随便用哪个都可以！
    # 训练集采样
    # ✅ 随机抽取	⚠️ 适用于单机多 GPU，但不同 GPU 可能会采样到相同数据	
    # ZIP 压缩模式，部分缓存数据
    if config.DATA.ZIP_MODE and config.DATA.CACHE_MODE == 'part':
        indices = np.arange(dist.get_rank(), len(dataset_train), dist.get_world_size())
        sampler_train = SubsetRandomSampler(indices)
    else:
        # 分布式采样（可选shuffle）
        # ✅ 支持 shuffle	✅ 多机多 GPU 训练，数据自动分配	标准的多 GPU 分布式训练
        sampler_train = torch.utils.data.DistributedSampler(
            dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
        )
    # 测试集采样
    # ❌ 不会打乱数据	❌ 不支持多 GPU 训练
    # 单 GPU 测试 / 评估
    if config.TEST.SEQUENTIAL:
        # 顺序采样（无随机性）
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)
    # ✅ 支持 shuffle（可选）	✅ 适用于多 GPU 训练	分布式训练 / 评估
    else:
        # 分布式验证采样（可选shuffle）
        sampler_val = torch.utils.data.distributed.DistributedSampler(
            dataset_val, shuffle=config.TEST.SHUFFLE
        )

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=config.DATA.BATCH_SIZE,  # 每卡的实际批次大小
        num_workers=config.DATA.NUM_WORKERS,# 数据加载线程数（推荐8-16）通常设为GPU数量的4倍（如4卡设16线程），但需避免超过CPU核心数
        pin_memory=config.DATA.PIN_MEMORY,  # 启用CUDA内存锁定（加速到GPU的数据传输）
            drop_last=True,                 # 丢弃无法整除批次的数据
        )

    data_loader_val = torch.utils.data.DataLoader(
        dataset_val, sampler=sampler_val,
        batch_size=config.DATA.BATCH_SIZE,
        shuffle=False,                      # 验证阶段禁用 shuffle
        num_workers=config.DATA.NUM_WORKERS,
        pin_memory=config.DATA.PIN_MEMORY,
        drop_last=False                     # 保留所有数据（用于准确计算指标）
    )

    # setup mixup / cutmix
    mixup_fn = None
    #若配置了 MIXUP >0 或 CUTMIX >0，则创建对应增强对象
    mixup_active = config.AUG.MIXUP > 0 or config.AUG.CUTMIX > 0. or config.AUG.CUTMIX_MINMAX is not None
    if mixup_active:
        # BP4D 和 DISFA 数据集使用自定义的 AUMixup
        if config.DATA.DATASET=="BP4D" or config.DATA.DATASET=="DISFA":
            mixup_fn = AUMixup(
                mixup_alpha=config.AUG.MIXUP, cutmix_alpha=config.AUG.CUTMIX, cutmix_minmax=config.AUG.CUTMIX_MINMAX,
                prob=config.AUG.MIXUP_PROB, switch_prob=config.AUG.MIXUP_SWITCH_PROB, mode=config.AUG.MIXUP_MODE,
                label_smoothing=config.MODEL.LABEL_SMOOTHING, num_classes=config.MODEL.NUM_CLASSES)
        else:
            mixup_fn = Mixup(
                mixup_alpha=config.AUG.MIXUP, cutmix_alpha=config.AUG.CUTMIX, cutmix_minmax=config.AUG.CUTMIX_MINMAX,
                prob=config.AUG.MIXUP_PROB, switch_prob=config.AUG.MIXUP_SWITCH_PROB, mode=config.AUG.MIXUP_MODE,
                label_smoothing=config.MODEL.LABEL_SMOOTHING, num_classes=config.MODEL.NUM_CLASSES)

    #dataset_train/dataset_val: 训练/验证数据容器
    #loader_train/loader_val: 训练/验证批次数据生成器
    #mixup_fn 数据增强处理器，在训练循环中对批次数据进行增强处理
    return dataset_train, dataset_val, data_loader_train, data_loader_val, mixup_fn

# build_dataset()：根据配置创建数据集
def build_dataset(is_train, config):
    transform = build_transform(is_train, config)
    #标准 ImageNet (1000 类)
    if config.DATA.DATASET == 'imagenet':
        prefix = 'train' if is_train else 'val'
        if config.DATA.ZIP_MODE:
            ann_file = prefix + "_map.txt"
            prefix = prefix + ".zip@/"
            dataset = CachedImageFolder(config.DATA.DATA_PATH, ann_file, prefix, transform,
                                        cache_mode=config.DATA.CACHE_MODE if is_train else 'part')
        else:
            root = os.path.join(config.DATA.DATA_PATH, prefix)
            dataset = datasets.ImageFolder(root, transform=transform)
        nb_classes = 1000
    #ImageNet-22K (21841 类)
    elif config.DATA.DATASET == 'imagenet22K':
        prefix = 'ILSVRC2011fall_whole'
        if is_train:
            ann_file = prefix + "_map_train.txt"
        else:
            ann_file = prefix + "_map_val.txt"
        dataset = IN22KDATASET(config.DATA.DATA_PATH, ann_file, transform)
        nb_classes = 21841
    #BP4D/DISFA（表情识别数据集)
    elif config.DATA.DATASET == 'BP4D' or config.DATA.DATASET=="DISFA":
        # if is_train:
        #     dataset = BP4DDATASET(config.DATA.IMG_SIZE, config.DATA.TRAIN_PATH_PREFIX, phase='train', transform=image_train(crop_size=config.DATA.IMG_SIZE), target_transform=land_transform(img_size=config.DATA.IMG_SIZE, flip_reflect=np.loadtxt('data/list/reflect_49.txt')), config=config)
        # else:
        #     dataset = BP4DDATASET(config.DATA.IMG_SIZE, config.DATA.TEST_PATH_PREFIX, phase='test', transform=image_test(crop_size=config.DATA.IMG_SIZE), target_transform=land_transform(img_size=config.DATA.IMG_SIZE, flip_reflect=np.loadtxt('data/list/reflect_49.txt')), config=config)
        if is_train:
            ann_file = config.DATA.TRAIN_PATH_PREFIX
        else:
            ann_file = config.DATA.TEST_PATH_PREFIX
        dataset = BP4DDATASET(config.DATA.IMG_SIZE, ann_file, transform, config=config)
        nb_classes = config.MODEL.NUM_CLASSES
        # if config.DATA.DATASET == 'BP4D':
        #     nb_classes = 12
        # elif config.DATA.DATASET=="DISFA":
        #     nb_classes = 8
        # else:
        #     raise TypeError

    else:
        raise NotImplementedError("We only support ImageNet Now.")

    return dataset, nb_classes

# build_transform()：构建数据预处理流程
def build_transform(is_train, config):
    resize_im = config.DATA.IMG_SIZE > 32
    #训练阶段：通过多样化增强提升模型对光照、遮挡、尺度的鲁棒性。
    if is_train:
        #使用 create_transform 创建包含随机裁剪、颜色抖动等增强的组合
        transform = create_transform(
            input_size=config.DATA.IMG_SIZE,
            is_training=True,
            # hflip=0.5,
            # vflip=0.,   # 50% 水平翻转0% 垂直翻转
            color_jitter=config.AUG.COLOR_JITTER if config.AUG.COLOR_JITTER > 0 else None,# 颜色抖动 (ColorJitter): 当 AUG.COLOR_JITTER > 0 时启用色彩随机扰动。
            auto_augment=config.AUG.AUTO_AUGMENT if config.AUG.AUTO_AUGMENT != 'none' else None,#自动增强策略 (AutoAugment): 根据预设策略（如 v0, rand-m9-mstd0.5 等）选择组合增强。
            re_prob=config.AUG.REPROB, # 随机擦除 (RandomErasing): 随机遮蔽部分图像区域概率由 re_prob 控制。
            re_mode=config.AUG.REMODE,
            re_count=config.AUG.RECOUNT,
            interpolation=config.DATA.INTERPOLATION,# 插值方法: 使用 BILINEAR 或 BICUBIC 等算法调整图像尺寸。
        )
        # 如果图像尺寸较小 (IMG_SIZE <=32)，改用 RandomCrop 代替默认的 RandomResizedCrop
        # 目的: 避免经典小尺寸数据集（如 CIFAR-10）在随机裁剪后过度丢失信息。
        if not resize_im:
            transform.transforms[0] = transforms.RandomCrop(config.DATA.IMG_SIZE, padding=4)
        return transform
    #验证阶段：保持确定的预处理逻辑以准确评估模型性能。
    t = []
    if resize_im:
        #根据 TEST.CROP 决定是否进行中心裁剪
        if config.TEST.CROP:
            #若开启 TEST.CROP，先放大至 256-based 比例再中心裁剪（模仿 CNN 经典训练策略）
            size = int((256 / 224) * config.DATA.IMG_SIZE) # 适应224→256的缩放比例
            t.append(
                transforms.Resize(size, interpolation=_pil_interp(config.DATA.INTERPOLATION)),
                # to maintain same ratio w.r.t. 224 images
            )
            t.append(transforms.CenterCrop(config.DATA.IMG_SIZE))
        # 直接缩放模式
        else:
            t.append(
                transforms.Resize((config.DATA.IMG_SIZE, config.DATA.IMG_SIZE),
                                  interpolation=_pil_interp(config.DATA.INTERPOLATION))
            )
    # 转为张量并归一化到 [0,1]
    t.append(transforms.ToTensor())
    # 标准化处理（Normalize)
    # Normalize参数:
    # IMAGENET_DEFAULT_MEAN = [0.485, 0.456, 0.406]
    # IMAGENET_DEFAULT_STD = [0.229, 0.224, 0.225]
    t.append(transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD))
    return transforms.Compose(t)
