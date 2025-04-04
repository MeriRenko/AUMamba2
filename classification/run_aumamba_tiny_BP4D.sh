#!/bin/bash

#定义配置
config1=configs/vssm/vmambav2v_tiny_224_BP4D_23for1.yaml
config2=configs/vssm/vmambav2v_tiny_224_BP4D_13for2.yaml
config3=configs/vssm/vmambav2v_tiny_224_BP4D_12for3.yaml


configs=("$config1" "$config2" "$config3")

#预训练权重
pretrain_path=pretrain_weights/vssm1_tiny_0230s_ckpt_epoch_264.pth

for i in ${!configs[@]}; do
    config=${configs[$i]}
    python -m torch.distributed.launch  --nnodes=1 --node_rank=0 --nproc_per_node=1  --master_addr="127.0.0.1" --master_port=29501 main.py --model_ema False --use_wandb --cfg $config --pretrained $pretrain_path
done
# 等待所有后台进程完成
wait
echo "所有实验已完成。"