#!/bin/bash
export WANDB_API_KEY='16e6b55945dd4bd08f7b789d4d2996fed8783355'
export WANDB__SERVICE_WAIT='300'

# for deepspeed
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
# export MASTER_ADDR="g1101"
# export SLURM_NNODES=1
# export SLURM_PROCID=0
echo "========== Environment Variables =========="
echo "MASTER_ADDR    = $MASTER_ADDR"
echo "SLURM_NNODES   = $SLURM_NNODES"
echo "SLURM_PROCID   = $SLURM_PROCID"
echo "==========================================="

export MASTER_PORT=29100
export DEFAULT_GPUS_PER_NODE=4


# 检测GPU数量（这里仅为示例，具体实现可能需要根据你的集群环境调整）
GPUS_PER_NODE=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)
# 如果检测到的GPU数量不合理（例如，没有设置CUDA_VISIBLE_DEVICES），则使用默认值
if [ "$GPUS_PER_NODE" -le 0 ]; then
  GPUS_PER_NODE=$DEFAULT_GPUS_PER_NODE
fi

# GPUS_PER_NODE=$DEFAULT_GPUS_PER_NODE

export LAUNCHER="python -m torch.distributed.run \
    --nproc_per_node $GPUS_PER_NODE \
    --nnodes $SLURM_NNODES \
    --node_rank $SLURM_PROCID \
    --rdzv_endpoint $MASTER_ADDR:$MASTER_PORT \
    --rdzv_backend c10d \
    --max_restarts 3 \
    --tee 2 \
    "

export Stage1_CMD=" \
    main.py \
    --cfg configs/bi_ttt/bi_vttt_small_224_DISFA_12for3_week.yaml \
    --model_ema False \
    --tag disfa_12for3_h6_m14_2waycat_passconv_219_ft_week \
    --pretrained checkpoint/imagenet1k_preconv_norm_passconv/ckpt_epoch_219.pth \
    --use_wandb
    "

bash -c "$LAUNCHER $Stage1_CMD"

