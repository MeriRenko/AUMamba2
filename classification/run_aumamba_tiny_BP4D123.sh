CUDA_LAUNCH_BLOCKING=1
python -m torch.distributed.run --nnodes=1 --node_rank=0 --nproc_per_node=1 main.py --use_wandb --cfg configs/vssm/vmambav2v_tiny_224_BP4D_12for3.yaml --model_ema False --pretrained pretrain_weights/vssm1_tiny_0230s_ckpt_epoch_264.pth

# output/bi_vttt_small_nd/disfa_23for1_h6_m14_2waycat_passconv_219_ft_mixup/ckpt_epoch_18.pth