#!/bin/bash

#SBATCH --job-name=train_12
#SBATCH --account=project_2006362
#SBATCH --partition=gpusmall
#SBATCH --time=24:00:00

#SBATCH --nodes=1
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=32

#SBATCH --output=vttt_disfa_23for1_219.out
#SBATCH --error=vttt_disfa_23for1_219.eer

## Please remember to load the environment your application may need.
source ~/.bashrc
module load gcc/10.4.0
module load cuda/12.6.1
conda activate AUTTT

## And use the variable $LOCAL_SCRATCH in your batch job script 
## to access the local fast storage on each node.
srun bash slurm_train_script/slurm_disfa_torchrun_23for1.sh
