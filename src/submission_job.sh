#!/bin/bash
#SBATCH -A m2616_g
#SBATCH -C gpu
##SBATCH -C "gpu&hbm80g"
#SBATCH -q premium
#SBATCH -t 3:00:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH -c 128
#SBATCH --gpus-per-task=1
#SBATCH --gpu-bind=none

export SLURM_CPU_BIND="cores"

echo "Job is running on partition: $SLURM_JOB_PARTITION"
echo "Job is running under account: $SLURM_JOB_ACCOUNT"

source /global/homes/t/tianle/useful_script/conda_surp_2024
cd /pscratch/sd/t/tianle/lucid/other_source/SURP_2024/src

#Uncomment this if device does not match
#export CUDA_VISIBLE_DEVICES=0


#Before submitting job, make sure the following are set correctly:
#Model name argument
#Model name + dataset name in model_trainer.py
#json file name in model_trainer.py
#epoch, max_length, batch_size in model_trainer.py

srun python3 /pscratch/sd/t/tianle/lucid/other_source/SURP_2024/src/run_model.py --model_name Llama3 --n_ensemble 5 --seed 237
