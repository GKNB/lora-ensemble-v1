#!/bin/bash
#SBATCH --job-name=Lucid      # create a short name for your job
#SBATCH --account=csigeneral           # csigeneral-v mlg-core oddr
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --cpus-per-task=8        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem=80G                # total memory per node (4 GB per cpu-core is default)
#SBATCH --partition=csi
#SBATCH --qos=csi
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00

##SBATCH --mail-type=BEGIN        # send email when job begins
##SBATCH --mail-type=END          # send email when job ends
##SBATCH --mail-user=sjantre@bnl.gov

#SBATCH --output=run_Bayesian-LoRA_%j.log

source ~/.bashrc
conda activate lucid

## Bayesian LoRA
python3 /hpcgpfs01/work/sjantre/lora-ensemble-v1/src/run_model_sanket.py \
    --model_name biomistral \
    --dataset 6 \
    --prior_var 0.1 \
    --seed 4 \
    --repo_dir /hpcgpfs01/work/sjantre/lora-ensemble-v1/ \
    --config /hpcgpfs01/work/sjantre/lora-ensemble-v1/src/config_sanket.json \
    --bayesian_lora \
    --run_every_step

# # LoRA Ensemble
# python3 /hpcgpfs01/work/sjantre/lora-ensemble-v1/src/run_model_sanket.py \
#     --model_name biomedgpt \
#     --dataset 6 \
#     --n_ensemble 3 \
#     --seed 6 \
#     --repo_dir /hpcgpfs01/work/sjantre/lora-ensemble-v1/ \
#     --config /hpcgpfs01/work/sjantre/lora-ensemble-v1/src/config_sanket.json \
#     --lora_ensemble \
#     --run_every_step

# Single LoRA
# python3 /hpcgpfs01/work/sjantre/lora-ensemble-v1/src/run_model_sanket.py \
#     --model_name llama3 \
#     --dataset 5 \
#     --seed 3 \
#     --ensemble_seed  4 \
#     --repo_dir /hpcgpfs01/work/sjantre/lora-ensemble-v1/ \
#     --config /hpcgpfs01/work/sjantre/lora-ensemble-v1/src/config_sanket.json \
#     --single_lora

conda deactivate