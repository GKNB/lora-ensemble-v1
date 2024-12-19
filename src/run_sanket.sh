#!/bin/bash

export CUDA_VISIBLE_DEVICES=0 
python3 /hpcgpfs01/work/sjantre/lora-ensemble-v1/src/run_model.py --model_name llama3 #--use_tqdm

# python3 /hpcgpfs01/work/sjantre/lora-ensemble-v1/src/run_model.py --model_name phi2 --blora --use_tqdm #--run_every_step

# python3 /hpcgpfs01/work/sjantre/lora-ensemble-v1/src/run_model.py --model_name llama2 --blora --use_tqdm --run_every_step