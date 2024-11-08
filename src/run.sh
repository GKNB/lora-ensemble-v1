#!/bin/bash

export CUDA_VISIBLE_DEVICES=0 
python3 /pscratch/sd/t/tianle/lucid/other_source/SURP_2024/src/run_model.py --model_name Llama3 --n_ensemble 10
