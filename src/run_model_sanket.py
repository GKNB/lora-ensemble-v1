import os
import sys
import json
from transformers import AutoModelForCausalLM, AutoTokenizer

# Set the cache directory
hf_cache_dir = "/hpcgpfs01/work/sjantre/.cache/huggingface"

# Ensure that the TRANSFORMERS_CACHE environment variable is set
os.environ["TRANSFORMERS_CACHE"] = hf_cache_dir
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

sys.path.append('/hpcgpfs01/work/sjantre/lora-ensemble-v1')

def load_model(args):
    config = {}
    with open(args.config, 'r') as f:
        config = json.load(f)
    
    model_path = config["models"][args.model_name]
    
    if 'mistral' in args.model_name.lower():
        tokenizer = AutoTokenizer.from_pretrained(model_path, add_eos_token=True, add_bos_token=True, padding_side="left")
        model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto")

    elif 'llama2' in args.model_name.lower():
        tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side="left")
        model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto")

    elif 'llama3' in args.model_name.lower():
        tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side="left")
        model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto")
        
    elif 'biomedgpt' in args.model_name.lower():
        tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side="left")
        model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto")

    else:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(model_path)
        
    print(f"{args.model_name} has been loaded successfully.")
    return model, tokenizer


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()

    # Load model
    parser.add_argument('--model_name', type=str, required=True)
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--repo_dir', type=str, required=True)
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--run_every_step', default=False, action='store_true')
    parser.add_argument('--use_tqdm', default=False, action='store_true')
    
    parser.add_argument('--prior_var', default=0.1, type=float)
    parser.add_argument('--n_ensemble', default=3, type=int)
    parser.add_argument('--ensemble_seed', default=1, type=int)

    parser.add_argument('--bayesian_lora', default=False, action='store_true')
    parser.add_argument('--lora_ensemble', default=False, action='store_true')
    parser.add_argument('--single_lora', default=False, action='store_true')

    args = parser.parse_args()
    model, tokenizer = load_model(args)

    # Initialize trainer class
    if args.bayesian_lora:
        from blora_trainer import BloraTrainer
        trainer = BloraTrainer(model, tokenizer, args)
    elif args.lora_ensemble:
        from lora_ensemble_trainer import LoraEnsembleTrainer
        trainer = LoraEnsembleTrainer(model, tokenizer, args)
    elif args.single_lora:
        from single_lora_trainer import SingleLoraTrainer
        trainer = SingleLoraTrainer(model, tokenizer, args)
    
    trainer.load_train_test_data()

    # Train a Model
    trainer.train_model(args)