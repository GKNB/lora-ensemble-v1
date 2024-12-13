import os
import json
from transformers import AutoModelForCausalLM, AutoTokenizer

# Set the cache directory
hf_cache_dir = "/pscratch/sd/t/tianle/myWork/transformers/cache"

# Ensure that the TRANSFORMERS_CACHE environment variable is set
os.environ["TRANSFORMERS_CACHE"] = hf_cache_dir
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

def load_model(args):
    config = {}
    with open(args.config, 'r') as f:
        config = json.load(f)
    if args.use_model_snapshot:
        model_path = os.path.join(r"{hf_cache_dir}", config["models"][args.model_name]["snapshot"])
    else:
        model_path = config["models"][args.model_name]["path"]

    if 'Mistral' in args.model_name:
        tokenizer = AutoTokenizer.from_pretrained(model_path, add_eos_token=True, add_bos_token=True, padding_side="left")
        model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto")

    elif 'Llama2' in args.model_name:
        tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side="left")
        model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto")

    elif 'Llama3' in args.model_name:
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
    
    # Use this code to analyze the results
    # from data_processors.result_analysis import result_analysis
    # analyzer = result_analysis()
    # analyzer.init(file_name="Mistral-7B-set-6_fold_3_results")
    # analyzer.run()
    # quit()

    # Use this code to conduct the cross reference analysis
    # from data_processors.cross_reference_analysis import cross_reference_analysis
    # analyzer = cross_reference_analysis()
    # analyzer.run()
    # quit()

    # Use this code to load each dataset and analyze the distribution
    # from data_processors.data_loader import dataset_loader
    # loader = dataset_loader()
    # loader.run()
    # quit()

    # Load model
    parser.add_argument('--model_name', type=str, required=True)
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--repo_dir', type=str, required=True)
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--n_ensemble', type=int, required=True)
    parser.add_argument('--seed', type=int, required=True)
    args = parser.parse_args() 
    model, tokenizer = load_model(args.model_name)

    # Initialize trainer class
    from model_trainer import model_trainer
    trainer = model_trainer()
    trainer.init(model, tokenizer, args)

    # Load data prompts into JSON file
    # trainer.load_data()

    # Load prompts from JSON file into Datasets
    trainer.load_train_test_data()

    # Train a Model
    trainer.train_model()

    # Load Saved Model
    # trainer.load_saved_model(model)

    # Use Pre-Trained Model
    # trainer.pretrained_model_inference()
