import os
from transformers import AutoModelForCausalLM, AutoTokenizer

# Set the cache directory
hf_cache_dir = "/pscratch/sd/t/tianle/myWork/transformers/cache"

# Ensure that the TRANSFORMERS_CACHE environment variable is set
os.environ["TRANSFORMERS_CACHE"] = hf_cache_dir
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"


def load_model(model_name):
    # Define the model path as used in Hugging Face Hub
    model_paths = {
        # 'Llama2': "meta-llama/Llama-2-7b-chat-hf",
        'Llama2': r"{hf_cache_dir}/models--meta-llama--Llama-2-7b-chat-hf/snapshots/c1b0db933684edbfe29a06fa47eb19cc48025e93",

        'Llama3': "meta-llama/Meta-Llama-3-8B-Instruct", 
        #'Llama3': r"{hf_cache_dir}/models--meta-llama--Meta-Llama-3-8B-Instruct/snapshots/5f0b02c75b57c5855da9ae460ce51323ea669d8a",

         'Mistral': "mistralai/Mistral-7B-Instruct-v0.3"
        #'Mistral': r"{hf_cache_dir}/models--mistralai--Mistral-7B-Instruct-v0.2/snapshots/41b61a33a2483885c981aa79e0df6b32407ed873",
        } 

    model_path = model_paths.get(model_name)
    if model_path is None:
        raise ValueError("Invalid model name: " + model_name)
    
    if 'Mistral' in model_name:
        tokenizer = AutoTokenizer.from_pretrained(model_path, add_eos_token=True, add_bos_token=True, padding_side="left")
        model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto")

    elif 'Llama2' in model_name:
        tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side="left")
        model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto")

    elif 'Llama3' in model_name:
        tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side="left")
        model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto")

    else:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(model_path)
        
    print(f"{model_name} has been loaded successfully.")
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
    trainer.preprocess_data()

    # Train a Model
    trainer.train_model()

    # Load Saved Model
    # trainer.load_saved_model(model)

    # Use Pre-Trained Model
    # trainer.pretrained_model_inference()
