import os

# Set the cache directory
hf_cache_dir = "/hpcgpfs01/scratch/rengel/.cache/huggingface"

# Ensure that the TRANSFORMERS_CACHE environment variable is set
os.environ["TRANSFORMERS_CACHE"] = hf_cache_dir
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

from transformers import AutoModelForCausalLM, AutoTokenizer, GPT2LMHeadModel, BioGptTokenizer, BioGptForCausalLM


def load_model(model_name):
    # Define the model path as used in Hugging Face Hub
    model_paths = {
        #'Llama2': "meta-llama/Llama-2-7b-chat-hf",
        'Llama2': "/hpcgpfs01/scratch/rengel/.cache/huggingface/models--meta-llama--Llama-2-7b-chat-hf/snapshots/c1b0db933684edbfe29a06fa47eb19cc48025e93",

        #'Galactica': "facebook/galactica-6.7b", 
        'Galactica': "/hpcgpfs01/scratch/rengel/.cache/huggingface/models--facebook--galactica-6.7b/snapshots/3a85eca7b4eff2d76ed6b3fa3e287940eed85b10",

        #'Falcon': "tiiuae/falcon-7b",
        'Falcon': "/hpcgpfs01/scratch/rengel/.cache/huggingface/models--tiiuae--falcon-7b/snapshots/898df1396f35e447d5fe44e0a3ccaaaa69f30d36",

        #'MPT': "mosaicml/mpt-7b",
        'MPT': "/hpcgpfs01/scratch/rengel/.cache/huggingface/models--mosaicml--mpt-7b/snapshots/ada218f9a93b5f1c6dce48a4cc9ff01fcba431e7",

        #'BioGPT': "microsoft/BioGPT-Large",
        'BioGPT': "/hpcgpfs01/scratch/rengel/.cache/huggingface/models--microsoft--BioGPT-Large/snapshots/c6a5136a91c5e3150d9f05ab9d33927a3210a22e",

        #'BioMedLM': "stanford-crfm/BioMedLM"
        'BioMedLM': "/hpcgpfs01/scratch/rengel/.cache/huggingface/models--stanford-crfm--BioMedLM/snapshots/cad400dd9e158cac5e3c71a7b5c407c62e76202c"

    }
    


    model_path = model_paths.get(model_name)
    if model_path is None:
        raise ValueError("Invalid model name: " + model_name)
    
    if 'BioGPT' in model_name:
        tokenizer = BioGptTokenizer.from_pretrained(model_path)
        model = BioGptForCausalLM.from_pretrained(model_path)
    elif 'BioMedLM' in model_name:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = GPT2LMHeadModel.from_pretrained(model_path)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_path, use_auth_token="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")
        model = AutoModelForCausalLM.from_pretrained(model_path, use_auth_token="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")
    
    print(f"{model_name} has been loaded successfully.")
    return model, tokenizer

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    
    # Load model
    parser.add_argument('--model_name', required=False)
    args = parser.parse_args()
    model, tokenizer = load_model(args.model_name)

    num_parameters = model.num_parameters()
    print(f"Model size: {num_parameters}")  

    # Setup the initial training environment
    from data_trainers.necare_trainer_v2 import necare_trainer
    trainer = necare_trainer()
    trainer.init()
    trainer.load_peft_config(model, tokenizer)
    trainer.load_dataset()
    trainer.preprocess_data()
    trainer.create_folds() 

    # Perform training and inference of each data fold
    # for fold in range(5):
    #     # Process the data fold
    #     trainer.process_fold(fold)

    #     # Load new model for next iteration, but skip on the last iteration
    #     if fold < 4:  
    #         model, tokenizer = load_model(args.model_name)
    #         print("Model Loaded Successfully...")
    #         trainer.load_peft_config(model, tokenizer)

    # Save the best model
    #best_model_filepath = f"/hpcgpfs01/scratch/rengel/.cache/huggingface/best_model_{args.model_name}"

    # File path for the loaded model
    best_model_filepath = f"/hpcgpfs01/scratch/rengel/.cache/huggingface/best_model_{args.model_name}_4"

    trainer.evaluate_best_model(best_model_filepath)

