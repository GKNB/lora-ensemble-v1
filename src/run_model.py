import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GPT2LMHeadModel, BioGptTokenizer, BioGptForCausalLM, BitsAndBytesConfig

# Set the cache directory
hf_cache_dir = "/hpcgpfs01/scratch/rengel/.cache/huggingface"

# Ensure that the TRANSFORMERS_CACHE environment variable is set
os.environ["TRANSFORMERS_CACHE"] = hf_cache_dir
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"


def load_model(model_name):
    # Define the model path as used in Hugging Face Hub
    model_paths = {
        # 'Llama2': "meta-llama/Llama-2-70b-chat-hf",
        # 'Llama2': "/hpcgpfs01/scratch/rengel/.cache/huggingface/models--meta-llama--Llama-2-70b-chat-hf/snapshots/e1ce257bd76895e0864f3b4d6c7ed3c4cdec93e2",
        'Llama2': "/hpcgpfs01/scratch/rengel/.cache/huggingface/models--meta-llama--Llama-2-7b-chat-hf/snapshots/c1b0db933684edbfe29a06fa47eb19cc48025e93",

        # 'Llama3': "meta-llama/Meta-Llama-3-8B-Instruct",
        'Llama3': "/hpcgpfs01/scratch/rengel/.cache/huggingface/models--meta-llama--Meta-Llama-3-8B-Instruct/snapshots/1448453bdb895762499deb4176c1dd83b145fac1",

        # 'Galactica': "facebook/galactica-30b", 
        # 'Galactica': "/hpcgpfs01/scratch/rengel/.cache/huggingface/models--facebook--galactica-30b/snapshots/80bd55898b06c7c363c467dec877b8b32702a2c4",
        # 'Galactica': "/hpcgpfs01/scratch/rengel/.cache/huggingface/models--facebook--galactica-6.7b/snapshots/3a85eca7b4eff2d76ed6b3fa3e287940eed85b10",

        # 'Falcon': "tiiuae/falcon-7b",
        # 'Falcon': "/hpcgpfs01/scratch/rengel/.cache/huggingface/models--tiiuae--falcon-7b/snapshots/898df1396f35e447d5fe44e0a3ccaaaa69f30d36",

        # 'MPT': "mosaicml/mpt-7b",
        # 'MPT': "/hpcgpfs01/scratch/rengel/.cache/huggingface/models--mosaicml--mpt-7b/snapshots/ada218f9a93b5f1c6dce48a4cc9ff01fcba431e7",

        # 'BioGPT': "microsoft/BioGPT-Large",
        # 'BioGPT': "/hpcgpfs01/scratch/rengel/.cache/huggingface/models--microsoft--BioGPT-Large/snapshots/c6a5136a91c5e3150d9f05ab9d33927a3210a22e",

        # 'BioMedLM': "stanford-crfm/BioMedLM"
        # 'BioMedLM': "/hpcgpfs01/scratch/rengel/.cache/huggingface/models--stanford-crfm--BioMedLM/snapshots/cad400dd9e158cac5e3c71a7b5c407c62e76202c",

        # 'Solar': "Upstage/SOLAR-10.7B-v1.0"
        # 'Solar': "/hpcgpfs01/scratch/rengel/.cache/huggingface/models--Upstage--SOLAR-10.7B-v1.0/snapshots/399afd2ff676489c2712feb0f92286a77b8d0cd5",

        # 'Mistral': "mistralai/Mistral-7B-Instruct-v0.2"
        'Mistral': "/hpcgpfs01/scratch/rengel/.cache/huggingface/models--mistralai--Mistral-7B-Instruct-v0.2/snapshots/41b61a33a2483885c981aa79e0df6b32407ed873",

        # 'Mistral': "mistralai/Mixtral-8x7B-Instruct-v0.1"
        'Mixtral': "/hpcgpfs01/scratch/rengel/.cache/huggingface/models--mistralai--Mixtral-8x7B-Instruct-v0.1/snapshots/125c431e2ff41a156b9f9076f744d2f35dd6e67a",
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
        
    elif 'Mistral' in model_name:
        tokenizer = AutoTokenizer.from_pretrained(model_path, token="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", add_eos_token=True, add_bos_token=True, padding_side="left")
        model = AutoModelForCausalLM.from_pretrained(model_path, token="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", device_map="auto")

    elif 'Mixtral' in model_name:
        # Quantize model         
        # https://huggingface.co/blog/4bit-transformers-bitsandbytes
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16
        )
        model = AutoModelForCausalLM.from_pretrained(model_path, quantization_config=bnb_config, device_map="auto")
        tokenizer = AutoTokenizer.from_pretrained(model_path, token="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", add_eos_token=True, add_bos_token=True, padding_side="left")

    elif 'Llama2' in model_name:
        tokenizer = AutoTokenizer.from_pretrained(model_path, token="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", padding_side="left")
        model = AutoModelForCausalLM.from_pretrained(model_path, token="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", device_map="auto")

    elif 'Llama3' in model_name:
        # Quantize model         
        # https://huggingface.co/blog/4bit-transformers-bitsandbytes
        # bnb_config = BitsAndBytesConfig(
        #     load_in_4bit=True,
        #     bnb_4bit_use_double_quant=True, # Try removing double quantization
        #     bnb_4bit_compute_dtype=torch.bfloat16
        # )
        # model = AutoModelForCausalLM.from_pretrained(model_path, quantization_config=bnb_config, device_map="auto")
        model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto")
        tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side="left")
       

    else:
        tokenizer = AutoTokenizer.from_pretrained(model_path, token="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")
        model = AutoModelForCausalLM.from_pretrained(model_path, token="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")
        
    print(f"{model_name} has been loaded successfully.")
    return model, tokenizer




if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()

    
    # Here we will run a scirpt to analyze results
    # from data_processors.result_analysis import result_analysis
    # analyzer = result_analysis() 
    # analyzer.init(file_name="Llama3-8B-set-6_results")
    # analyzer.run()
    # quit()

    # from data_processors.cross_reference_analysis import cross_reference_analysis
    # analyzer = cross_reference_analysis() 
    # analyzer.run()
    # quit()
 

    # Load model
    parser.add_argument('--model_name', required=False)
    args = parser.parse_args()
    model, tokenizer = load_model(args.model_name)

    # Calculate the total number of parameters in the model
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # Print the total number of parameters
    print(f'Total Number of Parameters: {total_params}')

    # Initialize trainer class
    from model_trainers.SURP_2024.model_trainer import model_trainer
    trainer = model_trainer()
    trainer.init(model, tokenizer)

    # Use this script to load data prompts inot json file
    # trainer.load_data()

    # Load data from json and use Datasets library
    trainer.preprocess_data()

    # Either train a model, or use pretrained model
    trainer.train_model()
    # trainer.pretrained_model_inference()


