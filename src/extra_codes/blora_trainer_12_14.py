# Necessary Imports
import torch
import os
import sys
import json
from datasets import Dataset

from Bayesian_LoRA import train_and_evaluate_bayesian_lora

sys.path.append('/hpcgpfs01/work/sjantre/lora-ensemble-v1')

class model_trainer():

    def init(self, model, tokenizer, args):

        print(args)
        # Initialize device and random seed
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"

        with open(args.config, 'r') as f:
            self.config = json.load(f)

        self.model_name = self.config["models"][args.model_name]["name"] + "-set-" + args.dataset

        self.json_file_path_train = os.path.join(args.repo_dir, self.config["datasets"][args.dataset]["train_data_path_suffix"])
        self.json_file_path_test  = os.path.join(args.repo_dir, self.config["datasets"][args.dataset]["test_data_path_suffix"])
        self.output_dir = os.path.join(args.repo_dir, self.config["output_dir_suffix"], self.model_name)
        self.fold_dir = os.path.join(args.repo_dir, self.config["fold_dir_suffix"], self.model_name)
        self.log_file_path = os.path.join(args.repo_dir, self.config["experiments_dir_suffix"], f"{self.model_name}-seed-{args.seed}-results.txt")
        self.plot_file_path = os.path.join(args.repo_dir, self.config["experiments_dir_suffix"], f"{self.model_name}-losses.png")
        self.plot_title = f"Loss values for {self.model_name}"

        self.bayesian_lora_tmp_dir = os.path.join(args.repo_dir, self.config["bayesian_lora_tmp_suffix"], self.model_name)
        os.makedirs(self.bayesian_lora_tmp_dir, exist_ok=True)

        # Open the log file in write mode, this will clear previous contents 
        with open(self.log_file_path, 'w') as file: 
            file.write("") 

        # Hyperparameters 
        self.lr = 1e-4 # Learning rate remains the same for all experiments
        self.new_tokens = 5  # New tokens remains the same for all experiments
        self.num_epochs = self.config["datasets"][args.dataset]["num_epochs"]
        self.batch_size = self.config["datasets"][args.dataset]["batch_size"]
        self.max_length = self.config["datasets"][args.dataset]["max_length"]
        print(f"self.num_epochs = {self.num_epochs}, self.batch_size = {self.batch_size}, self.max_length = {self.max_length}")

        # Load Model & Tokenizer
        self.model = model
        self.base_model = model
        self.tokenizer = tokenizer
        self.tokenizer.pad_token = self.tokenizer.eos_token 

        # Flag for printing model outputs during testing
        self.testing = False
        print("Model: ", self.model)

    def load_train_test_data(self, args):
        with open(self.json_file_path_train, 'r') as file:
            train_dataset_dict = json.load(file)
        full_train_dataset = Dataset.from_dict(train_dataset_dict)

        # Split the dataset into training and validation sets
        # Example: 80% training, 20% validation
        train_valid_split = full_train_dataset.train_test_split(test_size=args.validation_split, seed=args.seed)

        # Access the train and validation datasets
        self.train_dataset = train_valid_split['train']
        self.valid_dataset = train_valid_split['test']

        with open(self.json_file_path_test, 'r') as file:
            test_dataset_dict = json.load(file)
        self.test_dataset  = Dataset.from_dict(test_dataset_dict)

        # Print the sizes of each split to verify
        print(f"Training samples: {len(self.train_dataset)}")
        print(f"Validation samples: {len(self.valid_dataset)}")
        print(f"Testing samples: {len(self.test_dataset)}")

    def train_model(self, args):
            
        uq_config = {}
        uq_config["max_length"]     = self.max_length
        uq_config["batch_size"]     = self.batch_size
        uq_config["seed"]           = args.seed
        uq_config["device"]         = self.device
        uq_config["lr"]             = self.lr
        uq_config["num_epochs"]     = self.num_epochs
        uq_config["run_every_step"] = self.run_every_step
        uq_config["use_tqdm"]       = self.use_tqdm
        uq_config["prior_var"]      = args.prior_var

        train_and_evaluate_bayesian_lora(self.train_dataset, self.valid_dataset, self.test_dataset, self.bayesian_lora_tmp_dir, self.model, self.tokenizer, uq_config)