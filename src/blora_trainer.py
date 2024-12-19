# Necessary Imports
import torch
import os
import sys
import json
from datasets import Dataset

from Bayesian_LoRA import train_and_evaluate_bayesian_lora

sys.path.append('/hpcgpfs01/work/sjantre/lora-ensemble-v1')

class BloraTrainer:
    def __init__(self, model, tokenizer, args):
        print(args)

        # Device and config initialization
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"

        with open(args.config, 'r') as f:
            self.config = json.load(f)

        # Model and dataset identifiers
        self.model_name = f"{args.model_name}-set-{args.dataset}-seed-{args.seed}"
        self.dataset_config = self.config["datasets"][args.dataset]

        # Directories and logging setup
        self.repo_dir = args.repo_dir
        self.setup_paths(args)

        # Load Model & Tokenizer
        self.model = model
        self.tokenizer = tokenizer
        self.tokenizer.pad_token = self.tokenizer.eos_token
        print("Model: ", self.model)

    def setup_paths(self, args):
        """Sets up file paths and directories."""
        suffixes = self.config
        self.paths = {
            "train_data": os.path.join(self.repo_dir, suffixes["datasets"][args.dataset]["train_data_path_suffix"]),
            "test_data": os.path.join(self.repo_dir, suffixes["datasets"][args.dataset]["test_data_path_suffix"]),
            "log_file": os.path.join(self.repo_dir, suffixes["bayesian_lora_suffix"], f"{self.model_name}-results.txt"),
            "bayesian_lora_models_dir": os.path.join(self.repo_dir, suffixes["bayesian_lora_models_suffix"], self.model_name),
        }

        os.makedirs(self.paths["bayesian_lora_models_dir"], exist_ok=True)

        # Clear previous log file
        with open(self.paths["log_file"], 'w') as file:
            file.write("")

    def load_train_test_data(self):
        """Loads and splits datasets."""
        with open(self.paths["train_data"], 'r') as file:
            train_dataset_dict = json.load(file)
        full_train_dataset = Dataset.from_dict(train_dataset_dict)
        train_valid_split = full_train_dataset.train_test_split(test_size=self.dataset_config["valid_split"], seed=1)

        self.train_dataset = train_valid_split["train"]
        self.valid_dataset = train_valid_split["test"]

        with open(self.paths["test_data"], 'r') as file:
            test_dataset_dict = json.load(file)
        self.test_dataset = Dataset.from_dict(test_dataset_dict)

        print(f"Training samples: {len(self.train_dataset)}")
        print(f"Validation samples: {len(self.valid_dataset)}")
        print(f"Testing samples: {len(self.test_dataset)}")

    def train_model(self, args):
        """Trains the model using Bayesian LoRA."""
        uq_config = {
            "max_length": self.dataset_config["max_length"],
            "batch_size": self.dataset_config["batch_size"],
            "test_batch_size": self.dataset_config["test_batch_size"],
            "device": self.device,
            "seed": args.seed,
            "lr": 1e-4,
            "num_epochs": self.dataset_config["num_epochs"],
            "run_every_step": args.run_every_step,
            "use_tqdm": args.use_tqdm,
            "prior_var": args.prior_var,
            "log_file_path": self.paths["log_file"]
        }

        train_and_evaluate_bayesian_lora(
            self.train_dataset,
            self.valid_dataset,
            self.test_dataset,
            self.paths["bayesian_lora_models_dir"],
            self.model,
            self.tokenizer,
            uq_config,
        )