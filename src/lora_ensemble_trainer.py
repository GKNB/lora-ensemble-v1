# Necessary Imports
import torch
import os
import sys
import json
from datasets import Dataset

from util import set_seeds
from src.Lora_Ensemble import train_and_evaluate_lora_ensemble

sys.path.append('/hpcgpfs01/work/sjantre/lora-ensemble-v1')

class LoraEnsembleTrainer:
    def __init__(self, model, tokenizer, args):
        print(args)

        # Device and random seeds initialization
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.seeds = [args.seed + i * 10093 for i in range(args.n_ensemble)]
        self.generator = set_seeds(args.seed)

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
            "log_file": os.path.join(self.repo_dir, suffixes["lora_ensemble_suffix"], f"{self.model_name}-results.txt"),
            "lora_ensemble_models_dir": os.path.join(self.repo_dir, suffixes["lora_ensemble_models_suffix"], self.model_name),
        }

        os.makedirs(self.paths["lora_ensemble_models_dir"], exist_ok=True)

        # Clear previous log file
        with open(self.paths["log_file"], 'w') as file:
            file.write("")

    def load_train_test_data(self):
        """Loads and splits datasets."""
        with open(self.paths["train_data"], 'r') as file:
            train_dataset_dict = json.load(file)
        self.train_dataset = Dataset.from_dict(train_dataset_dict)

        with open(self.paths["test_data"], 'r') as file:
            test_dataset_dict = json.load(file)
        self.test_dataset = Dataset.from_dict(test_dataset_dict)

        print(f"Training samples: {len(self.train_dataset)}")
        print(f"Testing samples: {len(self.test_dataset)}")

    def train_model(self, args):
        """Trains the model using Bayesian LoRA."""
        uq_config = {
            "max_length": self.dataset_config["max_length"],
            "batch_size": self.dataset_config["batch_size"],
            "test_batch_size": self.dataset_config["test_batch_size"],
            "device": self.device,
            "seeds": self.seeds,
            "lr": 1e-4,
            "num_epochs": self.dataset_config["num_epochs"],
            "run_every_step": args.run_every_step,
            "use_tqdm": args.use_tqdm,
            "n_ensemble": args.n_ensemble,
            "log_file_path": self.paths["log_file"]
        }

        train_and_evaluate_lora_ensemble(
            self.train_dataset,
            self.test_dataset,
            self.paths["lora_ensemble_models_dir"],
            self.model,
            self.tokenizer,
            uq_config,
        )