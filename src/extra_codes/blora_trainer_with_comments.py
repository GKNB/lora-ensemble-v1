# Necessary Imports
import torch
import random
import os
import time
import sys
import json
import shutil
import numpy as np
import re
# import gc
import importlib
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch.nn as nn
from typing import Any
import peft
from datasets import Dataset
from peft import TaskType, get_peft_model, LoraConfig
from sklearn.model_selection import KFold, train_test_split
# from sklearn.metrics import accuracy_score, precision_score, f1_score, matthews_corrcoef, roc_auc_score, confusion_matrix
from torchmetrics.classification import (
    BinaryAccuracy,
    BinaryMatthewsCorrCoef,
    BinaryAUROC,
    BinaryConfusionMatrix,
    BinarySpecificity,
    MulticlassPrecision,
    MulticlassF1Score,
    BinaryCalibrationError,
)

from Bayesian_LoRA import train_and_evaluate_bayesian_lora

# from torchmetrics import Accuracy, CalibrationError
from transformers.modeling_outputs import ModelOutput

sys.path.append('/hpcgpfs01/work/sjantre/lora-ensemble-v1')
from bayesian_lora import (
    calculate_kronecker_factors,
    cholesky_decompose_small_factors,
    model_evidence,
    variance,
    stable_cholesky,
)

from bayesian_lora.main import jacobian_mean

class model_trainer():

    def init(self, model, tokenizer, args):

        print(args)
        # Initialize device and random seed
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.generator = set_seeds(args.seed)

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
        os.makedirs(self.lora_ensemble_tmp_dir, exist_ok=True)

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
        # max length is 425 for multishot (Pre-trained) Experiments

        # Loss values for plots
        self.train_losses = []
        self.valid_losses = []

        # Load Model & Tokenizer
        self.model = model
        self.base_model = model
        self.tokenizer = tokenizer
        self.tokenizer.pad_token = self.tokenizer.eos_token 

        # Flag for printing model outputs during testing
        self.testing = False
        print("Model: ", self.model)

    def log(self, message):
        # This function handles writing logs to the specified file
        try:
            with open(self.log_file_path, 'a') as f:
                f.write(message + '\n')
        except Exception as e:
            raise Exception(f"Failed to open and write to file: {self.log_file_path}") from e

    # # Load data prompts into JSON file
    # def load_data(self):
    #     # Append path to data loader
    #     os.chdir('data_processors')
    #     sys.path.append(os.getcwd())

    #     # Make call to data loader
    #     from data_loader import dataset_loader 
    #     loader = dataset_loader()
    #     loader.load_datasets()
    #     print("Done")

    # def save_fold_data(self, fold_name, train_dataset, valid_dataset=None, test_dataset=None):
    #     data = {'train': train_dataset.to_dict()}
    #     if valid_dataset:
    #         data['validation'] = valid_dataset.to_dict()
    #     if test_dataset:
    #         data['test'] = test_dataset.to_dict()

    #     # Save fold data as a JSON file
    #     fold_path = os.path.join(self.fold_dir, f'{fold_name}.json')
    #     os.makedirs(os.path.dirname(fold_path), exist_ok=True)
    #     with open(fold_path, 'w') as f:
    #         json.dump(data, f, indent=4)

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

    # def load_saved_model(self, model):
    #     # Evaluate the trained model on the test set
    #     self.testing = True
    #     self.base_model = model
    #     self.train_model()

    # def plot_losses(self, log_history):
    #     train_losses = []
    #     eval_losses = []
    #     epochs = []

    #     for entry in log_history:
    #         if 'loss' in entry and 'epoch' in entry:
    #             train_losses.append(entry['loss'])
    #             epochs.append(entry['epoch'])
    #         elif 'eval_loss' in entry and 'epoch' in entry:
    #             eval_losses.append(entry['eval_loss'])

    #     plt.figure(figsize=(10, 5))
    #     if train_losses:
    #         plt.plot(epochs[:len(train_losses)], train_losses, 'r-', label='Training Loss')
    #     if eval_losses:
    #         plt.plot(epochs[:len(eval_losses)], eval_losses, 'b-', label='Validation Loss')

    #     plt.xlabel('Epoch')
    #     plt.ylabel('Loss')
    #     plt.title(self.plot_title)
    #     plt.legend()
        
    #     # Check if the file exists and back it up
    #     if os.path.exists(self.plot_file_path):
    #         backup_path = self.plot_file_path + ".backup"
    #         shutil.copyfile(self.plot_file_path, backup_path)
    #         print(f"Existing file backed up as: {backup_path}")
    #     try:
    #         plt.savefig(self.plot_file_path)
    #         plt.show()
    #     except OSError as e:
    #         print(f"Error saving or showing plot: {e}. Plotting operation skipped.")


    def train_model(self, args):

        # def formatting_prompts_func(example):
        #     output_texts = []
        #     for i in range(len(example['question'])):

        #         # Llama 
        #         if "llama2" in self.model_name.lower() or "llama3" in self.model_name.lower():
        #             text = f"### Question: {example['question'][i]}\n### Answer: {example['answer'][i]}"

        #         # Mistral 
        #         if "mistral" in self.model_name.lower() or "mixtral" in self.model_name.lower():
        #             text = f"{self.tokenizer.eos_token}[INST]### Question: {example['question'][i]}[/INST]\n### Answer: {example['answer'][i]}"
                
        #         output_texts.append(text)
        #     return output_texts
        
        # def process_test_set(example):
        #     tokenized_inputs = {key: [] for key in ['input_ids', 'attention_mask', 'token_type_ids'] if key in self.tokenizer.model_input_names}

        #     for (q, a) in zip(example['question'], example['answer']):
        #         # Log and Debug
        #         self.count += 1
        #         self.log(f"Prompt {self.count}: {q}\nTrue Label: {a}\n")  
        #         #print(f"Prompt {i+1}: {q}\nTrue Label: {a}\n")

        #         # Format Prompt
        #         formatted_qa = f"### Question: {q}\n### Answer: {a}"
        #         if "mistral" in self.model_name.lower() or "mixtral" in self.model_name.lower():
        #             formatted_qa = f"{self.tokenizer.eos_token}[INST]### Question: {q}[/INST]\n### Answer: {a}"
                
        #         # Tokenize Immediately
        #         tokenized = self.tokenizer(
        #             formatted_qa,
        #             padding="max_length",
        #             truncation=True,
        #             max_length=self.max_length,
        #             return_tensors="pt"
        #         )
                
        #         # Convert batch encoding data structure to python list
        #         for key in tokenized_inputs.keys():
        #             tokenized_inputs[key].extend(tokenized[key].numpy().tolist())

        #     return tokenized_inputs

        # def compute_metrics(eval_preds):       

        #     torch.cuda.empty_cache()
        #     logits = eval_preds.predictions
        #     label_ids = eval_preds.label_ids

        #     # Convert logits from NumPy array to PyTorch tensor
        #     logits_tensor = torch.from_numpy(logits).to('cuda:0')

        #     # Split logits_tensor into smaller chunks
        #     chunk_size = 1
        #     num_chunks = (logits_tensor.size(0) + chunk_size - 1) // chunk_size

        #     # Initialize lists to store probabilities and predictions
        #     all_probabilities = []
        #     all_predictions = []

        #     # Convert the logits to probabilities in chunks for memory efficiency
        #     for i in range(num_chunks):
        #         start = i * chunk_size
        #         end = start + chunk_size
        #         chunk = logits_tensor[start:end]

        #         # Apply softmax on GPU for each chunk
        #         probabilities = F.softmax(chunk, dim=-1)
        #         predictions = torch.argmax(probabilities, dim=-1)

        #         # Append to lists
        #         all_probabilities.append(probabilities.cpu().numpy())
        #         all_predictions.append(predictions.cpu().numpy())

        #     # Concatenate probabilities and predictions
        #     probabilities = np.concatenate(all_probabilities)
        #     predictions = np.concatenate(all_predictions)

        #     # Filter out -100 values before decoding
        #     label_ids = [label[label != -100] for label in label_ids]

        #     # Pre-decode all necessary token IDs to text
        #     decoded_texts = self.tokenizer.batch_decode(predictions, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        #     decoded_labels = self.tokenizer.batch_decode(label_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)

        #     # Initialize true_labels and preds lists
        #     true_labels, preds = [], []
            
        #     # Process decoded texts to classify responses 
        #     i = 0
        #     for decoded_text, decoded_label in zip(decoded_texts, decoded_labels):

        #         # Normalize and find "yes" or "no"
        #         normalized_text = decoded_text.lower()
        #         normalized_label = decoded_label.lower()

        #         # Determine true label from decoded_label
        #         if "yes" in normalized_label: true_label = 1
        #         elif "no" in normalized_label: true_label = 0
        #         else:
        #             print("\nDEBUG\nTrue Label is: ", normalized_label)
        #             true_label = 0 # For debugging
        #         true_labels.append(true_label)

        #         # Determine predicted answer
        #         answer_start_idx = normalized_text.lower().find("answer: ") + len("answer: ")

        #         # Extract the substring starting from the answer
        #         answer_text = normalized_text[answer_start_idx:]

        #         # Parse the model output to get the first word after the "answer:"
        #         matches = re.findall(r'\byes\b|\bno\b', answer_text.lower())
        #         if matches: first_word = matches[0]  
        #         else:
        #             # If we cannot immediately find the answer in the first word, find the first word after a newline character
        #             answer_start_idx = normalized_text.lower().find("\n") + len("\n")

        #             # Extract the substring starting from the answer
        #             answer_text = normalized_text[answer_start_idx:]

        #             # Parse the model output to get the first word after the "answer:"
        #             matches = re.findall(r'\byes\b|\bno\b', answer_text)
        #             if matches: first_word = matches[0] 
        #             else: first_word = answer_text.split()[0]

        #         if "yes" in first_word.lower(): preds.append(1)
        #         elif "no" in first_word.lower(): preds.append(0)
        #         else: 
        #             # Append the opposite of the true label, checking for None
        #             if true_labels[i] == 0:
        #                 opposite_value = 1 
        #                 first_word = "yes"
        #             else:
        #                 opposite_value = 0
        #                 first_word = "no"
        #             preds.append(opposite_value)

        #         # Print model outputs
        #         if self.testing: 
        #             self.log(f"Model Prediction {i+1}: {first_word}")
        #         i += 1

        #     # Compute metrics
        #     accuracy = accuracy_score(true_labels, preds)
        #     mcc = matthews_corrcoef(true_labels, preds)
        #     auc = roc_auc_score(true_labels, preds)
        #     tn, fp, fn, tp = confusion_matrix(true_labels, preds).ravel()
        #     specificity = tn / (tn+fp)
        #     precision_macro = precision_score(true_labels, preds, average='macro')
        #     f1_macro = f1_score(true_labels, preds, average='macro')

        #     metrics = [("Accuracy", accuracy),
        #             ("MCC", mcc), 
        #             ("AUC", auc), 
        #             ("Specificity", specificity), 
        #             ("Macro Precision", precision_macro), 
        #             ("Macro F1 Score", f1_macro)]
            
        #     # Convert list of tuples into a dictionary
        #     metrics_dict = {metric_name: metric_value for metric_name, metric_value in metrics}

        #     # Print each metric name and value
        #     if self.testing:
        #         for metric_name, metric_value in metrics_dict.items():
        #             self.log(f"{metric_name}: {metric_value}")
        #     return metrics_dict
            
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