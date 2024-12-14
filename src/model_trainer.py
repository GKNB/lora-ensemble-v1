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
import gc
import importlib
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch.nn as nn
from datasets import Dataset
from trl import SFTTrainer 
from sklearn.metrics import accuracy_score, precision_score, f1_score, matthews_corrcoef, roc_auc_score, confusion_matrix

from util import set_seeds
from util import custom_collate_fn
from lora_ensemble import train_and_evaluate_lora_ensemble 

class model_trainer():

    def init(self, model, tokenizer, args):

        print(args)
        # Initialize device and random seed
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.n_ensemble = args.n_ensemble
        self.seeds = [args.seed + i * 10093 for i in range(self.n_ensemble)]
        self.generator = set_seeds(args.seed)

        with open(args.config, 'r') as f:
            self.config = json.load(f)
        # os.environ["WANDB_DISABLED"] = "true" 

        self.model_name = self.config["models"][args.model_name]["name"] + "-set-" + args.dataset
        self.json_file_path_train = os.path.join(args.repo_dir, self.config["datasets"][args.dataset]["train_data_path_suffix"])
        self.json_file_path_test  = os.path.join(args.repo_dir, self.config["datasets"][args.dataset]["test_data_path_suffix"])
        self.output_dir = os.path.join(args.repo_dir, self.config["output_dir_suffix"], self.model_name)
        self.fold_dir = os.path.join(args.repo_dir, self.config["fold_dir_suffix"], self.model_name)
        self.log_file_path = os.path.join(args.repo_dir, self.config["experiments_dir_suffix"], f"{self.model_name}-results.txt")
        self.plot_file_path = os.path.join(args.repo_dir, self.config["experiments_dir_suffix"], f"{self.model_name}-losses.png")
        self.plot_title = f"Loss values for {self.model_name}"
 
        self.lora_ensemble_tmp_dir = os.path.join(args.repo_dir, self.config["lora_ensemble_tmp_dir_suffix"], self.model_name)
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


    # Load data prompts into JSON file
    def load_data(self):
        # Append path to data loader
        os.chdir('data_processors')
        sys.path.append(os.getcwd())

        # Make call to data loader
        from data_loader import dataset_loader 
        loader = dataset_loader()
        loader.load_datasets()
        print("Done")


    def save_fold_data(self, fold_name, train_dataset, valid_dataset=None, test_dataset=None):
        data = {'train': train_dataset.to_dict()}
        if valid_dataset:
            data['validation'] = valid_dataset.to_dict()
        if test_dataset:
            data['test'] = test_dataset.to_dict()

        # Save fold data as a JSON file
        fold_path = os.path.join(self.fold_dir, f'{fold_name}.json')
        os.makedirs(os.path.dirname(fold_path), exist_ok=True)
        with open(fold_path, 'w') as f:
            json.dump(data, f, indent=4)



    def load_train_test_data(self):
        with open(self.json_file_path_train, 'r') as file:
            train_dataset_dict = json.load(file)
        self.train_dataset = Dataset.from_dict(train_dataset_dict)
        print("Train dataset looks like: ", self.train_dataset)

        with open(self.json_file_path_test, 'r') as file:
            test_dataset_dict = json.load(file)
        self.test_dataset  = Dataset.from_dict(test_dataset_dict)
        print("Test dataset looks like: ", self.test_dataset)

    def load_saved_model(self, model):
        # Evaluate the trained model on the test set
        self.testing = True
        self.base_model = model
        self.train_model()


    def plot_losses(self, log_history):
        train_losses = []
        eval_losses = []
        epochs = []

        for entry in log_history:
            if 'loss' in entry and 'epoch' in entry:
                train_losses.append(entry['loss'])
                epochs.append(entry['epoch'])
            elif 'eval_loss' in entry and 'epoch' in entry:
                eval_losses.append(entry['eval_loss'])

        plt.figure(figsize=(10, 5))
        if train_losses:
            plt.plot(epochs[:len(train_losses)], train_losses, 'r-', label='Training Loss')
        if eval_losses:
            plt.plot(epochs[:len(eval_losses)], eval_losses, 'b-', label='Validation Loss')

        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title(self.plot_title)
        plt.legend()
        
        # Check if the file exists and back it up
        if os.path.exists(self.plot_file_path):
            backup_path = self.plot_file_path + ".backup"
            shutil.copyfile(self.plot_file_path, backup_path)
            print(f"Existing file backed up as: {backup_path}")
        try:
            plt.savefig(self.plot_file_path)
            plt.show()
        except OSError as e:
            print(f"Error saving or showing plot: {e}. Plotting operation skipped.")


    def train_model(self):

        def formatting_prompts_func(example):
            output_texts = []
            for i in range(len(example['question'])):

                # Llama 
                if "llama2" in self.model_name.lower() or "llama3" in self.model_name.lower():
                    text = f"### Question: {example['question'][i]}\n### Answer: {example['answer'][i]}"

                # Mistral 
                if "mistral" in self.model_name.lower() or "mixtral" in self.model_name.lower():
                    text = f"{self.tokenizer.eos_token}[INST]### Question: {example['question'][i]}[/INST]\n### Answer: {example['answer'][i]}"
                
                output_texts.append(text)
            return output_texts
        
        def process_test_set(example):
            tokenized_inputs = {key: [] for key in ['input_ids', 'attention_mask', 'token_type_ids'] if key in self.tokenizer.model_input_names}

            for (q, a) in zip(example['question'], example['answer']):
                # Log and Debug
                self.count += 1
                self.log(f"Prompt {self.count}: {q}\nTrue Label: {a}\n")  
                #print(f"Prompt {i+1}: {q}\nTrue Label: {a}\n")

                # Format Prompt
                formatted_qa = f"### Question: {q}\n### Answer: {a}"
                if "mistral" in self.model_name.lower() or "mixtral" in self.model_name.lower():
                    formatted_qa = f"{self.tokenizer.eos_token}[INST]### Question: {q}[/INST]\n### Answer: {a}"
                
                # Tokenize Immediately
                tokenized = self.tokenizer(
                    formatted_qa,
                    padding="max_length",
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt"
                )
                
                # Convert batch encoding data structure to python list
                for key in tokenized_inputs.keys():
                    tokenized_inputs[key].extend(tokenized[key].numpy().tolist())

            return tokenized_inputs

        def compute_metrics(eval_preds):       

            torch.cuda.empty_cache()
            logits = eval_preds.predictions
            label_ids = eval_preds.label_ids

            # Convert logits from NumPy array to PyTorch tensor
            logits_tensor = torch.from_numpy(logits).to('cuda:0')

            # Split logits_tensor into smaller chunks
            chunk_size = 1
            num_chunks = (logits_tensor.size(0) + chunk_size - 1) // chunk_size

            # Initialize lists to store probabilities and predictions
            all_probabilities = []
            all_predictions = []

            # Convert the logits to probabilities in chunks for memory efficiency
            for i in range(num_chunks):
                start = i * chunk_size
                end = start + chunk_size
                chunk = logits_tensor[start:end]

                # Apply softmax on GPU for each chunk
                probabilities = F.softmax(chunk, dim=-1)
                predictions = torch.argmax(probabilities, dim=-1)

                # Append to lists
                all_probabilities.append(probabilities.cpu().numpy())
                all_predictions.append(predictions.cpu().numpy())

            # Concatenate probabilities and predictions
            probabilities = np.concatenate(all_probabilities)
            predictions = np.concatenate(all_predictions)

            # Filter out -100 values before decoding
            label_ids = [label[label != -100] for label in label_ids]

            # Pre-decode all necessary token IDs to text
            decoded_texts = self.tokenizer.batch_decode(predictions, skip_special_tokens=True, clean_up_tokenization_spaces=True)
            decoded_labels = self.tokenizer.batch_decode(label_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)

            # Initialize true_labels and preds lists
            true_labels, preds = [], []
            
            # Process decoded texts to classify responses 
            i = 0
            for decoded_text, decoded_label in zip(decoded_texts, decoded_labels):

                # Normalize and find "yes" or "no"
                normalized_text = decoded_text.lower()
                normalized_label = decoded_label.lower()

                # Determine true label from decoded_label
                if "yes" in normalized_label: true_label = 1
                elif "no" in normalized_label: true_label = 0
                else:
                    print("\nDEBUG\nTrue Label is: ", normalized_label)
                    true_label = 0 # For debugging
                true_labels.append(true_label)

                # Determine predicted answer
                answer_start_idx = normalized_text.lower().find("answer: ") + len("answer: ")

                # Extract the substring starting from the answer
                answer_text = normalized_text[answer_start_idx:]

                # Parse the model output to get the first word after the "answer:"
                matches = re.findall(r'\byes\b|\bno\b', answer_text.lower())
                if matches: first_word = matches[0]  
                else:
                    # If we cannot immediately find the answer in the first word, find the first word after a newline character
                    answer_start_idx = normalized_text.lower().find("\n") + len("\n")

                    # Extract the substring starting from the answer
                    answer_text = normalized_text[answer_start_idx:]

                    # Parse the model output to get the first word after the "answer:"
                    matches = re.findall(r'\byes\b|\bno\b', answer_text)
                    if matches: first_word = matches[0] 
                    else: first_word = answer_text.split()[0]

                if "yes" in first_word.lower(): preds.append(1)
                elif "no" in first_word.lower(): preds.append(0)
                else: 
                    # Append the opposite of the true label, checking for None
                    if true_labels[i] == 0:
                        opposite_value = 1 
                        first_word = "yes"
                    else:
                        opposite_value = 0
                        first_word = "no"
                    preds.append(opposite_value)

                # Print model outputs
                if self.testing: 
                    self.log(f"Model Prediction {i+1}: {first_word}")
                i += 1

            # Compute metrics
            accuracy = accuracy_score(true_labels, preds)
            mcc = matthews_corrcoef(true_labels, preds)
            auc = roc_auc_score(true_labels, preds)
            tn, fp, fn, tp = confusion_matrix(true_labels, preds).ravel()
            specificity = tn / (tn+fp)
            precision_macro = precision_score(true_labels, preds, average='macro')
            f1_macro = f1_score(true_labels, preds, average='macro')

            metrics = [("Accuracy", accuracy), 
                    ("MCC", mcc), 
                    ("AUC", auc), 
                    ("Specificity", specificity), 
                    ("Macro Precision", precision_macro), 
                    ("Macro F1 Score", f1_macro)]
            
            # Convert list of tuples into a dictionary
            metrics_dict = {metric_name: metric_value for metric_name, metric_value in metrics}

            # Print each metric name and value
            if self.testing:
                for metric_name, metric_value in metrics_dict.items():
                    self.log(f"{metric_name}: {metric_value}")
            return metrics_dict

        uq_config = {}
        uq_config["max_length"] = self.max_length
        uq_config["batch_size"] = self.batch_size
        uq_config["n_ensemble"] = self.n_ensemble
        uq_config["seeds"]      = self.seeds
        uq_config["device"]     = self.device
        uq_config["lr"]         = self.lr
        uq_config["num_epochs"] = self.num_epochs

        train_and_evaluate_lora_ensemble(self.train_dataset, self.test_dataset, self.lora_ensemble_tmp_dir, self.model, self.tokenizer, uq_config)

##            trainer = SFTTrainer(
##                model=self.model,
##                peft_config=peft_config,
##                args=training_arguments,
##                train_dataset=self.train_dataset,
##                eval_dataset=self.valid_dataset, 
##                tokenizer=self.tokenizer,
##                max_seq_length=self.max_length,
##                data_collator=collator,
##                packing=False,
##                formatting_func=formatting_prompts_func,
##                compute_metrics=compute_metrics,
##            )
##
##            trainer.train()
##            print("LOG HISTORY: ", trainer.state.log_history)
##            self.plot_losses(trainer.state.log_history)
##
##            # Save the model after training
##            model_path = os.path.join('/pscratch/sd/t/tianle/myWork/transformers/cache/saved_models/', f'{self.model_name}.pth')
##            with open(model_path, 'wb') as f:
##                torch.save(self.model.state_dict(), f)
##
##            # Clear GPU memory before evaluation
##            torch.cuda.empty_cache() # Clear unused memory from the cache
##            gc.collect() # Manual garbage collection
##            torch.cuda.synchronize() # Ensure that CUDA memory is freed
##
##            # Evaluate the model on the test set 
##            self.testing = True
##            self.count = 0
##            self.log("Evaluation on test set:\n")
##            tokenized_test_dataset = self.test_dataset.map(process_test_set, batched=True, batch_size=1)
##            with torch.no_grad():
##                results = trainer.predict(tokenized_test_dataset)
##            print("Evaluation Results:", results)


#            fold = self.fold_data[self.fold_idx]
#            print(f"Training Fold {self.fold_idx}")
#            print("Fold Data: ", fold)
#            self.model = self.base_model
#            print(f"Start train_and_evaluate_lora_ensemble for fold {self.fold_idx}")
#            train_and_evaluate_lora_ensemble(fold[0], fold[1], self.lora_ensemble_tmp_dir)
#            print(f"Finish train_and_evaluate_lora_ensemble for fold {self.fold_idx}")
#
#            for fold in self.fold_data:
#                i += 1
#                print(f"Training Fold {i}")
#                print("Fold Data: ", fold)
#                self.model = self.base_model
#               
#                print(f"Start train_and_evaluate_lora_ensemble for fold {i}")
#                train_and_evaluate_lora_ensemble(fold[0], fold[1], self.lora_ensemble_tmp_dir)
#                print(f"Finish train_and_evaluate_lora_ensemble for fold {i}")
#            print(f"ALL finish train_and_evaluate_lora_ensemble")
#
##                self.testing = False 
##                self.count = 0
##
##                trainer = SFTTrainer(
##                    model=self.model,
##                    peft_config=peft_config,
##                    args=training_arguments,
##                    train_dataset=fold[0],
##                    tokenizer=self.tokenizer,
##                    max_seq_length=self.max_length,
##                    data_collator=collator,
##                    packing=False,
##                    formatting_func=formatting_prompts_func,
##                    compute_metrics=compute_metrics,
##                )
##
##                # Train the model
##                trainer.train()
##                print("LOG HISTORY: ", trainer.state.log_history)
##
##                # Plot the training loss
##                self.plot_file_path = f"/pscratch/sd/t/tianle/lucid/other_source/SURP_2024/results/experiments/{self.model_name}-fold-{i}-losses.png"
##                self.plot_losses(trainer.state.log_history)
##
##                # Save the model after training
##                model_path = os.path.join('/pscratch/sd/t/tianle/myWork/transformers/cache/saved_models/', f'{self.model_name}-fold-{i}.pth')
##                with open(model_path, 'wb') as f:
##                    torch.save(self.model.state_dict(), f)
##
##                # Clear GPU memory before evaluation
##                torch.cuda.empty_cache() # Clear unused memory from the cache
##                gc.collect() # Manual garbage collection
##                torch.cuda.synchronize() # Ensure that CUDA memory is freed
##
##                # Evaluate on the test set
##                self.testing = True  
##                self.count = 0 
##                self.log_file_path = f"/pscratch/sd/t/tianle/lucid/other_source/SURP_2024/results/experiments/{self.model_name}-fold-{i}-results.txt"
##                self.log("Evaluation on test set:\n")
##                tokenized_test_dataset = fold[1].map(process_test_set, batched=True, batch_size=self.batch_size)
##                with torch.no_grad():
##                    results = trainer.predict(tokenized_test_dataset)
##                print("Evaluation Results:", results)

#            fold = self.fold_data[self.fold_idx]
#            print(f"Training Fold {self.fold_idx}")
#            print("Fold Data: ", fold)
#            self.model = self.base_model
#            print(f"Start train_and_evaluate_lora_ensemble for fold {self.fold_idx}")
#            train_and_evaluate_lora_ensemble(fold[0], fold[2], self.lora_ensemble_tmp_dir)
#            print(f"Finish train_and_evaluate_lora_ensemble for fold {self.fold_idx}")
#
#            for fold in self.fold_data:
#                i += 1
#                print(f"Training Fold {i}")
#                print("Fold Data: ", fold)
#                self.model = self.base_model
#               
#                print(f"Start train_and_evaluate_lora_ensemble for fold {i}")
#                train_and_evaluate_lora_ensemble(fold[0], fold[2], self.lora_ensemble_tmp_dir)
#                print(f"Finish train_and_evaluate_lora_ensemble for fold {i}")
#            print(f"ALL finish train_and_evaluate_lora_ensemble")
#
##                self.testing = False
##                self.count = 0 
##        
##                trainer = SFTTrainer(
##                    model=self.model,
##                    peft_config=peft_config,
##                    args=training_arguments,
##                    train_dataset=fold[0],
##                    eval_dataset=fold[1], 
##                    tokenizer=self.tokenizer,
##                    max_seq_length=self.max_length,
##                    data_collator=collator,
##                    packing=False,
##                    formatting_func=formatting_prompts_func,
##                    compute_metrics=compute_metrics,
##                )
##
##                # Train the model
##                trainer.train()
##                print("LOG HISTORY: ", trainer.state.log_history)
##
##                # Plot the training loss 
##                self.plot_file_path = f"/pscratch/sd/t/tianle/lucid/other_source/SURP_2024/results/experiments/{self.model_name}-fold-{i}-losses.png"
##                self.plot_losses(trainer.state.log_history)
##
##                # Save the model after training
##                model_path = os.path.join('/pscratch/sd/t/tianle/myWork/transformers/cache/saved_models/', f'{self.model_name}-fold-{i}.pth')
##                with open(model_path, 'wb') as f:
##                    torch.save(self.model.state_dict(), f)
##
##                # Clear GPU memory before evaluation
##                torch.cuda.empty_cache() # Clear unused memory from the cache
##                gc.collect() # Manual garbage collection
##                torch.cuda.synchronize() # Ensure that CUDA memory is freed
##
##                # Evaluate on the test set 
##                self.testing = True
##                self.log_file_path = f"/pscratch/sd/t/tianle/lucid/other_source/SURP_2024/results/experiments/{self.model_name}-fold-{i}-results.txt"
##                self.log("Evaluation on test set:\n")
##                tokenized_test_dataset = fold[2].map(process_test_set, batched=True, batch_size=1)
##                with torch.no_grad():
##                    results = trainer.predict(tokenized_test_dataset)
##                print("Evaluation Results:", results)


    def pretrained_model_inference(self):
  
        self.tokenizer.padding_side='left'
        self.tokenizer.pad_token_id=self.tokenizer.eos_token_id

        def tokenize_test_set(example):
            # Log the prompt and label with the current counter
            tokenized_inputs = {key: [] for key in ['input_ids', 'attention_mask', 'token_type_ids'] if key in self.tokenizer.model_input_names}

            for index, question in enumerate(example['question']):
                self.log(f"Prompt {self.prompt_counter}: {question}")
                self.prompt_counter += 1  
                self.log(f"Label {self.label_counter}: {example['answer'][index]}\n")
                self.label_counter += 1

            # The following prompts were used for the multi-shot prompting strategy
            # Copy the prompt that corresponds to the dataset being used

            # Dataset 1 prompts:
            # Llama
            # [f"### Question: Given the options Yes or No, will there be significant deregulation of ACTB (Actin) 24 months after exposure to low-dose radiation at 0.5 Gy?\n### Answer: No\n### Question: Given the options Yes or No, will there be significant deregulation of TUBA1A (Tubulin) 24 months after exposure to low-dose radiation at 0.5 Gy?\n### Answer: No\n### Question: Given the options Yes or No, will there be significant deregulation of MYH7 (Myosin) 24 months after exposure to low-dose radiation at 0.5 Gy?\n### Answer: Yes\n### Question: {q}\n### Answer: " for q in example['question']] if "llama2" in self.model_name.lower() or "llama3" in self.model_name.lower() else 
            # # Mistral
            # [f"{self.tokenizer.eos_token}[INST]### Question: Given the options Yes or No, will there be significant deregulation of ACTB (Actin) 24 months after exposure to low-dose radiation at 0.5 Gy?[/INST]\n### Answer: No\n[INST]### Question: Given the options Yes or No, will there be significant deregulation of TUBA1A (Tubulin) 24 months after exposure to low-dose radiation at 0.5 Gy?[/INST]\n### Answer: No\n[INST]### Question: Given the options Yes or No, will there be significant deregulation of MYH7 (Myosin) 24 months after exposure to low-dose radiation at 0.5 Gy?[/INST]\n### Answer: Yes\n[INST]### Question: {q}[/INST]\n### Answer: " for q in example['question']] if "mistral" in self.model_name.lower() or "mixtral" in self.model_name.lower() else [],  
                
            # Dataset 2 prompts:
            # Llama
            # [f"### Question: Given the options yes or no, will there be significant deregulation of the protein KRT5 72 hours after exposure to low dose radiation at 2.0 Gy?\n### Answer: No\n### Question: Given the options yes or no, will there be significant deregulation of the protein GAPDH 72 hours after exposure to low dose radiation at 2.0 Gy?\n### Answer: No\n### Question: Given the options yes or no, will there be significant deregulation of the protein ALB 72 hours after exposure to low dose radiation at 2.0 Gy?\n### Answer: Yes\n### Question: {q}\n### Answer: " for q in example['question']] if "llama2" in self.model_name.lower() or "llama3" in self.model_name.lower() else 
            # # Mistral
            # [f"{self.tokenizer.eos_token}[INST]### Question: Given the options yes or no, will there be significant deregulation of the protein KRT5 72 hours after exposure to low dose radiation at 2.0 Gy?[/INST]\n### Answer: No\n[INST]### Question: Given the options yes or no, will there be significant deregulation of the protein GAPDH 72 hours after exposure to low dose radiation at 2.0 Gy?[/INST]\n### Answer: No\n[INST]### Question: Given the options yes or no, will there be significant deregulation of the protein ALB 72 hours after exposure to low dose radiation at 2.0 Gy?[/INST]\n### Answer: Yes\n[INST]### Question: {q}[/INST]\n### Answer: " for q in example['question']] if "mistral" in self.model_name.lower() or "mixtral" in self.model_name.lower() else [],  
                
            # Dataset 3 prompts:
            # Llama
            # [f"### Question: Given the options yes or no, will there be an altered acetylation status of protein G6PD 4 hours after exposure to low dose radiation at 0.5 Gy?\n### Answer: No\n### Question: Given the options yes or no, will there be an altered acetylation status of protein FGFR1 4 hours after exposure to low dose radiation at 0.5 Gy?\n### Answer: No\n### Question: Given the options yes or no, will there be an altered acetylation status of protein CDKN1A 4 hours after exposure to low dose radiation at 0.5 Gy?\n### Answer: Yes\n### Question: {q}\n### Answer: " for q in example['question']] if "llama2" in self.model_name.lower() or "llama3" in self.model_name.lower() else 
            # # Mistral
            # [f"{self.tokenizer.eos_token}[INST]### Question: Given the options yes or no, will there be an altered acetylation status of protein G6PD 4 hours after exposure to low dose radiation at 0.5 Gy?[/INST]\n### Answer: No\n[INST]### Question: Given the options yes or no, will there be an altered acetylation status of protein FGFR1 4 hours after exposure to low dose radiation at 0.5 Gy?[/INST]\n### Answer: No\n[INST]### Question: Given the options yes or no, will there be an altered acetylation status of protein CDKN1A 4 hours after exposure to low dose radiation at 0.5 Gy?[/INST]\n### Answer: Yes\n[INST]### Question: {q}[/INST]\n### Answer: " for q in example['question']] if "mistral" in self.model_name.lower() or "mixtral" in self.model_name.lower() else [],  
                
            # Dataset 3c prompts:
            # Llama
            # [f"### Question: Given the options yes or no, will there be deregulation of the protein TP53 after low-dose radiation exposure?\n### Answer: No\n### Question: Given the options yes or no, will there be deregulation of the protein PTEN after low-dose radiation exposure?\n### Answer: No\n### Question: Given the options yes or no, will there be deregulation of the protein BCL2 after low-dose radiation exposure?\n### Answer: Yes\n### Question: {q}\n### Answer: " for q in example['question']] if "llama2" in self.model_name.lower() or "llama3" in self.model_name.lower() else 
            # # Mistral
            # [f"{self.tokenizer.eos_token}[INST]### Question: Given the options yes or no, will there be deregulation of the protein TP53 after low-dose radiation exposure?[/INST]\n### Answer: No\n[INST]### Question: Given the options yes or no, will there be deregulation of the protein PTEN after low-dose radiation exposure?[/INST]\n### Answer: No\n[INST]### Question: Given the options yes or no, will there be deregulation of the protein BCL2 after low-dose radiation exposure?[/INST]\n### Answer: Yes\n[INST]### Question: {q}[/INST]\n### Answer: " for q in example['question']] if "mistral" in self.model_name.lower() or "mixtral" in self.model_name.lower() else [],  
              
            # Dataset 4 prompts:
            # Llama
            # [f"### Question: Given the options yes or no, is there a protein interaction between ENOL and KRT5 in the presence of neurodegenerative diseases?\n### Answer: Yes\n### Question: Given the options yes or no, is there a protein interaction between HBA1 and LDHA in the presence of neurodegenerative diseases?\n### Answer: No\n### Question: Given the options yes or no, is there a protein interaction between CFTR and INS in the presence of neurodegenerative diseases?\n### Answer: Yes\n### Question: {q}\n### Answer: " for q in example['question']] if "llama2" in self.model_name.lower() or "llama3" in self.model_name.lower() else 
            # # Mistral
            # [f"{self.tokenizer.eos_token}[INST]### Question: Given the options yes or no, is there a protein interaction between ENOL and KRT5 in the presence of neurodegenerative diseases?[/INST]\n### Answer: Yes\n[INST]### Question: Given the options yes or no, is there a protein interaction between HBA1 and LDHA in the presence of neurodegenerative diseases?[/INST]\n### Answer: No\n[INST]### Question: Given the options yes or no, is there a protein interaction between CFTR and INS in the presence of neurodegenerative diseases?[/INST]\n### Answer: Yes\n[INST]### Question: {q}[/INST]\n### Answer: " for q in example['question']] if "mistral" in self.model_name.lower() or "mixtral" in self.model_name.lower() else [],  
              
            # Dataset 5 prompts:
            # Llama
            # [f"### Question: Given the options yes or no, is there a protein interaction between BCL2 and FGFR1 in the presence of metabolic diseases?\n### Answer: Yes\n### Question: Given the options yes or no, is there a protein interaction between MYH7 and KRT5 in the presence of metabolic diseases?\n### Answer: No\n### Question: Given the options yes or no, is there a protein interaction between CFTR and ACTB in the presence of metabolic diseases?\n### Answer: No\n### Question: {q}\n### Answer: " for q in example['question']] if "llama2" in self.model_name.lower() or "llama3" in self.model_name.lower() else 
            # # Mistral
            # [f"{self.tokenizer.eos_token}[INST]### Question: Given the options yes or no, is there a protein interaction between BCL2 and FGFR1 in the presence of metabolic diseases?[/INST]\n### Answer: Yes\n[INST]### Question: Given the options yes or no, is there a protein interaction between MYH7 and KRT5 in the presence of metabolic diseases?[/INST]\n### Answer: No\n[INST]### Question: Given the options yes or no, is there a protein interaction between CFTR and ACTB in the presence of metabolic diseases?[/INST]\n### Answer: No\n[INST]### Question: {q}[/INST]\n### Answer: " for q in example['question']] if "mistral" in self.model_name.lower() or "mixtral" in self.model_name.lower() else [],  

            # Dataset 6 prompts:
            # Llama
            # [f"### Question: Given the options yes or no, is there a protein interaction between ALB and CDKN1A in the presence of cancer?\n### Answer: Yes\n### Question: Given the options yes or no, is there a protein interaction between GAPDH and TUBA1A in the presence of cancer?\n### Answer: Yes\n### Question: Given the options yes or no, is there a protein interaction between TP53 and PTEN in the presence of cancer?\n### Answer: No\n### Question: {q}\n### Answer: " for q in example['question']] if "llama2" in self.model_name.lower() or "llama3" in self.model_name.lower() else 
            # # Mistral
            # [f"{self.tokenizer.eos_token}[INST]### Question: Given the options yes or no, is there a protein interaction between ALB and CDKN1A in the presence of cancer?[/INST]\n### Answer: Yes\n[INST]### Question: Given the options yes or no, is there a protein interaction between GAPDH and TUBA1A in the presence of cancer?[/INST]\n### Answer: Yes\n[INST]### Question: Given the options yes or no, is there a protein interaction between TP53 and PTEN in the presence of cancer?[/INST]\n### Answer: No\n[INST]### Question: {q}[/INST]\n### Answer: " for q in example['question']] if "mistral" in self.model_name.lower() or "mixtral" in self.model_name.lower() else [],  
             
            # Tokenize Inputs
            tokenized = self.tokenizer(
                # Paste Prompt here:
                # Llama
                [f"### Question: Given the options yes or no, will there be an altered acetylation status of protein G6PD 4 hours after exposure to low dose radiation at 0.5 Gy?\n### Answer: No\n### Question: Given the options yes or no, will there be an altered acetylation status of protein FGFR1 4 hours after exposure to low dose radiation at 0.5 Gy?\n### Answer: No\n### Question: Given the options yes or no, will there be an altered acetylation status of protein CDKN1A 4 hours after exposure to low dose radiation at 0.5 Gy?\n### Answer: Yes\n### Question: {q}\n### Answer: " for q in example['question']] if "llama2" in self.model_name.lower() or "llama3" in self.model_name.lower() else 
                # Mistral
                [f"{self.tokenizer.eos_token}[INST]### Question: Given the options yes or no, will there be an altered acetylation status of protein G6PD 4 hours after exposure to low dose radiation at 0.5 Gy?[/INST]\n### Answer: No\n[INST]### Question: Given the options yes or no, will there be an altered acetylation status of protein FGFR1 4 hours after exposure to low dose radiation at 0.5 Gy?[/INST]\n### Answer: No\n[INST]### Question: Given the options yes or no, will there be an altered acetylation status of protein CDKN1A 4 hours after exposure to low dose radiation at 0.5 Gy?[/INST]\n### Answer: Yes\n[INST]### Question: {q}[/INST]\n### Answer: " for q in example['question']] if "mistral" in self.model_name.lower() or "mixtral" in self.model_name.lower() else [],  
                padding="max_length",
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt" 
            )

            # Convert batch encoding data structure to python list 
            for key in tokenized_inputs.keys():
                tokenized_inputs[key].extend(tokenized[key].numpy().tolist())
            return tokenized_inputs
        
        # Use this for datasets 1-3
        # tokenized_test_dataset = self.test_dataset.map(tokenize_test_set, batched=True)      
        if "set-1" in self.model_name or "set-2" in self.model_name or "set-3" in self.model_name:
            # Train Normally
            pass

        else:
            # Evaluate the model
            i = 0
            for fold in self.fold_data:
                i += 1
                print("Fold Data: ", fold)
                self.log_file_path = f"/pscratch/sd/t/tianle/lucid/other_source/SURP_2024/results/experiments/{self.model_name}-fold-{i}-results.txt"
                self.log("Evaluation on test set:\n")
                self.model.eval()
                self.prompt_counter = 1 
                self.label_counter = 1   
                self.output_counter = 1 
                predictions, labels = [], []

                # Clear GPU memory before evaluation
                torch.cuda.empty_cache() # Clear unused memory from the cache
                gc.collect() # Manual garbage collection
                torch.cuda.synchronize() # Ensure that CUDA memory is freed
                
                if "set-4" in self.model_name or "set-5" in self.model_name: tokenized_test_dataset = fold[1].map(tokenize_test_set, batched=True, batch_size=1)
                elif "set-6" in self.model_name: tokenized_test_dataset = fold[2].map(tokenize_test_set, batched=True, batch_size=1)
                with torch.no_grad():
                    index = 0
                    for batch in tokenized_test_dataset:
                        inputs = {
                            k: torch.tensor(v, dtype=torch.long) for k, v in batch.items() if k in ['input_ids', 'attention_mask']
                        }

                        # Ensure each tensor in inputs has a batch dimension
                        inputs = {k: v.unsqueeze(0) if v.dim() == 1 else v for k, v in inputs.items()}
                        # Move tensors to the device after unsqueezing
                        inputs = {k: v.to(self.device) for k, v in inputs.items()}  
                        true_label = batch['answer']  
                        generated_sequences = self.model.generate(
                            **inputs,
                            max_new_tokens=self.new_tokens
                        )

                        for idx, sequence in enumerate(generated_sequences):
                            text = self.tokenizer.decode(sequence, skip_special_tokens=True).lower().strip()

                            # Find the start of the answer (0-shot)
                            answer_start_idx = text.lower().find("answer: ") + len("answer: ")

                            # Find the 3rd answer keyword that appears (3-shot)
                            idx = -len("answer: ")  # Start at -length of search term to compensate for the addition inside the loop
                            for _ in range(4):
                                idx = text.lower().find("answer: ", idx + len("answer: "))
                            answer_start_idx = idx + len("answer: ")
                                
                            # Extract the substring starting from the answer
                            answer_text = text[answer_start_idx:]

                            # Parse the model output to get the first word after the "answer:"
                            matches = re.findall(r'\byes\b|\bno\b', answer_text.lower())
                            if matches: answer = matches[0]  
                            else:
                                # If we cannot immediately find the answer in the first word, find the first word after a newline character
                                answer = answer_text.split()[0]

                            if "yes" in true_label.lower(): labels.append(1)
                            elif "no" in true_label.lower(): labels.append(0)

                            if "yes" in answer.lower(): predictions.append(1)
                            elif "no" in answer.lower(): predictions.append(0)
                            else: 
                                # Append the opposite of the true label, checking for None
                                if labels[index] is not None:
                                    if labels[index] == 0:
                                        opposite_value = 1 
                                        answer = "yes"
                                    else:
                                        opposite_value = 0
                                        answer = "no"
                                    predictions.append(opposite_value)

                            # Print model outputs
                            self.log(f"Model Prediction {self.output_counter}: {answer}")
                            self.output_counter += 1
                            index += 1

                metrics = []
                try:
                    accuracy = accuracy_score(labels, predictions)
                    metrics.append(("Accuracy", accuracy))
                except Exception as e:
                    print(f"Error calculating Accuracy: {e}")
                    metrics.append(("Accuracy", None))
                try:
                    mcc = matthews_corrcoef(labels, predictions)
                    metrics.append(("MCC", mcc))
                except Exception as e:
                    print(f"Error calculating MCC: {e}")
                    metrics.append(("MCC", None))
                try:
                    auc = roc_auc_score(labels, predictions)
                    metrics.append(("AUC", auc))
                except Exception as e:
                    print(f"Error calculating AUC: {e}")
                    metrics.append(("AUC", None))
                try:
                    tn, fp, fn, tp = confusion_matrix(labels, predictions).ravel()
                    specificity = tn / (tn + fp)
                    metrics.append(("Specificity", specificity))
                except Exception as e:
                    print(f"Error calculating Specificity: {e}")
                    metrics.append(("Specificity", None))
                try:
                    precision = precision_score(labels, predictions, average="macro")
                    metrics.append((f"Macro Precision", precision))
                except Exception as e:
                    print(f"Error calculating Macro Precision: {e}")
                    metrics.append((f"Macro Precision", None))
                try:
                    f1 = f1_score(labels, predictions, average="macro")
                    metrics.append((f"Macro F1 Score", f1))
                except Exception as e:
                    print(f"Error calculating Macro F1 Score: {e}")
                    metrics.append((f"Macro F1 Score", None))

                # Log and print metrics
                self.log("")
                for metric_name, metric_value in metrics:
                    print(f"{metric_name}: {metric_value}")
                    self.log(f"{metric_name}: {metric_value}")
