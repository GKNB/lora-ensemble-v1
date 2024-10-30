# Necessary Imports
import torch
import random
import os
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
from typing import Any
from datasets import Dataset
from peft import TaskType, get_peft_model, LoraConfig
from transformers import TrainingArguments
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import accuracy_score, precision_score, f1_score, matthews_corrcoef, roc_auc_score, confusion_matrix
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


# Set random seeds for consistent experiments
def set_seeds(seed=237):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    generator = torch.Generator().manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    return generator
     

class model_trainer():

    def init(self, model, tokenizer, args):

        # Initialize device and random seed
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.n_ensemble = args.n_ensemble
        self.seeds = [237 + i * 10093 for i in range(self.n_ensemble)]
        self.generator = set_seeds(237)
        # os.environ["WANDB_DISABLED"] = "true" 

        # Initialize file paths 
#        self.model_name = "Mistral-7B-set-3c"
#        self.model_name = "Llama3-8B-set-3.1"
        self.model_name = "Llama3-8B-set-3c"

#        self.json_file_path = '/pscratch/sd/t/tianle/lucid/other_source/SURP_2024/data/dataset_3_v1_prompts.json'
        self.json_file_path = '/pscratch/sd/t/tianle/lucid/other_source/SURP_2024/data/dataset_3c_prompts.json'

        self.output_dir = f"/pscratch/sd/t/tianle/lucid/other_source/SURP_2024/results/models/{self.model_name}"
        self.fold_dir = f"/pscratch/sd/t/tianle/lucid/other_source/SURP_2024/results/folds/{self.model_name}"
        self.log_file_path = f"/pscratch/sd/t/tianle/lucid/other_source/SURP_2024/results/experiments/{self.model_name}-results.txt"
        self.plot_file_path = f"/pscratch/sd/t/tianle/lucid/other_source/SURP_2024/results/experiments/{self.model_name}-losses.png"
        self.plot_title = f"Loss values for {self.model_name}"

        # Open the log file in write mode, this will clear previous contents 
        with open(self.log_file_path, 'w') as file: 
            file.write("") 

        # Hyperparameters 
        self.lr = 1e-4 # Learning rate remains the same for all experiments
        self.new_tokens = 5  # New tokens remains the same for all experiments
        self.num_epochs = 4 
        self.batch_size = 16
        self.max_length = 120
        # max length is 425 for multishot (Pre-trained) Experiments
 
        # datasets 1-3 
            # max_length = 120 
            # num_epochs = 4 
            # dataset 1.1 - Batch Size = 8 
            # dataset 1.2 - Batch Size = 16 
            # dataset 1.3 - Batch Size = 4
            # dataset 2.1 - Batch Size = 2 
            # dataset 2.2 - Batch Size = 1 
            # dataset 2.3 - Batch Size = 2
            # dataset 2.4 - Batch Size = 1 
            # dataset 3.1 - Batch Size = 2 
            # dataset 3.2 - Batch Size = 1 
            # dataset 3c - Batch Size = 16

        # datasets 4-5
            # max_length = 50
            # num_epochs = 4
            # batch_size = 4

        # dataset 6
            # max_length = 50
            # num_epochs = 4
            # batch_size = 16
        
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



    def preprocess_data(self):
        
        # Initialize dataset 
        self.dataset_examples = []

        # Open and read the JSON file
        with open(self.json_file_path, 'r') as file:
            prompts_data = json.load(file)

        print("Size of dataset: ", len(prompts_data))

        # Iterate through each entry in the JSON file
        for prompt in prompts_data:
            self.dataset_examples.append(prompt)

        # Shuffle the dataset to ensure a mix of 'Yes' and 'No' answers throughout
        random.shuffle(self.dataset_examples)

        # Set up 80 / 10 / 10 split for training / validation / testing for datasets 1-3
        if "set-1" in self.model_name or "set-2" in self.model_name or "set-3" in self.model_name:

            # Split the dataset into training, validation, and possibly test sets
            total_items = len(self.dataset_examples)
            train_end = int(total_items * 0.8)
            valid_end = train_end + int(total_items * 0.1)
            
            train_dataset = self.dataset_examples[:train_end]
            valid_dataset = self.dataset_examples[train_end:valid_end]
            test_dataset = self.dataset_examples[valid_end:]

            # Convert list of dictionaries into Hugging Face Dataset
            self.train_dataset = Dataset.from_dict({'question': [i['question'] for i in train_dataset], 'answer': [i['answer'] for i in train_dataset]})
            self.valid_dataset = Dataset.from_dict({'question': [i['question'] for i in valid_dataset], 'answer': [i['answer'] for i in valid_dataset]})
            self.test_dataset = Dataset.from_dict({'question': [i['question'] for i in test_dataset], 'answer': [i['answer'] for i in test_dataset]})


        # Set up 5-fold cross validation for datasets 4 and 5
        if "set-4" in self.model_name or "set-5" in self.model_name:
            kf = KFold(n_splits=5, shuffle=True, random_state=42)
            self.fold_data = []

            for fold, (train_index, test_index) in enumerate(kf.split(self.dataset_examples), start=1):
                train_fold = [self.dataset_examples[i] for i in train_index]
                test_fold = [self.dataset_examples[i] for i in test_index]

                train_fold_dataset = Dataset.from_dict({'question': [i['question'] for i in train_fold], 'answer': [i['answer'] for i in train_fold]})
                test_fold_dataset = Dataset.from_dict({'question': [i['question'] for i in test_fold], 'answer': [i['answer'] for i in test_fold]})

                fold_name = f'fold-{fold}'
                self.save_fold_data(fold_name, train_dataset=train_fold_dataset, test_dataset=test_fold_dataset)
                self.fold_data.append((train_fold_dataset, test_fold_dataset))


        # Set up 5-fold cross validation for dataset 6
        elif "set-6" in self.model_name:
            kf = KFold(n_splits=5, shuffle=True, random_state=42)
            self.fold_data = []

            for fold, (train_index, test_index) in enumerate(kf.split(self.dataset_examples), start=1):
                train_index, validation_index = train_test_split(train_index, test_size=0.2, random_state=42)

                train_fold = [self.dataset_examples[i] for i in train_index]
                valid_fold = [self.dataset_examples[i] for i in validation_index]
                test_fold = [self.dataset_examples[i] for i in test_index]

                train_fold_dataset = Dataset.from_dict({'question': [i['question'] for i in train_fold], 'answer': [i['answer'] for i in train_fold]})
                valid_fold_dataset = Dataset.from_dict({'question': [i['question'] for i in valid_fold], 'answer': [i['answer'] for i in valid_fold]})
                test_fold_dataset = Dataset.from_dict({'question': [i['question'] for i in test_fold], 'answer': [i['answer'] for i in test_fold]})

                fold_name = f'fold-{fold}'
                self.save_fold_data(fold_name, train_dataset=train_fold_dataset, valid_dataset=valid_fold_dataset, test_dataset=test_fold_dataset)
                self.fold_data.append((train_fold_dataset, valid_fold_dataset, test_fold_dataset))


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

        def custom_collate_fn(batch):
            # Extract questions and answers from the batch
            prompt = """Answer the following question with Yes or No.\n\nQuestion: {question}\n\nAnswer (Yes or No):"""
            prompts = [prompt.format(question=item['question']) for item in batch]
            classes = torch.tensor([1 if item['answer'] == 'Yes' else 0 for item in batch])
            return prompts, classes
        
        labels = [f" Yes", f" No"]
        target_ids = self.tokenizer(
            labels, return_tensors="pt", add_special_tokens=False
        ).input_ids[:, -1:]

        tokenizer_run_kwargs = {
                        "return_tensors": "pt",
                        "padding": "max_length",
                        "truncation": True,
                        "max_length": self.max_length,
                    }

        # LoRA CONFIG 
        # https://moon-ci-docs.huggingface.co/docs/peft/pr_721/en/package_reference/tuners#peft.LoraConfig
        target_modules = ['q_proj', 'v_proj']
        
        # Here we differentiate the training process depending on the dataset
        if "set-1" in self.model_name or "set-2" in self.model_name or "set-3" in self.model_name:

            test_ensemble_probabilities = []

            #Here use the uniform test_loader without shuffle to handle all lora instance
            #This is important since we later need to average prob, and need to make sure all models access test dataset in the same order
            test_loader = torch.utils.data.DataLoader(self.test_dataset, collate_fn=custom_collate_fn, batch_size = self.batch_size, shuffle=False)

            for i in range(self.n_ensemble):
                self.log(f"Training lora instance {i}")

                self.generator = set_seeds(self.seeds[i])
                train_loader = torch.utils.data.DataLoader(self.train_dataset, collate_fn=custom_collate_fn, batch_size = self.batch_size)
                
                peft_config = LoraConfig(
                    task_type=TaskType.CAUSAL_LM, 
                    target_modules=target_modules, 
                    inference_mode=False, 
                    r=16, 
                    lora_alpha=32, 
                    lora_dropout=0.05, 
                    bias="none"
                )
                lora_model = get_peft_model(self.model, peft_config)
                model_instance_path = f"{self.output_dir}/model_instance_{i}.pth"
#                total_params = sum(p.numel() for p in lora_model.parameters())
#                trainable_params = sum(p.numel() for p in lora_model.parameters() if p.requires_grad)
#                print(f"DEBUG: Total parameters: {total_params}")
#                print(f"DEBUG: Trainable parameters: {trainable_params}")
  
                opt_cfg = {
                    "module": "torch.optim",
                    "classname": "AdamW",
                    "lr": self.lr,
                    "betas": (0.9, 0.999),
                    "eps": 1e-8,  # 1e-5
                }
                
                optclass = getattr(
                    importlib.import_module(opt_cfg.pop("module")),
                    opt_cfg.pop("classname"),
                )
    
                opt = optclass(lora_model.parameters(), **opt_cfg)
                lora_model.train()

                grad_steps = 0
                for epoch in range(self.num_epochs):
                    self.log(f"Beginning epoch {epoch}")
                    for batch in train_loader:
                        opt.zero_grad()
                        prompts, classes = batch
                        inputs = self.tokenizer(prompts, **tokenizer_run_kwargs).to(self.device)
                        logits = lora_model(**inputs).logits[:, -1, target_ids.squeeze(-1)]
                        loss = F.cross_entropy(logits, classes.to(self.device))
                        print(f"In grad_steps = {grad_steps}, loss = {loss}")
#                        print(f"logits = {logits} \nclasses = {classes}")
                        loss.backward()
                        opt.step()
                        grad_steps += 1
                
#                self.log(f"Saving lora instance {i} after finetuning to {model_instance_path}")
#                lora_model.save_pretrained(model_instance_path)


                lora_model.eval()
                
                test_probabilities = []
                test_true_class = []
                with torch.no_grad():
                    for batch in test_loader:
                        prompts, classes = batch
                        inputs = self.tokenizer(prompts, **tokenizer_run_kwargs).to(self.device)
                        logits = lora_model(**inputs).logits[:, -1, target_ids.squeeze(-1)]
                        probabilities = F.softmax(logits, dim=-1)
                        test_probabilities.append(probabilities.cpu().numpy())
                        test_true_class.append(classes.cpu().numpy())

                test_ensemble_probabilities.append(np.concatenate(test_probabilities))
                test_true_class = np.concatenate(test_true_class)
                print(f"i = {i}, Test ensemble probabilities = \n{test_ensemble_probabilities}")
                print(f"i = {i}, Test true class= \n{test_true_class}")
                self.log(f"lora instance i = {i} Successfully finished.")

            test_average_probabilities = np.mean(test_ensemble_probabilities, axis=0)
            print(f"Final, Test average ensemble probabilities = \n{test_average_probabilities}")

            prob_positive = test_average_probabilities[:, 1]
            pred_label = (prob_positive >= 0.5).astype(int)
            test_true_class = torch.from_numpy(test_true_class)
            prob_positive = torch.from_numpy(prob_positive)
            pred_label = torch.from_numpy(pred_label)

            accuracy_metric = BinaryAccuracy()
            mcc_metric = BinaryMatthewsCorrCoef()
            auroc_metric = BinaryAUROC()
            confmat_metric = BinaryConfusionMatrix()
            specificity_metric = BinarySpecificity()
            precision_macro_metric = MulticlassPrecision(num_classes=2, average='macro')
            f1_macro_metric = MulticlassF1Score(num_classes=2, average='macro')
            ece_metric = BinaryCalibrationError()
            
            accuracy = accuracy_metric(pred_label, test_true_class)
            mcc_score = mcc_metric(pred_label, test_true_class)
            roc_auc = auroc_metric(prob_positive, test_true_class)
            confusion_matrix = confmat_metric(pred_label, test_true_class)
            specificity = specificity_metric(pred_label, test_true_class)
            precision_macro = precision_macro_metric(pred_label, test_true_class)
            f1_macro = f1_macro_metric(pred_label, test_true_class)
            ece = ece_metric(prob_positive, test_true_class)
            nll = -np.mean(np.log(test_average_probabilities[np.arange(len(test_true_class)), test_true_class]))
            
            print(f"Accuracy: {accuracy.item():.4f}")
            print(f"MCC: {mcc_score.item():.4f}")
            print(f"AUC: {roc_auc.item():.4f}")
            print(f"Confusion Matrix:\n{confusion_matrix}")
            print(f"Specificity: {specificity.item():.4f}")
            print(f"Precision (Macro): {precision_macro.item():.4f}")
            print(f"F1 Score (Macro): {f1_macro.item():.4f}")
            print(f"Expected Calibration Error (ECE): {ece.item():.4f}")
            print(f"NLL loss: {nll:.4f}")

            print("Main task is done! Can finish")

#            trainer = SFTTrainer(
#                model=self.model,
#                peft_config=peft_config,
#                args=training_arguments,
#                train_dataset=self.train_dataset,
#                eval_dataset=self.valid_dataset, 
#                tokenizer=self.tokenizer,
#                max_seq_length=self.max_length,
#                data_collator=collator,
#                packing=False,
#                formatting_func=formatting_prompts_func,
#                compute_metrics=compute_metrics,
#            )
#
#            trainer.train()
#            print("LOG HISTORY: ", trainer.state.log_history)
#            self.plot_losses(trainer.state.log_history)
#
#            # Save the model after training
#            model_path = os.path.join('/pscratch/sd/t/tianle/myWork/transformers/cache/saved_models/', f'{self.model_name}.pth')
#            with open(model_path, 'wb') as f:
#                torch.save(self.model.state_dict(), f)
#
#            # Clear GPU memory before evaluation
#            torch.cuda.empty_cache() # Clear unused memory from the cache
#            gc.collect() # Manual garbage collection
#            torch.cuda.synchronize() # Ensure that CUDA memory is freed
#
#            # Evaluate the model on the test set 
#            self.testing = True
#            self.count = 0
#            self.log("Evaluation on test set:\n")
#            tokenized_test_dataset = self.test_dataset.map(process_test_set, batched=True, batch_size=1)
#            with torch.no_grad():
#                results = trainer.predict(tokenized_test_dataset)
#            print("Evaluation Results:", results)

        elif "set-4" in self.model_name or "set-5" in self.model_name:
            i = 0
            for fold in self.fold_data:
                i += 1
                print(f"Training Fold {i}")
                print("Fold Data: ", fold)
                self.model = self.base_model
                self.testing = False 
                self.count = 0
        
                trainer = SFTTrainer(
                    model=self.model,
                    peft_config=peft_config,
                    args=training_arguments,
                    train_dataset=fold[0],
                    tokenizer=self.tokenizer,
                    max_seq_length=self.max_length,
                    data_collator=collator,
                    packing=False,
                    formatting_func=formatting_prompts_func,
                    compute_metrics=compute_metrics,
                )

                # Train the model
                trainer.train()
                print("LOG HISTORY: ", trainer.state.log_history)

                # Plot the training loss
                self.plot_file_path = f"/pscratch/sd/t/tianle/lucid/other_source/SURP_2024/results/experiments/{self.model_name}-fold-{i}-losses.png"
                self.plot_losses(trainer.state.log_history)

                # Save the model after training
                model_path = os.path.join('/pscratch/sd/t/tianle/myWork/transformers/cache/saved_models/', f'{self.model_name}-fold-{i}.pth')
                with open(model_path, 'wb') as f:
                    torch.save(self.model.state_dict(), f)

                # Clear GPU memory before evaluation
                torch.cuda.empty_cache() # Clear unused memory from the cache
                gc.collect() # Manual garbage collection
                torch.cuda.synchronize() # Ensure that CUDA memory is freed

                # Evaluate on the test set
                self.testing = True  
                self.count = 0 
                self.log_file_path = f"/pscratch/sd/t/tianle/lucid/other_source/SURP_2024/results/experiments/{self.model_name}-fold-{i}-results.txt"
                self.log("Evaluation on test set:\n")
                tokenized_test_dataset = fold[1].map(process_test_set, batched=True, batch_size=self.batch_size)
                with torch.no_grad():
                    results = trainer.predict(tokenized_test_dataset)
                print("Evaluation Results:", results)

        elif "set-6" in self.model_name:
            i = 0
            for fold in self.fold_data:
                i += 1
                print(f"Training Fold {i}")
                print("Fold Data: ", fold)
                self.model = self.base_model
                self.testing = False
                self.count = 0 
        
                trainer = SFTTrainer(
                    model=self.model,
                    peft_config=peft_config,
                    args=training_arguments,
                    train_dataset=fold[0],
                    eval_dataset=fold[1], 
                    tokenizer=self.tokenizer,
                    max_seq_length=self.max_length,
                    data_collator=collator,
                    packing=False,
                    formatting_func=formatting_prompts_func,
                    compute_metrics=compute_metrics,
                )

                # Train the model
                trainer.train()
                print("LOG HISTORY: ", trainer.state.log_history)

                # Plot the training loss 
                self.plot_file_path = f"/pscratch/sd/t/tianle/lucid/other_source/SURP_2024/results/experiments/{self.model_name}-fold-{i}-losses.png"
                self.plot_losses(trainer.state.log_history)

                # Save the model after training
                model_path = os.path.join('/pscratch/sd/t/tianle/myWork/transformers/cache/saved_models/', f'{self.model_name}-fold-{i}.pth')
                with open(model_path, 'wb') as f:
                    torch.save(self.model.state_dict(), f)

                # Clear GPU memory before evaluation
                torch.cuda.empty_cache() # Clear unused memory from the cache
                gc.collect() # Manual garbage collection
                torch.cuda.synchronize() # Ensure that CUDA memory is freed

                # Evaluate on the test set 
                self.testing = True
                self.log_file_path = f"/pscratch/sd/t/tianle/lucid/other_source/SURP_2024/results/experiments/{self.model_name}-fold-{i}-results.txt"
                self.log("Evaluation on test set:\n")
                tokenized_test_dataset = fold[2].map(process_test_set, batched=True, batch_size=1)
                with torch.no_grad():
                    results = trainer.predict(tokenized_test_dataset)
                print("Evaluation Results:", results)


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
