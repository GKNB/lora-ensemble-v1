#Necessary Imports

import os
import re
import sys
import json
import torch
import random
import shutil
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F

from datasets import Dataset
from peft import TaskType, LoraConfig
from transformers import TrainingArguments
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
from sklearn.metrics import accuracy_score, precision_score, f1_score, matthews_corrcoef, roc_auc_score, confusion_matrix


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

    def init(self, model, tokenizer):

        # Initialize device and random seed
        self.device = "cuda"
        self.generator = set_seeds(237)
        os.environ["WANDB_DISABLED"] = "true" 

        # Initialize file paths
        self.model_name = "Llama3-8B-set-4" 
        self.json_file_path = '/direct/sdcc+u/rengel/data/dataset_4_prompts.json'
        self.log_file_path = f"/direct/sdcc+u/rengel/results/experiments_v2/{self.model_name}_results.txt"
        self.plot_file_path = f"/direct/sdcc+u/rengel/results/experiments_v2/{self.model_name}_losses.png"
        self.plot_title = f"Loss values for {self.model_name}" 

        # Open the log file in write mode, this will clear previous contents
        with open(self.log_file_path, 'w') as file:
            file.write("") 

        # Hyperparameters
        self.lr = 1e-4   
        self.num_epochs = 4
        self.batch_size = 4
        self.new_tokens = 5 
        self.max_length = 50
        # max length is 420 for multishot
 
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
            # batch_size = 16 

        # dataset 6
            # max_length = 50
            # num_epochs = 4
            # batch_size = 8, Llama-3 uses 16
        
        # Loss values for plots
        self.train_losses = []
        self.valid_losses = []

        # Load Model & Tokenizer
        self.model = model
        self.tokenizer = tokenizer
        self.tokenizer.pad_token = self.tokenizer.eos_token 
         
        print("Model: ", self.model)

        

    def log(self, message):
        # This function handles writing logs to the specified file
        try:
            with open(self.log_file_path, 'a') as f:
                f.write(message + '\n')
        except Exception as e:
            raise Exception(f"Failed to open and write to file: {self.log_file_path}") from e


    def load_data(self):
        # Append path to data loader
        os.chdir('../src/data_processors')
        sys.path.append(os.getcwd())

        # Make call to data loader
        from data_loader import dataset_loader 
        loader = dataset_loader()
        loader.load_datasets()
        print("Done")


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
        
        # Split the dataset into training, validation, and possibly test sets
        total_items = len(self.dataset_examples)
        train_end = int(total_items * 0.8)
        valid_end = train_end + int(total_items * 0.1)
        
        train_dataset = self.dataset_examples[:train_end]
        valid_dataset = self.dataset_examples[train_end:valid_end]
        test_dataset = self.dataset_examples[valid_end:]

        # Convert list of dictionaries into Hugging Face Dataset
        train_dataset = Dataset.from_dict({'question': [i['question'] for i in train_dataset], 'answer': [i['answer'] for i in train_dataset]})
        valid_dataset = Dataset.from_dict({'question': [i['question'] for i in valid_dataset], 'answer': [i['answer'] for i in valid_dataset]})
        test_dataset = Dataset.from_dict({'question': [i['question'] for i in test_dataset], 'answer': [i['answer'] for i in test_dataset]})

        self.train_dataset = train_dataset
        self.valid_dataset = valid_dataset
        self.test_dataset = test_dataset

        print("Training dataset: ", self.train_dataset)
        print("Validation dataset: ", self.valid_dataset)
        print("Testing dataset: ", self.test_dataset)

        # Determine distribution of tokens in each prompt
        prompt_lengths = []
        for example in self.test_dataset: 
            # Llama
            if "llama2" in self.model_name.lower() or "llama3" in self.model_name.lower():
                prompt = f"### Question: {example['question']}\n### Answer: "

            # Mistral
            if "mistral" in self.model_name.lower() or "mixtral" in self.model_name.lower():
                prompt = f"{self.tokenizer.eos_token}[INST]### Question: {example['question']}[/INST]\n### Answer: "
            
            prompt_length = len(self.tokenizer.tokenize(prompt))
            prompt_lengths.append(prompt_length)

        # Analyze the distribution of prompt lengths
        print(f"Maximum prompt length: {max(prompt_lengths)}")
        print(f"Average prompt length: {sum(prompt_lengths) / len(prompt_lengths)}")


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
        # Flag for printing model outputs during testing
        testing = False

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

            for i, (q, a) in enumerate(zip(example['question'], example['answer'])):
                # Log and Debug
                self.log(f"Prompt {i+1}: {q}\nTrue Label: {a}\n")  
                print(f"Prompt {i+1}: {q}\nTrue Label: {a}\n")

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

            print("\nDEBUG: Computing Metrics")  
            torch.cuda.empty_cache()
            
            logits = eval_preds.predictions
            label_ids = eval_preds.label_ids

            # Convert logits from NumPy array to PyTorch tensor
            logits_tensor = torch.from_numpy(logits).to('cuda')

            # Split logits_tensor into smaller chunks
            chunk_size = 1024 
            num_chunks = (logits_tensor.size(0) + chunk_size - 1) // chunk_size

            # Initialize lists to store probabilities and predictions
            all_probabilities = []
            all_predictions = []

            # This code converts the logits to probabilities in chunks for memory efficiency
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

            # Pre-decode all necessary token IDs to text (on CPU)
            decoded_texts = self.tokenizer.batch_decode(predictions, skip_special_tokens=True, clean_up_tokenization_spaces=True)
            decoded_labels = self.tokenizer.batch_decode(label_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)

            print(f"Size of logits: {logits.nbytes}")
            print(f"Size of probabilities: {probabilities.nbytes}")

            # Initialize true_labels and preds lists
            true_labels, preds = [], []
            
            # Process decoded texts to classify responses 
            i = 0
            for decoded_text, decoded_label in zip(decoded_texts, decoded_labels):

                # Normalize and find "yes" or "no"
                normalized_text = decoded_text.lower()
                normalized_label = decoded_label.lower()

                # Determine true label from decoded_label
                true_label = 1 if "yes" in normalized_label else 0 if "no" in normalized_label else None
                true_labels.append(true_label)

                # Determine predicted answer
                answer_start_idx = normalized_text.lower().find("answer: ") + len("answer: ")

                # DEBUG
                # print("------------------------------")
                
                # Extract the substring starting from the answer
                answer_text = normalized_text[answer_start_idx:]

                # Parse the model output to get the first word after the "answer:"
                matches = re.findall(r'\byes\b|\bno\b', answer_text.lower())
                if matches:
                    first_word = matches[0]  
                else:
                    # If we cannot immediately find the answer in the first word, find the first word after a newline character
                    answer_start_idx = normalized_text.lower().find("\n") + len("\n")

                    # Extract the substring starting from the answer
                    answer_text = normalized_text[answer_start_idx:]

                    # Parse the model output to get the first word after the "answer:"
                    matches = re.findall(r'\byes\b|\bno\b', answer_text)
                    if matches:
                        first_word = matches[0] 
                    else:
                        first_word = answer_text.split()[0]

                if testing:
                    # DEBUG
                    print("model output: ", normalized_text)
                    print("predicted answer: ", first_word)
                    print("True Label: ", true_labels[i])

                if "yes" in first_word.lower(): 
                    preds.append(1)
                elif "no" in first_word.lower():  
                    preds.append(0)
                else: 
                    # Append the opposite of the true label, checking for None
                    if true_labels[i] is not None:
                        if true_labels[i] == 0:
                            opposite_value = 1 
                            first_word = "yes"
                        else:
                            opposite_value = 0
                            first_word = "no"
                        # print("DEBUG: Opposite value: ", opposite_value, "\n\n\n")
                        preds.append(opposite_value)

                # Print model outputs
                if testing: 
                    #self.log(f"Prompt: {decoded_texts}")
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

            # For demonstration, print each metric name and value
            if testing:
                for metric_name, metric_value in metrics_dict.items():
                    self.log(f"{metric_name}: {metric_value}")

            # print("\n\n")
            return metrics_dict

        # LoRA CONFIG 
        # https://moon-ci-docs.huggingface.co/docs/peft/pr_721/en/package_reference/tuners#peft.LoraConfig
        target_modules = ['q_proj', 'v_proj']

        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM, 
            target_modules=target_modules, 
            inference_mode=False, 
            r=16, 
            lora_alpha=32, 
            lora_dropout=0.05, 
            bias="none"
        )

        training_arguments = TrainingArguments(
            output_dir=f"/direct/sdcc+u/rengel/results/{self.model_name}", 
            num_train_epochs=self.num_epochs,
            per_device_train_batch_size=self.batch_size,
            optim="adamw_torch", 
            learning_rate=self.lr,
            warmup_steps=0, 
            lr_scheduler_type='constant',
            evaluation_strategy="epoch",
            logging_strategy="epoch"
        )
        
        # Llama 
        if "llama2" in self.model_name.lower() or "llama3" in self.model_name.lower():
            response_template_with_context = "\n### Answer:" 

        # Mistral
        if "mistral" in self.model_name.lower() or "mixtral" in self.model_name.lower():
            response_template_with_context = "[/INST]\n### Answer:" 

        response_template_ids = self.tokenizer.encode(response_template_with_context, add_special_tokens=False)[2:]
        collator = DataCollatorForCompletionOnlyLM(response_template=response_template_ids, tokenizer=self.tokenizer)

        trainer = SFTTrainer(
            model=self.model,
            peft_config=peft_config,
            args=training_arguments,
            train_dataset=self.train_dataset,
            eval_dataset=self.valid_dataset,
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
        self.plot_losses(trainer.state.log_history)

        # Evaluate the model on the test set
        testing = True
        self.log("Evaluation on test set:\n")
        tokenized_test_dataset = self.test_dataset.map(process_test_set, batched=True, batch_size=self.batch_size)
        results = trainer.predict(tokenized_test_dataset)
        print("Evaluation Results:", results)




    def pretrained_model_inference(self):

        self.model.eval()
        predictions, labels = [], []
        self.prompt_counter = 1 
        self.label_counter = 1   
        self.output_counter = 1   
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

            # The following prompts are used for the multi-shot prompting strategy

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
                # Llama
                [f"### Question: Given the options Yes or No, will there be significant deregulation of ACTB (Actin) 24 months after exposure to low-dose radiation at 0.125 Gy?\n### Answer: No\n### Question: Given the options Yes or No, will there be significant deregulation of TUBA1A (Tubulin) 24 months after exposure to low-dose radiation at 0.125 Gy?\n### Answer: No\n### Question: Given the options Yes or No, will there be significant deregulation of MYH7 (Myosin) 24 months after exposure to low-dose radiation at 0.125 Gy?\n### Answer: Yes\n### Question: {q}\n### Answer: " for q in example['question']] if "llama2" in self.model_name.lower() or "llama3" in self.model_name.lower() else 
                # Mistral
                [f"{self.tokenizer.eos_token}[INST]### Question: Given the options Yes or No, will there be significant deregulation of ACTB (Actin) 24 months after exposure to low-dose radiation at 0.5 Gy?[/INST]\n### Answer: No\n[INST]### Question: Given the options Yes or No, will there be significant deregulation of TUBA1A (Tubulin) 24 months after exposure to low-dose radiation at 0.5 Gy?[/INST]\n### Answer: No\n[INST]### Question: Given the options Yes or No, will there be significant deregulation of MYH7 (Myosin) 24 months after exposure to low-dose radiation at 0.5 Gy?[/INST]\n### Answer: Yes\n[INST]### Question: {q}[/INST]\n### Answer: " for q in example['question']] if "mistral" in self.model_name.lower() or "mixtral" in self.model_name.lower() else [],  
                padding="max_length",
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt"
            ) 

            # Convert batch encoding data structure to python list
            for key in tokenized_inputs.keys():
                tokenized_inputs[key].extend(tokenized[key].numpy().tolist())

            return tokenized_inputs
        

        tokenized_test_dataset = self.test_dataset.map(tokenize_test_set, batched=True)

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
                    if matches:
                        answer = matches[0]  
                    else:
                        # If we cannot immediately find the answer in the first word, find the first word after a newline character
                        answer = answer_text.split()[0]

                    # DEBUG
                    print("\n------------------------------")
                    print("model output: ", text)
                    print("answer: ", answer)
                    print("true label: ", true_label)
                    print("IDX: ", index)

                    if "yes" in true_label.lower():
                        labels.append(1)
                    elif "no" in true_label.lower():
                        labels.append(0)
                    else:
                        print("DEBUG")
                        quit()

                    if "yes" in answer.lower(): 
                        predictions.append(1)
                    elif "no" in answer.lower():  
                        predictions.append(0)
                    else: 
                        # Append the opposite of the true label, checking for None
                        print("DEBUG: True Label: ", labels[index])
                        if labels[index] is not None:
                            if labels[index] == 0:
                                opposite_value = 1 
                                answer = "yes"
                            else:
                                opposite_value = 0
                                answer = "no"
                            print("DEBUG: Opposite value: ", opposite_value, "\n\n\n")
                            predictions.append(opposite_value)

                    print("\n")
                    # Print model outputs
                    self.log(f"Model Prediction {self.output_counter}: {answer}")
                    self.output_counter += 1
                    index += 1

        print("Preds: ", predictions)
        print("Labels: ", labels)
        print("Preds: ", len(predictions))
        print("Labels: ", len(labels))

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
