#Necessary Imports
from peft import get_peft_model, TaskType, PeftModel, LoraConfig
import torch
from datasets import Dataset
import random
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef, roc_auc_score, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import re
import pandas as pd
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
from transformers import TrainingArguments
import sys
import os
import torch.nn.functional as F
import shutil
import json

    
def worker_init_fn(worker_id):
    worker_info = torch.utils.data.get_worker_info()
    seed = worker_info.seed % (2**32) 
    seed += worker_id
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)


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
        model_name = "TEMP"
        self.log_file_path = f"/direct/sdcc+u/rengel/results/{model_name}_results.txt"
        self.plot_file_path = f"/direct/sdcc+u/rengel/results/{model_name}_losses.png"
        self.plot_title = f"Loss values for {model_name}"

        # Open the log file in write mode, this will clear previous contents
        with open(self.log_file_path, 'w') as file:
            file.write("")  
 
        # Hyperparameters
        self.lr = 1e-4 
        self.num_epochs = 4
        self.batch_size = 4
        self.new_tokens = 5 
        self.max_length = 120

        # Both Models
        # datasets 1-3
            # max_length = 120
            # num_epochs = 4
        # datasets 4-5
            # max_length = 60
            # num_epochs = 5
            # batch_size = 32
        # dataset 6
            # max_length = 60
            # num_epochs = 5
            # batch_size = 16

        # dataset 1.1 - Batch Size = 4
        # dataset 1.2 - Batch Size = 4
        # dataset 1.3 - Batch Size = 1
        # dataset 2.1 - Batch Size = 2
        # dataset 2.2 - Batch Size = 4
        # dataset 2.3 - Batch Size = 2
        # dataset 2.4 - Batch Size = 2
        # dataset 3.1 - Batch Size = 2 
        # dataset 3.2 - Batch Size = 1


        # Loss values for plots
        self.train_losses = []
        self.valid_losses = []

        # Load Model & Tokenizer
        self.model = model
        self.tokenizer = tokenizer
        self.tokenizer.pad_token = self.tokenizer.eos_token 
         
        # print("Model: ", self.model)

        

    def log(self, message):
        # This function handles writing logs to the specified file
        try:
            with open(self.log_file_path, 'a') as f:
                f.write(message + '\n')
        except Exception as e:
            raise Exception(f"Failed to open and write to file: {self.log_file_path}") from e


    def load_data(self):

        os.chdir('../src/data_processors')
        sys.path.append(os.getcwd())

        # Uncomment the code below for the dataset and prompt you are using
        # This is only necessary for changing the prompts and saving them as json files
        # For loading the current json files skip to the next function
        self.list1 = []
        self.list2 = []

        # Dataset 1
        # set 1: 446 x 2 = 892
        # set 2: 666 x 2 = 1,332
        # set 3: 102 x 2 = 204
        # prompt = f"Given the options Yes or No, will there be significant deregulation of {protein} 24 months after exposure to low-dose radiation at 0.063 Gy?" 
        # prompt = f"Given the options Yes or No, will there be significant deregulation of {protein} 24 months after exposure to low-dose radiation at 0.125 Gy?"
        # prompt = f"Given the options Yes or No, will there be significant deregulation of {protein} 24 months after exposure to low-dose radiation at 0.5 Gy?"
        from dataset_1_processor import d1_processor
        data_file_path = "/direct/sdcc+u/rengel/data/dataset_1_original.xls"
        p1 = d1_processor(data_file_path)
        d1, d2, d3 = p1.load_excel_spreadsheets() 
        for item in d3: # Change this to d1, d2, or d3 depending on the subset of data
            self.list1.append(item[0])  
            self.list2.append(item[1]) 
        print("Length of dataset: ", len(self.list1))
        print("Length of dataset: ", len(self.list2))


        # Dataset 2
        # set 1: 80 x 2 = 160
        # set 2: 99 x 2 = 198
        # set 3: 37 x 2 = 74
        # set 4: 47 x 2 = 94
        # prompt = f"Given the options yes or no, will there be significant deregulation of the protein {protein} 72 hours after exposure to low dose radiation at 2.0 Gy?"
        # prompt = f"Given the options yes or no, will there be significant deregulation of the protein {protein} 1 month after exposure to low dose radiation at 2.0 Gy?"
        # prompt = f"Given the options yes or no, will there be significant deregulation of the protein {protein} 3 months after exposure to low dose radiation at 2.0 Gy?"  
        # prompt = f"Given the options yes or no, will there be significant deregulation of the protein {protein} 6 months after exposure to low dose radiation at 2.0 Gy?"
        # from dataset_2_processor import d2_processor
        # data_file_path = "/direct/sdcc+u/rengel/data/dataset_2_original.xlsx"
        # p2 = d2_processor(data_file_path)
        # d1, d2, d3, d4 = p2.load_excel_spreadsheets()
        # for item in d4[0]:
        #     self.list1.append(item)  
        # for item in d4[1]:
        #     self.list2.append(item)  
        # print("Length of dataset: ", len(self.list1))
        # print("Length of dataset: ", len(self.list2))
        

        # Dataset 3
        # set 1: 49 x 2 = 98
        # set 2: 77 x 2 = 154
        # prompt = f"Given the options yes or no, will there be an altered acetylation status of protein {protein} 4 hours after exposure to low dose radiation at 0.5 Gy?" 
        # prompt = f"Given the options yes or no, will there be an altered acetylation status of protein {protein} 24 hours after exposure to low dose radiation at 0.5 Gy?" 
        # from dataset_3_processor import d3_processor
        # data_file_path = "/direct/sdcc+u/rengel/data/dataset_3_original.xlsx"
        # p3 = d3_processor(data_file_path)
        # d1, d2  = p3.load_excel_spreadsheets()
        # for item in d1[0]:
        #     self.list1.append(item)  
        # for item in d1[1]:
        #     self.list2.append(item)  
        # print("Length of dataset: ", len(self.list1))
        # print("Length of dataset: ", len(self.list2))


        # Dataset 3c
        # 1,111 x 2 = 2,222 
        # prompt = f"Given the options yes or no, will there be deregulation of the protein {protein} after low-dose radiation exposure?""
        # from dataset_3c_processor import d3c_processor
        # data_file_path_1 = "/direct/sdcc+u/rengel/data/dataset_1_original.xls"
        # data_file_path_2 = "/direct/sdcc+u/rengel/data/dataset_2_original.xlsx"
        # data_file_path_3 = "/direct/sdcc+u/rengel/data/dataset_3_original.xlsx"
        # p3c = d3c_processor(data_file_path_1, data_file_path_2, data_file_path_3)
        # d1, d2 = p3c.load_excel_spreadsheets()
        # for item in d1:
        #     self.list1.append(item)
        # for item in d2:
        #     self.list2.append(item)
        # print("Length of dataset: ", len(self.list1))
        # print("Length of dataset: ", len(self.list2))


        # Dataset 4
        # 5,881 x 2 = 11,762 pairs
        # prompt = f"Given the options yes or no, is there a protein interaction between {protein1} and {protein2} in the presence of neurodegenerative diseases?"
        # from dataset_4_5_processor import d4_5_processor
        # data_file_path_1 = "/direct/sdcc+u/rengel/data/dataset_4_original_pros.txt"
        # data_file_path_2 = "/direct/sdcc+u/rengel/data/dataset_4_original_index.txt"
        # p4 = d4_5_processor(data_file_path_1, data_file_path_2)
        # d1, d2  = p4.load_data()
        # for item in d1: self.list1.append(item)   
        # for item in d2: self.list2.append(item) 
        # print("Length of dataset: ", len(self.list1))
        # print("Length of dataset: ", len(self.list2))
        

        # Dataset 5
        # 5,131 x 2 = 10,262 pairs
        # prompt = f"Given the options yes or no, is there a protein interaction between {protein1} and {protein2} in the presence of metabolic diseases?"
        # from dataset_4_5_processor import d4_5_processor
        # data_file_path_1 = "/direct/sdcc+u/rengel/data/dataset_5_original_pros.txt"
        # data_file_path_2 = "/direct/sdcc+u/rengel/data/dataset_5_original_index.txt"
        # p5 = d4_5_processor(data_file_path_1, data_file_path_2)
        # d1, d2  = p5.load_data()
        # for item in d1: self.list1.append(item)   
        # for item in d2: self.list2.append(item)
        # print("Length of dataset: ", len(self.list1))
        # print("Length of dataset: ", len(self.list2))

        # Dataset 6
        # 933 x 2 = 1,866 pairs
        # prompt = f"Given the options yes or no, is there a protein interaction between {protein1} and {protein2} in the presence of cancer?"
        # from dataset_6_processor import d6_processor
        # data_file_path = "/direct/sdcc+u/rengel/data/dataset_6_original.txt"
        # p4 = d6_processor(data_file_path)
        # d1, d2  = p4.load_data()
        # for item in d1: self.list1.append(item)   
        # for item in d2: self.list2.append(item)
        # print("Length of dataset: ", len(self.list1))
        # print("Length of dataset: ", len(self.list2))



        # The following code was used to save the above datasets/prompts into json files
       
        # Datasets 1-3
        # Copy/paste the prompt from above to use for each list
        self.dataset_examples = []
        for protein in self.list1:
            prompt = f"Given the options Yes or No, will there be significant deregulation of {protein} 24 months after exposure to low-dose radiation at 0.063 Gy?" 
            self.dataset_examples.append({'question': prompt, 'answer': 'Yes'})
        for protein in self.list2:
            prompt = f"Given the options Yes or No, will there be significant deregulation of {protein} 24 months after exposure to low-dose radiation at 0.063 Gy?" 
            self.dataset_examples.append({'question': prompt, 'answer': 'No'})

        # Datasets 4-6
        # Copy/paste the prompt from above to use for each list
        # self.dataset_examples = [] 
        # for pos_pair in self.list1:
        #     protein1 = pos_pair[0]
        #     protein2 = pos_pair[1]
        #     prompt = f"Given the options yes or no, is there a protein interaction between {protein1} and {protein2} in the presence of cancer?"
        #     self.dataset_examples.append({'question': prompt, 'answer': 'Yes'})
        # for neg_pair in self.list2:
        #     protein1 = neg_pair[0]
        #     protein2 = neg_pair[1]
        #     prompt = f"Given the options yes or no, is there a protein interaction between {protein1} and {protein2} in the presence of cancer?"
        #     self.dataset_examples.append({'question': prompt, 'answer': 'No'})

        # Use this code to save the prompts to json files
        # with open('/direct/sdcc+u/rengel/data/dataset_1_v3_prompts.json', 'w') as file:
        #     json.dump(self.dataset_examples, file, indent=4)
        # quit()





    def preprocess_data(self):
        
        # Initialize dataset 
        self.dataset_examples = []

        # Path to the JSON file
        json_file_path = '/direct/sdcc+u/rengel/data/dataset_1_v3_prompts.json'

        # Open and read the JSON file
        with open(json_file_path, 'r') as file:
            prompts_data = json.load(file)

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

        # print("Training dataset: ", self.train_dataset)
        # print("Validation dataset: ", self.valid_dataset)
        # print("Testing dataset: ", self.test_dataset)

        # Determine distribution of tokens in each prompt
        # prompt_lengths = []
        # for example in self.test_dataset:  # Assuming dataset_examples is your dataset
        #     # Llama
        #     # prompt = f"### Question: {example['question']}\n### Answer: "

        #     # Mixtral
        #     prompt = f"{self.tokenizer.eos_token}[INST]### Question: {example['question']}[/INST]\n### Answer: "
            
        #     prompt_length = len(self.tokenizer.tokenize(prompt))
        #     prompt_lengths.append(prompt_length)

        # # Analyze the distribution of prompt lengths
        # print(f"Maximum prompt length: {max(prompt_lengths)}")
        # print(f"Average prompt length: {sum(prompt_lengths) / len(prompt_lengths)}")



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
                text = f"### Question: {example['question'][i]}\n### Answer: {example['answer'][i]}"

                # Mistral 
                # text = f"{self.tokenizer.eos_token}[INST]### Question: {example['question'][i]}[/INST]\n### Answer: {example['answer'][i]}"
                
                output_texts.append(text)
            return output_texts
        
        def process_test_set(example):

            for i, (q, a) in enumerate(zip(example['question'], example['answer'])):
                self.log(f"Prompt {i+1}: {q}\nTrue Label: {a}\n")  

            tokenized_inputs = self.tokenizer(
                # Llama
                [f"### Question: {q}\n### Answer: {a}" for q, a in zip(example['question'], example['answer'])],

                # Mistral
                # [f"{self.tokenizer.eos_token}[INST]### Question: {q}[/INST]\n### Answer: {a}" for q, a in zip(example['question'], example['answer'])],

                padding="max_length",
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt"
            )
            return tokenized_inputs
        

        def compute_metrics(eval_preds):         
            
            label_ids = eval_preds.label_ids
            true_labels = []

            # Iterate over each label_id sequence in label_ids
            for i, single_label_ids in enumerate(label_ids):
                if torch.is_tensor(single_label_ids):
                    single_label_ids = single_label_ids.tolist()

                # Filter out -100 values before decoding
                valid_ids = [id_ for id_ in single_label_ids if id_ != -100]

                # Convert valid_ids to a tensor and decode
                ids = torch.tensor(valid_ids)
                decoded_label = self.tokenizer.decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)

                if "yes" in decoded_label.lower(): true_labels.append(1)
                elif "no" in decoded_label.lower():  true_labels.append(0)
                else: true_labels.append(None)
 
            logits = eval_preds.predictions
            probabilities = F.softmax(torch.tensor(logits), dim=-1)
            token_ids = torch.argmax(probabilities, dim=-1).tolist()
            decoded_texts = [self.tokenizer.decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=True) for ids in token_ids]

            preds = []
            for i, text in enumerate(decoded_texts):
                # Find the start of the answer
                answer_start_idx = text.lower().find("answer: ") + len("answer: ")

                # DEBUG
                # print("------------------------------")
                
                # Extract the substring starting from the answer
                answer_text = text[answer_start_idx:]

                # Parse the model output to get the first word after the "answer:"
                matches = re.findall(r'\byes\b|\bno\b', answer_text.lower())
                if matches:
                    first_word = matches[0]  
                else:
                    # If we cannot immediatelyl find the answer in the first word, find the first word after a newline character
                    answer_start_idx = text.lower().find("\n") + len("\n")

                    # Extract the substring starting from the answer
                    answer_text = text[answer_start_idx:]

                    # Parse the model output to get the first word after the "answer:"
                    matches = re.findall(r'\byes\b|\bno\b', answer_text)
                    if matches:
                        first_word = matches[0] 
                    else:
                        first_word = answer_text.split()[0]

                # DEBUG
                # print("decoded text: ", text)
                # print("answer text: ", answer_text)
                # print("first word: ", first_word)

                if "yes" in first_word.lower(): 
                    preds.append(1)
                elif "no" in first_word.lower():  
                    preds.append(0)
                else: 
                    # Append the opposite of the true label, checking for None
                    print("DEBUG: True Label: ", true_labels[i])
                    if true_labels[i] is not None:
                        if true_labels[i] == 0:
                            opposite_value = 1 
                            first_word = "yes"
                        else:
                            opposite_value = 0
                            first_word = "no"
                        print("DEBUG: Opposite value: ", opposite_value, "\n\n\n")
                        preds.append(opposite_value)

                # Print model outputs
                if testing: 
                    #self.log(f"Prompt: {decoded_texts}")
                    self.log(f"Model Prediction {i+1}: {first_word}")

            # Compute metrics
            accuracy = accuracy_score(true_labels, preds)
            mcc = matthews_corrcoef(true_labels, preds)
            auc = roc_auc_score(true_labels, preds)
            tn, fp, fn, tp = confusion_matrix(true_labels, preds).ravel()
            specificity = tn / (tn+fp)

            # Calculate precision, recall, and F1-score for both micro and macro averaging
            precision_micro = precision_score(true_labels, preds, average='micro')
            recall_micro = recall_score(true_labels, preds, average='micro')
            f1_micro = f1_score(true_labels, preds, average='micro')

            precision_macro = precision_score(true_labels, preds, average='macro')
            recall_macro = recall_score(true_labels, preds, average='macro')
            f1_macro = f1_score(true_labels, preds, average='macro')

            metrics = [("Accuracy", accuracy), 
                    ("MCC", mcc), 
                    ("AUC", auc), 
                    ("Specificity", specificity), 
                    ("Micro Precision", precision_micro), 
                    ("Micro Recall", recall_micro), 
                    ("Micro F1 Score", f1_micro),
                    ("Macro Precision", precision_macro), 
                    ("Macro Recall", recall_macro), 
                    ("Macro F1 Score", f1_macro)]
            
            # Convert list of tuples into a dictionary
            metrics_dict = {metric_name: metric_value for metric_name, metric_value in metrics}

            # For demonstration, print each metric name and value
            if testing:
                for metric_name, metric_value in metrics_dict.items():
                    self.log(f"{metric_name}: {metric_value}")

            print("\n\n")
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
            output_dir="/direct/sdcc+u/rengel/results", 
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
        response_template_with_context = "\n### Answer:" 

        # Mistral
        # response_template_with_context = "[/INST]\n### Answer:" 

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
        tokenized_test_dataset = self.test_dataset.map(process_test_set, batched=True)
        results = trainer.predict(tokenized_test_dataset)
        print("Evaluation Results:", results)




    def pretrained_model_inference(self):

        self.model.eval()
        predictions, labels = [], []

        def tokenize_test_set(example):
            # Tokenize Inputs
            tokenized_inputs = self.tokenizer(
                [f"{self.tokenizer.eos_token}[INST]### Question: {q}[/INST]\n### Answer: " # Note this must change based on each model
                for q in example['question']],  
                padding="max_length",
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt"
            )
            
            return tokenized_inputs

        tokenized_test_dataset = self.test_dataset.map(
            tokenize_test_set, 
            batched=True, 
            batch_size=self.batch_size 
        )

        with torch.no_grad():
            for batch in tokenized_test_dataset:
                # Modify the part where you prepare inputs for the model
                inputs = {
                    k: torch.tensor(v, dtype=torch.long) for k, v in batch.items() if k in ['input_ids', 'attention_mask']
                }

                # Ensure each tensor in inputs has a batch dimension
                inputs = {k: v.unsqueeze(0) if v.dim() == 1 else v for k, v in inputs.items()}
                inputs = {k: v.to(self.device) for k, v in inputs.items()}  # Move tensors to the device after unsqueezing

                true_label = batch['answer']  

                generated_sequences = self.model.generate(
                    **inputs,
                    max_new_tokens=self.new_tokens
                )

                for idx, sequence in enumerate(generated_sequences):
                    text = self.tokenizer.decode(sequence, skip_special_tokens=True).lower().strip()

                    # Find the start of the answer
                    answer_start_idx = text.lower().find("answer: ") + len("answer: ")

                    # Extract the substring starting from the answer
                    answer_text = text[answer_start_idx:]

                    # Parse the model output to get the first word after the "answer:"
                    matches = re.findall(r'\byes\b|\bno\b', answer_text)
                    first_word = matches[0] if matches else answer_text.split()[0]

                    # DEBUG
                    # print("------------------------------")
                    # print("decoded text: ", text)
                    # print("answer text: ", answer_text)
                    # print("first word: ", first_word)

                    if "yes" in true_label:
                        labels.append(1)
                    else:
                        labels.append(0)
                    
                    if "yes" in first_word.lower(): 
                        predictions.append(1)
                    elif "no" in first_word.lower():  
                        predictions.append(0)
                    else: 
                        # Append the opposite of the true label, checking for None
                        print("DEBUG: True Label: ", labels[idx])
                        if labels[idx] is not None:
                            if labels[idx] == 0:
                                opposite_value = 1 
                                first_word = "yes"
                            else:
                                opposite_value = 0
                                first_word = "no"
                            print("DEBUG: Opposite value: ", opposite_value, "\n\n\n")
                            predictions.append(opposite_value)

                    # Print model outputs
                    self.log(f"Model Prediction {idx+1}: {first_word}")

        self.predictions = predictions
        self.labels = labels

        print("Preds: ", len(self.predictions))
        print("Labels: ", len(self.labels))



    def calculate_metrics(self):
        labels = self.labels
        predictions = self.predictions

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

        # Micro and Macro Precision, Recall, and F1 Score
        for average in ['micro', 'macro']:
            try:
                precision = precision_score(labels, predictions, average=average)
                metrics.append((f"{average.capitalize()} Precision", precision))
            except Exception as e:
                print(f"Error calculating {average.capitalize()} Precision: {e}")
                metrics.append((f"{average.capitalize()} Precision", None))
                
            try:
                recall = recall_score(labels, predictions, average=average)
                metrics.append((f"{average.capitalize()} Recall", recall))
            except Exception as e:
                print(f"Error calculating {average.capitalize()} Recall: {e}")
                metrics.append((f"{average.capitalize()} Recall", None))
                
            try:
                f1 = f1_score(labels, predictions, average=average)
                metrics.append((f"{average.capitalize()} F1 Score", f1))
            except Exception as e:
                print(f"Error calculating {average.capitalize()} F1 Score: {e}")
                metrics.append((f"{average.capitalize()} F1 Score", None))

        # Log and print metrics
        for metric_name, metric_value in metrics:
            print(f"{metric_name}: {metric_value}")
            self.log(f"{metric_name}: {metric_value}")

