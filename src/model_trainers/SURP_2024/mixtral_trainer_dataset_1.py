 #Necessary Imports
from peft import get_peft_model, PromptTuningInit, PromptTuningConfig, TaskType, PeftModel, PeftConfig, LoraConfig, IA3Config
import torch
from datasets import Dataset
import random
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef, roc_auc_score, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import re
import pandas as pd
from sklearn.model_selection import train_test_split
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
from transformers import BitsAndBytesConfig, TrainingArguments, pipeline, logging
import sys
import os
import torch.nn.functional as F
import shutil


            

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

        print("Initializing...")

        # Initialize device and random seed
        self.device = "cuda"
        self.generator = set_seeds(237)

        # Initialize file paths
        model_name = "Test-Mixtral-8x7B" 
        
        self.data_file_path = "/direct/sdcc+u/rengel/data/Table_S1_S9.xls"
        self.log_file_path = f"/direct/sdcc+u/rengel/results/{model_name}_results.txt"
        self.plot_file_path = f"/direct/sdcc+u/rengel/results/{model_name}_losses.png"
        self.plot_title = f"Average Loss for {model_name}"

        # Open the log file in write mode, this will clear previous contents
        with open(self.log_file_path, 'w') as file:
            file.write("")  

        # Hyperparameters
        self.lr = 1e-4 
        self.num_epochs = 10
        self.batch_size = 16
        self.weight_decay = 0.25
        self.new_tokens = 5 
        self.max_length = 100

        # Loss values for plots
        self.train_losses = []
        self.valid_losses = []

        # Initialize PEFT config
        self.model = model
        self.tokenizer = tokenizer



    def load_data(self):

        print("Loading Dataset...")
        self.log("Loading Dataset...")

        # Call dataset_1_processor and get the 3 lists
        os.chdir('../src/data_processors')
        sys.path.append(os.getcwd())
        from dataset_1_processor import d1_processor
        self.data_file_path = "/direct/sdcc+u/rengel/data/Table_S1_S9.xls"
        p1 = d1_processor(self.data_file_path)
        d1, d2, d3 = p1.load_excel_spreadsheets()
        # Each of the 3 datasets represents a different radiation group
        # list1 represents Deregulated Proteins
        # list2 represents Proteins Unaffected
        self.list1 = []
        self.list2 = []
        for item in d1:
            self.list1.append(item[0])  
            self.list2.append(item[1]) 

        print("Length of dataset: ", len(self.list1))
        print("Length of dataset: ", len(self.list2))

        # Call dataset_2_processor and get the 4 lists
        # from dataset_2_processor import d2_processor
        # self.data_file_path = "/direct/sdcc+u/rengel/data/Proteinlist.xlsx"
        # p2 = d2_processor(self.data_file_path)
        # d1, d2, d3, d4 = p2.load_excel_spreadsheets()
        # set 1: 80 x 2 = 160
        # set 2: 99 x 2 = 198
        # set 3: 37 x 2 = 74
        # set 4: 47 x 2 = 94

        # Call dataset_3_processor and get the 2 lists
        # from dataset_3_processor import d3_processor
        # self.data_file_path = "/direct/sdcc+u/rengel/data/Protein_Acetylation.xlsx"
        # p3 = d3_processor(self.data_file_path)
        # d1, d2  = p3.load_excel_spreadsheets()
        # print("Test: ", d2[0], "\n", d2[1])
        # print("Test: ", len(d2[0]), "\n", len(d2[1]))

        # Call dataset_4_processor and get the 2 lists for Dataset 4
        # from dataset_4_processor import d4_processor
        # self.data_file_path_1 = "/direct/sdcc+u/rengel/data/neurodegenerative_pros_AB.txt"
        # self.data_file_path_2 = "/direct/sdcc+u/rengel/data/neurodegenerative_pro_index.txt"
        # p4 = d4_processor(self.data_file_path_1, self.data_file_path_2)
        # d1, d2  = p4.load_data()
        # print("Test: ", d1[0], "\n", d2[0])
        # print("Test: ", len(d1), "\n", len(d2))

        # Call dataset_4_processor and get the 2 lists for Dataset 5
        # from dataset_4_processor import d4_processor
        # self.data_file_path_1 = "/direct/sdcc+u/rengel/data/metabolic_pros_AB.txt"
        # self.data_file_path_2 = "/direct/sdcc+u/rengel/data/metabolic_pro_index.txt"
        # p4 = d4_processor(self.data_file_path_1, self.data_file_path_2)
        # d1, d2  = p4.load_data()
        # print("Test: ", d1[0], "\n", d2[0])
        # print("Test: ", len(d1), "\n", len(d2))


    

    def log(self, message):
        # This function handles writing logs to the specified file
        try:
            with open(self.log_file_path, 'a') as f:
                f.write(message + '\n')
        except Exception as e:
            raise Exception(f"Failed to open and write to file: {self.log_file_path}") from e




    def preprocess_data(self):

        print("Preprocessing Dataset...")
        self.tokenizer.pad_token = self.tokenizer.eos_token

        dataset_examples = []  # This will be a list of dictionaries
        for protein in self.list1:
            prompt = f"Given the options Yes, or No, will there be a change in the hippocampal protein {protein} after exposure to low-dose ionizing radiation?"
            dataset_examples.append({'question': prompt, 'answer': 'Yes'})

        for protein in self.list2:
            prompt = f"Given the options Yes, or No, will there be a change in the hippocampal protein {protein} after exposure to low-dose ionizing radiation?"
            dataset_examples.append({'question': prompt, 'answer': 'No'})

        # Shuffle the dataset to ensure a mix of 'Yes' and 'No' answers throughout
        random.shuffle(dataset_examples)
        
        # Split the dataset into training, validation, and possibly test sets
        total_items = len(dataset_examples)
        train_end = int(total_items * 0.8)
        valid_end = train_end + int(total_items * 0.1)
        
        train_dataset = dataset_examples[:train_end]
        valid_dataset = dataset_examples[train_end:valid_end]
        test_dataset = dataset_examples[valid_end:]

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
        prompt_lengths = []
        for example in self.test_dataset:  # Assuming dataset_examples is your dataset
            prompt = f"{self.tokenizer.eos_token}[INST]Question: {example['question']}\n[/INST]Answer: {self.tokenizer.eos_token}"
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

        def formatting_prompts_func(example):
            output_texts = []
            for i in range(len(example['question'])):
                text = f"{self.tokenizer.eos_token}[INST]Question: {example['question'][i]}\n[/INST]Answer: {self.tokenizer.eos_token} {example['answer'][i]}"
                output_texts.append(text)
            return output_texts
        
        def process_test_set(example):
            tokenized_inputs = self.tokenizer(
                [f"{self.tokenizer.eos_token}[INST]Question: {q}\n[/INST]Answer: {a}" for q, a in zip(example['question'], example['answer'])],
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
            for single_label_ids in label_ids:
                if torch.is_tensor(single_label_ids):
                    single_label_ids = single_label_ids.tolist()

                # Filter out -100 values before decoding
                valid_ids = [id_ for id_ in single_label_ids if id_ != -100]

                # Convert valid_ids to a tensor and decode
                ids = torch.tensor(valid_ids)
                decoded_label = self.tokenizer.decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)

                # Debug
                #print("Label: ", decoded_label)

                # Check for "yes" or "no" in the decoded label
                if "yes" in decoded_label.lower(): true_labels.append(1)
                elif "no" in decoded_label.lower():  true_labels.append(0)
                else: true_labels.append(None)
 

            ###
                    

            logits = eval_preds.predictions
            probabilities = F.softmax(torch.tensor(logits), dim=-1)
            token_ids = torch.argmax(probabilities, dim=-1).tolist()
            decoded_texts = [self.tokenizer.decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=True) for ids in token_ids]

            preds = []
            for i, text in enumerate(decoded_texts):
                # Find the start of the answer
                answer_start_idx = text.find("Answer: ") + len("Answer: ")
                
                # Extract the substring starting from the answer
                answer_text = text[answer_start_idx:]
                
                # Split the answer text into words and take the first one
                first_word = answer_text.split()[0] if answer_text.split() else "No answer found"
                
                #print(f"First word in Answer {i+1}: ", first_word)

                if "yes" in first_word.lower(): preds.append(1)
                elif "no" in first_word.lower():  preds.append(0)
                else: true_labels.append(None)

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
            # for metric_name, metric_value in metrics_dict.items():
            #     print(f"{metric_name}: {metric_value}")

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
            weight_decay=self.weight_decay,
            evaluation_strategy="epoch",
            logging_strategy="epoch"
        )
        
        response_template_with_context = "\n[/INST]Answer:" 
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

        trainer.train()
        print("LOG HISTORY: ", trainer.state.log_history)
        self.plot_losses(trainer.state.log_history)

        # Apply the processing function to the entire test dataset
        tokenized_test_dataset = self.test_dataset.map(process_test_set, batched=True)
        # print("Tokenized Test Dataset: ", tokenized_test_dataset)

        results = trainer.predict(tokenized_test_dataset)
        print("Evaluation Results:", results)




    def inference(self):
        print("Inference...")
        self.log("Inference...")

        #self.model.to(self.device)
        self.model.eval()

        predictions, all_true_labels = [], []
        with torch.no_grad():

            for batch in self.test_dataloader:

                input_ids, labels = batch

                # Remove the unexpected second dimension
                input_ids = input_ids.squeeze(1).to(self.device)

                # Pass the reshaped input_ids to the model
                outputs = self.model.generate(input_ids, max_new_tokens=self.new_tokens)
                decoded_outputs = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)

                # Decode batch labels just once per batch
                true_labels = labels 

                for decoded_output, true_label in zip(decoded_outputs, true_labels):
                    print("Generated Sequence: ", decoded_output)
                    self.log(f"Generated Sequence: {decoded_output}")

                    # Extract first 'yes' or 'no' from the decoded output
                    print("New Tokens: ", decoded_output[self.max_length:].lower())
                    matches = re.findall(r'yes, |yes,|yes|no, |no,|\'no|`no', decoded_output[self.max_length:].lower())
                    first_word = matches[0] if matches else ''

                    # Determine true label based on decoded label
                    if "yes" in true_label:
                        true_label = 1
                    elif "no" in true_label:
                        true_label = 0
                    else:
                        # DEBUG
                        print("Error: ", true_label)

                    # # Create binary predictions based on the model's text output
                    if "yes, " == first_word or "yes," == first_word or "yes" == first_word:
                        prediction = 1
                    elif "no, " == first_word or "no," == first_word or "\'no\'" == first_word or "`no" == first_word:
                        prediction = 0
                    else:
                        # Append the opposite value of x 
                        print("Model Unsure, marking as incorrect")
                        if true_label == 0:
                            prediction = 1
                        else:
                            prediction = 0

                    print(f"Prediction: {prediction}")
                    print(f"True Label: {true_label}\n\n")
                    self.log(f"Prediction: {prediction}")
                    self.log(f"True Label: {true_label}\n\n")

                    predictions.append(prediction)
                    all_true_labels.append(true_label)

        self.predictions = predictions
        self.labels = all_true_labels


    def calculate_metrics(self):

        print("Calculating Metrics...")
        self.log("Calculating Metrics...")

        labels = self.labels
        predictions = self.predictions

        accuracy = accuracy_score(labels, predictions)
        mcc = matthews_corrcoef(labels, predictions)
        auc = roc_auc_score(labels, predictions)
        tn, fp, fn, tp = confusion_matrix(labels, predictions).ravel()
        specificity = tn / (tn+fp)

        # Calculate precision, recall, and F1-score for both micro and macro averaging
        precision_micro = precision_score(labels, predictions, average='micro')
        recall_micro = recall_score(labels, predictions, average='micro')
        f1_micro = f1_score(labels, predictions, average='micro')

        precision_macro = precision_score(labels, predictions, average='macro')
        recall_macro = recall_score(labels, predictions, average='macro')
        f1_macro = f1_score(labels, predictions, average='macro')

        metrics = [
            ("Accuracy", accuracy),
            ("MCC", mcc),
            ("AUC", auc),
            ("Specificity", specificity),
            ("Micro Precision", precision_micro), 
            ("Micro Recall", recall_micro),       
            ("Micro F1 Score", f1_micro),        
            ("Macro Precision", precision_macro), 
            ("Macro Recall", recall_macro),       
            ("Macro F1 Score", f1_macro)          
        ]

        # Assuming you want to print out the metrics
        for metric_name, metric_value in metrics:
            print(f"{metric_name}: {metric_value}")
            self.log(f"{metric_name}: {metric_value}")


    def plot_losses(self):
        epochs = range(1, self.num_epochs + 1)
        plt.figure()
        plt.plot(epochs, self.train_losses, 'r', label='Training Loss')
        plt.plot(epochs, self.valid_losses, 'b', label='Validation Loss')
        plt.title(self.plot_title)
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig(self.plot_file_path) 
        plt.close() 

