#Necessary Imports
from transformers import AutoModelForCausalLM, get_linear_schedule_with_warmup
from peft import get_peft_model, PromptTuningInit, PromptTuningConfig, TaskType, PeftModel, PeftConfig, LoraConfig, IA3Config
from huggingface_hub import login, HfApi
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
from transformers import AdamW
from torch.nn import BCEWithLogitsLoss
import random
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef, roc_auc_score, roc_curve, confusion_matrix
import numpy as np
import sys
import os
import matplotlib.pyplot as plt

class CustomDataset(Dataset):
    def __init__(self, data_items):
        self.data = data_items

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        prompt, details = item
        return details['input_ids'].squeeze(), details['attention_mask'].squeeze(), torch.tensor(details['label'])
    
def worker_init_fn(worker_id):
    worker_info = torch.utils.data.get_worker_info()
    seed = worker_info.seed % (2**32) 
    seed += worker_id
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)

def set_seeds(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    generator = torch.Generator().manual_seed(seed)  
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    return generator  
     

class necare_trainer():

    def init(self):

        # Initialize device and random seed
        self.device = "cuda"
        self.generator = set_seeds(42)

        if torch.cuda.is_available():
            print("CUDA is available")
        else:
            print("CUDA is not available")

        # Initialize file paths
        model_name = "Test" 
        self.data_file_path = "/direct/sdcc+u/rengel/data/NECARE_TrainingData.txt"
        self.log_file_path = f"/direct/sdcc+u/rengel/results/{model_name}_results.txt"
        self.plot_file_path = f"/direct/sdcc+u/rengel/results/{model_name}_losses.png"
        self.plot_title = f"Average Loss for {model_name}"

        # Open the log file in write mode, this will clear previous contents
        with open(self.log_file_path, 'w') as file:
            file.write("")  
            
        print("Initializing model trainer...")
        self.log("Initializing model trainer...")

        # Hyperparameters
        self.data_samples = 2500
        self.lr = .0001
        self.num_epochs = 5
        self.batch_size = 16
        self.max_length = 20
        self.weight_decay = .1

        # Loss values for plots
        self.train_losses = []
        self.valid_losses = []

        # Metrics 
        self.accuracies = []
        self.mccs = []
        self.aucs = []
        self.specificities = []
        self.micro_precisions = []
        self.micro_recalls = []
        self.micro_f1_scores = []
        self.macro_precisions = []
        self.macro_recalls = []
        self.macro_f1_scores = []


    def load_peft_config(self, model, tokenizer):

        print("Loading PEFT Configuration...")
        self.log("Loading PEFT Configuration...")

        self.model = model
        self.tokenizer = tokenizer

        # LoRA CONFIG 
        # https://moon-ci-docs.huggingface.co/docs/peft/pr_721/en/package_reference/tuners#peft.LoraConfig
        #self.peft_config = LoraConfig(task_type=TaskType.CAUSAL_LM, inference_mode=False, r=16, lora_alpha=8, lora_dropout=.1, bias="lora_only")
        #self.model = get_peft_model(self.model, self.peft_config)

        # IA3 CONFIG
        # https://moon-ci-docs.huggingface.co/docs/peft/pr_721/en/package_reference/tuners#peft.IA3Config
        #self.peft_config = IA3Config(target_modules=1, feedforward_modules=1, modules_to_save=1)
        #self.model = get_peft_model(self.model, self.peft_config)

        

    def log(self, message):
        # This function handles writing logs to the specified file
        try:
            with open(self.log_file_path, 'a') as f:
                f.write(message + '\n')
        except Exception as e:
            raise Exception(f"Failed to open and write to file: {self.log_file_path}") from e


    def load_dataset(self):

        print("Loading Dataset...")
        self.log("Loading Dataset...")

        data_lists = []

        num_lines = self.data_samples
        try:
            with open(self.data_file_path, 'r') as file:
                # Skipping the header line
                next(file)

                # Reading specified number of lines
                for count, line in enumerate(file, start=1):  # start=1 to account for the header
                    if count == num_lines:
                        #print("Reached the specified number of lines.")
                        break

                    parts = line.strip().split()
                    if parts:  # checking if the line is not empty
                        # Creating a list with proteinA, proteinB and interaction boolean value
                        data_list = [parts[0], parts[1], parts[2]]
                        data_lists.append(data_list)

            self.data = data_lists

        except FileNotFoundError as e:
            print(f"The file does not exist. Error: {str(e)}")
        except StopIteration:
            print("An error occurred while reading the file.")


    def preprocess_data(self):

        print("Preprocessing Dataset...")
        self.log("Preprocessing Dataset...")

        # Create prompts for training the LLM
        prompted_data = {}
        for proteinA, proteinB, interaction in self.data:
            prompt = f"Is there a protein interaction between {proteinA} and {proteinB}?"
            #prompt = f"In the context of cancer, do proteins {proteinA} and {proteinB} interact?"
            #prompt = f"In cancer biology, do proteins {proteinA} and {proteinB} interact?"

            prompted_data[prompt] = interaction

        data = prompted_data
        self.tokenizer.pad_token = "0"

        tokenized_data = {}  # A new dictionary to store the tokenized data
        for prompt, label in data.items():

            # Tokenize the prompt using the model's tokenizer
            encoded_prompt = self.tokenizer.encode_plus(
                prompt,
                add_special_tokens=True,
                max_length=self.max_length,  
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            )

            # Save the tokenized data
            tokenized_data[prompt] = {
                'input_ids': encoded_prompt['input_ids'],
                'attention_mask': encoded_prompt['attention_mask'],
                'label': 1 if label == '1' else 0 
            }

        # Replace the original data with tokenized data
        data = tokenized_data  

        # Convert the dictionary items into a list
        data_items = list(data.items())

        # Randomly shuffle the data items
        random.seed(42)
        random.shuffle(data_items)

        self.data = data_items


    def create_folds(self):

        print("Creating 5-Folds...")
        self.log("Creating 5-Folds...")
        
        kfold = KFold(n_splits=5, shuffle=True, random_state=42)
    
        # Convert the generator to a list before subscripting
        splits = list(kfold.split(self.data))
        
        self.folds = [
            {
                'train': train_idx,
                'valid': test_idx,
                'test': splits[(i+1)%5][1]
            }
            for i, (train_idx, test_idx) in enumerate(splits)
        ]

    def data_loader(self, fold):
            
        print(f"Loading Data for Fold {fold+1}...")
        self.log(f"Loading Data for Fold {fold+1}...")
        
        train_idx, valid_idx, test_idx = self.folds[fold]['train'], self.folds[fold]['valid'], self.folds[fold]['test']
        
        # Since self.data is already a list, you can directly use it
        data_items = self.data  

        train_items = [data_items[i] for i in train_idx]
        valid_items = [data_items[i] for i in valid_idx]
        test_items = [data_items[i] for i in test_idx]

        # Create instances of the custom dataset for each split
        train_dataset = CustomDataset(train_items)
        valid_dataset = CustomDataset(valid_items)
        test_dataset = CustomDataset(test_items)

        # Initialize the DataLoaders with the datasets
        self.train_dataloader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, worker_init_fn=worker_init_fn, generator=self.generator)
        self.valid_dataloader = DataLoader(valid_dataset, batch_size=self.batch_size, shuffle=False)
        self.test_dataloader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)


    def train_model(self):

        print("Training Model...")
        self.log("Training Model...")

        # Define optimizer
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        # Define loss function
        self.loss_fn = BCEWithLogitsLoss()

        # Move model to GPU if available
        self.model.to(self.device)
        print(f"Device: {self.device}")
        self.log(f"Device: {self.device}")

        for epoch in range(self.num_epochs):

            # Training
            self.model.train()
            total_train_loss = 0
            for batch in self.train_dataloader:
                optimizer.zero_grad()

                input_ids, attention_mask, labels = [b.to(self.device) for b in batch]

                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask).logits
                outputs = outputs.view(outputs.size(0), -1).mean(dim=1)

                labels = labels.float()
                loss = self.loss_fn(outputs, labels)

                loss.backward()
                optimizer.step()

                total_train_loss += loss.item()

            avg_train_loss = total_train_loss / len(self.train_dataloader)
            self.train_losses.append(avg_train_loss)

            # Evaluation
            self.model.eval()
            total_valid_loss = 0
            with torch.no_grad():
                for batch in self.valid_dataloader:
                    input_ids, attention_mask, labels = [b.to(self.device) for b in batch]

                    outputs = self.model(input_ids=input_ids, attention_mask=attention_mask).logits
                    outputs = outputs.view(outputs.size(0), -1).mean(dim=1)

                    labels = labels.float()
                    loss = self.loss_fn(outputs, labels)

                    total_valid_loss += loss.item()

            avg_valid_loss = total_valid_loss / len(self.valid_dataloader)
            self.valid_losses.append(avg_valid_loss)

            print(f'Epoch [{epoch+1}/{self.num_epochs}], Training Loss: {avg_train_loss}, Validation Loss: {avg_valid_loss}')
            self.log(f'Epoch [{epoch+1}/{self.num_epochs}], Training Loss: {avg_train_loss}, Validation Loss: {avg_valid_loss}')


    def inference(self, dataloader):

        print("Performing Inference...")
        self.log("Performing Inference...")
        
        # Move model to GPU if available
        self.model.to(self.device)
        self.model.eval()
        
        predictions, true_labels = [], []
        with torch.no_grad():
            for batch in dataloader:
                input_ids, attention_mask, labels = [b.to(self.device) for b in batch]
                
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask).logits
                outputs = outputs.view(outputs.size(0), -1).mean(dim=1)
                
                # Applying sigmoid and rounding off to get binary predictions
                preds = torch.sigmoid(outputs).round().long()  

                predictions.extend(preds.cpu().numpy())
                true_labels.extend(labels.cpu().numpy())
                
        return predictions, true_labels

    
    def calculate_metrics(self, predictions, labels):

        print("Calculating Metrics...")
        self.log("Calculating Metrics...")

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

        print(f"Accuracy: {accuracy}")
        print(f"MCC: {mcc}")
        print(f"AUC: {auc}")
        print(f"Specificity: {specificity}")

        print(f"Micro Precision: {precision_micro}")
        print(f"Micro Recall: {recall_micro}")
        print(f"Micro F1 Score: {f1_micro}")

        print(f"Macro Precision: {precision_macro}")
        print(f"Macro Recall: {recall_macro}")
        print(f"Macro F1 Score: {f1_macro}\n")

        
        self.log(f"Accuracy: {accuracy}")
        self.log(f"MCC: {mcc}")
        self.log(f"AUC: {auc}")
        self.log(f"Specificity: {specificity}")

        self.log(f"Micro Precision: {precision_micro}")
        self.log(f"Micro Recall: {recall_micro}")
        self.log(f"Micro F1 Score: {f1_micro}")

        self.log(f"Macro Precision: {precision_macro}")
        self.log(f"Macro Recall: {recall_macro}")
        self.log(f"Macro F1 Score: {f1_macro}\n")

        self.accuracies.append(accuracy)
        self.mccs.append(mcc)
        self.aucs.append(auc)
        self.specificities.append(specificity)

        self.micro_precisions.append(precision_micro)
        self.micro_recalls.append(recall_micro)
        self.micro_f1_scores.append(f1_micro)

        self.macro_precisions.append(precision_macro)
        self.macro_recalls.append(recall_macro)
        self.macro_f1_scores.append(f1_macro)
    

    def process_fold(self, fold):
        
        self.data_loader(fold)
        #self.train_model() 
        predictions, labels = self.inference(self.valid_dataloader)
        self.calculate_metrics(predictions, labels)


    def plot_losses(self):

        epochs = range(1, self.num_epochs + 1)
        
        # Calculate the average loss per epoch across all folds for training and validation
        self.avg_train_losses = [np.mean(self.train_losses[i::self.num_epochs]) for i in range(self.num_epochs)]
        self.avg_valid_losses = [np.mean(self.valid_losses[i::self.num_epochs]) for i in range(self.num_epochs)]
        
        plt.figure()
        plt.plot(epochs, self.avg_train_losses, 'r', label='Average Training Loss')
        plt.plot(epochs, self.avg_valid_losses, 'b', label='Average Validation Loss')
        plt.title(self.plot_title)
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig(self.plot_file_path) 
        plt.close() 


    
    def log_average_metrics(self):
        
        print("Calculating Average Metrics Across Folds...")
        self.log("Calculating Average Metrics Across Folds...")

        #print("Average training loss per epoch: ", self.avg_train_losses)
        #print("Average validation loss per epoch: ", self.avg_valid_losses)

        #self.log(f"Average training loss per epoch: {self.avg_train_losses}")
        #self.log(f"Average validation loss per epoch: {self.avg_valid_losses}")

        metrics = [("Accuracy", self.accuracies), ("MCC", self.mccs), ("AUC", self.aucs), ("Specificity", self.specificities), 
                   ("Micro Precision", self.micro_precisions), ("Micro Recall", self.micro_recalls), ("Micro F1 Score", self.micro_f1_scores),
                   ("Macro Precision", self.macro_precisions), ("Macro Recall", self.macro_recalls), ("Macro F1 Score", self.macro_f1_scores)]
        
        for metric_name, metric_values in metrics:
            avg_metric = np.mean(metric_values)
            print(f"Average {metric_name}: {avg_metric}")
            self.log(f"Average {metric_name}: {avg_metric}")









