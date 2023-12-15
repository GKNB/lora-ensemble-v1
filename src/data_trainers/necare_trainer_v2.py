#Necessary Imports
from peft import get_peft_model, PromptTuningInit, PromptTuningConfig, TaskType, PeftModel, PeftConfig, LoraConfig, IA3Config
import torch
from torch.utils.data import DataLoader, Dataset
import random
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef, roc_auc_score, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import re


class CustomDataset(Dataset):
    def __init__(self, data_items):
        self.data = data_items

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        prompt, details = item
        return details['input_ids'].squeeze(), details['attention_mask'].squeeze(), details['label'].squeeze()

    
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
     

class necare_trainer():

    def init(self):

        # Initialize device and random seed
        self.device = "cuda"
        self.generator = set_seeds(237)

        # Initialize file paths
        model_name = "BioMedLM-2.7b-Pretrained" 
        
        self.data_file_path = "/direct/sdcc+u/rengel/data/NECARE_TrainingData.txt"
        self.log_file_path = f"/direct/sdcc+u/rengel/results/{model_name}_results.txt"
        self.plot_file_path = f"/direct/sdcc+u/rengel/results/{model_name}_losses.png"
        self.plot_title = f"Average Loss for {model_name}"

        # Open the log file in write mode, this will clear previous contents
        with open(self.log_file_path, 'w') as file:
            file.write("")  

        # Hyperparameters
        self.data_samples = 2500 # This loads the entire dataset, 2241 data points, 933 pos and 1308 neg
        self.lr = 1e-5 
        self.num_epochs = 20
        self.batch_size = 16
        self.weight_decay = .05
        self.max_length = 18 # Max length is roughly the size of the prompt

        # Use only for training?
        self.new_tokens = 20 # Galactica = 3, Llama = 2, MPT = 2, Falcon = 2, BioGPT = 2, BioMedLM = 2

        # Loss values for plots
        self.train_losses = []
        self.valid_losses = []

        # Initialize best fold 
        self.best_fold_valid_losses = None
        self.best_fold_train_losses = None
        self.best_fold = None
        self.best_model = None

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

        #print("Loading PEFT Configuration...")
        #self.log("Loading PEFT Configuration...")

        self.model = model
        self.tokenizer = tokenizer

        print("Model: ", self.model)

        # For the experiments I used the standard target_modules for Galactica, Llama-2, MPT, and Falcon
        # I used these defined target modules for BioGPT and BioMedLM

        # Galactica, Llama-2, BioGPT 
        #target_modules = ['q_proj', 'v_proj']

        # Falcon
        #target_modules = ['query_key_value']

        # MPT
        #target_modules = ['Wqkv']

        # BioMedLM
        #target_modules = ['c_attn', 'c_proj']

        # LoRA CONFIG 
        # https://moon-ci-docs.huggingface.co/docs/peft/pr_721/en/package_reference/tuners#peft.LoraConfig

        # With no target modules set
        #self.peft_config = LoraConfig(task_type=TaskType.CAUSAL_LM, inference_mode=False, r=16, lora_alpha=8, lora_dropout=.1, bias="lora_only")

        # With target modules set
        #self.peft_config = LoraConfig(task_type=TaskType.CAUSAL_LM, target_modules=target_modules, inference_mode=False, r=16, lora_alpha=8, lora_dropout=.1, bias="lora_only")
        #self.model = get_peft_model(self.model, self.peft_config)

        #print("Peft config: ", self.peft_config)
        #input('enter...')

        # IA3 CONFIG
        # https://moon-ci-docs.huggingface.co/docs/peft/pr_721/en/package_reference/tuners#peft.IA3Config
        #self.peft_config = IA3Config(target_modules=1, feedforward_modules=1, modules_to_save=1)
        #self.model = get_peft_model(self.model, self.peft_config)

        # QLoRA CONFIG (Must be used with LoRA configuration)
        
        

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
                        # Creating a list with proteinA, proteinB and interaction ("yes" or "no")
                        interaction = "yes" if parts[2] == 'True' or parts[2] == '1' else "no"  # Adjusting based on how boolean values are represented
                        data_list = [parts[0], parts[1], interaction]
                        data_lists.append(data_list)

            self.data = data_lists


        except FileNotFoundError as e:
            print(f"The file does not exist. Error: {str(e)}")
        except StopIteration:
            print("An error occurred while reading the file.")



    def preprocess_data(self):

        print("Preprocessing Dataset...")
        self.log("Preprocessing Dataset...")

        # Create prompts for training
        prompted_data = {}
        for proteinA, proteinB, interaction in self.data:

            # Prompt 
            prompt = f"Is there a protein interaction between {proteinA} and {proteinB}? "

            # 1-shot Prompt
            #prompt = f"Is there a protein interaction between proteinA and proteinB? No. Is there a protein interaction between {proteinA} and {proteinB}? "

            # 2-shot Prompt v1
            #prompt = f"Is there a protein interaction between proteinA and proteinB? No. Is there a protein interaction between proteinA and proteinB? Yes. Is there a protein interaction between {proteinA} and {proteinB}? "

            # 2-shot Prompt v2
            #prompt = f"Is there a protein interaction between proteinA and proteinB? Yes. Is there a protein interaction between proteinA and proteinB? No. Is there a protein interaction between {proteinA} and {proteinB}? "


            prompted_data[prompt] = interaction

        data = prompted_data
        self.tokenizer.pad_token = "0"
        self.tokenizer.padding_side = 'left'

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

            # Tokenize the label
            encoded_label = self.tokenizer.encode(
                label,
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
                'label': encoded_label,
            }

        # Replace the original data with tokenized data
        data = tokenized_data  

        # Convert the dictionary items into a list
        self.data = list(data.items())




    def create_folds(self):
        print("Creating 5-Folds...")

        self.folds = []

        # Number of samples in the dataset
        num_samples = len(self.data)

        # print("self.data: ", self.data)

        # Generate a shuffled list of indices
        indices = np.arange(num_samples)
        np.random.shuffle(indices)

        # Split the indices into 5 equal parts
        split_indices = np.array_split(indices, 5)

        # Create 5 folds with rotating test, valid, and train splits
        for i in range(5):
            fold_data = {'train': [], 'valid': [], 'test': []}
            fold_data['test'] = split_indices[i]
            fold_data['valid'] = split_indices[(i + 1) % 5]
            for j in range(2, 5):
                fold_data['train'].extend(split_indices[(i + j) % 5])
            self.folds.append(fold_data)

        # Convert index lists to numpy arrays for easier handling later
        for fold in self.folds:
            for key in fold:
                fold[key] = np.array(fold[key])


        # Debug
        # Convert lists to sets
        #set1 = set(self.folds[0]['train'])
        #set2 = set(self.folds[0]['valid'])

        # Find common elements
        #common_elements = set1.intersection(set2)

        # Print or process the common elements
        #print("Common Elements test 1: ", common_elements)




    def data_loader(self, fold):
        
        print(f"Loading Data for Fold {fold+1}...")
        self.log(f"Loading Data for Fold {fold+1}...")
        
        train_idx, valid_idx, test_idx = self.folds[fold]['train'], self.folds[fold]['valid'], self.folds[fold]['test']
        
        data_items = self.data  
        
        train_items = [data_items[i] for i in train_idx]
        valid_items = [data_items[i] for i in valid_idx]
        test_items = [data_items[i] for i in test_idx]

        # Ensure no overlap between train, validation, and test sets
        assert not set(train_idx) & set(valid_idx)
        assert not set(valid_idx) & set(test_idx)
        assert not set(test_idx) & set(train_idx)

        #print("Length of train_items: ", len(train_items))
        #print("Length of val_items: ", len(valid_items))

        # Create instances of the custom dataset for each split
        train_dataset = CustomDataset(train_items)
        valid_dataset = CustomDataset(valid_items)
        test_dataset = CustomDataset(test_items)

        # Initialize the DataLoaders with the datasets
        self.train_dataloader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, worker_init_fn=worker_init_fn, generator=self.generator)
        self.valid_dataloader = DataLoader(valid_dataset, batch_size=self.batch_size, shuffle=False)
        self.test_dataloader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)

        num_pos_train, num_neg_train = 0, 0
        num_pos_valid, num_neg_valid = 0, 0
        num_pos_test, num_neg_test = 0, 0

        for dataloader, (pos_count, neg_count) in zip(
            [self.train_dataloader, self.valid_dataloader, self.test_dataloader],
            [(num_pos_train, num_neg_train), (num_pos_valid, num_neg_valid), (num_pos_test, num_neg_test)]
        ):
            for batch in dataloader:
                input_ids, attention_mask, labels = [b.to(self.device) for b in batch]
                ls = [self.tokenizer.decode(label, skip_special_tokens=True).lower() for label in labels]
                for label in ls:
                    if 'yes' in label:
                        pos_count += 1
                    elif 'no' in label:
                        neg_count += 1

        print("Positive classes in Train: ", num_pos_train)
        print("Negative classes in Train ", num_neg_train)
        self.log(f"Positive classes in Train: {num_pos_train}")
        self.log(f"Negative classes in Train: {num_neg_train}")

        print("Positive classes in Validation: ", num_pos_valid)
        print("Negative classes in Validation: ", num_neg_valid)
        self.log(f"Positive classes in Validation: {num_pos_valid}")
        self.log(f"Negative classes in Validation: {num_neg_valid}")

        print("Positive classes in Test: ", num_pos_test)
        print("Negative classes in Test: ", num_neg_test)
        self.log(f"Positive classes in Test: {num_pos_test}")
        self.log(f"Negative classes in Test: {num_neg_test}")





    def train_model(self):
        print("Training and Evaluating Model...")
        self.log("Training and Evaluating Model...")

        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        self.model.to(self.device)


        print(f"Device: {self.device}")
        self.log(f"Device: {self.device}")

        for epoch in range(self.num_epochs):
            # Training
            self.model.train()
            total_train_loss = 0

            # Read about Supervised fine tuning trainer
            
            for batch in self.train_dataloader:
                optimizer.zero_grad()
                input_ids, attention_mask, labels = [b.to(self.device) for b in batch]
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                loss.backward()
                optimizer.step()
                total_train_loss += loss.item()

            avg_train_loss = total_train_loss / len(self.train_dataloader)
            self.train_losses.append(avg_train_loss)

            # Evaluation
            total_valid_loss = 0
            self.model.eval()
            with torch.no_grad():
                for batch in self.valid_dataloader:
                    input_ids, attention_mask, labels = [b.to(self.device) for b in batch]
                    outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                    loss = outputs.loss
                    total_valid_loss += loss.item()

            avg_valid_loss = total_valid_loss / len(self.valid_dataloader)
            self.valid_losses.append(avg_valid_loss)

            print(f'Epoch [{epoch+1}/{self.num_epochs}], Training Loss: {avg_train_loss}, Validation Loss: {avg_valid_loss}')
            self.log(f'Epoch [{epoch+1}/{self.num_epochs}], Training Loss: {avg_train_loss}, Validation Loss: {avg_valid_loss}')





    def inference(self, data):
        print("Performing Inference...")
        self.log("Performing Inference...")

        self.model.to(self.device)
        self.model.eval()

        predictions, all_true_labels = [], []
        with torch.no_grad():
            for batch in data:
                input_ids, attention_mask, labels = [b.to(self.device) for b in batch]

                # For the pretrained model try using generate function
                generated_sequences = self.model.generate(input_ids=input_ids, attention_mask=attention_mask, max_length=self.max_length + 100)

                # Do this only with the trained model
                # Direct model output 
                # outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)

                # # Greedy decoding for generating a small number of tokens
                # next_tokens = torch.argmax(outputs.logits, dim=-1)
                # generated_sequences = next_tokens[:, -self.new_tokens:]  # Take last n tokens as the generated sequence

                # Decode batch labels just once per batch
                true_labels = [self.tokenizer.decode(label, skip_special_tokens=True).lower() for label in labels]

                for i, seq in enumerate(generated_sequences):

                    # Decoding the tokens to text
                    decoded_output = self.tokenizer.decode(seq, skip_special_tokens=True).replace('0', ' ').strip()
                    print("Generated Sequence: ", decoded_output)
                    self.log(f"Generated Sequence: {decoded_output}")

                    # Extract first 'yes' or 'no' from the decoded output
                    matches = re.findall(r'Yes|yes|No|no', decoded_output)
                    first_word = matches[0] if matches else ''

                    # Determine true label based on decoded label
                    if "yes" in true_labels[i]:
                        true_label = 1
                    elif "no" in true_labels[i]:
                        true_label = 0

                    # Creating binary predictions based on the model's text output
                    if "yes" == first_word or "Yes" == first_word:
                        prediction = 1
                    elif "no" == first_word or "No" == first_word:
                        prediction = 0
                    else:
                        # Append the opposite value of x 
                        if true_label == 0:
                            prediction = 1
                        else:
                            prediction = 0

                    print("Prediction: ", prediction)
                    print("True Label: ", true_label)
                    self.log(f"Prediction: {prediction}")
                    self.log(f"True Label: {true_label}")

                    predictions.append(prediction)
                    all_true_labels.append(true_label)

        return predictions, all_true_labels




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


        metrics = [("Accuracy", accuracy), ("MCC", mcc), ("AUC", auc), ("Specificity", specificity), 
                   ("Micro Precision", precision_micro), ("Micro Recall", recall_micro), ("Micro F1 Score", f1_micro),
                   ("Macro Precision", precision_macro), ("Macro Recall", recall_macro), ("Macro F1 Score", f1_macro)]
        
        for metric_name, metric_value in metrics:
            print(f"{metric_name}: {metric_value}")
            self.log(f"{metric_name}: {metric_value}")


    def plot_losses(self):

        epochs = range(1, self.num_epochs + 1)
        
        plt.figure()
        plt.plot(epochs, self.best_fold_train_losses, 'r', label='Training Loss')
        plt.plot(epochs, self.best_fold_valid_losses, 'b', label='Validation Loss')
        plt.title(self.plot_title)
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig(self.plot_file_path) 
        plt.close() 



    def process_fold(self, fold):
        # This function should be called k times for the k folds, and picks the fold that trains the best model
        # We decide which model is "best" based on the lowest validation loss of that model
        
        # Load the data for the current fold
        self.data_loader(fold)
        #print("successfully loaded data fold: ", fold)

        # Train the model on that fold's train set
        self.train_model() 

        # Evaluate on the model's validation set
        predictions, labels = self.inference(self.valid_dataloader)
        self.calculate_metrics(predictions, labels) 

        # Check if the current fold has the lowest validation loss
        if not self.best_fold_valid_losses or self.valid_losses[-1] < self.best_fold_valid_losses[-1]:

            # Save the loss values for the model just trained, because the new model is better
            self.best_fold_train_losses = self.train_losses.copy()
            self.best_fold_valid_losses = self.valid_losses.copy()
            self.best_fold = fold
            self.best_model = self.model

        # Reset the values in self.train_losses and self.valid_losses for the next fold
        # We do this because we only care about the loss values for the best model
        self.train_losses = []
        self.valid_losses = []


    def evaluate_best_model(self, best_model_filepath):

        # Save model
        #best_model_filepath = f"{best_model_filepath}_{self.best_fold}"
        #torch.save(self.best_model.state_dict(), best_model_filepath)
        #self.model = self.best_model

        

        # Load model
        #self.model.load_state_dict(torch.load(best_model_filepath))
        #match = re.search(r'(\d+)$', best_model_filepath)
        #self.best_fold = int(match.group(1))

        # Pretrained model
        self.best_fold = 4

        # Load the data for the best fold
        self.data_loader(self.best_fold)

        print("best model: ", self.best_fold)
        self.log(f"best model: {self.best_fold}")
    
        # Evaluate the best model on that fold's test set
        predictions, labels = self.inference(self.test_dataloader)

        # Calculate metrics for the best model and plot losses graph
        self.calculate_metrics(predictions, labels)  
        #self.plot_losses()









