 #Necessary Imports
from peft import get_peft_model, PromptTuningInit, PromptTuningConfig, TaskType, PeftModel, PeftConfig, LoraConfig, IA3Config
import torch
from torch.utils.data import DataLoader, Dataset, Subset
import random
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef, roc_auc_score, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from collections import Counter



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
        model_name = "Galactica-7b-LoRA" 
        
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
        self.num_epochs = 5
        self.batch_size = 16
        self.weight_decay = .1
        self.max_length = 18 # Max length is the size of the prompt
        self.new_tokens = 3 # Galactica needs 3 tokens

        # Loss values for plots
        self.train_losses = []
        self.valid_losses = []

        # Initialize best fold 
        self.best_fold_valid_losses = None
        self.best_fold_train_losses = None
        self.best_fold = None

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

        # LoRA CONFIG 
        # https://moon-ci-docs.huggingface.co/docs/peft/pr_721/en/package_reference/tuners#peft.LoraConfig
        self.peft_config = LoraConfig(task_type=TaskType.CAUSAL_LM, inference_mode=False, r=16, lora_alpha=8, lora_dropout=.1, bias="lora_only")
        self.model = get_peft_model(self.model, self.peft_config)

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

        yes_interactions = []
        no_interactions = []
        data_lists = [] # Used for the unbalanced dataset

        num_lines = self.data_samples
        try:
            with open(self.data_file_path, 'r') as file:
                # Skipping the header line
                next(file)

                # Reading specified number of lines
                for count, line in enumerate(file, start=1):  # start=1 to account for the header
                    if count > num_lines:
                        break

                    parts = line.strip().split()
                    if parts:  # checking if the line is not empty
                        # Creating a list with proteinA, proteinB, and interaction ("yes" or "no")
                        interaction = "yes" if parts[2] == 'True' or parts[2] == '1' else "no"
                        data_list = [parts[0], parts[1], interaction]

                        # Used for the unbalanced dataset
                        data_lists.append(data_list)
                        self.data = data_lists

                        '''
                        if interaction == "yes":
                            yes_interactions.append(data_list)
                        else:
                            no_interactions.append(data_list)

            # Balancing the dataset
            min_size = min(len(yes_interactions), len(no_interactions))
            balanced_data = yes_interactions[:min_size] + no_interactions[:min_size]

            print("Yes Amount: ", len(yes_interactions[:min_size]))
            print("No Amount: ", len(no_interactions[:min_size]))
            
            # Optionally shuffle the balanced dataset
            np.random.shuffle(balanced_data)

            self.data = balanced_data
            '''
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
        print("Creating 60/20/20 Train/Validation/Test Split...")

        self.folds = []

        # Number of samples in the dataset
        num_samples = len(self.data)

        # Generate a shuffled list of indices
        indices = np.arange(num_samples)
        np.random.shuffle(indices)

        # Split the indices into train and temp with 60/40 split
        train_idx, temp_idx = train_test_split(indices, test_size=0.4, random_state=42)

        # Split the temp_idx further into validation and test each with 50% of temp_idx (which is 20% of the original dataset)
        valid_idx, test_idx = train_test_split(temp_idx, test_size=0.5, random_state=42)

        # Create the fold data with the static splits
        fold_data = {'train': train_idx, 'valid': valid_idx, 'test': test_idx}
        self.folds.append(fold_data)

        # Convert index lists to numpy arrays for easier handling later
        for key in fold_data:
            fold_data[key] = np.array(fold_data[key])



    def data_loader(self, fold):
        print(f"Loading Data for Fold {fold+1}...")
        self.log(f"Loading Data for Fold {fold+1}...")

        train_idx, valid_idx, test_idx = self.folds[fold]['train'], self.folds[fold]['valid'], self.folds[fold]['test']

        # Create datasets for each split
        train_dataset = CustomDataset([self.data[i] for i in train_idx])
        valid_dataset = CustomDataset([self.data[i] for i in valid_idx])
        test_dataset = CustomDataset([self.data[i] for i in test_idx])

        # Print size of each split
        print(f"Training Data Size: {len(train_dataset)}")
        print(f"Validation Data Size: {len(valid_dataset)}")
        print(f"Test Data Size: {len(test_dataset)}")

        # Initialize the DataLoaders
        self.train_dataloader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, worker_init_fn=worker_init_fn, generator=self.generator)
        self.valid_dataloader = DataLoader(valid_dataset, batch_size=self.batch_size, shuffle=False)
        self.test_dataloader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)

        true_labels = []
        for batch in self.train_dataloader:
            input_ids, attention_mask, labels = [b.to(self.device) for b in batch]
            ls = [self.tokenizer.decode(label, skip_special_tokens=True).lower() for label in labels]
            for i in ls:
                if 'yes' in i:
                    true_label = 1
                elif 'no' in i:
                    true_label = 0
                true_labels.append(true_label)
        num_pos = 0
        num_neg = 0
        for l in true_labels:
            if l == 1:
                num_pos = num_pos + 1
            elif l == 0:
                num_neg = num_neg + 1
        print("Positive classes in Train: ", num_pos)
        print("Negative classes in Train ", num_neg)
        self.log(f"Positive classes in Train: {num_pos}")
        self.log(f"Negative classes in Train: {num_neg}")

        true_labels = []
        for batch in self.valid_dataloader:
            input_ids, attention_mask, labels = [b.to(self.device) for b in batch]
            ls = [self.tokenizer.decode(label, skip_special_tokens=True).lower() for label in labels]
            for i in ls:
                if 'yes' in i:
                    true_label = 1
                elif 'no' in i:
                    true_label = 0
                true_labels.append(true_label)
        num_pos = 0
        num_neg = 0
        for l in true_labels:
            if l == 1:
                num_pos = num_pos + 1
            elif l == 0:
                num_neg = num_neg + 1
        print("Positive classes in Validation: ", num_pos)
        print("Negative classes in Validation: ", num_neg)
        self.log(f"Positive classes in Validation: {num_pos}")
        self.log(f"Negative classes in Validation: {num_neg}")

        true_labels = []
        for batch in self.test_dataloader:
            input_ids, attention_mask, labels = [b.to(self.device) for b in batch]
            ls = [self.tokenizer.decode(label, skip_special_tokens=True).lower() for label in labels]
            for i in ls:
                if 'yes' in i:
                    true_label = 1
                elif 'no' in i:
                    true_label = 0
                true_labels.append(true_label)
        num_pos = 0
        num_neg = 0
        for l in true_labels:
            if l == 1:
                num_pos = num_pos + 1
            elif l == 0:
                num_neg = num_neg + 1
        print("Positive classes in Test: ", num_pos)
        print("Negative classes in Test: ", num_neg)
        self.log(f"Positive classes in Test: {num_pos}")
        self.log(f"Negative classes in Test: {num_neg}")
        

        # Print class ratio for each dataset
        #train_class_ratio = calculate_class_ratio(train_dataset)
        #valid_class_ratio = calculate_class_ratio(valid_dataset)
        #test_class_ratio = calculate_class_ratio(test_dataset)

        #print(f"Class Ratio in Training Set: {train_class_ratio}")
        #print(f"Class Ratio in Validation Set: {valid_class_ratio}")
        #print(f"Class Ratio in Test Set: {test_class_ratio}")




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





    def inference(self):
        print("Performing Inference...")
        self.log("Performing Inference...")

        self.model.to(self.device)
        self.model.eval()

        predictions, all_true_labels = [], []
        with torch.no_grad():
            for batch in self.test_dataloader:
                input_ids, attention_mask, labels = [b.to(self.device) for b in batch]

                # Direct model output 
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)

                # Greedy decoding for generating a small number of tokens
                next_tokens = torch.argmax(outputs.logits, dim=-1)
                generated_sequences = next_tokens[:, -self.new_tokens:]  # Take last n tokens as the generated sequence

                # Decode batch labels just once per batch
                true_labels = [self.tokenizer.decode(label, skip_special_tokens=True).lower() for label in labels]

                for i, seq in enumerate(generated_sequences):

                    # Decoding the tokens to text
                    decoded_output = self.tokenizer.decode(seq, skip_special_tokens=True)
                    print("Generated Sequence: ", decoded_output)
                    self.log(f"Generated Sequence: {decoded_output}")

                    # Determine true label based on decoded label
                    if "yes" in true_labels[i]:
                        true_label = 1
                    elif "no" in true_labels[i]:
                        true_label = 0

                    # Creating binary predictions based on the model's text output
                    if "yes" in decoded_output.lower():
                        prediction = 1
                    elif "no" in decoded_output.lower():
                        prediction = 0
                    else:
                        if true_labels:
                            # Get the value of the corresponding true label and store as x
                            x = true_labels[-1]

                            # Append the opposite value of x 
                            if x == 0:
                                prediction = 1
                            else:
                                prediction = 0
                        else:
                            print("True labels is empty.")

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

        # Check if the current fold has the lowest validation loss
        if not self.best_fold_valid_losses or self.valid_losses[-1] < self.best_fold_valid_losses[-1]:

            # Save the loss values for the model just trained, because the new model is better
            self.best_fold_train_losses = self.train_losses.copy()
            self.best_fold_valid_losses = self.valid_losses.copy()
            self.best_fold = fold

        # Reset the values in self.train_losses and self.valid_losses for the next fold
        # We do this because we only care about the loss values for the best model
        self.train_losses = []
        self.valid_losses = []


    def evaluate_best_model(self):

        # Load the data for the best fold
        self.data_loader(self.best_fold)

        print("best model: ", self.best_fold + 1)
    
        # Evaluate the best model on that fold's test set
        predictions, labels = self.inference()

        # Calculate metrics for the best model and plot losses graph
        self.calculate_metrics(predictions, labels)  
        self.plot_losses()









