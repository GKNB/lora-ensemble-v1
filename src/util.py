import random
import numpy as np
import torch

# Set random seeds for consistent experiments
def set_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    generator = torch.Generator().manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    return generator
 
def custom_collate_fn(batch):
    # Extract questions and answers from the batch
    prompt = """Answer the following question with Yes or No.\n\nQuestion: {question}\n\nAnswer (Yes or No):"""
    prompts = [prompt.format(question=item['question']) for item in batch]
    classes = torch.tensor([1 if item['answer'] == 'Yes' else 0 for item in batch])
    return prompts, classes


# Below are obsolote functions, that we will not use anymore
#def preprocess_data(self):
#    
#    # Initialize dataset 
#    self.dataset_examples = []
#
#    # Open and read the JSON file
#    with open(self.json_file_path, 'r') as file:
#        prompts_data = json.load(file)
#
#    print("Size of dataset: ", len(prompts_data))
#
#    # Iterate through each entry in the JSON file
#    for prompt in prompts_data:
#        self.dataset_examples.append(prompt)
#
#    # Shuffle the dataset to ensure a mix of 'Yes' and 'No' answers throughout
#    random.shuffle(self.dataset_examples)
#
#    # Set up 80 / 10 / 10 split for training / validation / testing for datasets 1-3
#    if "set-1" in self.model_name or "set-2" in self.model_name or "set-3" in self.model_name:
#
#        # Split the dataset into training, validation, and possibly test sets
#        total_items = len(self.dataset_examples)
#        train_end = int(total_items * 0.8)
#        valid_end = train_end + int(total_items * 0.1)
#        
#        train_dataset = self.dataset_examples[:train_end]
#        valid_dataset = self.dataset_examples[train_end:valid_end]
#        test_dataset = self.dataset_examples[valid_end:]
#
#        # Convert list of dictionaries into Hugging Face Dataset
#        self.train_dataset = Dataset.from_dict({'question': [i['question'] for i in train_dataset], 'answer': [i['answer'] for i in train_dataset]})
#        self.valid_dataset = Dataset.from_dict({'question': [i['question'] for i in valid_dataset], 'answer': [i['answer'] for i in valid_dataset]})
#        self.test_dataset = Dataset.from_dict({'question': [i['question'] for i in test_dataset], 'answer': [i['answer'] for i in test_dataset]})
#
#
#    # Set up 5-fold cross validation for datasets 4 and 5
#    if "set-4" in self.model_name or "set-5" in self.model_name:
#        kf = KFold(n_splits=5, shuffle=True, random_state=42)
#        self.fold_data = []
#
#        for fold, (train_index, test_index) in enumerate(kf.split(self.dataset_examples), start=1):
#            train_fold = [self.dataset_examples[i] for i in train_index]
#            test_fold = [self.dataset_examples[i] for i in test_index]
#
#            train_fold_dataset = Dataset.from_dict({'question': [i['question'] for i in train_fold], 'answer': [i['answer'] for i in train_fold]})
#            test_fold_dataset = Dataset.from_dict({'question': [i['question'] for i in test_fold], 'answer': [i['answer'] for i in test_fold]})
#
#            fold_name = f'fold-{fold}'
#            self.save_fold_data(fold_name, train_dataset=train_fold_dataset, test_dataset=test_fold_dataset)
#            self.fold_data.append((train_fold_dataset, test_fold_dataset))
#
#
#    # Set up 5-fold cross validation for dataset 6
#    elif "set-6" in self.model_name:
#        kf = KFold(n_splits=5, shuffle=True, random_state=42)
#        self.fold_data = []
#
#        for fold, (train_index, test_index) in enumerate(kf.split(self.dataset_examples), start=1):
#            train_index, validation_index = train_test_split(train_index, test_size=0.2, random_state=42)
#
#            train_fold = [self.dataset_examples[i] for i in train_index]
#            valid_fold = [self.dataset_examples[i] for i in validation_index]
#            test_fold = [self.dataset_examples[i] for i in test_index]
#
#            train_fold_dataset = Dataset.from_dict({'question': [i['question'] for i in train_fold], 'answer': [i['answer'] for i in train_fold]})
#            valid_fold_dataset = Dataset.from_dict({'question': [i['question'] for i in valid_fold], 'answer': [i['answer'] for i in valid_fold]})
#            test_fold_dataset = Dataset.from_dict({'question': [i['question'] for i in test_fold], 'answer': [i['answer'] for i in test_fold]})
#
#            fold_name = f'fold-{fold}'
#            self.save_fold_data(fold_name, train_dataset=train_fold_dataset, valid_dataset=valid_fold_dataset, test_dataset=test_fold_dataset)
#            self.fold_data.append((train_fold_dataset, valid_fold_dataset, test_fold_dataset))


