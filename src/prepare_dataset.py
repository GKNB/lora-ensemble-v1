import json
import os
import random
import argparse
from datasets import Dataset

def save_dataset(dataset, file_name):
    dataset_dict = dataset.to_dict()
    with open(file_name, 'w') as file:
        json.dump(dataset_dict, file)
    print(f"Dataset saved to {file_name}")

def load_dataset(file_name):
    with open(file_name, 'r') as file:
        dataset_dict = json.load(file)
    dataset = Dataset.from_dict(dataset_dict)
    print(f"Dataset loaded from {file_name}")
    return dataset

def train_test_split_and_save(file_name_in, file_name_out_prefix, train_ratio, random_seed):

    random.seed(random_seed)
    dataset_examples = []
    with open(file_name_in, 'r') as file:
        prompts_data = json.load(file)
    print("Size of dataset: ", len(prompts_data))

    for prompt in prompts_data:
        dataset_examples.append(prompt)
    random.shuffle(dataset_examples)

    total_items = len(dataset_examples)
    train_end = int(total_items * train_ratio)
    
    train_dataset_examples = dataset_examples[:train_end]
    test_dataset_examples = dataset_examples[train_end:]
    
    train_dataset = Dataset.from_dict({
        'question': [i['question'] for i in train_dataset_examples], 
        'answer': [i['answer'] for i in train_dataset_examples]
    })
    
    test_dataset = Dataset.from_dict({
        'question': [i['question'] for i in test_dataset_examples], 
        'answer': [i['answer'] for i in test_dataset_examples]
    })

    save_dataset(train_dataset, f"{file_name_out_prefix}_train_prompts.json")
    save_dataset(test_dataset,  f"{file_name_out_prefix}_test_prompts.json")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split dataset into train and test sets.")
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--train_ratio", type=float, required=True)
    args = parser.parse_args()

    root_dir_in = "/pscratch/sd/t/tianle/lucid/other_source/SURP_2024/data"
    root_dir_out = "/pscratch/sd/t/tianle/lucid/other_source/SURP_2024/data"

    dataset = args.dataset
    input_name = dataset + "_prompts.json"
    train_ratio = args.train_ratio
    random_seed = 42

    file_name_in = os.path.join(root_dir_in, input_name)
    file_name_out_prefix = os.path.join(root_dir_out, dataset)
    train_test_split_and_save(file_name_in, file_name_out_prefix, train_ratio, random_seed)

    loaded_train_dataset = load_dataset(f"{file_name_out_prefix}_train_prompts.json")
    print(f"trainset of {dataset} is:", loaded_train_dataset)
    loaded_test_dataset = load_dataset(f"{file_name_out_prefix}_test_prompts.json")
    print(f"testset of {dataset} is:",  loaded_test_dataset)
