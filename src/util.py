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