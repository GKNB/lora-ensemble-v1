# Evaluating Large Language Models for Predicting Protein Behavior under Radiation Exposure and Disease Conditions

This repository contains the code used to run the experiments presented in the paper titled "Evaluating Large Language Models for Predicting Protein Behavior under Radiation Exposure and Disease Conditions."

### Prerequisites
- python 3.11
- Install the required packages using: pip install -r requirements.txt

### Running Experiments
To run the experiments, you need to modify certain parameters in the model_trainer.py script:

If running experiments for training a model:
- Change self.model_name to determine the naming convention for the saved training and inference results.
- Update self.json_file_path to point to the dataset you are using.
- Modify the hyperparameters based on the dataset to replicate the results presented in the paper.

If running experiments with a pre-trained model:
- Copy the few-shot prompt for the corresponding dataset from the pretrained_model_inference function.
- Paste the copied prompt into the tokenizer section immediately below.
