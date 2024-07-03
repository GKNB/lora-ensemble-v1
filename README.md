# Evaluating Large Language Models for Predicting Protein Behavior under Radiation Exposure and Disease Conditions

This repository contains the code used to run the experiments presented in the paper titled "Evaluating Large Language Models for Predicting Protein Behavior under Radiation Exposure and Disease Conditions."

### Prerequisites
- python 3.11
- Install the required packages using: pip install -r requirements.txt

### Running Experiments
All file paths should be updated for each user.
To run the experiments, modify the parameters in the model_trainer.py script:

If running experiments for training a model:
- Change self.model_name to determine the naming convention for the results file.
- Update self.json_file_path to use to the correct dataset.
- Modify hyperparameters based on the dataset to replicate the results in the paper.

If running experiments with a pre-trained model:
- Change self.model_name to determine the naming convention for the results file.
- Update self.json_file_path to use to the correct dataset.
- Modify the self.max_length hyperparameter to be 425, for the few-shot prompts.
- Copy the few-shot prompt for the corresponding dataset from the pretrained_model_inference function.
- Paste the copied prompt into the tokenizer section immediately below.

Use run.sh to run an experiment.
