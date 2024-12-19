import torch
import torch.nn.functional as F
import time
import sys
import numpy as np
import os
import peft
from tqdm import tqdm
from peft import TaskType, get_peft_model, LoraConfig
from torchmetrics.classification import (
    BinaryAccuracy,
    BinaryMatthewsCorrCoef,
    BinaryAUROC,
    BinaryConfusionMatrix,
    BinarySpecificity,
    MulticlassPrecision,
    MulticlassF1Score,
    BinaryCalibrationError,
)

from util import custom_collate_fn
from util import set_seeds

def log_function(log_file_path):
    def log(message):
        # This function handles writing logs to the specified file
        try:
            with open(log_file_path, 'a') as f:
                f.write(message + '\n')
        except Exception as e:
            raise Exception(f"Failed to open and write to file: {log_file_path}") from e
    return log


def train_and_evaluate_lora_ensemble(train_dataset, test_dataset, output_dir,
                                     model, tokenizer, uq_config):

    max_length      = uq_config["max_length"]
    batch_size      = uq_config["batch_size"]
    test_batch_size = uq_config["test_batch_size"]
    device          = uq_config["device"]
    seeds           = uq_config["seeds"]
    lr              = uq_config["lr"]        
    num_epochs      = uq_config["num_epochs"]
    run_every_step  = uq_config["run_every_step"]
    use_tqdm        = uq_config["use_tqdm"]
    n_ensemble      = uq_config["n_ensemble"]
    log_file_path   = uq_config["log_file_path"]

    log = log_function(log_file_path)

    labels = [f" Yes", f" No"]
    target_ids = tokenizer(
        labels, return_tensors="pt", add_special_tokens=False
    ).input_ids[:, -1:]

    tokenizer_run_kwargs = {
                    "return_tensors": "pt",
                    "padding": "max_length",
                    "truncation": True,
                    "max_length": max_length,
                }

    # LoRA CONFIG 
    # https://moon-ci-docs.huggingface.co/docs/peft/pr_721/en/package_reference/tuners#peft.LoraConfig
    target_modules = ['q_proj', 'v_proj']

    train_loader = torch.utils.data.DataLoader(train_dataset, collate_fn=custom_collate_fn, batch_size = batch_size)
    test_loader = torch.utils.data.DataLoader(test_dataset, collate_fn=custom_collate_fn, batch_size = test_batch_size, shuffle=False)

    test_ensemble_probabilities = []
    for i in range(n_ensemble):
        print(f"LoRA instance {i+1}")
        log(f"LoRA instance {i+1}")

        generator = set_seeds(seeds[i])
        lora_instance_path = f"{output_dir}/lora_instance_{i+1}_params.pth"

        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            target_modules=target_modules,
            inference_mode=False,
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            bias="none"
        )
        lora_model = get_peft_model(model, peft_config).to(device)

        if not os.path.exists(lora_instance_path) or run_every_step:
            opt = torch.optim.AdamW(lora_model.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-8)
            print(f"Training LoRA instance {i+1} parameters")
            log(f"Training LoRA instance {i+1} parameters")

            start_time = time.time()
            for epoch in range(num_epochs):
                print(f"Beginning epoch {epoch+1}")
                log(f"Beginning epoch {epoch+1}")
                for batch in tqdm(train_loader, disable=not use_tqdm, file=sys.stdout):
                    opt.zero_grad()
                    prompts, classes = batch
                    inputs = tokenizer(prompts, **tokenizer_run_kwargs).to(device)
                    logits = lora_model(**inputs).logits[:, -1, target_ids.squeeze(-1)]
                    loss = F.cross_entropy(logits, classes.to(device))
                    assert not torch.isnan(loss).any(), "NaN in loss for LoRA model training."
                    loss.backward()
                    opt.step()
            end_time = time.time()
            elapsed_time = end_time - start_time
            print(f"Elapsed time: {elapsed_time} seconds for LoRA instance {i+1} training over {num_epochs} epochs")
            log(f"Elapsed time: {elapsed_time} seconds for LoRA instance {i+1} training over {num_epochs} epochs")
            print(f"Saving LoRA instance {i+1} parameters after finetuning to {lora_instance_path}")
            log(f"Saving LoRA instance {i+1} parameters after finetuning to {lora_instance_path}")
            lora_model.save_pretrained(lora_instance_path)
        else:
            print(f"Loading LoRA instance {i+1} from {lora_instance_path}")
            log(f"Loading LoRA instance {i+1} from {lora_instance_path}")
            del lora_model
            lora_model = peft.PeftModel.from_pretrained(model, lora_instance_path, is_trainable=True)
            lora_model = lora_model.to(device)

        lora_model.eval()
        test_probabilities, test_true_classes = [], []
        with torch.no_grad():
            for batch in tqdm(test_loader, disable=not use_tqdm, file=sys.stdout):
                prompts, classes = batch
                inputs = tokenizer(prompts, **tokenizer_run_kwargs).to(device)
                logits = lora_model(**inputs).logits[:, -1, target_ids.squeeze(-1)]
                probabilities = F.softmax(logits, dim=-1)
                test_probabilities.append(probabilities.cpu().numpy())
                test_true_classes.append(classes.cpu().numpy())

        test_instance_path = os.path.join(output_dir, f"test_data_instance_{i+1}_seed_{seeds[i]}.npz")
        np.savez(test_instance_path, 
                 seed=seeds[i],
                 test_probabilities=np.concatenate(test_probabilities), 
                 test_true_classes=np.concatenate(test_true_classes))
        print(f"LoRA instance {i+1} evaluation complete. Data saved to {test_instance_path}.")
        log(f"LoRA instance {i+1} evaluation complete. Data saved to {test_instance_path}.")

        test_ensemble_probabilities.append(np.concatenate(test_probabilities))
        test_true_classes = np.concatenate(test_true_classes)
        print(f"LoRA instance {i+1} Successfully finished.")
        log(f"LoRA instance {i+1} Successfully finished.")

    test_average_probabilities = np.mean(test_ensemble_probabilities, axis=0)
    prob_positive = test_average_probabilities[:, 1]
    pred_labels = (prob_positive >= 0.5).astype(int)

    test_true_classes = torch.from_numpy(test_true_classes)
    prob_positive = torch.from_numpy(prob_positive)
    pred_labels = torch.from_numpy(pred_labels)
    
    accuracy_metric = BinaryAccuracy()
    mcc_metric = BinaryMatthewsCorrCoef()
    auroc_metric = BinaryAUROC()
    confmat_metric = BinaryConfusionMatrix()
    specificity_metric = BinarySpecificity()
    precision_macro_metric = MulticlassPrecision(num_classes=2, average='macro')
    f1_macro_metric = MulticlassF1Score(num_classes=2, average='macro')
    ece_metric = BinaryCalibrationError()
    
    accuracy = accuracy_metric(pred_labels, test_true_classes)
    mcc_score = mcc_metric(pred_labels, test_true_classes)
    roc_auc = auroc_metric(prob_positive, test_true_classes)
    confusion_matrix = confmat_metric(pred_labels, test_true_classes)
    specificity = specificity_metric(pred_labels, test_true_classes)
    precision_macro = precision_macro_metric(pred_labels, test_true_classes)
    f1_macro = f1_macro_metric(pred_labels, test_true_classes)
    ece = ece_metric(prob_positive, test_true_classes)
    nll = -np.mean(np.log(test_average_probabilities[np.arange(len(test_true_classes)), test_true_classes]))
    
    print(f"Accuracy: {accuracy.item():.4f}")
    log(f"Accuracy: {accuracy.item():.4f}")
    print(f"MCC: {mcc_score.item():.4f}")
    log(f"MCC: {mcc_score.item():.4f}")
    print(f"AUC: {roc_auc.item():.4f}")
    log(f"AUC: {roc_auc.item():.4f}")
    print(f"Confusion Matrix:\n{confusion_matrix}")
    log(f"Confusion Matrix:\n{confusion_matrix}")
    print(f"Specificity: {specificity.item():.4f}")
    log(f"Specificity: {specificity.item():.4f}")
    print(f"Precision (Macro): {precision_macro.item():.4f}")
    log(f"Precision (Macro): {precision_macro.item():.4f}")
    print(f"F1 Score (Macro): {f1_macro.item():.4f}")
    log(f"F1 Score (Macro): {f1_macro.item():.4f}")
    print(f"Expected Calibration Error (ECE): {ece.item():.4f}")
    log(f"Expected Calibration Error (ECE): {ece.item():.4f}")
    print(f"NLL loss: {nll:.4f}")
    log(f"NLL loss: {nll:.4f}")
    
    print("LoRA Ensemble model successfully finished.")
    log("LoRA Ensemble model successfully finished.")