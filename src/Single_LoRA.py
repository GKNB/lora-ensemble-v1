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


def train_and_evaluate_single_lora(train_dataset, test_dataset, single_output_dir, ensemble_output_dir,
                                     model, tokenizer, uq_config):

    max_length      = uq_config["max_length"]
    batch_size      = uq_config["batch_size"]
    test_batch_size = uq_config["test_batch_size"]
    device          = uq_config["device"]
    single_seed     = uq_config["single_seed"]
    lr              = uq_config["lr"]        
    num_epochs      = uq_config["num_epochs"]
    run_every_step  = uq_config["run_every_step"]
    use_tqdm        = uq_config["use_tqdm"]
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

    ensemble_lora_instance_path = f"{ensemble_output_dir}/lora_instance_{single_seed}_params.pth"
    single_lora_instance_path = f"{single_output_dir}/single_lora_instance.pth"

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

    path_missing = not os.path.exists(ensemble_lora_instance_path) and not os.path.exists(single_lora_instance_path)
    run_every_step = path_missing or run_every_step

    if run_every_step:
        opt = torch.optim.AdamW(lora_model.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-8)
        print(f"Training LoRA parameters")
        log(f"Training LoRA parameters")

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
        print(f"Elapsed time: {elapsed_time} seconds for LoRA training over {num_epochs} epochs")
        log(f"Elapsed time: {elapsed_time} seconds for LoRA training over {num_epochs} epochs")
        print(f"Saving LoRA parameters after finetuning to {single_lora_instance_path}")
        log(f"Saving LoRA parameters after finetuning to {single_lora_instance_path}")
        lora_model.save_pretrained(single_lora_instance_path)
    elif os.path.exists(ensemble_lora_instance_path):
        print(f"Loading LoRA instance {single_seed} from {ensemble_lora_instance_path}")
        log(f"Loading LoRA instance {single_seed} from {ensemble_lora_instance_path}")
        del lora_model
        lora_model = peft.PeftModel.from_pretrained(model, ensemble_lora_instance_path, is_trainable=True)
        lora_model = lora_model.to(device)
    elif os.path.exists(single_lora_instance_path):
        print(f"Loading LoRA from {single_lora_instance_path}")
        log(f"Loading LoRA from {single_lora_instance_path}")
        del lora_model
        lora_model = peft.PeftModel.from_pretrained(model, single_lora_instance_path, is_trainable=True)
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

    test_probabilities = np.concatenate(test_probabilities)
    test_true_classes = np.concatenate(test_true_classes)

    prob_positive = test_probabilities[:, 1]
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
    nll = -np.mean(np.log(test_probabilities[np.arange(len(test_true_classes)), test_true_classes]))     

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

    print("Single LoRA model successfully finished.")
    log("Single LoRA model successfully finished.")