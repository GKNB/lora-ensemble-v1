import torch
import torch.nn.functional as F
import time
import numpy as np
import os
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


def train_and_evaluate_lora_ensemble(train_dataset, test_dataset, output_dir,
                                     model, tokenizer, uq_config):

    max_length  = uq_config["max_length"]
    batch_size  = uq_config["batch_size"]
    n_ensemble  = uq_config["n_ensemble"]
    seeds       = uq_config["seeds"]     
    device      = uq_config["device"]    
    lr          = uq_config["lr"]        
    num_epochs  = uq_config["num_epochs"]

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

    test_loader = torch.utils.data.DataLoader(test_dataset, collate_fn=custom_collate_fn, batch_size=batch_size, shuffle=False)

    test_ensemble_probabilities = []
    for i in range(n_ensemble):
        print(f"Training lora instance {i}")

        generator = set_seeds(seeds[i])
        train_loader = torch.utils.data.DataLoader(train_dataset, collate_fn=custom_collate_fn, batch_size=batch_size)

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
        opt = torch.optim.AdamW(lora_model.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-8)
        lora_model.train()

        total_params = sum(p.numel() for p in lora_model.parameters())
        trainable_params = sum(p.numel() for p in lora_model.parameters() if p.requires_grad)
        print(f"DEBUG: Total parameters: {total_params}")
        print(f"DEBUG: Trainable parameters: {trainable_params}")

        grad_steps = 0

        start_time = time.time()
        total_token_count = 0
        for epoch in range(num_epochs):
            print(f"Beginning epoch {epoch + 1}")
            for batch in train_loader:
                opt.zero_grad()
                prompts, classes = batch
                inputs = tokenizer(prompts, **tokenizer_run_kwargs).to(device)
                logits = lora_model(**inputs).logits[:, -1, target_ids.squeeze(-1)]
                loss = F.cross_entropy(logits, classes.to(device))
                print(f"In grad_steps = {grad_steps}, loss = {loss}")
                loss.backward()
                opt.step()
                grad_steps += 1

        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Elapsed time: {elapsed_time} seconds for ensemble {i} with {num_epochs} epochs")

        lora_model.eval()
        test_probabilities, test_true_classes = [], []
        with torch.no_grad():
            for batch in test_loader:
                prompts, classes = batch
                inputs = tokenizer(prompts, **tokenizer_run_kwargs).to(device)
                logits = lora_model(**inputs).logits[:, -1, target_ids.squeeze(-1)]
                probabilities = F.softmax(logits, dim=-1)
                test_probabilities.append(probabilities.cpu().numpy())
                test_true_classes.append(classes.cpu().numpy())

        test_instance_path = os.path.join(output_dir, f"test_data_instance_{i}_seed_{seeds[i]}.npz")
        np.savez(test_instance_path, 
                 seed=seeds[i],
                 test_probabilities=np.concatenate(test_probabilities), 
                 test_true_classes=np.concatenate(test_true_classes))
        print(f"LoRA instance {i} evaluation complete. Data saved to {test_instance_path}.")

        test_ensemble_probabilities.append(np.concatenate(test_probabilities))
        test_true_classes = np.concatenate(test_true_classes)
        print(f"lora instance i = {i} Successfully finished.")

    test_average_probabilities = np.mean(test_ensemble_probabilities, axis=0)
    print(f"Final, Test average ensemble probabilities = \n{test_average_probabilities}")
    prob_positive = test_average_probabilities[:, 1]
    pred_labels = (prob_positive >= 0.5).astype(int)
    
    accuracy_metric = BinaryAccuracy()
    mcc_metric = BinaryMatthewsCorrCoef()
    auroc_metric = BinaryAUROC()
    confmat_metric = BinaryConfusionMatrix()
    specificity_metric = BinarySpecificity()
    precision_macro_metric = MulticlassPrecision(num_classes=2, average='macro')
    f1_macro_metric = MulticlassF1Score(num_classes=2, average='macro')
    ece_metric = BinaryCalibrationError()
    
    accuracy = accuracy_metric(torch.tensor(pred_labels), torch.tensor(test_true_classes))
    mcc_score = mcc_metric(torch.tensor(pred_labels), torch.tensor(test_true_classes))
    roc_auc = auroc_metric(torch.tensor(prob_positive), torch.tensor(test_true_classes))
    confusion_matrix = confmat_metric(torch.tensor(pred_labels), torch.tensor(test_true_classes))
    specificity = specificity_metric(torch.tensor(pred_labels), torch.tensor(test_true_classes))
    precision_macro = precision_macro_metric(torch.tensor(pred_labels), torch.tensor(test_true_classes))
    f1_macro = f1_macro_metric(torch.tensor(pred_labels), torch.tensor(test_true_classes))
    ece = ece_metric(torch.tensor(prob_positive), torch.tensor(test_true_classes))
    nll = -np.mean(np.log(test_average_probabilities[np.arange(len(test_true_classes)), test_true_classes]))
    
    print(f"Accuracy: {accuracy.item():.4f}")
    print(f"MCC: {mcc_score.item():.4f}")
    print(f"AUC: {roc_auc.item():.4f}")
    print(f"Confusion Matrix:\n{confusion_matrix}")
    print(f"Specificity: {specificity.item():.4f}")
    print(f"Precision (Macro): {precision_macro.item():.4f}")
    print(f"F1 Score (Macro): {f1_macro.item():.4f}")
    print(f"Expected Calibration Error (ECE): {ece.item():.4f}")
    print(f"NLL loss: {nll:.4f}")
    
    print("Ensemble evaluation complete.")
