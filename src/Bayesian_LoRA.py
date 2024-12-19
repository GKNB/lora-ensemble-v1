import torch
import torch.nn.functional as F
import torch.nn as nn
import time
import sys
import numpy as np
import os
import importlib
import peft
from typing import Any
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

# from torchmetrics import Accuracy, CalibrationError
from transformers.modeling_outputs import ModelOutput

sys.path.append('/hpcgpfs01/work/sjantre/lora-ensemble-v1')
from bayesian_lora import (
    calculate_kronecker_factors,
    cholesky_decompose_small_factors,
    model_evidence,
    variance,
    stable_cholesky,
)

from bayesian_lora.main import jacobian_mean

def log_function(log_file_path):
    def log(message):
        # This function handles writing logs to the specified file
        try:
            with open(log_file_path, 'a') as f:
                f.write(message + '\n')
        except Exception as e:
            raise Exception(f"Failed to open and write to file: {log_file_path}") from e
    return log

def set_seeds(seed=1):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    return

def train_and_evaluate_bayesian_lora(train_dataset, valid_dataset, test_dataset, output_dir,
                                     model, tokenizer, uq_config):

    max_length      = uq_config["max_length"]
    batch_size      = uq_config["batch_size"]
    test_batch_size = uq_config["test_batch_size"]
    device          = uq_config["device"]
    seed            = uq_config["seed"]
    lr              = uq_config["lr"]        
    num_epochs      = uq_config["num_epochs"]
    run_every_step  = uq_config["run_every_step"]
    use_tqdm        = uq_config["use_tqdm"]
    prior_var       = uq_config["prior_var"]
    log_file_path   = uq_config["log_file_path"]

    log = log_function(log_file_path)
    set_seeds(seed)

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
    val_loader = torch.utils.data.DataLoader(valid_dataset, collate_fn=custom_collate_fn, batch_size = batch_size, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_dataset, collate_fn=custom_collate_fn, batch_size = test_batch_size, shuffle=False)

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

    # 1. Do MAP training
    map_param_path = f"{output_dir}/MAP_params.pth"
    if not os.path.exists(map_param_path) or run_every_step:
        # setup optimiser
        # New optimizer configuration
        opt_cfg = {
            "module": "torch.optim",
            "classname": "AdamW",
            "lr": lr, # 0.0001, 
            "betas": (0.9, 0.999),
            "eps": 1e-8,  # 1e-5
        }
        # add prior / regularization for MAP objective:
        opt_cfg |= {"weight_decay": 1 / prior_var}
        optclass = getattr(
            importlib.import_module(opt_cfg.pop("module")),
            opt_cfg.pop("classname"),
        )
        opt = optclass(lora_model.parameters(), **opt_cfg)
        print("Training MAP parameters")
        log("Training MAP parameters")

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
                assert not torch.isnan(loss).any(), "NaN in loss for MAP training."
                loss.backward()
                opt.step()
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Elapsed time: {elapsed_time} seconds for MAP training over {num_epochs} epochs")
        log(f"Elapsed time: {elapsed_time} seconds for MAP training over {num_epochs} epochs")
        print(f"Saving MAP parameters after finetuning to {map_param_path}")
        log(f"Saving MAP parameters after finetuning to {map_param_path}")
        lora_model.save_pretrained(map_param_path)
    else:
        print(f"Loading MAP parameters from {map_param_path}")
        log(f"Loading MAP parameters from {map_param_path}")
        del lora_model
        lora_model = peft.PeftModel.from_pretrained(model, map_param_path, is_trainable=True)
        lora_model = lora_model.to(device)

    # 2. Evaluate the log likelihood
    ll_path = f"{output_dir}/ll.pth"
    if not os.path.exists(ll_path) or run_every_step:
        print("Evaluating the MAP log likelihood")
        log("Evaluating the MAP log likelihood")
        LL = 0.0
        with torch.no_grad(), torch.inference_mode():
            for batch in tqdm(val_loader, disable=not use_tqdm, file=sys.stdout):
                prompts, classes = batch
                inputs = tokenizer(prompts, **tokenizer_run_kwargs).to(device)
                logits = lora_model(**inputs).logits[:, -1, target_ids.squeeze(-1)]
                probs = logits.softmax(-1)
                LL += probs.gather(1, classes[:, None].to(device)).log().sum()
        torch.save(LL, ll_path)
    else:
        print(f"Loading LL from {ll_path}")
        log(f"Loading LL from {ll_path}")
        LL = torch.load(ll_path)

    # 3. Calculate the (low-rank) Kronecker factors
    def fwd_call(model: nn.Module, batch: Any) -> torch.Tensor:
        prompts, _ = batch
        tok_kwargs = tokenizer_run_kwargs | {
            "padding": True,
            "return_tensors": "pt",
        }
        inputs = tokenizer(prompts, **tok_kwargs).to(device)
        outputs = lora_model(**inputs)
        logits = (
            outputs.logits[:, -1, target_ids.squeeze(-1)]
        )
        logits = logits.softmax(-1)
        return logits
    
    kfac_path = f"{output_dir}/kronecker_factors.pth"
    if not os.path.exists(kfac_path) or run_every_step:
        print("Computing the low-rank Kronecker factors")
        log("Computing the low-rank Kronecker factors")
        factors = calculate_kronecker_factors(
            lora_model,
            fwd_call,
            train_loader,
            n_kfac=10,
            lr_threshold=100,
            target_module_keywords=["lora"],
            use_tqdm=use_tqdm,
        )
        # Calculate Cholesky decomposition of the smaller factors
        factors = cholesky_decompose_small_factors(
            factors, lr_threshold=100, device=device, dtype=torch.float32
        )
        torch.save({"factors": factors}, kfac_path)

    else:
        print(f"Loading low-rank Kronecker factors from {kfac_path}")
        log(f"Loading low-rank Kronecker factors from {kfac_path}")
        factors = torch.load(kfac_path)["factors"]

    # 4. Use the marginal likelihood to optimise the prior variance
    prior_path = f"{output_dir}/prior_params.pth"
    if not os.path.exists(prior_path) or run_every_step:
        print("Optimising priors using marginal likelihood")
        log("Optimising priors using marginal likelihood")
        s2 = torch.tensor(prior_var, requires_grad=True)
        opt = torch.optim.AdamW([s2], lr=1e-2)

        for _ in range(200):
            opt.zero_grad()
            loss = model_evidence(
                lora_model, LL, factors, n_lora=16, n_kfac=10, s2=s2
            ).log()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(s2, 1.0)
            opt.step()
        torch.save({"s2": s2}, prior_path)
        print(f"prior variance is: {s2.item()}")
        log(f"prior variance is: {s2.item()}")
    else:
        print("Loading prior parameters (optimised using marginal likelihood)")
        log("Loading prior parameters (optimised using marginal likelihood)")
        priors = torch.load(prior_path)
        s2 = priors["s2"]

    # 5. Make linearized predictions
    del lora_model
    torch.cuda.empty_cache()
    print("Doing linearized prediction")
    log("Doing linearized prediction")

    lora_model = peft.PeftModel.from_pretrained(model, map_param_path, is_trainable=True)
    lora_model = lora_model.to(device)

    def output_callback(outputs: ModelOutput) -> torch.Tensor:
        """Post process model outputs.

        This function will be passed the results of model(**batch_inputs), and
        should return the relevant logits. For multiple-choice tasks, this is
        the class logits, but for full next-token prediction, this would just
        be all the logits.
        """
        # Get the last token for CausalLM
        logits = outputs.logits[:, -1]
        # Select the logits corresponding to our target classes
        target_logits = logits[:, target_ids.squeeze(-1)]
        return target_logits
    
    lora_model.eval()

    pred_mu, pred_var, pred_logits = [], [], []
    test_probabilities, test_true_classes = [], []
    with torch.no_grad():
        for batch in tqdm(test_loader, disable=not use_tqdm, file=sys.stdout):
            prompts, classes = batch
            classes = classes.to(device)
            batch_inputs = tokenizer(prompts, **tokenizer_run_kwargs).to(device)
            # Predict the output logit locations
            jacobian, f_mu = jacobian_mean(
                lora_model, batch_inputs, output_callback=output_callback
            )
            pred_mu.append(f_mu.clone().cpu())

            # Predict the output logit variances
            f_var = variance(
                batch_inputs,
                jacobian,
                factors,
                s2,
                n_logits=2,
                n_lora=16,
                n_kfac=10,
                device=device,
            )
            pred_var.append(f_var.clone().cpu())

            # Sample logits from a Gaussian parametrised by f_mu, f_var
            L = stable_cholesky(f_var)
            samples = 100_000
            f_mu = f_mu.expand(samples, *f_mu.shape)
            L = L.expand(samples, *L.shape)
            eps = torch.randn_like(f_mu).unsqueeze(-1)
            logits = f_mu[..., None] + L @ eps
            logits = logits.squeeze(-1).mean(0)

            probabilities = F.softmax(logits, dim=-1)
            test_probabilities.append(probabilities.cpu().numpy())
            test_true_classes.append(classes.cpu().numpy())
            pred_logits.append(logits.cpu())
        
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

        output_path = f"{output_dir}/predicted_logits.pth"
        torch.save(
            {"pred_mu": pred_mu, "pred_var": pred_var, "pred_logits": pred_logits},
            output_path,
        )

        print("Bayesian LoRA model successfully finished.")
        log("Bayesian LoRA model successfully finished.")