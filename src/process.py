import os
import numpy as np
import torch
from torchmetrics.classification import (
    BinaryAccuracy,
    BinaryMatthewsCorrCoef,
    BinaryAUROC,
    BinaryConfusionMatrix,
    BinarySpecificity,
    MulticlassPrecision,
    MulticlassF1Score,
    BinaryCalibrationError
)
import argparse

def process(output_dir):
    test_ensemble_probabilities = []
    test_true_classes = None
    loaded_seeds = set()

    for file_name in os.listdir(output_dir):
        if file_name.endswith(".npz"):
            data = np.load(os.path.join(output_dir, file_name))
            
            seed = data["seed"].item()
            if seed in loaded_seeds:
                raise ValueError(f"Duplicate seed detected: {seed} in file {file_name}.")
            loaded_seeds.add(seed)
            
            test_ensemble_probabilities.append(data["test_probabilities"])
            if test_true_classes is None:
                test_true_classes = data["test_true_classes"]
            elif not np.array_equal(test_true_classes, data["test_true_classes"]):
                raise ValueError(f"Mismatch in true classes between files. Problematic file: {file_name}")

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

    print("Metrics computation complete.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Process metrics from lora_ensemble")
    parser.add_argument('--tmp_dir', type=str, required=True, help='Path to the temporary directory.')
    args = parser.parse_args()
    process(args.tmp_dir)
