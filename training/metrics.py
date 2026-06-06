import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score, precision_score, recall_score, roc_auc_score, confusion_matrix

def compute_top2_accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    """Computes the Top-2 accuracy."""
    _, top2_preds = logits.topk(2, dim=1)
    correct = top2_preds.eq(targets.view(-1, 1).expand_as(top2_preds))
    return correct.any(dim=1).float().mean().item()

def compute_classification_metrics(all_preds: np.ndarray, all_targets: np.ndarray, all_probs: np.ndarray, num_classes: int = 8) -> dict:
    """Compute standard classification metrics."""
    top1_acc = accuracy_score(all_targets, all_preds)
    
    # Calculate one-vs-rest AUC for each class and macro
    try:
        if all_probs.shape[1] == num_classes:
            per_class_auc_roc = roc_auc_score(all_targets, all_probs, multi_class='ovr', average=None).tolist()
            macro_auc_roc = roc_auc_score(all_targets, all_probs, multi_class='ovr', average='macro')
        else:
            raise ValueError("Probs shape mismatch")
    except Exception:
        # Handled cases where some classes might not be present in the target array
        per_class_auc_roc = [0.0] * num_classes
        macro_auc_roc = 0.0

    return {
        "top1_acc": top1_acc,
        "macro_f1": f1_score(all_targets, all_preds, average='macro'),
        "weighted_f1": f1_score(all_targets, all_preds, average='weighted'),
        "balanced_acc": balanced_accuracy_score(all_targets, all_preds),
        "per_class_precision": precision_score(all_targets, all_preds, average=None, zero_division=0).tolist(),
        "per_class_recall": recall_score(all_targets, all_preds, average=None, zero_division=0).tolist(),
        "per_class_f1": f1_score(all_targets, all_preds, average=None, zero_division=0).tolist(),
        "per_class_auc_roc": per_class_auc_roc,
        "macro_auc_roc": macro_auc_roc,
        "confusion_matrix": confusion_matrix(all_targets, all_preds, labels=list(range(num_classes)))
    }
