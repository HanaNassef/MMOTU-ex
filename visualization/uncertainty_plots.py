import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path

BLUE_DARK = "#003f5c"
BLUE_MEDIUM = "#2f4b7c"

def plot_reliability_diagram(probs_pre: np.ndarray, probs_post: np.ndarray, 
                             labels: np.ndarray, save_path: str, n_bins: int = 15):
    plt.figure(figsize=(8, 6))
    
    for probs, color, label_str in zip([probs_pre, probs_post], [BLUE_DARK, "#ffa600"], ["Pre-Scaling", "Post-Scaling"]):
        confidences = probs.max(axis=1)
        predictions = probs.argmax(axis=1)
        accuracies = (predictions == labels).astype(np.float64)
        
        bins = np.linspace(0, 1, n_bins + 1)
        bin_accs, bin_confs = [], []
        
        for lo, hi in zip(bins[:-1], bins[1:]):
            mask = (confidences > lo) & (confidences <= hi)
            if mask.sum() > 0:
                bin_accs.append(accuracies[mask].mean())
                bin_confs.append(confidences[mask].mean())
                
        plt.plot(bin_confs, bin_accs, marker='o', color=color, label=label_str, linewidth=2)
        
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Perfect Calibration')
    plt.xlabel('Confidence')
    plt.ylabel('Accuracy')
    plt.title('Reliability Diagram')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()

def plot_risk_coverage_curves(rc_curves_by_backbone: dict, save_path: str):
    plt.figure(figsize=(8, 6))
    colors = sns.color_palette("husl", len(rc_curves_by_backbone))
    
    for (model_name, metrics), color in zip(rc_curves_by_backbone.items(), colors):
        plt.plot(metrics["coverages"], metrics["risks"], label=f"{model_name} (AURC: {metrics['aurc']:.3f})", color=color, linewidth=2)
        
    plt.xlabel('Coverage')
    plt.ylabel('Selective Risk (Error Rate)')
    plt.title('Risk-Coverage Curve')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()

def plot_coverage_per_class(conformal_eval_by_backbone: dict, save_path: str, alpha: float = 0.10):
    # Setup data for grouped bar chart
    models = list(conformal_eval_by_backbone.keys())
    classes = list(conformal_eval_by_backbone[models[0]]["per_class_coverage"].keys())
    
    x = np.arange(len(classes))
    width = 0.8 / len(models)
    
    plt.figure(figsize=(12, 6))
    colors = sns.color_palette("Blues_d", len(models))
    
    for i, model in enumerate(models):
        coverages = [conformal_eval_by_backbone[model]["per_class_coverage"].get(c, 0) for c in classes]
        plt.bar(x + i * width - 0.4 + width/2, coverages, width, label=model, color=colors[i])
        
    plt.axhline(y=1.0 - alpha, color='red', linestyle='--', label=f'Target ({1-alpha:.2f})')
    plt.xticks(x, [f'Class {c}' for c in classes])
    plt.ylabel('Coverage')
    plt.title('Class-Conditional Conformal Coverage')
    plt.legend()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()

def plot_set_size_distribution(pred_sets_by_backbone: dict, save_path: str):
    plt.figure(figsize=(10, 6))
    for model_name, pred_sets in pred_sets_by_backbone.items():
        set_sizes = [len(s) for s in pred_sets]
        sns.kdeplot(set_sizes, label=model_name, bw_adjust=1.5)
        
    plt.xlabel('Prediction Set Size')
    plt.ylabel('Density')
    plt.title('Distribution of Conformal Set Sizes')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
