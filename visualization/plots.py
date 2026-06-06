import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Common colors from project spec
BLUE_DARK = "#003366"
BLUE_MEDIUM = "#3366CC"
BLUE_LIGHT = "#6699FF"
BLUE_PALE = "#99CCFF"

def plot_training_curves(log_csv_path: str, save_path: str):
    df = pd.read_csv(log_csv_path)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    ax1.plot(df['epoch'], df['train_loss'], label='Train Loss', color=BLUE_DARK)
    ax1.plot(df['epoch'], df['val_loss'], label='Val Loss', color=BLUE_MEDIUM)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    
    ax2.plot(df['epoch'], df['train_macro_f1'], label='Train Macro F1', color=BLUE_DARK)
    ax2.plot(df['epoch'], df['val_macro_f1'], label='Val Macro F1', color=BLUE_MEDIUM)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Macro F1')
    ax2.legend()
    
    ax2_lr = ax2.twinx()
    ax2_lr.plot(df['epoch'], df['lr_backbone'], label='LR Backbone', color='r', linestyle='--')
    ax2_lr.set_ylabel('Learning Rate')
    ax2_lr.legend(loc='lower right')
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close(fig)

def plot_confusion_matrix(cm: np.ndarray, class_names: list, model_name: str, save_path: str):
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title(f"Confusion Matrix - {model_name}")
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_backbone_comparison(results_dict: dict, metric: str = "exbale", save_path: str = None):
    data = []
    for bb, df in results_dict.items():
        if df.empty: continue
        means = df.groupby('xai_method')[metric].mean().reset_index()
        means['backbone'] = bb
        data.append(means)
        
    if not data: return
    
    plot_df = pd.concat(data)
    plt.figure(figsize=(12, 6))
    sns.barplot(data=plot_df, x='backbone', y=metric, hue='xai_method', palette='Blues')
    plt.title(f'Mean {metric.upper()} by Backbone and XAI Method')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.close()

def plot_xai_comparison_violin(results_df: pd.DataFrame, metric: str = "exbale", save_path: str = None):
    if results_df.empty: return
    plt.figure(figsize=(10, 6))
    sns.violinplot(data=results_df, x='xai_method', y=metric, palette='Blues')
    plt.title(f'{metric.upper()} Distribution by XAI Method')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.close()

def plot_threshold_heatmap(sweep_df: pd.DataFrame, metric: str = "f1_screening", save_path: str = None):
    if sweep_df.empty: return
    pivot = sweep_df.pivot(index='sc_thresh', columns='cc_thresh', values=metric)
    plt.figure(figsize=(8, 6))
    sns.heatmap(pivot, annot=True, cmap='Blues', fmt=".3f")
    plt.title(f'{metric} for Threshold Sweeps')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.close()

def plot_exbale_vs_correctness(results_df: pd.DataFrame, save_path: str = None):
    if results_df.empty: return
    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=results_df, x='confidence', y='exbale', hue='is_correct', palette={True: BLUE_MEDIUM, False: 'red'}, alpha=0.5)
    plt.title('ExBale vs Model Confidence')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.close()

def plot_screening_results_table(sweep_df: pd.DataFrame, save_path: str):
    fig, ax = plt.subplots(figsize=(12, min(8, len(sweep_df)*0.5)))
    ax.axis('tight')
    ax.axis('off')
    display_df = sweep_df.round(4)
    table = ax.table(cellText=display_df.values, colLabels=display_df.columns, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    plt.savefig(save_path, bbox_inches='tight')
    plt.close(fig)

def plot_per_class_alignment(per_class_df: pd.DataFrame, save_path: str):
    if per_class_df.empty: return
    metrics = ['sc', 'cc', 'wcis', 'exbale']
    plot_df = per_class_df.reset_index().melt(id_vars='class_label', value_vars=metrics, var_name='Metric', value_name='Score')
    
    plt.figure(figsize=(12, 6))
    sns.barplot(data=plot_df, x='class_label', y='Score', hue='Metric', palette='Blues')
    plt.title('Alignment Metrics per Class')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_roc_curves(model_name: str, all_probs: np.ndarray, all_targets: np.ndarray, num_classes: int, save_path: str):
    from sklearn.metrics import roc_curve, auc
    plt.figure(figsize=(8, 8))
    
    colors = plt.cm.get_cmap('tab10', num_classes)
    for c in range(num_classes):
        y_true = (all_targets == c).astype(int)
        if y_true.sum() == 0: continue
        fpr, tpr, _ = roc_curve(y_true, all_probs[:, c])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, color=colors(c), lw=2, label=f'Class {c} (AUC = {roc_auc:.2f})')
        
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'Multi-class ROC - {model_name}')
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_insertion_deletion(results_df: pd.DataFrame, save_path: str):
    if results_df.empty or 'insertion_auc' not in results_df.columns: return
    
    means = results_df.groupby('xai_method')[['insertion_auc', 'deletion_auc']].mean().reset_index()
    plot_df = means.melt(id_vars='xai_method', var_name='Metric', value_name='AUC')
    
    plt.figure(figsize=(10, 6))
    sns.barplot(data=plot_df, x='xai_method', y='AUC', hue='Metric', palette='Blues')
    plt.title('Insertion and Deletion AUC by XAI Method')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_grad_norm_history(log_csv_path: str, save_path: str):
    df = pd.read_csv(log_csv_path)
    plt.figure(figsize=(10, 5))
    plt.plot(df['epoch'], df['grad_norm_mean'], label='Mean Grad Norm', color=BLUE_MEDIUM)
    plt.plot(df['epoch'], df['grad_norm_max'], label='Max Grad Norm', color=BLUE_DARK)
    plt.xlabel('Epoch')
    plt.ylabel('Gradient Norm')
    plt.title('Gradient Norm History')
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
