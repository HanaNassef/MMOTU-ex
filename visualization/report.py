import json
import pandas as pd
from pathlib import Path
import datetime

def generate_summary_report(all_xai_results: dict, trained_models: dict, config: any, save_path: str):
    with open(save_path, 'w') as f:
        f.write("====================================================\n")
        f.write("      MMOTU XAI Pipeline - Summary Report\n")
        f.write("====================================================\n\n")
        
        f.write(f"Date Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Run Name: {config.experiment.run_name}\n")
        f.write(f"Device: {config.experiment.device}\n\n")
        
        f.write("--- 1. Configuration ---\n")
        f.write(f"Models Trained: {', '.join(config.training.models_to_train)}\n")
        f.write(f"Epochs: {config.training.num_epochs}, Batch Size: {config.training.batch_size}\n")
        f.write(f"Loss Function: {config.training.loss_fn}\n")
        f.write(f"XAI Methods: {', '.join(config.xai.cam_methods + config.xai.gradient_methods)}\n")
        if config.xai.run_shap:
            f.write(f"SHAP enabled: True\n")
        f.write("\n")
        
        f.write("--- 2. Dataset Splitting ---\n")
        try:
            df = pd.read_csv(config.data.splits_csv)
            f.write(f"Total Images: {len(df)}\n")
            counts = df.groupby(['split', 'class_label']).size().unstack(fill_value=0)
            f.write(counts.to_string())
            f.write("\n\n")
        except Exception as e:
            f.write(f"Could not load split stats: {e}\n\n")
            
        f.write("--- 3. Classification Results ---\n")
        for model in trained_models.keys():
            log_dir = Path(config.output.logs_dir)
            json_file = log_dir / f"{config.experiment.run_name}_{model}_best_metrics.json"
            if json_file.exists():
                with open(json_file, 'r') as jf:
                    m = json.load(jf)
                f.write(f"{model}:\n")
                f.write(f"  Best Epoch: {m.get('best_epoch')}\n")
                f.write(f"  Top-1 Acc:  {m.get('best_val_top1_acc', 0):.4f}\n")
                f.write(f"  Top-2 Acc:  {m.get('best_val_top2_acc', 0):.4f}\n")
                f.write(f"  Macro F1:   {m.get('best_val_macro_f1', 0):.4f}\n")
                f.write(f"  Bal Acc:    {m.get('best_val_balanced_acc', 0):.4f}\n\n")
            else:
                f.write(f"{model}: No metric JSON found.\n\n")
                
        f.write("--- 4. XAI Alignment Results (ExBale) ---\n")
        for bb, df in all_xai_results.items():
            if df.empty: continue
            f.write(f"Backbone: {bb}\n")
            means = df.groupby('xai_method')[['sc', 'cc', 'wcis', 'exbale']].mean().round(4)
            f.write(means.to_string())
            f.write("\n\n")
            
        f.write("--- 5. Saved Files Path ---\n")
        f.write(f"Results Dir: {config.output.results_dir}\n")
        f.write(f"Figures Dir: {config.output.figures_dir}\n")
        f.write("Check visual plots for thresholds, ROCs, and qualitative comparisons.\n")
        f.write("====================================================\n")
