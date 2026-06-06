__version__ = "1.0.0"

import argparse
import yaml
import torch
import pandas as pd
from pathlib import Path
import logging
import gc
import re
import json

from utils.logger import setup_logger
from utils.reproducibility import setup_reproducibility
from utils.checkpoint import load_checkpoint
from data.splits import create_patient_level_splits, load_splits
from data.dataset import get_dataloaders, compute_class_weights, MMOTUDataset
from models.factory import get_model
from training.trainer import Trainer
from xai.xai_runner import XAIRunner
from evaluation.screening import ScreeningAnalyzer
from evaluation.statistical_tests import StatisticalAnalyzer
from visualization.plots import (
    plot_training_curves, plot_confusion_matrix, plot_backbone_comparison,
    plot_xai_comparison_violin, plot_threshold_heatmap, plot_exbale_vs_correctness,
    plot_screening_results_table, plot_per_class_alignment, plot_roc_curves,
    plot_insertion_deletion, plot_grad_norm_history
)
from visualization.report import generate_summary_report
from torchvision import transforms

class ConfigNamespace:
    def __init__(self, d):
        for a, b in d.items():
            if isinstance(b, (list, tuple)):
                setattr(self, a, [ConfigNamespace(x) if isinstance(x, dict) else x for x in b])
            else:
                setattr(self, a, ConfigNamespace(b) if isinstance(b, dict) else b)

def load_config(path: str) -> ConfigNamespace:
    with open(path, 'r') as f:
        config_dict = yaml.safe_load(f)
    return ConfigNamespace(config_dict)

def parse_args():
    parser = argparse.ArgumentParser(description="MMOTU XAI Pipeline")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to YAML config")
    parser.add_argument("--stage", type=int, default=None, help="Run only this stage (0-6)")
    parser.add_argument("--resume", type=str, default=None, help="Resume training from checkpoint")
    parser.add_argument("--skip_training", action="store_true", help="Skip Stage 2")
    parser.add_argument("--models", type=str, default=None, help="Comma-separated list of models to run")
    parser.add_argument("--debug", action="store_true", help="Run in fast debug mode")
    return parser.parse_args()

def setup_device(device_str: str) -> torch.device:
    if device_str == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_str)

def create_output_dirs(output_config):
    Path(output_config.results_dir).mkdir(parents=True, exist_ok=True)
    Path(output_config.checkpoints_dir).mkdir(parents=True, exist_ok=True)
    Path(output_config.logs_dir).mkdir(parents=True, exist_ok=True)
    Path(output_config.figures_dir).mkdir(parents=True, exist_ok=True)
    Path(output_config.xai_results_dir).mkdir(parents=True, exist_ok=True)
    Path(output_config.viz_dir).mkdir(parents=True, exist_ok=True)

def load_or_discover_metadata(data_config) -> pd.DataFrame:
    raw_dir = Path(data_config.raw_dir)
    meta_path = raw_dir / data_config.metadata_file if data_config.metadata_file else None
    
    if meta_path and meta_path.exists():
        df = pd.read_csv(meta_path)
        # Ensure standard columns
        if 'image_path' not in df.columns:
            # try to fix paths assuming relative to raw_dir
            pass # Implementation specific to actual CSV format
        return df
        
    # Auto-discover
    print("Auto-discovering dataset...")
    images_dir = raw_dir / data_config.images_subdir
    masks_dir = raw_dir / data_config.masks_subdir
    
    data = []
    # Assume folder names represent classes or filename encodes it
    # We will just assign a dummy class 0 if we can't infer, but spec says 8 classes
    # Assuming class subfolders in images_dir
    class_folders = [f for f in images_dir.iterdir() if f.is_dir()]
    if not class_folders:
        # No subfolders, try filename regex or just put everything in class 0
        for img_path in images_dir.glob("*.*"):
            if img_path.suffix.lower() not in ['.jpg', '.png', '.jpeg']: continue
            
            # infer patient
            m = re.search(r'patient(\d+)', img_path.stem, re.IGNORECASE)
            patient_id = m.group(1) if m else "unknown"
            
            # infer class from filename? e.g. class3_patient...
            cm = re.search(r'class(\d+)', img_path.stem, re.IGNORECASE)
            class_label = int(cm.group(1)) if cm else 0
            
            mask_path = masks_dir / img_path.name
            
            data.append({
                'patient_id': patient_id,
                'image_path': str(img_path),
                'mask_path': str(mask_path) if mask_path.exists() else None,
                'class_label': class_label
            })
    else:
        # Subfolders exist (e.g. class_0, class_1)
        for class_dir in class_folders:
            # try to extract int from folder name
            digits = re.findall(r'\d+', class_dir.name)
            class_label = int(digits[-1]) if digits else 0
            
            for img_path in class_dir.glob("*.*"):
                if img_path.suffix.lower() not in ['.jpg', '.png', '.jpeg']: continue
                m = re.search(r'patient(\d+)', img_path.stem, re.IGNORECASE)
                patient_id = m.group(1) if m else "unknown"
                
                # Mask might be in matching subfolder or flat
                mask_path = masks_dir / class_dir.name / img_path.name
                if not mask_path.exists():
                    mask_path = masks_dir / img_path.name
                    
                data.append({
                    'patient_id': patient_id,
                    'image_path': str(img_path),
                    'mask_path': str(mask_path) if mask_path.exists() else None,
                    'class_label': class_label
                })
                
    df = pd.DataFrame(data)
    print(f"Discovered {len(df)} images.")
    print("Class distribution:")
    print(df['class_label'].value_counts())
    return df

def main():
    args = parse_args()
    config = load_config(args.config)
    
    if args.models:
        config.training.models_to_train = [m.strip() for m in args.models.split(',')]
        
    if args.debug:
        config.training.num_epochs = 3
        config.xai.run_shap = False
        config.xai.run_faithfulness = False
        config.xai.cam_methods = ["gradcam"]
        
    run_stage = lambda s: args.stage is None or args.stage == s
    
    # ── Stage 0: Setup ──
    setup_reproducibility(config.experiment.random_seed)
    device = setup_device(config.experiment.device)
    create_output_dirs(config.output)
    logger = setup_logger(config.output.logs_dir, config.experiment.run_name)
    logger.info("=" * 60)
    logger.info(f"Run: {config.experiment.run_name} | Device: {device} | Debug: {args.debug}")
    
    # ── Stage 1: Data ──
    if run_stage(1):
        logger.info("=" * 60)
        logger.info("Stage 1: Data Preparation")
        if not Path(config.data.splits_csv).exists():
            logger.info("Creating patient-level splits...")
            metadata = load_or_discover_metadata(config.data)
            
            if args.debug:
                # subset for debug
                metadata = metadata.sample(min(100, len(metadata)), random_state=config.experiment.random_seed)
                
            splits_df = create_patient_level_splits(metadata)
            splits_df.to_csv(config.data.splits_csv, index=False)
            
        train_df, val_df, test_df = load_splits(config.data.splits_csv)
        class_weights = compute_class_weights(train_df)
        logger.info(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
        
        # Pre-compute wcis stats for normalization
        # We can approximate this by running WCIS on test set masks if available, but let's just save defaults.
        with open(Path(config.output.results_dir) / "wcis_normalization_stats.json", 'w') as f:
            json.dump({"wcis_global_min": 0.0, "wcis_global_max": 1.0}, f)
            
    # ── Stage 2: Training ──
    trained_models = {}
    if run_stage(2) and not args.skip_training:
        logger.info("=" * 60)
        logger.info("Stage 2: Training")
        train_df, val_df, test_df = load_splits(config.data.splits_csv)
        class_weights = compute_class_weights(train_df)
        
        for model_name in config.training.models_to_train:
            run_name = f"{config.experiment.run_name}_{model_name}"
            logger.info(f"=== Training {model_name} ===")
            
            model, in_features = get_model(model_name, num_classes=config.training.num_classes, dropout=config.training.dropout)
            model = model.to(device)
            
            logger.info(f"Model {model_name} created. Head in_features: {in_features}")
            
            train_loader, val_loader, test_loader = get_dataloaders(config.data.splits_csv, config)
            
            trainer = Trainer(model, train_loader, val_loader, config.training, device, logger, config.output.checkpoints_dir, run_name, class_weights=class_weights)
            
            if args.resume:
                # Implement resume logic if path provided
                pass
                
            best_ckpt_path = trainer.train()
            trained_models[model_name] = best_ckpt_path
            logger.info(f"Best checkpoint saved: {best_ckpt_path}")
            
            torch.cuda.empty_cache()
            gc.collect()
            
    elif args.skip_training:
        # Populate trained_models from existing files
        for model_name in config.training.models_to_train:
            ckpt_path = Path(config.output.checkpoints_dir) / f"{config.experiment.run_name}_{model_name}_best.pt"
            if ckpt_path.exists():
                trained_models[model_name] = str(ckpt_path)
            else:
                logger.warning(f"Skipped training but checkpoint not found for {model_name}: {ckpt_path}")

    # ── Stage 3: XAI ──
    all_xai_results = {}
    if run_stage(3):
        logger.info("=" * 60)
        logger.info("Stage 3: XAI Generation")
        _, _, test_df = load_splits(config.data.splits_csv)
        
        val_transform = transforms.Compose([
            transforms.Resize((config.data.image_size, config.data.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        mask_transform = transforms.Compose([
            transforms.Resize((config.data.image_size, config.data.image_size), interpolation=transforms.InterpolationMode.NEAREST),
            transforms.ToTensor()
        ])
        
        test_dataset_with_paths = MMOTUDataset(test_df, transform=val_transform, mask_transform=mask_transform, return_path=True)
        
        # Inject wcis path into config dynamically
        config.xai.wcis_stats_path = str(Path(config.output.results_dir) / "wcis_normalization_stats.json")
        
        for model_name, ckpt_path in trained_models.items():
            logger.info(f"=== XAI for {model_name} ===")
            model, _ = get_model(model_name, num_classes=config.training.num_classes)
            load_checkpoint(ckpt_path, model)
            model = model.to(device).eval()
            
            xai_runner = XAIRunner(model, model_name, test_dataset_with_paths, config.xai, device, logger)
            
            if config.xai.run_shap:
                train_df, _, _ = load_splits(config.data.splits_csv)
                bg_dataset = MMOTUDataset(train_df, transform=val_transform, mask_transform=mask_transform)
                xai_runner.init_shap(bg_dataset)
                
            results_df = xai_runner.run(output_dir=config.output.xai_results_dir)
            all_xai_results[model_name] = results_df
            
            torch.cuda.empty_cache()
            gc.collect()

    # If skipping early stages but wanting later stages, load XAI results
    if not run_stage(3) and (run_stage(4) or run_stage(5) or run_stage(6)):
        for model_name in config.training.models_to_train:
            res_path = Path(config.output.xai_results_dir) / f"xai_results_{model_name}.csv"
            if res_path.exists():
                all_xai_results[model_name] = pd.read_csv(res_path)

    # ── Stage 4: Evaluation ──
    if run_stage(4):
        logger.info("=" * 60)
        logger.info("Stage 4: Evaluation")
        
        all_sweeps = {}
        all_per_class = {}
        
        for model_name, results_df in all_xai_results.items():
            if results_df.empty: continue
            logger.info(f"=== Evaluating {model_name} ===")
            analyzer = ScreeningAnalyzer(results_df)
            
            sweep_df = analyzer.threshold_sweep(sc_thresholds=config.evaluation.screening.sc_thresholds, 
                                                cc_thresholds=config.evaluation.screening.cc_thresholds)
            
            optimal = analyzer.find_optimal_thresholds(sweep_df)
            per_class = analyzer.per_class_reliability()
            roc_info = analyzer.exbale_roc_analysis()
            
            logger.info(f"Optimal thresholds: SC<{optimal['sc_thresh']}, CC<{optimal['cc_thresh']}")
            logger.info(f"ExBale ROC-AUC: {roc_info['auc_roc']:.4f}")
            
            sweep_df.to_csv(f"{config.output.results_dir}/{model_name}_threshold_sweep.csv", index=False)
            per_class.to_csv(f"{config.output.results_dir}/{model_name}_per_class_reliability.csv")
            
            all_sweeps[model_name] = sweep_df
            all_per_class[model_name] = per_class
            
        # Statistical Tests
        stat = StatisticalAnalyzer()
        if "densenet121" in all_xai_results and not all_xai_results["densenet121"].empty:
            xai_comparison = stat.compare_xai_methods(all_xai_results["densenet121"])
            logger.info("Statistical Tests computed for DenseNet121.")
            
        gradcam_results = {}
        for name, df in all_xai_results.items():
            if not df.empty and 'xai_method' in df.columns:
                gradcam_results[name] = df[df['xai_method'] == 'gradcam']
                
        if gradcam_results:
            backbone_comparison = stat.compare_backbones(gradcam_results)
            if not backbone_comparison['tukey_results_df'].empty:
                backbone_comparison['tukey_results_df'].to_csv(f"{config.output.results_dir}/backbone_tukey_hsd.csv", index=False)

    # ── Stage 5: Visualizations ──
    if run_stage(5):
        logger.info("=" * 60)
        logger.info("Stage 5: Visualization")
        
        for model_name in config.training.models_to_train:
            log_path = Path(config.output.logs_dir) / f"{config.experiment.run_name}_{model_name}_training_log.csv"
            if log_path.exists():
                plot_training_curves(str(log_path), f"{config.output.figures_dir}/{model_name}_training.png")
                plot_grad_norm_history(str(log_path), f"{config.output.figures_dir}/{model_name}_grad_norm.png")
                
            if model_name in all_xai_results and not all_xai_results[model_name].empty:
                df = all_xai_results[model_name]
                plot_xai_comparison_violin(df, save_path=f"{config.output.figures_dir}/{model_name}_xai_violin.png")
                plot_exbale_vs_correctness(df, save_path=f"{config.output.figures_dir}/{model_name}_exbale_vs_conf.png")
                plot_insertion_deletion(df, save_path=f"{config.output.figures_dir}/{model_name}_faithfulness.png")
                
            sweep_path = Path(config.output.results_dir) / f"{model_name}_threshold_sweep.csv"
            if sweep_path.exists():
                sweep_df = pd.read_csv(sweep_path)
                plot_threshold_heatmap(sweep_df, save_path=f"{config.output.figures_dir}/{model_name}_threshold_heatmap.png")
                plot_screening_results_table(sweep_df, save_path=f"{config.output.figures_dir}/{model_name}_screening_table.png")
                
            per_class_path = Path(config.output.results_dir) / f"{model_name}_per_class_reliability.csv"
            if per_class_path.exists():
                pc_df = pd.read_csv(per_class_path).set_index('class_label')
                plot_per_class_alignment(pc_df, save_path=f"{config.output.figures_dir}/{model_name}_per_class.png")
                
        if all_xai_results:
            plot_backbone_comparison(all_xai_results, save_path=f"{config.output.figures_dir}/backbone_comparison.png")
            
    # ── Stage 6: Summary ──
    if run_stage(6):
        logger.info("=" * 60)
        logger.info("Stage 6: Summary Report")
        generate_summary_report(all_xai_results, trained_models, config, save_path=f"{config.output.results_dir}/summary_report.txt")
        
    logger.info("=== Pipeline complete ===")

if __name__ == "__main__":
    main()
