import json
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path

# Import your existing data loader function
from data.dataset import get_dataloaders
from models.factory import get_model
from utils.checkpoint import load_checkpoint

from .calibration import TemperatureScaler, expected_calibration_error
from .conformal import MondrianAPSConformalPredictor, evaluate_conformal_sets
from .selective_prediction import risk_coverage_curve
from .estimators import MCDropoutEstimator
from .uncertainty_stats import compute_per_image_risk_contributions, run_repeated_measures_tests

from visualization.uncertainty_plots import (
    plot_reliability_diagram,
    plot_risk_coverage_curves,
    plot_coverage_per_class,
    plot_set_size_distribution
)

def _get_logits_and_labels(model, dataloader, device):
    """Helper function to collect logits and labels over a dataloader"""
    model.eval()
    all_logits, all_labels = [], []
    with torch.no_grad():
        for images, _, labels in dataloader:
            images = images.to(device)
            logits = model(images)
            all_logits.append(logits.cpu().numpy())
            all_labels.append(labels.cpu().numpy()) 
    return np.vstack(all_logits), np.concatenate(all_labels)


def _resolve_model_name(model_key: str) -> str:
    if "_fold" in model_key:
        return model_key.split("_fold")[0]
    return model_key


def _load_model_from_checkpoint(model_key: str, checkpoint_path: str, config, device: torch.device):
    model_name = _resolve_model_name(model_key)
    model, _ = get_model(model_name, num_classes=config.training.num_classes, dropout=config.training.dropout)
    load_checkpoint(checkpoint_path, model)
    return model.to(device).eval()

def run_uncertainty_pipeline(trained_models: dict, config: dict, device: torch.device, alpha: float | None = None, logger=None):
    
    # 1. Initialize Data Loaders 
    splits_path = getattr(config.data, "splits_path", None) or getattr(config.data, "splits_csv")
    _, val_loader, test_loader = get_dataloaders(splits_path, config)

    summary_data = []
    risk_contributions_by_model = {}
    

    rc_curves_by_backbone = {}
    conformal_eval_by_backbone = {}
    pred_sets_by_backbone = {}
    summary_rows = []
    
    figures_dir = Path(config.output.figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)
    results_dir = Path(config.output.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    alpha = float(alpha if alpha is not None else getattr(config.uncertainty, "alpha", 0.10))
    
    for model_name, checkpoint_path in trained_models.items():
        model = _load_model_from_checkpoint(model_name, checkpoint_path, config, device)

        # 2. Run forward pass on Val split
        val_logits, val_labels = _get_logits_and_labels(model, val_loader, device)
        
        # 3. Fit Temperature Scaler
        scaler = TemperatureScaler().to(device)
        fitted_temperature = scaler.fit(torch.tensor(val_logits).to(device), torch.tensor(val_labels).to(device))
        if fitted_temperature <= 1.05 or fitted_temperature > 10:
            message = f"[uncertainty] {model_name}: fitted temperature={fitted_temperature:.3f}"
            if logger is not None:
                logger.warning(message)
            else:
                print(message)
        
        # 4. Apply Scaler to Val Data
        val_calibrated_logits = scaler(torch.tensor(val_logits).to(device)).detach().cpu().numpy()
        val_calibrated_probs = F.softmax(torch.tensor(val_calibrated_logits), dim=1).numpy()
        
        # 5. Run test split + Calibrate
        test_logits, test_labels = _get_logits_and_labels(model, test_loader, device)
        test_calibrated_logits = scaler(torch.tensor(test_logits).to(device)).detach().cpu().numpy()
        test_calibrated_probs = F.softmax(torch.tensor(test_calibrated_logits), dim=1).numpy()
        test_raw_probs = F.softmax(torch.tensor(test_logits), dim=1).numpy()
        
        # 6. Evaluation (ECE)
        ece_pre = expected_calibration_error(test_raw_probs, test_labels, config.uncertainty.ece_n_bins)
        ece_post = expected_calibration_error(test_calibrated_probs, test_labels, config.uncertainty.ece_n_bins)

        #  Draw the Reliability Diagram for this specific model right now
        plot_reliability_diagram(
            test_raw_probs, 
            test_calibrated_probs, 
            test_labels, 
            str(figures_dir / f"{model_name}_reliability.png"),
            n_bins=config.uncertainty.ece_n_bins
        )

        predictor = MondrianAPSConformalPredictor(alpha=alpha, min_calibration_per_class=config.uncertainty.min_calibration_per_class)
        predictor.calibrate(val_calibrated_probs, val_labels)

        pred_sets = predictor.predict_sets(test_calibrated_probs)
        conformal_eval = evaluate_conformal_sets(pred_sets, test_labels, num_classes=config.training.num_classes)

        uncertainty_scores_softmax = -np.log(test_calibrated_probs.max(axis=1))
        rc_curve_softmax = risk_coverage_curve(test_calibrated_probs, test_labels, uncertainty_scores_softmax)

        # 8. MC Dropout (AURC comparison)
        mc_estimator = MCDropoutEstimator(model, device, n_samples=config.uncertainty.mc_dropout_samples)

        all_mc_entropy = []
        for images, _, _ in test_loader:
            mc_results = mc_estimator.predict(images)
            all_mc_entropy.append(mc_results["predictive_entropy"])

        uncertainty_scores_mc = np.concatenate(all_mc_entropy)
        rc_curve_mc = risk_coverage_curve(test_calibrated_probs, test_labels, uncertainty_scores_mc)

        summary_rows.append({
            "Backbone": model_name,
            "Alpha": alpha,
            "Temperature": fitted_temperature,
            "ECE_Pre": ece_pre,
            "ECE_Post": ece_post,
            "Marginal_Coverage": conformal_eval["marginal_coverage"],
            "Avg_Set_Size": conformal_eval["avg_set_size"],
            "Singleton_Rate": conformal_eval["singleton_rate"],
            "AURC_Softmax": rc_curve_softmax["aurc"],
            "AURC_MC_Dropout": rc_curve_mc["aurc"],
        })

        with open(results_dir / f"{model_name}_riskcoverage.json", "w") as f:
            json.dump(rc_curve_softmax, f)

        rc_curves_by_backbone[model_name] = rc_curve_softmax
        conformal_eval_by_backbone[model_name] = {"per_class_coverage": conformal_eval["per_class_coverage"]}
        pred_sets_by_backbone[model_name] = pred_sets
        
        # Store for stats (using the calibrated probabilities)
        risk_contributions_by_model[model_name] = compute_per_image_risk_contributions(test_calibrated_probs, test_labels)
            
    # Draw the three big comparison charts using the boxes
    plot_risk_coverage_curves(rc_curves_by_backbone, str(figures_dir / "all_risk_coverage.png"))
    plot_coverage_per_class(conformal_eval_by_backbone, str(figures_dir / "all_class_coverage.png"), alpha=0.10)
    plot_set_size_distribution(pred_sets_by_backbone, str(figures_dir / "all_set_sizes.png"))

    # 9. Write DataFrames
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(results_dir / "uncertainty_conformal_summary.csv", index=False)
    
    stats_df = run_repeated_measures_tests(risk_contributions_by_model)
    stats_df.to_csv(results_dir / "backbone_uncertainty_friedman.csv", index=False)
    
    return summary_df
