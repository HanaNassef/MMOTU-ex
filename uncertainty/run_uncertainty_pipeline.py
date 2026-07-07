import json
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path


from .calibration import TemperatureScaler, expected_calibration_error
from .conformal import MondrianAPSConformalPredictor, evaluate_conformal_sets
from .selective_prediction import risk_coverage_curve
from .estimators import MCDropoutEstimator
from .uncertainty_stats import compute_per_image_risk_contributions, run_repeated_measures_tests

def _get_logits_and_labels(model, dataloader, device):
    """Helper function to collect logits and labels over a dataloader"""
    model.eval()
    all_logits, all_labels = [], []
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            logits = model(images)
            all_logits.append(logits.cpu().numpy())
            all_labels.append(labels.numpy())
    return np.vstack(all_logits), np.concatenate(all_labels)

def run_uncertainty_pipeline(trained_models: dict, config: dict, device: torch.device, alpha: float):

    summary_data = []
    risk_contributions_by_model = {}
    
    for model_name, model in trained_models.items():
        # 2. Run forward pass on Val split
        val_logits, val_labels = _get_logits_and_labels(model, val_loader, device)
        
        # 3. Fit Temperature Scaler
        scaler = TemperatureScaler().to(device)
        scaler.fit(torch.tensor(val_logits).to(device), torch.tensor(val_labels).to(device))
        
        # 4. Calibrate Val Probs + Fit Mondrian Conformal
        val_calibrated_logits = scaler(torch.tensor(val_logits).to(device)).cpu().numpy()
        val_calibrated_probs = F.softmax(torch.tensor(val_calibrated_logits), dim=1).numpy()
        
        predictor = MondrianAPSConformalPredictor(alpha=alpha, min_calibration_per_class=config.uncertainty.min_calibration_per_class)
        predictor.calibrate(val_calibrated_probs, val_labels)
        
        # 5. Run test split + Calibrate
        test_logits, test_labels = _get_logits_and_labels(model, test_loader, device)
        test_calibrated_logits = scaler(torch.tensor(test_logits).to(device)).cpu().numpy()
        test_calibrated_probs = F.softmax(torch.tensor(test_calibrated_logits), dim=1).numpy()
        test_raw_probs = F.softmax(torch.tensor(test_logits), dim=1).numpy()
        
        # 6. Evaluation 
        ece_pre = expected_calibration_error(test_raw_probs, test_labels, config.uncertainty.ece_n_bins)
        ece_post = expected_calibration_error(test_calibrated_probs, test_labels, config.uncertainty.ece_n_bins)
        pred_sets = predictor.predict_sets(test_calibrated_probs)
        conformal_eval = evaluate_conformal_sets(pred_sets, test_labels, num_classes=8) 
        
        uncertainty_scores = -np.log(test_calibrated_probs.max(axis=1))
        rc_curve = risk_coverage_curve(test_calibrated_probs, test_labels, uncertainty_scores)
        
        # 7. MC Dropout 
        mc_estimator = MCDropoutEstimator(model, device, n_samples=config.uncertainty.mc_dropout_samples)

        
        # Log to summary
        summary_data.append({
            "Backbone": model_name,
            "Temperature": scaler.temperature.item(),
            "ECE_Pre": ece_pre,
            "ECE_Post": ece_post,
            "Marginal_Coverage": conformal_eval["marginal_coverage"],
            "Avg_Set_Size": conformal_eval["avg_set_size"],
            "Singleton_Rate": conformal_eval["singleton_rate"],
            "AURC": rc_curve["aurc"]
        })
        
        # Store for stats
        risk_contributions_by_model[model_name] = compute_per_image_risk_contributions(test_calibrated_probs, test_labels)
        
        # Output JSON
        with open(f"results/{model_name}_riskcoverage.json", "w") as f:
            json.dump(rc_curve, f)
            
    # 8. Write DataFrames
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv("results/uncertainty_conformal_summary.csv", index=False)
    
    stats_df = run_repeated_measures_tests(risk_contributions_by_model)
    stats_df.to_csv("results/backbone_uncertainty_friedman.csv", index=False)
    
    return summary_df
