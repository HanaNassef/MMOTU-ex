import numpy as np
import pytest
from uncertainty.conformal import MondrianAPSConformalPredictor, evaluate_conformal_sets

def test_mondrian_aps_coverage():
    num_classes = 8
    n_cal = 500
    n_test = 500
    alpha = 0.10
    
    # Generate synthetic well-calibrated probabilities
    np.random.seed(42)
    cal_logits = np.random.randn(n_cal, num_classes)
    cal_probs = np.exp(cal_logits) / np.sum(np.exp(cal_logits), axis=1, keepdims=True)
    cal_labels = np.array([np.random.choice(num_classes, p=prob) for prob in cal_probs])
    
    test_logits = np.random.randn(n_test, num_classes)
    test_probs = np.exp(test_logits) / np.sum(np.exp(test_logits), axis=1, keepdims=True)
    test_labels = np.array([np.random.choice(num_classes, p=prob) for prob in test_probs])
    
    # Initialize and calibrate
    predictor = MondrianAPSConformalPredictor(alpha=alpha, min_calibration_per_class=10)
    predictor.calibrate(cal_probs, cal_labels)
    
    # Predict and evaluate
    pred_sets = predictor.predict_sets(test_probs)
    eval_metrics = evaluate_conformal_sets(pred_sets, test_labels, num_classes)
    
    # Assert marginal coverage is >= 1 - alpha - 0.03 (finite-sample slack)
    assert eval_metrics["marginal_coverage"] >= (1.0 - alpha - 0.03), \
        f"Coverage {eval_metrics['marginal_coverage']} is too low for target {1-alpha}"
