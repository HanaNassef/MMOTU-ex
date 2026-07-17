import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, wilcoxon

def compute_per_image_risk_contributions(probs: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Calculates continuous per-image NLL as a proxy for risk contribution."""
    n = len(labels)
    contributions = np.zeros(n)
    for i in range(n):
        true_class_prob = probs[i, labels[i]]
        contributions[i] = -np.log(max(true_class_prob, 1e-12))
    return contributions

def run_repeated_measures_tests(risk_contributions_by_model: dict) -> pd.DataFrame:
    models = list(risk_contributions_by_model.keys())
    data = [risk_contributions_by_model[m] for m in models]
    
    # 1. Friedman Test
    stat, p_val = friedmanchisquare(*data)
    results = [{
        "Comparison": "Omnibus (Friedman)",
        "Statistic": stat,
        "p-value": p_val,
        "Mean_1": None,
        "Mean_2": None,
        "Lower_Risk_Model": None,
    }]
    
    # 2. Pairwise Wilcoxon with Bonferroni correction
    num_comparisons = (len(models) * (len(models) - 1)) / 2
    alpha_corrected = 0.05 / num_comparisons
    
    for i in range(len(models)):
        for j in range(i + 1, len(models)):
            m1, m2 = models[i], models[j]
            mean1, mean2 = float(np.mean(data[i])), float(np.mean(data[j]))
            try:
                w_stat, w_pval = wilcoxon(data[i], data[j])
                significant = "Yes" if w_pval < alpha_corrected else "No"
                results.append({
                    "Comparison": f"{m1} vs {m2}",
                    "Statistic": w_stat,
                    "p-value": w_pval,
                    "Significant (Bonferroni)": significant,
                    "Mean_1": mean1,
                    "Mean_2": mean2,
                    "Lower_Risk_Model": m1 if mean1 < mean2 else m2,
                })
            except ValueError:
                # Handles cases where all differences are exactly zero
                pass
                
    return pd.DataFrame(results)