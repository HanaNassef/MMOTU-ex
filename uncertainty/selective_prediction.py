import numpy as np


def risk_coverage_curve(probs: np.ndarray, labels: np.ndarray, uncertainty_scores: np.ndarray):
    probs = np.asarray(probs)
    labels = np.asarray(labels)
    uncertainty_scores = np.asarray(uncertainty_scores)

    order = np.argsort(uncertainty_scores)
    sorted_labels = labels[order]
    sorted_preds = probs[order].argmax(axis=1)

    coverages = []
    risks = []
    n = len(labels)

    for k in range(1, n + 1):
        retained_accuracy = np.mean(sorted_preds[:k] == sorted_labels[:k])
        coverages.append(k / n)
        risks.append(1.0 - retained_accuracy)

    aurc = float(np.trapz(risks, coverages)) if n > 1 else float(risks[0] if risks else 0.0)
    return {
        "coverages": coverages,
        "risks": risks,
        "aurc": aurc,
    }
