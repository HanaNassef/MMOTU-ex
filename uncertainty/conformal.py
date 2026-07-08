import numpy as np
from typing import Dict, List


def _aps_score(probs: np.ndarray, label: int) -> float:
    order = np.argsort(-probs)
    sorted_probs = probs[order]
    cumulative = np.cumsum(sorted_probs)
    label_position = int(np.where(order == label)[0][0])
    return float(cumulative[label_position])


class MondrianAPSConformalPredictor:
    def __init__(self, alpha: float = 0.1, min_calibration_per_class: int = 10):
        self.alpha = alpha
        self.min_calibration_per_class = min_calibration_per_class
        self.class_thresholds = {}
        self.global_threshold = None

    @staticmethod
    def _aps_score(probs_row: np.ndarray, true_label: int) -> float:
        return _aps_score(probs_row, true_label)

    def calibrate(self, probs: np.ndarray, labels: np.ndarray):
        probs = np.asarray(probs)
        labels = np.asarray(labels)

        scores_by_class: Dict[int, List[float]] = {}
        for probs_row, label in zip(probs, labels):
            scores_by_class.setdefault(int(label), []).append(self._aps_score(probs_row, int(label)))

        all_scores: List[float] = []
        for class_label, scores in scores_by_class.items():
            all_scores.extend(scores)
            n = len(scores)
            if n < self.min_calibration_per_class:
                continue
            q_level = min(np.ceil((n + 1) * (1 - self.alpha)) / n, 1.0)
            self.class_thresholds[class_label] = float(np.quantile(scores, q_level, method="higher"))

        n_all = len(all_scores)
        q_level_all = min(np.ceil((n_all + 1) * (1 - self.alpha)) / n_all, 1.0)
        self.global_threshold = float(np.quantile(all_scores, q_level_all, method="higher"))

        return self

    def predict_sets(self, probs: np.ndarray):
        probs = np.asarray(probs)
        prediction_sets = []

        for sample_probs in probs:
            order = np.argsort(-sample_probs)
            sorted_probs = sample_probs[order]
            cumsum = np.cumsum(sorted_probs)
            sample_set = []
            for rank, class_idx in enumerate(order):
                qhat = self.class_thresholds.get(int(class_idx), self.global_threshold)
                if cumsum[rank] <= qhat or rank == 0:
                    sample_set.append(int(class_idx))
                else:
                    break

            if not sample_set:
                sample_set = [int(np.argmax(sample_probs))]
            prediction_sets.append(sample_set)

        return prediction_sets


def evaluate_conformal_sets(pred_sets, labels, num_classes: int):
    labels = np.asarray(labels)
    marginal_coverage = np.mean([labels[i] in pred_sets[i] for i in range(len(labels))])
    avg_set_size = float(np.mean([len(s) for s in pred_sets])) if pred_sets else 0.0
    singleton_rate = float(np.mean([len(s) == 1 for s in pred_sets])) if pred_sets else 0.0

    per_class_coverage = {}
    for class_label in range(num_classes):
        class_indices = np.where(labels == class_label)[0]
        if len(class_indices) > 0:
            coverage = np.mean([labels[i] in pred_sets[i] for i in class_indices])
        else:
            coverage = float("nan")
        per_class_coverage[class_label] = float(coverage)

    return {
        "marginal_coverage": float(marginal_coverage),
        "avg_set_size": avg_set_size,
        "singleton_rate": singleton_rate,
        "per_class_coverage": per_class_coverage,
    }
