import numpy as np
import pandas as pd
from pathlib import Path
from typing import List

CLASS_NAMES = [
    "Chocolate cyst",
    "Serous cystadenoma",
    "Teratoma",
    "Theca cell tumour",
    "Clear cell carcinoma",
    "Dermoid cyst",
    "Simple cyst",
    "Normal ovary",
]


def export_conformal_sets(
    model_name: str,
    image_paths: List[str],
    true_labels: np.ndarray,
    test_calibrated_probs: np.ndarray,
    pred_sets: List[List[int]],
    output_dir: str = "results",
) -> pd.DataFrame:
    """
    Writes one row per test image with the predicted conformal set, its
    size, whether it is a singleton, whether the true label is covered,
    and the top-1 softmax prediction for reference. This does not change
    any existing summary file; it is purely additive.
    """
    n = len(true_labels)
    assert len(image_paths) == n
    assert len(pred_sets) == n
    assert test_calibrated_probs.shape[0] == n

    rows = []
    for i in range(n):
        set_indices = sorted(pred_sets[i])
        set_names = [CLASS_NAMES[c] for c in set_indices]
        top1_idx = int(np.argmax(test_calibrated_probs[i]))
        true_idx = int(true_labels[i])

        rows.append({
            "image_path": image_paths[i],
            "true_class_idx": true_idx,
            "true_class_name": CLASS_NAMES[true_idx],
            "top1_pred_idx": top1_idx,
            "top1_pred_name": CLASS_NAMES[top1_idx],
            "top1_confidence": float(test_calibrated_probs[i, top1_idx]),
            "conformal_set_idx": set_indices,
            "conformal_set_names": set_names,
            "set_size": len(set_indices),
            "is_singleton": len(set_indices) == 1,
            "true_label_covered": true_idx in set_indices,
            "top1_correct": top1_idx == true_idx,
        })

    df = pd.DataFrame(rows)
    out_path = Path(output_dir) / f"{model_name}_conformal_sets.csv"
    df.to_csv(out_path, index=False)
    return df


def select_qualitative_test_cases(
    df: pd.DataFrame, n_singleton: int = 2, n_large_set: int = 2, n_uncovered: int = 1
) -> pd.DataFrame:
    """
    Selects a small, representative sample for the Results section figure
    or table: confident singleton cases, ambiguous large-set cases, and
    (if any exist) a case where the true label falls outside the set, which
    is the clinically important failure mode to show honestly rather than
    hide.
    """
    singleton_correct = df[df["is_singleton"] & df["top1_correct"]].sort_values(
        "top1_confidence", ascending=False
    ).head(n_singleton)

    large_set = df.sort_values("set_size", ascending=False).head(n_large_set)

    uncovered = df[~df["true_label_covered"]].head(n_uncovered)

    combined = pd.concat([singleton_correct, large_set, uncovered]).drop_duplicates(
        subset="image_path"
    )
    combined["case_type"] = "unlabeled"
    combined.loc[combined["image_path"].isin(singleton_correct["image_path"]), "case_type"] = "confident_singleton"
    combined.loc[combined["image_path"].isin(large_set["image_path"]), "case_type"] = "ambiguous_large_set"
    combined.loc[combined["image_path"].isin(uncovered["image_path"]), "case_type"] = "coverage_failure"

    return combined.reset_index(drop=True)
