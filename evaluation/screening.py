import pandas as pd
import numpy as np
from sklearn.metrics import roc_curve, auc

class ScreeningAnalyzer:
    def __init__(self, results_df: pd.DataFrame):
        self.df = results_df.copy()

    def threshold_sweep(self, sc_thresholds=[0.1, 0.15, 0.2, 0.25, 0.3], cc_thresholds=[0.3, 0.4, 0.5, 0.6, 0.7]) -> pd.DataFrame:
        results = []
        
        # We need a single prediction per image. 
        # Since XAI methods just replicate predictions, we group by image to get the true prediction state.
        # But wait, we evaluate the flag rate based on the XAI method alignment.
        # It's better to evaluate this per XAI method.
        # The prompt says "For each (sc_thresh, cc_thresh) combination".
        # Assume we operate on the entire DF (which might have multiple XAI methods / cam_thresholds).
        # To be precise, we usually fix a CAM threshold and a method (e.g. GradCAM, cam_thresh=0.5).
        # We will filter to GradCAM and cam_threshold = 0.5 if there are multiple, to give a clean sweep.
        # Or we do it across the current slice provided.
        
        for sc_t in sc_thresholds:
            for cc_t in cc_thresholds:
                # Flag condition
                flagged = (self.df['sc'] < sc_t) & (self.df['cc'] < cc_t)
                
                # Ground truth for flag: incorrect prediction (is_correct == False)
                incorrect = ~self.df['is_correct']
                
                tp = (flagged & incorrect).sum()
                fp = (flagged & ~incorrect).sum()
                fn = (~flagged & incorrect).sum()
                tn = (~flagged & ~incorrect).sum()
                
                flag_rate = flagged.mean()
                
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                f1_screening = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
                
                sensitivity = recall # same as recall
                specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
                youden_j = sensitivity + specificity - 1.0
                
                results.append({
                    "sc_thresh": sc_t,
                    "cc_thresh": cc_t,
                    "flag_rate": flag_rate,
                    "precision": precision,
                    "recall": recall,
                    "f1_screening": f1_screening,
                    "youden_j": youden_j
                })
                
        return pd.DataFrame(results)

    def find_optimal_thresholds(self, sweep_df: pd.DataFrame) -> dict:
        best_idx = sweep_df['youden_j'].idxmax()
        return sweep_df.iloc[best_idx].to_dict()

    def exbale_roc_analysis(self) -> dict:
        # Treat 'incorrect prediction' as positive class
        y_true = (~self.df['is_correct']).astype(int)
        
        # Predictor is (1 - ExBale) since lower ExBale means higher chance of incorrect
        # Handle NaNs (e.g. drop them or treat as 1.0 - which means worst explanation)
        valid_idx = ~self.df['exbale'].isna()
        y_true_valid = y_true[valid_idx]
        y_score = 1.0 - self.df.loc[valid_idx, 'exbale']
        
        if len(y_true_valid) == 0:
            return {"auc_roc": 0.0, "optimal_threshold": 0.0, "fpr": [], "tpr": []}
            
        fpr, tpr, thresholds = roc_curve(y_true_valid, y_score)
        roc_auc = auc(fpr, tpr)
        
        youden = tpr - fpr
        opt_idx = np.argmax(youden)
        opt_thresh = thresholds[opt_idx]
        
        return {
            "auc_roc": roc_auc,
            "optimal_threshold": opt_thresh,
            "fpr": fpr,
            "tpr": tpr
        }

    def per_class_reliability(self) -> pd.DataFrame:
        agg = self.df.groupby('class_label').agg({
            'sc': 'mean',
            'cc': 'mean',
            'wcis': 'mean',
            'exbale': 'mean'
        }).reset_index()
        
        # Add flag rate using default thresholds SC<0.2, CC<0.5
        flagged = (self.df['sc'] < 0.2) & (self.df['cc'] < 0.5)
        self.df['flagged'] = flagged
        flag_rate_df = self.df.groupby('class_label')['flagged'].mean().reset_index()
        flag_rate_df.rename(columns={'flagged': 'flag_rate'}, inplace=True)
        
        return pd.merge(agg, flag_rate_df, on='class_label').set_index('class_label')
