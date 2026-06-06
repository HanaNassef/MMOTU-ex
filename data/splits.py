import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
import numpy as np

def create_patient_level_splits(metadata_df: pd.DataFrame, train_ratio: float = 0.68, val_ratio: float = 0.14, test_ratio: float = 0.18, random_state: int = 42) -> pd.DataFrame:
    """
    Split the dataset into train/val/test at the patient level.
    Ensures all images from one patient go to a single split and class ratios are balanced.
    """
    assert np.isclose(train_ratio + val_ratio + test_ratio, 1.0), "Ratios must sum to 1"
    
    # Sort patients by their majority class to help with stratified group k-fold
    patient_majority_class = metadata_df.groupby('patient_id')['class_label'].agg(lambda x: x.mode()[0]).reset_index()
    patient_majority_class.rename(columns={'class_label': 'majority_class'}, inplace=True)
    
    merged_df = metadata_df.merge(patient_majority_class, on='patient_id', how='left')
    
    # We will use StratifiedGroupKFold in two steps to get approximately train/val/test splits.
    # First split: Test vs (Train + Val)
    sgkf_test = StratifiedGroupKFold(n_splits=int(1/test_ratio), shuffle=True, random_state=random_state)
    
    X = merged_df.index.values
    y = merged_df['majority_class'].values
    groups = merged_df['patient_id'].values
    
    test_idx = []
    train_val_idx = []
    
    for train_val_i, test_i in sgkf_test.split(X, y, groups):
        train_val_idx = X[train_val_i]
        test_idx = X[test_i]
        break # Just need one fold
        
    # Second split: Train vs Val from the Train+Val set
    # Calculate the relative validation ratio in the remaining set
    relative_val_ratio = val_ratio / (train_ratio + val_ratio)
    
    sgkf_val = StratifiedGroupKFold(n_splits=int(1/relative_val_ratio), shuffle=True, random_state=random_state)
    
    X_train_val = X[train_val_i]
    y_train_val = y[train_val_i]
    groups_train_val = groups[train_val_i]
    
    train_idx = []
    val_idx = []
    
    for train_i, val_i in sgkf_val.split(X_train_val, y_train_val, groups_train_val):
        train_idx = X_train_val[train_i]
        val_idx = X_train_val[val_i]
        break # Just need one fold
        
    # Assign splits
    metadata_df['split'] = 'none'
    metadata_df.loc[train_idx, 'split'] = 'train'
    metadata_df.loc[val_idx, 'split'] = 'val'
    metadata_df.loc[test_idx, 'split'] = 'test'
    
    # Print distribution
    print("Class distribution across splits:")
    print(pd.crosstab(metadata_df['split'], metadata_df['class_label']))
    
    # Drop the temporary column if added to df (we didn't modify original, just created merged)
    return metadata_df

def load_splits(splits_csv: str):
    """Load the split CSV and return (train_df, val_df, test_df)."""
    df = pd.read_csv(splits_csv)
    train_df = df[df['split'] == 'train'].reset_index(drop=True)
    val_df = df[df['split'] == 'val'].reset_index(drop=True)
    test_df = df[df['split'] == 'test'].reset_index(drop=True)
    return train_df, val_df, test_df
