import numpy as np
import cv2

def binarize_cam(cam: np.ndarray, threshold: float) -> np.ndarray:
    """
    cam: float array in [0, 1], shape (H, W)
    threshold: float in (0, 1)
    Returns: bool array same shape
    """
    c_min, c_max = cam.min(), cam.max()
    if c_max > c_min:
        norm_cam = (cam - c_min) / (c_max - c_min)
    else:
        norm_cam = cam
    return norm_cam >= threshold

def compute_sc(cam_binary: np.ndarray, seg_mask: np.ndarray) -> float:
    """Segmentation Coverage = |CAM ∩ SEG| / |SEG|"""
    if seg_mask.shape != cam_binary.shape:
        cam_binary = cv2.resize(cam_binary.astype(np.uint8), (seg_mask.shape[1], seg_mask.shape[0]), interpolation=cv2.INTER_NEAREST)

    seg_sum = seg_mask.sum()
    if seg_sum == 0:
        return np.nan
        
    intersection = (cam_binary.astype(bool) & seg_mask.astype(bool)).sum()
    return float(intersection) / float(seg_sum)

def compute_cc(cam_binary: np.ndarray, seg_mask: np.ndarray) -> float:
    """CAM Containment = |CAM ∩ SEG| / |CAM|"""
    if seg_mask.shape != cam_binary.shape:
        cam_binary = cv2.resize(cam_binary.astype(np.uint8), (seg_mask.shape[1], seg_mask.shape[0]), interpolation=cv2.INTER_NEAREST)

    cam_sum = cam_binary.sum()
    if cam_sum == 0:
        return 0.0
        
    intersection = (cam_binary.astype(bool) & seg_mask.astype(bool)).sum()
    return float(intersection) / float(cam_sum)

def compute_wcis(cam_raw: np.ndarray, seg_mask: np.ndarray) -> float:
    """Weighted CAM Intensity in Segmentation = mean(cam_raw[seg_mask == 1])"""
    if seg_mask.shape != cam_raw.shape:
        cam_raw = cv2.resize(cam_raw, (seg_mask.shape[1], seg_mask.shape[0]), interpolation=cv2.INTER_LINEAR)
        
    mask_pixels = cam_raw[seg_mask.astype(bool)]
    if len(mask_pixels) == 0:
        return 0.0
        
    return float(np.mean(mask_pixels))

def compute_exbale(sc: float, cc: float, wcis: float, wcis_global_min: float = 0.0, wcis_global_max: float = 1.0) -> float:
    """ExBale (Explainability-Based Alignment Evaluation)"""
    if np.isnan(sc):
        return np.nan
        
    wcis_norm = (wcis - wcis_global_min) / (wcis_global_max - wcis_global_min + 1e-8)
    wcis_norm = np.clip(wcis_norm, 0.0, 1.0)
    
    sc = np.clip(sc, 0.0, 1.0)
    cc = np.clip(cc, 0.0, 1.0)
    
    return float((sc * cc * wcis_norm) ** (1.0 / 3.0))

def compute_all_metrics(cam_raw: np.ndarray, seg_mask: np.ndarray, thresholds: list[float], wcis_global_min: float, wcis_global_max: float) -> list[dict]:
    results = []
    
    wcis = compute_wcis(cam_raw, seg_mask)
    
    for t in thresholds:
        cam_bin = binarize_cam(cam_raw, t)
        sc = compute_sc(cam_bin, seg_mask)
        cc = compute_cc(cam_bin, seg_mask)
        exbale = compute_exbale(sc, cc, wcis, wcis_global_min, wcis_global_max)
        
        results.append({
            "threshold": t,
            "sc": sc,
            "cc": cc,
            "wcis": wcis,
            "exbale": exbale
        })
        
    return results
