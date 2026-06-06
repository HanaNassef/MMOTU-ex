import torch
import torch.nn.functional as F
import numpy as np
from torchvision.transforms import GaussianBlur
from sklearn.metrics import auc

def compute_insertion_deletion_auc(model, image_tensor: torch.Tensor, cam_raw: np.ndarray, class_idx: int, device: torch.device, n_steps: int = 50) -> dict:
    """
    Insertion/Deletion AUC.
    """
    model.eval()
    
    # 1. Base preparations
    C, H, W = image_tensor.shape[1], image_tensor.shape[2], image_tensor.shape[3]
    pixel_count = H * W
    step_size = pixel_count // n_steps
    
    # Baseline image for insertion
    blur = GaussianBlur(kernel_size=51, sigma=10.0)
    baseline_img = blur(image_tensor)
    
    # Flatten structures for sorting
    cam_flat = cam_raw.flatten()
    sorted_indices = np.argsort(cam_flat)[::-1] # descending importance
    
    # Storage for curves
    insertion_scores = []
    deletion_scores = []
    
    # Start states
    # Insertion starts with blurred image
    insert_img = baseline_img.clone()
    # Deletion starts with original image
    delete_img = image_tensor.clone()
    
    with torch.no_grad():
        for i in range(n_steps + 1):
            pixels_to_reveal = i * step_size
            if i == n_steps:
                pixels_to_reveal = pixel_count
                
            current_indices = sorted_indices[:pixels_to_reveal]
            
            if pixels_to_reveal > 0:
                y_coords = current_indices // W
                x_coords = current_indices % W
                
                # Insertion: reveal original pixels
                insert_img[0, :, y_coords, x_coords] = image_tensor[0, :, y_coords, x_coords]
                
                # Deletion: remove (blur) original pixels
                delete_img[0, :, y_coords, x_coords] = baseline_img[0, :, y_coords, x_coords]
            
            # Evaluate Insertion
            out_ins = model(insert_img)
            prob_ins = F.softmax(out_ins, dim=1)[0, class_idx].item()
            insertion_scores.append(prob_ins)
            
            # Evaluate Deletion
            out_del = model(delete_img)
            prob_del = F.softmax(out_del, dim=1)[0, class_idx].item()
            deletion_scores.append(prob_del)
            
    # Calculate AUC (x values from 0.0 to 1.0)
    x_axis = np.linspace(0.0, 1.0, len(insertion_scores))
    ins_auc = auc(x_axis, insertion_scores)
    del_auc = auc(x_axis, deletion_scores)
    
    return {
        "insertion_auc": ins_auc,
        "deletion_auc": del_auc
    }
