import numpy as np
import cv2
import matplotlib.pyplot as plt

def overlay_cam_on_image(?
    """
    Overlays CAM heatmap (jet) and Segmentation mask contour on the image.
    """
    # Normalize cam to [0, 255] uint8
    c_min, c_max = cam_np.min(), cam_np.max()
    if c_max > c_min:
        cam_uint8 = (255 * (cam_np - c_min) / (c_max - c_min)).astype(np.uint8)
    else:
        cam_uint8 = np.zeros_like(cam_np, dtype=np.uint8)
        
    heatmap = cv2.applyColorMap(cam_uint8, cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    
    # Overlay heatmap
    if original_img_np.shape[:2] != heatmap.shape[:2]:
        heatmap = cv2.resize(heatmap, (original_img_np.shape[1], original_img_np.shape[0]))
        
    overlay = cv2.addWeighted(original_img_np, 1.0 - alpha, heatmap, alpha, 0)
    
    # Draw contour for seg_mask
    if seg_mask_np is not None and seg_mask_np.max() > 0:
        if seg_mask_np.shape != overlay.shape[:2]:
            seg_mask_np = cv2.resize(seg_mask_np.astype(np.uint8), (overlay.shape[1], overlay.shape[0]), interpolation=cv2.INTER_NEAREST)
            
        contours, _ = cv2.findContours((seg_mask_np > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, contours, -1, (0, 255, 0), 2)
        
    return overlay

def save_comparison_figure(image_path: str, seg_mask: np.ndarray, cam_results_dict: dict, metrics_dict: dict, save_path: str):
    """
    Grid layout: rows=XAI methods, columns=[Original, Seg Mask, CAM Overlay, Binarized CAM]
    """
    import cv2
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    methods = list(cam_results_dict.keys())
    n_methods = len(methods)
    
    fig, axes = plt.subplots(nrows=max(1, n_methods), ncols=4, figsize=(16, 4 * max(1, n_methods)))
    
    if n_methods == 1:
        axes = [axes]
        
    for i, method in enumerate(methods):
        cam = cam_results_dict[method]
        metrics = metrics_dict.get(method, {})
        
        overlay = overlay_cam_on_image(img, cam, seg_mask)
        bin_cam = (cam > 0.5).astype(np.uint8) * 255 # Simple threshold for viz
        
        ax_orig = axes[i][0]
        ax_orig.imshow(img)
        ax_orig.set_title(f"Original ({method})")
        ax_orig.axis('off')
        
        ax_seg = axes[i][1]
        ax_seg.imshow(seg_mask, cmap='gray')
        ax_seg.set_title("Seg Mask")
        ax_seg.axis('off')
        
        ax_over = axes[i][2]
        ax_over.imshow(overlay)
        ax_over.set_title("CAM Overlay")
        ax_over.axis('off')
        
        ax_bin = axes[i][3]
        ax_bin.imshow(bin_cam, cmap='gray')
        m_str = f"SC:{metrics.get('sc',0):.2f} CC:{metrics.get('cc',0):.2f}\nWCIS:{metrics.get('wcis',0):.2f} Ex:{metrics.get('exbale',0):.2f}"
        ax_bin.set_title(f"Binarized\n{m_str}")
        ax_bin.axis('off')
        
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close(fig)
