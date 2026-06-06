import torch
import numpy as np
import shap
import logging

class SHAPExplainer:
    def __init__(self, model, background_dataset: torch.utils.data.Dataset, device: torch.device, n_background: int = 100):
        self.model = model
        self.device = device
        
        # Sample background images
        n_background = min(n_background, len(background_dataset))
        indices = np.random.choice(len(background_dataset), n_background, replace=False)
        
        background_tensors = []
        for i in indices:
            img, _, _ = background_dataset[i]
            background_tensors.append(img)
            
        background = torch.stack(background_tensors).to(self.device)
        self.explainer = shap.DeepExplainer(model, background)

    def compute_shap(self, image_tensor: torch.Tensor, class_idx: int) -> np.ndarray:
        """Returns (224, 224) SHAP value map, absolute values, normalized to [0,1]."""
        try:
            # shap_values expects batch, returns list of len(classes) of shape batch x C x H x W
            shap_values = self.explainer.shap_values(image_tensor)
            
            # shap_values could be a single tensor depending on shap version or model output
            if isinstance(shap_values, list):
                if class_idx < len(shap_values):
                    class_shap = shap_values[class_idx][0]
                else:
                    class_shap = shap_values[0][0]
            else:
                class_shap = shap_values[0] # assume batch 1

            # Mean abs across channels
            class_shap = np.mean(np.abs(class_shap), axis=0)
            
            shap_min, shap_max = class_shap.min(), class_shap.max()
            if shap_max > shap_min:
                class_shap = (class_shap - shap_min) / (shap_max - shap_min)
            else:
                class_shap = np.zeros_like(class_shap)
                
            return class_shap
            
        except Exception as e:
            logging.getLogger(__name__).error(f"SHAP computation failed: {e}")
            return np.zeros((image_tensor.shape[2], image_tensor.shape[3]), dtype=np.float32)
