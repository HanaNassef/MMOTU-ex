import torch
import numpy as np
import captum.attr as captum_attr
import logging

class GradientExplainer:
    def __init__(self, model, device: torch.device):
        self.model = model
        self.device = device

    def compute_saliency(self, image_tensor: torch.Tensor, class_idx: int) -> np.ndarray:
        """Vanilla gradient saliency. Returns (224,224) map, values in [0,1]."""
        image_tensor.requires_grad = True
        self.model.zero_grad()
        
        output = self.model(image_tensor)
        score = output[0, class_idx]
        score.backward()
        
        saliency = image_tensor.grad.abs().max(dim=1)[0].squeeze().cpu().numpy()
        
        sal_min, sal_max = saliency.min(), saliency.max()
        if sal_max > sal_min:
            saliency = (saliency - sal_min) / (sal_max - sal_min)
        else:
            saliency = np.zeros_like(saliency)
            
        return saliency

    def compute_integrated_gradients(self, image_tensor: torch.Tensor, class_idx: int, n_steps: int = 50) -> np.ndarray:
        """Integrated Gradients with zero baseline. Returns (224,224) map, values in [0,1]."""
        baseline = torch.zeros_like(image_tensor)
        ig = captum_attr.IntegratedGradients(self.model)
        
        try:
            attr, delta = ig.attribute(image_tensor, baseline, target=class_idx, return_convergence_delta=True, n_steps=n_steps)
            attr = attr.squeeze().cpu().numpy()
            
            # Aggregate across channels (mean abs)
            attr = np.mean(np.abs(attr), axis=0)
            
            # Normalize
            attr_min, attr_max = attr.min(), attr.max()
            if attr_max > attr_min:
                attr = (attr - attr_min) / (attr_max - attr_min)
            else:
                attr = np.zeros_like(attr)
            return attr
            
        except Exception as e:
            logging.getLogger(__name__).error(f"Integrated Gradients failed: {e}")
            return np.zeros((image_tensor.shape[2], image_tensor.shape[3]), dtype=np.float32)

    def compute_all_gradients(self, image_tensor: torch.Tensor, class_idx: int) -> dict:
        return {
            "saliency": self.compute_saliency(image_tensor.clone(), class_idx),
            "integrated_gradients": self.compute_integrated_gradients(image_tensor, class_idx)
        }
