import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def enable_mc_dropout(model: nn.Module) -> None:
    for module in model.modules():
        if isinstance(module, nn.Dropout):
            module.train()


class MCDropoutEstimator:
    def __init__(self, model: nn.Module, device, n_samples: int = 30):
        self.model = model.to(device).eval()
        self.device = device
        self.n_samples = n_samples
        enable_mc_dropout(self.model)

    @torch.no_grad()
    def predict(self, images: torch.Tensor):
        images = images.to(self.device)
        sample_probs = []
        for _ in range(self.n_samples):
            logits = self.model(images)
            sample_probs.append(F.softmax(logits, dim=1))

        all_probs = torch.stack(sample_probs)
        mean_probs = all_probs.mean(dim=0)
        epistemic_std = all_probs.std(dim=0)
        predictive_entropy = -(mean_probs * torch.log(mean_probs.clamp_min(1e-12))).sum(dim=1)

        return {
            "mean_probs": mean_probs.cpu().numpy(),
            "epistemic_std": epistemic_std.cpu().numpy(),
            "predictive_entropy": predictive_entropy.cpu().numpy(),
        }


class DeepEnsembleEstimator:
    def __init__(self, ensemble, device):
        self.ensemble = ensemble
        self.device = device

    @torch.no_grad()
    def predict(self, image_tensor: torch.Tensor):
        if image_tensor.dim() == 3:
            image_tensor = image_tensor.unsqueeze(0)
        image_tensor = image_tensor.to(self.device)

        member_probs = []
        for model in self.ensemble.models:
            logits = model(image_tensor)
            member_probs.append(F.softmax(logits, dim=1))

        stacked = torch.stack(member_probs)
        mean_probs = stacked.mean(dim=0)
        epistemic_std = stacked.std(dim=0)
        predictive_entropy = -(mean_probs * torch.log(mean_probs.clamp_min(1e-12))).sum(dim=1)

        return {
            "mean_probs": mean_probs.cpu().numpy().squeeze(0),
            "epistemic_std": epistemic_std.cpu().numpy().squeeze(0),
            "predictive_entropy": predictive_entropy.cpu().numpy().squeeze(0),
        }
