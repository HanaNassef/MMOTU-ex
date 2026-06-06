import torch
import torch.nn as nn
import torchvision.models as tv_models
from models.heads import ClassificationHead
from typing import Tuple

def get_model(model_name: str, num_classes: int = 8, pretrained: bool = True, dropout: float = 0.5) -> Tuple[nn.Module, int]:
    """
    Returns the requested model architecture with a custom classification head,
    and the number of input features to that head.
    """
    model_name = model_name.lower()
    
    if model_name == "densenet121":
        weights = tv_models.DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None
        model = tv_models.densenet121(weights=weights)
        in_features = model.classifier.in_features
        model.classifier = ClassificationHead(in_features, num_classes, dropout)
        
    elif model_name == "resnet50":
        weights = tv_models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        model = tv_models.resnet50(weights=weights)
        in_features = model.fc.in_features
        model.fc = ClassificationHead(in_features, num_classes, dropout)
        
    elif model_name == "resnet101":
        weights = tv_models.ResNet101_Weights.IMAGENET1K_V2 if pretrained else None
        model = tv_models.resnet101(weights=weights)
        in_features = model.fc.in_features
        model.fc = ClassificationHead(in_features, num_classes, dropout)
        
    elif model_name == "efficientnet_b3":
        weights = tv_models.EfficientNet_B3_Weights.IMAGENET1K_V1 if pretrained else None
        model = tv_models.efficientnet_b3(weights=weights)
        in_features = model.classifier[1].in_features
        model.classifier = ClassificationHead(in_features, num_classes, dropout)
        
    elif model_name == "mobilenet_v3_large":
        weights = tv_models.MobileNet_V3_Large_Weights.IMAGENET1K_V2 if pretrained else None
        model = tv_models.mobilenet_v3_large(weights=weights)
        in_features = model.classifier[0].weight.shape[1] # input to first linear
        # MobileNetV3 classifier is a bit complex, let's just replace it entirely using the pooled features
        # The pooling output is 960
        in_features = 960
        model.classifier = ClassificationHead(in_features, num_classes, dropout)
        
    elif model_name == "swin_t":
        weights = tv_models.Swin_T_Weights.IMAGENET1K_V1 if pretrained else None
        model = tv_models.swin_t(weights=weights)
        in_features = model.head.in_features
        model.head = ClassificationHead(in_features, num_classes, dropout)
        
    elif model_name == "vit_b_16":
        weights = tv_models.ViT_B_16_Weights.IMAGENET1K_V1 if pretrained else None
        model = tv_models.vit_b_16(weights=weights)
        in_features = model.heads.head.in_features
        model.heads = ClassificationHead(in_features, num_classes, dropout)
        
    else:
        raise ValueError(f"Unknown model name: {model_name}")
        
    return model, in_features
