"""
Ensemble classifier combining TinyViT, ResNet50, VGG19, GoogLeNet (Inception v1),
and EfficientNet — trained on crops produced by the YOLOv8 detection stage.

Two ensembling strategies are provided:
  1. Soft-voting (weighted average of softmax probabilities) — no extra training needed.
  2. Learned stacking head (small MLP on concatenated logits) — usually a bit stronger,
     needs a short extra training pass on held-out predictions.

Requires: torch, torchvision, timm (for TinyViT)  ->  pip install timm
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tv_models

try:
    import timm
except ImportError:
    timm = None


# ---------------------------- Individual model builders ----------------------------

def build_resnet50(num_classes, pretrained=True):
    model = tv_models.resnet50(weights=tv_models.ResNet50_Weights.DEFAULT if pretrained else None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def build_vgg19(num_classes, pretrained=True):
    model = tv_models.vgg19_bn(weights=tv_models.VGG19_BN_Weights.DEFAULT if pretrained else None)
    model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes)
    return model


def build_googlenet(num_classes, pretrained=True):
    model = tv_models.googlenet(weights=tv_models.GoogLeNet_Weights.DEFAULT if pretrained else None,
                                 aux_logits=True, init_weights=not pretrained)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    # Replace aux classifiers too (used only during training, ignored in eval)
    if model.aux1 is not None:
        model.aux1.fc2 = nn.Linear(model.aux1.fc2.in_features, num_classes)
    if model.aux2 is not None:
        model.aux2.fc2 = nn.Linear(model.aux2.fc2.in_features, num_classes)
    return model


def build_efficientnet(num_classes, pretrained=True):
    model = tv_models.efficientnet_b0(
        weights=tv_models.EfficientNet_B0_Weights.DEFAULT if pretrained else None)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    return model


def build_tinyvit(num_classes, pretrained=True):
    if timm is None:
        raise ImportError("timm is required for TinyViT: pip install timm")
    model = timm.create_model("tiny_vit_21m_224", pretrained=pretrained, num_classes=num_classes)
    return model


MODEL_BUILDERS = {
    "tinyvit": build_tinyvit,
    "resnet50": build_resnet50,
    "vgg19": build_vgg19,
    "googlenet": build_googlenet,
    "efficientnet": build_efficientnet,
}

# Each backbone's expected input size (TinyViT/GoogLeNet commonly use 224; keep uniform at 224
# for simplicity — all backbones below accept 224x224 without modification)
INPUT_SIZE = 224


# ---------------------------- Ensemble wrapper ----------------------------

class EnsembleClassifier(nn.Module):
    """
    Wraps all 5 backbones. forward() returns per-model logits (dict) so you can
    inspect individual model performance as well as the ensembled prediction.
    """
    def __init__(self, num_classes, model_names=None, pretrained=True):
        super().__init__()
        model_names = model_names or list(MODEL_BUILDERS.keys())
        self.model_names = model_names
        self.models = nn.ModuleDict({
            name: MODEL_BUILDERS[name](num_classes, pretrained=pretrained)
            for name in model_names
        })
        # Learnable per-model weights for soft voting (initialized equal, softmax-normalized)
        self.vote_weights = nn.Parameter(torch.ones(len(model_names)))

    def forward(self, x, return_individual=False):
        logits = {}
        for name, model in self.models.items():
            out = model(x)
            # googlenet returns a GoogLeNetOutputs namedtuple during training (has aux logits)
            if isinstance(out, tuple):
                out = out[0]
            elif hasattr(out, "logits"):
                out = out.logits
            logits[name] = out

        weights = F.softmax(self.vote_weights, dim=0)
        probs_stack = torch.stack(
            [F.softmax(logits[name], dim=1) * weights[i] for i, name in enumerate(self.model_names)],
            dim=0,
        )
        ensembled_probs = probs_stack.sum(dim=0)  # (B, num_classes), weighted soft-vote

        if return_individual:
            return ensembled_probs, logits
        return ensembled_probs


class StackingHead(nn.Module):
    """
    Optional stronger ensembler: a small MLP that learns how to combine the
    concatenated logits from all 5 models, instead of a simple weighted average.
    Train this AFTER the backbones are trained/frozen, on validation-set logits.
    """
    def __init__(self, num_classes, num_models=5, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(num_classes * num_models, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, concatenated_logits):
        return self.net(concatenated_logits)


def get_transforms(input_size=INPUT_SIZE, train=True):
    from torchvision import transforms
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]

    if train:
        return transforms.Compose([
            transforms.Resize((input_size, input_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(imagenet_mean, imagenet_std),
        ])
    return transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize(imagenet_mean, imagenet_std),
    ])


if __name__ == "__main__":
    model = EnsembleClassifier(num_classes=10, pretrained=False)
    dummy = torch.randn(2, 3, 224, 224)
    probs = model(dummy)
    print("Ensembled probs shape:", probs.shape)
    print("Sums to 1 per row:", probs.sum(dim=1))
