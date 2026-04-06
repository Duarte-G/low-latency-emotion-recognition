from __future__ import annotations

import warnings

import torch
import torch.nn as nn
from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0


class EmotionClassifier(nn.Module):
    def __init__(
        self,
        num_classes: int = 4,
        dropout: float = 0.3,
        pretrained: bool = True,
        use_landmarks: bool = False,
        landmark_dim: int = 936,
        landmark_hidden_dim: int = 128,
        fusion_hidden_dim: int = 256,
    ) -> None:
        super().__init__()
        self.use_landmarks = use_landmarks

        weights = None
        if pretrained:
            try:
                weights = EfficientNet_B0_Weights.DEFAULT
            except Exception as exc:
                warnings.warn(f"Nao foi possivel configurar pesos ImageNet: {exc}. Seguindo sem pretreino.")

        try:
            self.backbone = efficientnet_b0(weights=weights)
        except Exception as exc:
            warnings.warn(f"Falha ao carregar EfficientNet com pesos pretreinados: {exc}. Usando pesos aleatorios.")
            self.backbone = efficientnet_b0(weights=None)

        in_features = self.backbone.classifier[1].in_features
        if self.use_landmarks:
            self.backbone.classifier = nn.Identity()
            self.landmark_branch = nn.Sequential(
                nn.Linear(landmark_dim, landmark_hidden_dim),
                nn.ReLU(inplace=False),
                nn.Dropout(p=dropout),
                nn.Linear(landmark_hidden_dim, landmark_hidden_dim),
                nn.ReLU(inplace=False),
            )
            self.classifier = nn.Sequential(
                nn.Dropout(p=dropout, inplace=False),
                nn.Linear(in_features + landmark_hidden_dim, fusion_hidden_dim),
                nn.ReLU(inplace=False),
                nn.Dropout(p=dropout, inplace=False),
                nn.Linear(fusion_hidden_dim, num_classes),
            )
        else:
            self.backbone.classifier = nn.Sequential(
                nn.Dropout(p=dropout, inplace=False),
                nn.Linear(in_features, num_classes),
            )
            self.landmark_branch = None
            self.classifier = None

    def forward(self, x: torch.Tensor, landmarks: torch.Tensor | None = None) -> torch.Tensor:
        if not self.use_landmarks:
            return self.backbone(x)

        image_features = self.backbone(x)
        if landmarks is None:
            landmarks = torch.zeros(
                (x.shape[0], self.landmark_branch[0].in_features),
                device=x.device,
                dtype=image_features.dtype,
            )
        landmark_features = self.landmark_branch(landmarks)
        fused = torch.cat([image_features, landmark_features], dim=1)
        return self.classifier(fused)


def resolve_device(requested: str = "auto") -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
