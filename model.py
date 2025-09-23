# model.py
import torch
import torch.nn as nn
import torchvision.models as models

class MultiTaskEffNetB0(nn.Module):
    def __init__(self, num_emotions=7):
        super().__init__()
        base = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        in_features = base.classifier[1].in_features
        base.classifier = nn.Identity()
        self.backbone = base

        # SE-like gating on pooled features
        self.se_fc = nn.Sequential(
            nn.Linear(in_features, in_features // 16),
            nn.ReLU(),
            nn.Linear(in_features // 16, in_features),
            nn.Sigmoid()
        )

        # task-specific heads
        self.fc_emotion = nn.Linear(in_features, num_emotions)
        self.fc_auth = nn.Linear(in_features, 1)

        # uncertainty weighting parameters (MUST match your trained checkpoint)
        self.log_var_e = nn.Parameter(torch.zeros(1))
        self.log_var_a = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        feats = self.backbone(x)  # [B, in_features]
        gate = self.se_fc(feats)
        feats = feats * gate

        emo_logits = self.fc_emotion(feats)
        auth_logits = self.fc_auth(feats)
        return emo_logits, auth_logits
