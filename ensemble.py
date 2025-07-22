import torch
import torch.nn as nn
import torchvision.models as models

class PatchAttentionEnsemble(nn.Module):
    def __init__(self, output_dim=2):
        super().__init__()
        self.model1 = models.resnet50(pretrained=True)
        self.model2 = models.densenet121(pretrained=True)
        self.model3 = models.efficientnet_b0(pretrained=True)
        self.model1.fc = nn.Identity()
        self.model2.classifier = nn.Identity()
        self.model3.classifier = nn.Identity()
        self.fc_head1 = nn.Linear(2048, 512)
        self.fc_head2 = nn.Linear(1024, 512)
        self.fc_head3 = nn.Linear(1280, 512)
        self.attention = nn.Sequential(
            nn.Linear(512 * 3, 256), nn.ReLU(), nn.Linear(256, 3), nn.Softmax(dim=1))
        self.classifier = nn.Linear(512, output_dim)

    def forward(self, x):
        B, P, C, H, W = x.shape
        x = x.view(-1, C, H, W)
        feat1 = self.fc_head1(self.model1(x))
        feat2 = self.fc_head2(self.model2(x))
        feat3 = self.fc_head3(self.model3(x))
        feats = torch.stack([feat1, feat2, feat3], dim=1).view(B, P, 3, 512)
        attn_input = feats.view(B * P, -1)
        attn_weights = self.attention(attn_input).view(B, P, 3, 1)
        fused = (feats * attn_weights).sum(dim=(1, 2))
        return self.classifier(fused)
