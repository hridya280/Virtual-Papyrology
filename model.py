import torch
import torch.nn as nn


class SimpleUNet3D(nn.Module):
    def __init__(self):
        super().__init__()

        def block(i, o):
            return nn.Sequential(
                nn.Conv3d(i, o, 3, padding=1),
                nn.BatchNorm3d(o),
                nn.ReLU(inplace=True),
                nn.Conv3d(o, o, 3, padding=1),
                nn.BatchNorm3d(o),
                nn.ReLU(inplace=True),
            )

        self.enc1 = block(1, 16)
        self.enc2 = block(16, 32)
        self.enc3 = block(32, 64)
        self.pool = nn.MaxPool3d(2)
        self.bottleneck = block(64, 128)
        self.up3 = nn.ConvTranspose3d(128, 64, 2, 2)
        self.dec3 = block(128, 64)
        self.up2 = nn.ConvTranspose3d(64, 32, 2, 2)
        self.dec2 = block(64, 32)
        self.up1 = nn.ConvTranspose3d(32, 16, 2, 2)
        self.dec1 = block(32, 16)
        self.out = nn.Conv3d(16, 1, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        b = self.bottleneck(self.pool(e3))
        d3 = self.dec3(torch.cat([self.up3(b), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return self.out(d1)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())


def get_architecture_summary():
    """Return a structured summary of the architecture."""
    return {
        "Encoder": [
            {"stage": "Enc1", "in_ch": 1, "out_ch": 16, "ops": "Conv3d→BN→ReLU→Conv3d→BN→ReLU"},
            {"stage": "Pool", "in_ch": 16, "out_ch": 16, "ops": "MaxPool3d(2)"},
            {"stage": "Enc2", "in_ch": 16, "out_ch": 32, "ops": "Conv3d→BN→ReLU→Conv3d→BN→ReLU"},
            {"stage": "Pool", "in_ch": 32, "out_ch": 32, "ops": "MaxPool3d(2)"},
            {"stage": "Enc3", "in_ch": 32, "out_ch": 64, "ops": "Conv3d→BN→ReLU→Conv3d→BN→ReLU"},
            {"stage": "Pool", "in_ch": 64, "out_ch": 64, "ops": "MaxPool3d(2)"},
        ],
        "Bottleneck": [
            {"stage": "Bottleneck", "in_ch": 64, "out_ch": 128, "ops": "Conv3d→BN→ReLU→Conv3d→BN→ReLU"},
        ],
        "Decoder": [
            {"stage": "Up3", "in_ch": 128, "out_ch": 64, "ops": "ConvTranspose3d(2,2) + Skip(Enc3)"},
            {"stage": "Dec3", "in_ch": 128, "out_ch": 64, "ops": "Conv3d→BN→ReLU→Conv3d→BN→ReLU"},
            {"stage": "Up2", "in_ch": 64, "out_ch": 32, "ops": "ConvTranspose3d(2,2) + Skip(Enc2)"},
            {"stage": "Dec2", "in_ch": 64, "out_ch": 32, "ops": "Conv3d→BN→ReLU→Conv3d→BN→ReLU"},
            {"stage": "Up1", "in_ch": 32, "out_ch": 16, "ops": "ConvTranspose3d(2,2) + Skip(Enc1)"},
            {"stage": "Dec1", "in_ch": 32, "out_ch": 16, "ops": "Conv3d→BN→ReLU→Conv3d→BN→ReLU"},
        ],
        "Output": [
            {"stage": "Out", "in_ch": 16, "out_ch": 1, "ops": "Conv3d(1×1×1)"},
        ],
    }
