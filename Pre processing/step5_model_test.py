import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    """Residual block with two conv layers."""
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        return F.relu(x + residual)


class OpticalEncoder(nn.Module):
    """Extracts spatial features (edges, textures) from HR optical."""
    def __init__(self, in_channels=3, base_channels=64):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, base_channels, 3, padding=1)
        self.res1 = ResidualBlock(base_channels)
        self.res2 = ResidualBlock(base_channels)
        self.res3 = ResidualBlock(base_channels)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.res1(x)
        x = self.res2(x)
        x = self.res3(x)
        return x  # (B, 64, 256, 256)


class ThermalEncoder(nn.Module):
    """Extracts thermal features from LR thermal (after upscaling)."""
    def __init__(self, in_channels=1, base_channels=64):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, base_channels, 3, padding=1)
        self.res1 = ResidualBlock(base_channels)
        self.res2 = ResidualBlock(base_channels)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.res1(x)
        x = self.res2(x)
        return x  # (B, 64, 256, 256)


class FusionModule(nn.Module):
    """
    Attention-based fusion of optical and thermal features.
    Optical features guide WHERE to sharpen.
    Thermal features provide WHAT temperature values.
    """
    def __init__(self, channels=64):
        super().__init__()
        # Attention: learn which optical features are useful for thermal
        self.attention = nn.Sequential(
            nn.Conv2d(channels * 2, channels, 1),
            nn.ReLU(),
            nn.Conv2d(channels, channels, 1),
            nn.Sigmoid()
        )
        self.fuse_conv = nn.Conv2d(channels * 2, channels, 3, padding=1)
        self.bn = nn.BatchNorm2d(channels)

    def forward(self, optical_feat, thermal_feat):
        combined = torch.cat([optical_feat, thermal_feat], dim=1)  # (B, 128, H, W)
        attention_map = self.attention(combined)  # (B, 64, H, W)

        # Weight optical features by attention (only useful edges/textures pass)
        guided_optical = optical_feat * attention_map
        fused = torch.cat([guided_optical, thermal_feat], dim=1)  # (B, 128, H, W)
        fused = F.relu(self.bn(self.fuse_conv(fused)))  # (B, 64, H, W)
        return fused


class ReconstructionDecoder(nn.Module):
    """Reconstructs HR thermal from fused features."""
    def __init__(self, in_channels=64):
        super().__init__()
        self.res1 = ResidualBlock(in_channels)
        self.res2 = ResidualBlock(in_channels)
        self.res3 = ResidualBlock(in_channels)
        self.res4 = ResidualBlock(in_channels)
        self.output_conv = nn.Conv2d(in_channels, 1, 3, padding=1)

    def forward(self, x):
        x = self.res1(x)
        x = self.res2(x)
        x = self.res3(x)
        x = self.res4(x)
        x = self.output_conv(x)  # (B, 1, 256, 256)
        return x


class OpticalGuidedThermalSR(nn.Module):
    """
    Full model: Optical-Guided Super-Resolution for Thermal IR.

    Input:
        optical_hr: (B, 3, 256, 256)  - HR optical image
        thermal_lr: (B, 1, 64, 64)    - LR thermal image

    Output:
        thermal_sr: (B, 1, 256, 256)  - Super-resolved thermal image
    """
    def __init__(self, scale_factor=4, base_channels=64):
        super().__init__()
        self.scale_factor = scale_factor

        self.optical_encoder = OpticalEncoder(in_channels=3, base_channels=base_channels)
        self.thermal_encoder = ThermalEncoder(in_channels=1, base_channels=base_channels)
        self.fusion = FusionModule(channels=base_channels)
        self.decoder = ReconstructionDecoder(in_channels=base_channels)

    def forward(self, optical_hr, thermal_lr):
        # Step 1: Upscale LR thermal to HR size using bicubic
        thermal_up = F.interpolate(thermal_lr, scale_factor=self.scale_factor,
                                   mode='bicubic', align_corners=False)  # (B, 1, 256, 256)

        # Step 2: Extract features
        optical_feat = self.optical_encoder(optical_hr)     # (B, 64, 256, 256)
        thermal_feat = self.thermal_encoder(thermal_up)      # (B, 64, 256, 256)

        # Step 3: Fuse optical guidance with thermal
        fused = self.fusion(optical_feat, thermal_feat)      # (B, 64, 256, 256)

        # Step 4: Reconstruct HR thermal
        residual = self.decoder(fused)                        # (B, 1, 256, 256)

        # Step 5: Residual learning (predict difference from bicubic upscale)
        thermal_sr = thermal_up + residual

        return thermal_sr


# Quick test
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = OpticalGuidedThermalSR(scale_factor=4).to(device)
    optical = torch.randn(2, 3, 256, 256, device=device)
    thermal_lr = torch.randn(2, 1, 64, 64, device=device)

    output = model(optical, thermal_lr)
    print(f"Input  - Optical: {optical.shape}, Thermal LR: {thermal_lr.shape}")
    print(f"Output - Thermal SR: {output.shape}")

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
