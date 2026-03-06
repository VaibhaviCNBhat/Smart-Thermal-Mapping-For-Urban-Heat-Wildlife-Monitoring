import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
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
        return x

class ThermalEncoder(nn.Module):
    def __init__(self, in_channels=1, base_channels=64):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, base_channels, 3, padding=1)
        self.res1 = ResidualBlock(base_channels)
        self.res2 = ResidualBlock(base_channels)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.res1(x)
        x = self.res2(x)
        return x

class FusionModule(nn.Module):
    def __init__(self, channels=64):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Conv2d(channels * 2, channels, 1),
            nn.ReLU(),
            nn.Conv2d(channels, channels, 1),
            nn.Sigmoid()
        )
        self.fuse_conv = nn.Conv2d(channels * 2, channels, 3, padding=1)
        self.bn = nn.BatchNorm2d(channels)

    def forward(self, optical_feat, thermal_feat):
        combined = torch.cat([optical_feat, thermal_feat], dim=1)
        attention_map = self.attention(combined)
        guided_optical = optical_feat * attention_map
        fused = torch.cat([guided_optical, thermal_feat], dim=1)
        fused = F.relu(self.bn(self.fuse_conv(fused)))
        return fused

class ReconstructionDecoder(nn.Module):
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
        x = self.output_conv(x)
        return x

class OpticalGuidedThermalSR(nn.Module):
    def __init__(self, scale_factor=4, base_channels=64):
        super().__init__()
        self.scale_factor = scale_factor
        self.optical_encoder = OpticalEncoder(in_channels=3, base_channels=base_channels)
        self.thermal_encoder = ThermalEncoder(in_channels=1, base_channels=base_channels)
        self.fusion = FusionModule(channels=base_channels)
        self.decoder = ReconstructionDecoder(in_channels=base_channels)

    def forward(self, optical_hr, thermal_lr):
        thermal_up = F.interpolate(thermal_lr, scale_factor=self.scale_factor, mode='bicubic', align_corners=False)
        optical_feat = self.optical_encoder(optical_hr)
        thermal_feat = self.thermal_encoder(thermal_up)
        fused = self.fusion(optical_feat, thermal_feat)
        residual = self.decoder(fused)
        thermal_sr = thermal_up + residual
        return thermal_sr
