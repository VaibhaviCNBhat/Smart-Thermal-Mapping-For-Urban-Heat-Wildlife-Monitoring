import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from scipy.ndimage import zoom


class ThermalSRDataset(Dataset):
    """
    Dataset for Optical-Guided Thermal Super-Resolution.
    Returns:
        optical_hr:  (3, 256, 256) float32 normalized
        thermal_lr:  (1, 64, 64)   float32 normalized
        thermal_hr:  (1, 256, 256) float32 normalized (ground truth)
    """

    def __init__(self, dataset_root, split="train", scale_factor=4):
        self.dataset_root = dataset_root
        self.scale_factor = scale_factor

        # Load scene list
        split_file = os.path.join(dataset_root, f"{split}.txt")
        with open(split_file, "r") as f:
            self.scenes = [line.strip() for line in f if line.strip()]

        # Load normalization stats
        stats = np.load(os.path.join(dataset_root, "dataset_stats.npy"), allow_pickle=True).item()
        self.optical_mean = stats["optical_mean"].astype(np.float32)
        self.optical_std = stats["optical_std"].astype(np.float32)
        self.thermal_mean = float(stats["thermal_mean"])
        self.thermal_std = float(stats["thermal_std"])

        print(f"[{split}] Loaded {len(self.scenes)} scenes")

    def __len__(self):
        return len(self.scenes)

    def normalize_optical(self, optical):
        """Normalize optical: (H, W, 3) → (3, H, W) normalized."""
        optical = (optical - self.optical_mean) / (self.optical_std + 1e-8)
        optical = np.transpose(optical, (2, 0, 1))  # (3, H, W)
        return optical.astype(np.float32)

    def normalize_thermal(self, thermal):
        """Normalize thermal to zero mean, unit std."""
        thermal = (thermal - self.thermal_mean) / (self.thermal_std + 1e-8)
        return thermal.astype(np.float32)

    def __getitem__(self, idx):
        scene_dir = os.path.join(self.dataset_root, self.scenes[idx])

        optical_hr = np.load(os.path.join(scene_dir, "optical_hr.npy"))   # (256, 256, 3)
        thermal_hr = np.load(os.path.join(scene_dir, "thermal_hr.npy"))   # (256, 256)
        thermal_lr = np.load(os.path.join(scene_dir, "thermal_lr.npy"))   # (64, 64)

        # Normalize
        optical_hr = self.normalize_optical(optical_hr)       # (3, 256, 256)
        thermal_hr = self.normalize_thermal(thermal_hr)       # (256, 256)
        thermal_lr = self.normalize_thermal(thermal_lr)       # (64, 64)

        # Add channel dimension to thermal
        thermal_hr = thermal_hr[np.newaxis, :, :]  # (1, 256, 256)
        thermal_lr = thermal_lr[np.newaxis, :, :]  # (1, 64, 64)

        return {
            "optical_hr": torch.from_numpy(optical_hr),
            "thermal_lr": torch.from_numpy(thermal_lr),
            "thermal_hr": torch.from_numpy(thermal_hr),
            "scene_name": self.scenes[idx]
        }


def get_dataloaders(dataset_root, batch_size=8, num_workers=4):
    train_ds = ThermalSRDataset(dataset_root, split="train")
    val_ds = ThermalSRDataset(dataset_root, split="val")
    test_ds = ThermalSRDataset(dataset_root, split="test")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False,
                             num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader, test_loader


# Quick test
if __name__ == "__main__":
    DATASET_ROOT = r"C:\Users\Admin\Downloads\ssl4eo_l_oli_tirs_toa_benchmark\dataset"
    train_loader, val_loader, test_loader = get_dataloaders(DATASET_ROOT, batch_size=4)

    batch = next(iter(train_loader))
    print(f"\nBatch shapes:")
    print(f"  optical_hr: {batch['optical_hr'].shape}")   # (4, 3, 256, 256)
    print(f"  thermal_lr: {batch['thermal_lr'].shape}")   # (4, 1, 64, 64)
    print(f"  thermal_hr: {batch['thermal_hr'].shape}")   # (4, 1, 256, 256)
