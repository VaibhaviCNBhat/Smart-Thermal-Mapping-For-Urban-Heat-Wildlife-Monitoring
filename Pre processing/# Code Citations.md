# Code Citations

## License: unknown
https://github.com/Sumeet2807/Vid2Sound/blob/17e556d8a4446f4d7327c471129d5f54c938dc1f/model/c2d.py

```
# Complete Pipeline: Optical-Guided Thermal Super-Resolution

## Overview of All Steps

```
Step 1: Dataset Preprocessing (Create LR/HR pairs as .npy)
Step 2: Train/Val/Test Split (80/10/10)
Step 3: Dataset Statistics (Global min/max/mean/std for normalization)
Step 4: PyTorch Dataset & DataLoader
Step 5: Model Architecture (Optical-Guided Fusion SR Network)
Step 6: Training Loop (with PSNR/SSIM/RMSE tracking)
Step 7: Evaluation on Test Set
Step 8: Demo Script (Single image inference + visualization)
```

---

## Step 1: Dataset Preprocessing

This reads raw GeoTIFFs, crops to 256×256, creates LR thermal (64×64), and saves as `.npy` + preview `.png`.

````python
# filepath: c:\Users\Admin\Downloads\ssl4eo_l_oli_tirs_toa_benchmark\step1_preprocess.py
import os
import glob
import numpy as np
import rasterio
from PIL import Image
from scipy.ndimage import zoom

# === CONFIGURATION ===
DATASET_ROOT = r"C:\Users\Admin\Downloads\ssl4eo_l_oli_tirs_toa_benchmark"
OUTPUT_ROOT = os.path.join(DATASET_ROOT, "dataset")
SCALE_FACTOR = 4          # 4× super-resolution
CROP_SIZE = 256           # Target size (power of 2)
LR_SIZE = CROP_SIZE // SCALE_FACTOR  # 64×64

OPTICAL_BANDS = (4, 3, 2)  # Red, Green, Blue
THERMAL_BAND = 10           # TIRS-1


def center_crop(arr, size):
    """Center crop a 2D or 3D array to (size, size)."""
    if arr.ndim == 2:
        h, w = arr.shape
        top = (h - size) // 2
        left = (w - size) // 2
        return arr[top:top+size, left:left+size]
    elif arr.ndim == 3:
        h, w = arr.shape[1], arr.shape[2]
        top = (h - size) // 2
        left = (w - size) // 2
        return arr[:, top:top+size, left:left+size]


def normalize_for_preview(arr, percentile_low=2, percentile_high=98):
    """Normalize to uint8 for PNG preview only."""
    arr = arr.astype(np.float32)
    low = np.percentile(arr, percentile_low)
    high = np.percentile(arr, percentile_high)
    arr = np.clip(arr, low, high)
    if high - low == 0:
        return np.zeros_like(arr, dtype=np.uint8)
    return ((arr - low) / (high - low) * 255.0).astype(np.uint8)


def create_lr_thermal(hr_thermal, scale_factor):
    """Downsample HR thermal using bicubic interpolation."""
    lr = zoom(hr_thermal, 1.0 / scale_factor, order=3)  # order=3 = bicubic
    return lr


def process_scene(tif_path, output_dir):
    """Process one GeoTIFF into optical_hr, thermal_hr, thermal_lr."""
    with rasterio.open(tif_path) as src:
        if src.count < THERMAL_BAND:
            return False, "Not enough bands"

        # Read optical bands (4, 3, 2) as float32
        red = src.read(OPTICAL_BANDS[0]).astype(np.float32)
        green = src.read(OPTICAL_BANDS[1]).astype(np.float32)
        blue = src.read(OPTICAL_BANDS[2]).astype(np.float32)
        optical = np.stack([red, green, blue], axis=0)  # (3, H, W)

        # Read thermal band 10 as float32
        thermal = src.read(THERMAL_BAND).astype(np.float32)

    # Check for blank thermal
    if thermal.max() == 0 or np.mean(thermal == 0) > 0.5:
        return False, "Blank thermal"

    # Check for blank optical
    if optical.max() == 0:
        return False, "Blank optical"

    # Center crop to 256×256
    optical = center_crop(optical, CROP_SIZE)      # (3, 256, 256)
    thermal_hr = center_crop(thermal, CROP_SIZE)    # (256, 256)

    # Create LR thermal (64×64)
    thermal_lr = create_lr_thermal(thermal_hr, SCALE_FACTOR)  # (64, 64)

    # Convert optical to (H, W, C) for saving
    optical_hwc = np.transpose(optical, (1, 2, 0))  # (256, 256, 3)

    # Save as .npy (float32 — preserves raw values)
    os.makedirs(output_dir, exist_ok=True)
    np.save(os.path.join(output_dir, "optical_hr.npy"), optical_hwc)
    np.save(os.path.join(output_dir, "thermal_hr.npy"), thermal_hr)
    np.save(os.path.join(output_dir, "thermal_lr.npy"), thermal_lr)

    # Save preview PNGs
    # Optical preview
    opt_preview = np.stack([
        normalize_for_preview(optical_hwc[:, :, 0]),
        normalize_for_preview(optical_hwc[:, :, 1]),
        normalize_for_preview(optical_hwc[:, :, 2])
    ], axis=-1)
    Image.fromarray(opt_preview).save(os.path.join(output_dir, "optical_hr_preview.png"))

    # Thermal HR preview
    th_hr_preview = normalize_for_preview(thermal_hr, 1, 99)
    Image.fromarray(th_hr_preview, mode="L").save(os.path.join(output_dir, "thermal_hr_preview.png"))

    # Thermal LR preview
    th_lr_preview = normalize_for_preview(thermal_lr, 1, 99)
    Image.fromarray(th_lr_preview, mode="L").save(os.path.join(output_dir, "thermal_lr_preview.png"))

    return True, "OK"


def main():
    print("=" * 60)
    print("STEP 1: Dataset Preprocessing")
    print("=" * 60)
    print(f"Scale factor: {SCALE_FACTOR}×")
    print(f"HR size: {CROP_SIZE}×{CROP_SIZE}")
    print(f"LR size: {LR_SIZE}×{LR_SIZE}")

    search_pattern = os.path.join(DATASET_ROOT, "*", "*", "all_bands.tif")
    tif_files = sorted(glob.glob(search_pattern))

    # Exclude 'output' and 'dataset' folders
    tif_files = [f for f in tif_files if "\\output\\" not in f and "\\dataset\\" not in f]

    if not tif_files:
        print("[ERROR] No GeoTIFF files found!")
        return

    print(f"Found {len(tif_files)} scenes.")
    print(f"Output: {OUTPUT_ROOT}\n")

    os.makedirs(OUTPUT_ROOT, exist_ok=True)

    success = 0
    skipped = 0
    errors = []

    for i, tif_path in enumerate(tif_files, 1):
        parts = tif_path.split(os.sep)
        scene_id = parts[-2]
        location_id = parts[-3]
        folder_name = f"{location_id}_{scene_id}"
        output_dir = os.path.join(OUTPUT_ROOT, folder_name)

        if i % 500 == 0 or i == 1:
            print(f"[{i}/{len(tif_files)}] Processing: {folder_name}")

        try:
            ok, msg = process_scene(tif_path, output_dir)
            if ok:
                success += 1
            else:
                skipped += 1
                if os.path.exists(output_dir):
                    import shutil
                    shutil.rmtree(output_dir)
        except Exception as e:
            errors.append((folder_name, str(e)))
            skipped += 1

    print("\n" + "=" * 60)
    print("STEP 1 COMPLETE")
    print(f"  Valid scenes:   {success}")
    print(f"  Skipped:        {skipped}")
    print(f"  Errors:         {len(errors)}")
    print(f"  Output folder:  {OUTPUT_ROOT}")
    print("=" * 60)

    # Save scene list
    scenes = sorted([d for d in os.listdir(OUTPUT_ROOT) if os.path.isdir(os.path.join(OUTPUT_ROOT, d))])
    with open(os.path.join(OUTPUT_ROOT, "all_scenes.txt"), "w") as f:
        for s in scenes:
            f.write(s + "\n")
    print(f"Scene list saved: {len(scenes)} scenes in all_scenes.txt")


if __name__ == "__main__":
    main()
````

---

## Step 2: Train/Val/Test Split

````python
# filepath: c:\Users\Admin\Downloads\ssl4eo_l_oli_tirs_toa_benchmark\step2_split.py
import os
import random

DATASET_ROOT = r"C:\Users\Admin\Downloads\ssl4eo_l_oli_tirs_toa_benchmark\dataset"
SEED = 42
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1


def main():
    print("=" * 60)
    print("STEP 2: Train/Val/Test Split")
    print("=" * 60)

    scenes_file = os.path.join(DATASET_ROOT, "all_scenes.txt")
    with open(scenes_file, "r") as f:
        scenes = [line.strip() for line in f if line.strip()]

    print(f"Total scenes: {len(scenes)}")

    random.seed(SEED)
    random.shuffle(scenes)

    n = len(scenes)
    train_end = int(n * TRAIN_RATIO)
    val_end = train_end + int(n * VAL_RATIO)

    train_scenes = scenes[:train_end]
    val_scenes = scenes[train_end:val_end]
    test_scenes = scenes[val_end:]

    print(f"  Train: {len(train_scenes)}")
    print(f"  Val:   {len(val_scenes)}")
    print(f"  Test:  {len(test_scenes)}")

    # Save split files
    for split_name, split_scenes in [("train", train_scenes), ("val", val_scenes), ("test", test_scenes)]:
        split_file = os.path.join(DATASET_ROOT, f"{split_name}.txt")
        with open(split_file, "w") as f:
            for s in split_scenes:
                f.write(s + "\n")
        print(f"  Saved: {split_file}")

    print("\nDone!")


if __name__ == "__main__":
    main()
````

---

## Step 3: Compute Dataset Statistics

````python
# filepath: c:\Users\Admin\Downloads\ssl4eo_l_oli_tirs_toa_benchmark\step3_stats.py
import os
import numpy as np

DATASET_ROOT = r"C:\Users\Admin\Downloads\ssl4eo_l_oli_tirs_toa_benchmark\dataset"


def main():
    print("=" * 60)
    print("STEP 3: Compute Dataset Statistics")
    print("=" * 60)

    # Only use training set for computing stats
    train_file = os.path.join(DATASET_ROOT, "train.txt")
    with open(train_file, "r") as f:
        train_scenes = [line.strip() for line in f if line.strip()]

    print(f"Computing stats from {len(train_scenes)} training scenes...")

    # Running stats
    optical_sum = np.zeros(3, dtype=np.float64)
    optical_sq_sum = np.zeros(3, dtype=np.float64)
    thermal_sum = 0.0
    thermal_sq_sum = 0.0
    thermal_min = float('inf')
    thermal_max = float('-inf')
    optical_min = np.full(3, float('inf'))
    optical_max = np.full(3, float('-inf'))
    pixel_count = 0
    thermal_pixel_count = 0

    for i, scene in enumerate(train_scenes):
        if i % 1000 == 0:
            print(f"  [{i}/{len(train_scenes)}]")

        scene_dir = os.path.join(DATASET_ROOT, scene)
        optical = np.load(os.path.join(scene_dir, "optical_hr.npy"))      # (256, 256, 3)
        thermal = np.load(os.path.join(scene_dir, "thermal_hr.npy"))      # (256, 256)

        # Optical stats per channel
        for c in range(3):
            ch = optical[:, :, c].astype(np.float64)
            optical_sum[c] += ch.sum()
            optical_sq_sum[c] += (ch ** 2).sum()
            optical_min[c] = min(optical_min[c], ch.min())
            optical_max[c] = max(optical_max[c], ch.max())

        pixel_count += optical.shape[0] * optical.shape[1]

        # Thermal stats
        th = thermal.astype(np.float64)
        thermal_sum += th.sum()
        thermal_sq_sum += (th ** 2).sum()
        thermal_min = min(thermal_min, th.min())
        thermal_max = max(thermal_max, th.max())
        thermal_pixel_count += th.size

    # Compute mean and std
    optical_mean = optical_sum / pixel_count
    optical_std = np.sqrt(optical_sq_sum / pixel_count - optical_mean ** 2)
    thermal_mean = thermal_sum / thermal_pixel_count
    thermal_std = np.sqrt(thermal_sq_sum / thermal_pixel_count - thermal_mean ** 2)

    print("\n" + "=" * 60)
    print("DATASET STATISTICS (Training Set)")
    print("=" * 60)
    print(f"\nOptical (Bands 4, 3, 2):")
    print(f"  Mean: [{optical_mean[0]:.4f}, {optical_mean[1]:.4f}, {optical_mean[2]:.4f}]")
    print(f"  Std:  [{optical_std[0]:.4f}, {optical_std[1]:.4f}, {optical_std[2]:.4f}]")
    print(f"  Min:  [{optical_min[0]:.4f}, {optical_min[1]:.4f}, {optical_min[2]:.4f}]")
    print(f"  Max:  [{optical_max[0]:.4f}, {optical_max[1]:.4f}, {optical_max[2]:.4f}]")
    print(f"\nThermal (Band 10):")
    print(f"  Mean: {thermal_mean:.4f}")
    print(f"  Std:  {thermal_std:.4f}")
    print(f"  Min:  {thermal_min:.4f}")
    print(f"  Max:  {thermal_max:.4f}")

    # Save stats
    stats = {
        "optical_mean": optical_mean,
        "optical_std": optical_std,
        "optical_min": optical_min,
        "optical_max": optical_max,
        "thermal_mean": thermal_mean,
        "thermal_std": thermal_std,
        "thermal_min": thermal_min,
        "thermal_max": thermal_max,
    }
    np.save(os.path.join(DATASET_ROOT, "dataset_stats.npy"), stats)
    print(f"\nStats saved to: {os.path.join(DATASET_ROOT, 'dataset_stats.npy')}")


if __name__ == "__main__":
    main()
````

---

## Step 4: PyTorch Dataset & DataLoader

````python
# filepath: c:\Users\Admin\Downloads\ssl4eo_l_oli_tirs_toa_benchmark\step4_dataloader.py
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
````

---

## Step 5: Model Architecture

````python
# filepath: c:\Users\Admin\Downloads\ssl4eo_l_oli_tirs_toa_benchmark\step5_model.py
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
```


## License: unknown
https://github.com/Sumeet2807/Vid2Sound/blob/17e556d8a4446f4d7327c471129d5f54c938dc1f/model/c2d.py

```
# Complete Pipeline: Optical-Guided Thermal Super-Resolution

## Overview of All Steps

```
Step 1: Dataset Preprocessing (Create LR/HR pairs as .npy)
Step 2: Train/Val/Test Split (80/10/10)
Step 3: Dataset Statistics (Global min/max/mean/std for normalization)
Step 4: PyTorch Dataset & DataLoader
Step 5: Model Architecture (Optical-Guided Fusion SR Network)
Step 6: Training Loop (with PSNR/SSIM/RMSE tracking)
Step 7: Evaluation on Test Set
Step 8: Demo Script (Single image inference + visualization)
```

---

## Step 1: Dataset Preprocessing

This reads raw GeoTIFFs, crops to 256×256, creates LR thermal (64×64), and saves as `.npy` + preview `.png`.

````python
# filepath: c:\Users\Admin\Downloads\ssl4eo_l_oli_tirs_toa_benchmark\step1_preprocess.py
import os
import glob
import numpy as np
import rasterio
from PIL import Image
from scipy.ndimage import zoom

# === CONFIGURATION ===
DATASET_ROOT = r"C:\Users\Admin\Downloads\ssl4eo_l_oli_tirs_toa_benchmark"
OUTPUT_ROOT = os.path.join(DATASET_ROOT, "dataset")
SCALE_FACTOR = 4          # 4× super-resolution
CROP_SIZE = 256           # Target size (power of 2)
LR_SIZE = CROP_SIZE // SCALE_FACTOR  # 64×64

OPTICAL_BANDS = (4, 3, 2)  # Red, Green, Blue
THERMAL_BAND = 10           # TIRS-1


def center_crop(arr, size):
    """Center crop a 2D or 3D array to (size, size)."""
    if arr.ndim == 2:
        h, w = arr.shape
        top = (h - size) // 2
        left = (w - size) // 2
        return arr[top:top+size, left:left+size]
    elif arr.ndim == 3:
        h, w = arr.shape[1], arr.shape[2]
        top = (h - size) // 2
        left = (w - size) // 2
        return arr[:, top:top+size, left:left+size]


def normalize_for_preview(arr, percentile_low=2, percentile_high=98):
    """Normalize to uint8 for PNG preview only."""
    arr = arr.astype(np.float32)
    low = np.percentile(arr, percentile_low)
    high = np.percentile(arr, percentile_high)
    arr = np.clip(arr, low, high)
    if high - low == 0:
        return np.zeros_like(arr, dtype=np.uint8)
    return ((arr - low) / (high - low) * 255.0).astype(np.uint8)


def create_lr_thermal(hr_thermal, scale_factor):
    """Downsample HR thermal using bicubic interpolation."""
    lr = zoom(hr_thermal, 1.0 / scale_factor, order=3)  # order=3 = bicubic
    return lr


def process_scene(tif_path, output_dir):
    """Process one GeoTIFF into optical_hr, thermal_hr, thermal_lr."""
    with rasterio.open(tif_path) as src:
        if src.count < THERMAL_BAND:
            return False, "Not enough bands"

        # Read optical bands (4, 3, 2) as float32
        red = src.read(OPTICAL_BANDS[0]).astype(np.float32)
        green = src.read(OPTICAL_BANDS[1]).astype(np.float32)
        blue = src.read(OPTICAL_BANDS[2]).astype(np.float32)
        optical = np.stack([red, green, blue], axis=0)  # (3, H, W)

        # Read thermal band 10 as float32
        thermal = src.read(THERMAL_BAND).astype(np.float32)

    # Check for blank thermal
    if thermal.max() == 0 or np.mean(thermal == 0) > 0.5:
        return False, "Blank thermal"

    # Check for blank optical
    if optical.max() == 0:
        return False, "Blank optical"

    # Center crop to 256×256
    optical = center_crop(optical, CROP_SIZE)      # (3, 256, 256)
    thermal_hr = center_crop(thermal, CROP_SIZE)    # (256, 256)

    # Create LR thermal (64×64)
    thermal_lr = create_lr_thermal(thermal_hr, SCALE_FACTOR)  # (64, 64)

    # Convert optical to (H, W, C) for saving
    optical_hwc = np.transpose(optical, (1, 2, 0))  # (256, 256, 3)

    # Save as .npy (float32 — preserves raw values)
    os.makedirs(output_dir, exist_ok=True)
    np.save(os.path.join(output_dir, "optical_hr.npy"), optical_hwc)
    np.save(os.path.join(output_dir, "thermal_hr.npy"), thermal_hr)
    np.save(os.path.join(output_dir, "thermal_lr.npy"), thermal_lr)

    # Save preview PNGs
    # Optical preview
    opt_preview = np.stack([
        normalize_for_preview(optical_hwc[:, :, 0]),
        normalize_for_preview(optical_hwc[:, :, 1]),
        normalize_for_preview(optical_hwc[:, :, 2])
    ], axis=-1)
    Image.fromarray(opt_preview).save(os.path.join(output_dir, "optical_hr_preview.png"))

    # Thermal HR preview
    th_hr_preview = normalize_for_preview(thermal_hr, 1, 99)
    Image.fromarray(th_hr_preview, mode="L").save(os.path.join(output_dir, "thermal_hr_preview.png"))

    # Thermal LR preview
    th_lr_preview = normalize_for_preview(thermal_lr, 1, 99)
    Image.fromarray(th_lr_preview, mode="L").save(os.path.join(output_dir, "thermal_lr_preview.png"))

    return True, "OK"


def main():
    print("=" * 60)
    print("STEP 1: Dataset Preprocessing")
    print("=" * 60)
    print(f"Scale factor: {SCALE_FACTOR}×")
    print(f"HR size: {CROP_SIZE}×{CROP_SIZE}")
    print(f"LR size: {LR_SIZE}×{LR_SIZE}")

    search_pattern = os.path.join(DATASET_ROOT, "*", "*", "all_bands.tif")
    tif_files = sorted(glob.glob(search_pattern))

    # Exclude 'output' and 'dataset' folders
    tif_files = [f for f in tif_files if "\\output\\" not in f and "\\dataset\\" not in f]

    if not tif_files:
        print("[ERROR] No GeoTIFF files found!")
        return

    print(f"Found {len(tif_files)} scenes.")
    print(f"Output: {OUTPUT_ROOT}\n")

    os.makedirs(OUTPUT_ROOT, exist_ok=True)

    success = 0
    skipped = 0
    errors = []

    for i, tif_path in enumerate(tif_files, 1):
        parts = tif_path.split(os.sep)
        scene_id = parts[-2]
        location_id = parts[-3]
        folder_name = f"{location_id}_{scene_id}"
        output_dir = os.path.join(OUTPUT_ROOT, folder_name)

        if i % 500 == 0 or i == 1:
            print(f"[{i}/{len(tif_files)}] Processing: {folder_name}")

        try:
            ok, msg = process_scene(tif_path, output_dir)
            if ok:
                success += 1
            else:
                skipped += 1
                if os.path.exists(output_dir):
                    import shutil
                    shutil.rmtree(output_dir)
        except Exception as e:
            errors.append((folder_name, str(e)))
            skipped += 1

    print("\n" + "=" * 60)
    print("STEP 1 COMPLETE")
    print(f"  Valid scenes:   {success}")
    print(f"  Skipped:        {skipped}")
    print(f"  Errors:         {len(errors)}")
    print(f"  Output folder:  {OUTPUT_ROOT}")
    print("=" * 60)

    # Save scene list
    scenes = sorted([d for d in os.listdir(OUTPUT_ROOT) if os.path.isdir(os.path.join(OUTPUT_ROOT, d))])
    with open(os.path.join(OUTPUT_ROOT, "all_scenes.txt"), "w") as f:
        for s in scenes:
            f.write(s + "\n")
    print(f"Scene list saved: {len(scenes)} scenes in all_scenes.txt")


if __name__ == "__main__":
    main()
````

---

## Step 2: Train/Val/Test Split

````python
# filepath: c:\Users\Admin\Downloads\ssl4eo_l_oli_tirs_toa_benchmark\step2_split.py
import os
import random

DATASET_ROOT = r"C:\Users\Admin\Downloads\ssl4eo_l_oli_tirs_toa_benchmark\dataset"
SEED = 42
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1


def main():
    print("=" * 60)
    print("STEP 2: Train/Val/Test Split")
    print("=" * 60)

    scenes_file = os.path.join(DATASET_ROOT, "all_scenes.txt")
    with open(scenes_file, "r") as f:
        scenes = [line.strip() for line in f if line.strip()]

    print(f"Total scenes: {len(scenes)}")

    random.seed(SEED)
    random.shuffle(scenes)

    n = len(scenes)
    train_end = int(n * TRAIN_RATIO)
    val_end = train_end + int(n * VAL_RATIO)

    train_scenes = scenes[:train_end]
    val_scenes = scenes[train_end:val_end]
    test_scenes = scenes[val_end:]

    print(f"  Train: {len(train_scenes)}")
    print(f"  Val:   {len(val_scenes)}")
    print(f"  Test:  {len(test_scenes)}")

    # Save split files
    for split_name, split_scenes in [("train", train_scenes), ("val", val_scenes), ("test", test_scenes)]:
        split_file = os.path.join(DATASET_ROOT, f"{split_name}.txt")
        with open(split_file, "w") as f:
            for s in split_scenes:
                f.write(s + "\n")
        print(f"  Saved: {split_file}")

    print("\nDone!")


if __name__ == "__main__":
    main()
````

---

## Step 3: Compute Dataset Statistics

````python
# filepath: c:\Users\Admin\Downloads\ssl4eo_l_oli_tirs_toa_benchmark\step3_stats.py
import os
import numpy as np

DATASET_ROOT = r"C:\Users\Admin\Downloads\ssl4eo_l_oli_tirs_toa_benchmark\dataset"


def main():
    print("=" * 60)
    print("STEP 3: Compute Dataset Statistics")
    print("=" * 60)

    # Only use training set for computing stats
    train_file = os.path.join(DATASET_ROOT, "train.txt")
    with open(train_file, "r") as f:
        train_scenes = [line.strip() for line in f if line.strip()]

    print(f"Computing stats from {len(train_scenes)} training scenes...")

    # Running stats
    optical_sum = np.zeros(3, dtype=np.float64)
    optical_sq_sum = np.zeros(3, dtype=np.float64)
    thermal_sum = 0.0
    thermal_sq_sum = 0.0
    thermal_min = float('inf')
    thermal_max = float('-inf')
    optical_min = np.full(3, float('inf'))
    optical_max = np.full(3, float('-inf'))
    pixel_count = 0
    thermal_pixel_count = 0

    for i, scene in enumerate(train_scenes):
        if i % 1000 == 0:
            print(f"  [{i}/{len(train_scenes)}]")

        scene_dir = os.path.join(DATASET_ROOT, scene)
        optical = np.load(os.path.join(scene_dir, "optical_hr.npy"))      # (256, 256, 3)
        thermal = np.load(os.path.join(scene_dir, "thermal_hr.npy"))      # (256, 256)

        # Optical stats per channel
        for c in range(3):
            ch = optical[:, :, c].astype(np.float64)
            optical_sum[c] += ch.sum()
            optical_sq_sum[c] += (ch ** 2).sum()
            optical_min[c] = min(optical_min[c], ch.min())
            optical_max[c] = max(optical_max[c], ch.max())

        pixel_count += optical.shape[0] * optical.shape[1]

        # Thermal stats
        th = thermal.astype(np.float64)
        thermal_sum += th.sum()
        thermal_sq_sum += (th ** 2).sum()
        thermal_min = min(thermal_min, th.min())
        thermal_max = max(thermal_max, th.max())
        thermal_pixel_count += th.size

    # Compute mean and std
    optical_mean = optical_sum / pixel_count
    optical_std = np.sqrt(optical_sq_sum / pixel_count - optical_mean ** 2)
    thermal_mean = thermal_sum / thermal_pixel_count
    thermal_std = np.sqrt(thermal_sq_sum / thermal_pixel_count - thermal_mean ** 2)

    print("\n" + "=" * 60)
    print("DATASET STATISTICS (Training Set)")
    print("=" * 60)
    print(f"\nOptical (Bands 4, 3, 2):")
    print(f"  Mean: [{optical_mean[0]:.4f}, {optical_mean[1]:.4f}, {optical_mean[2]:.4f}]")
    print(f"  Std:  [{optical_std[0]:.4f}, {optical_std[1]:.4f}, {optical_std[2]:.4f}]")
    print(f"  Min:  [{optical_min[0]:.4f}, {optical_min[1]:.4f}, {optical_min[2]:.4f}]")
    print(f"  Max:  [{optical_max[0]:.4f}, {optical_max[1]:.4f}, {optical_max[2]:.4f}]")
    print(f"\nThermal (Band 10):")
    print(f"  Mean: {thermal_mean:.4f}")
    print(f"  Std:  {thermal_std:.4f}")
    print(f"  Min:  {thermal_min:.4f}")
    print(f"  Max:  {thermal_max:.4f}")

    # Save stats
    stats = {
        "optical_mean": optical_mean,
        "optical_std": optical_std,
        "optical_min": optical_min,
        "optical_max": optical_max,
        "thermal_mean": thermal_mean,
        "thermal_std": thermal_std,
        "thermal_min": thermal_min,
        "thermal_max": thermal_max,
    }
    np.save(os.path.join(DATASET_ROOT, "dataset_stats.npy"), stats)
    print(f"\nStats saved to: {os.path.join(DATASET_ROOT, 'dataset_stats.npy')}")


if __name__ == "__main__":
    main()
````

---

## Step 4: PyTorch Dataset & DataLoader

````python
# filepath: c:\Users\Admin\Downloads\ssl4eo_l_oli_tirs_toa_benchmark\step4_dataloader.py
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
````

---

## Step 5: Model Architecture

````python
# filepath: c:\Users\Admin\Downloads\ssl4eo_l_oli_tirs_toa_benchmark\step5_model.py
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
```


## License: unknown
https://github.com/Sumeet2807/Vid2Sound/blob/17e556d8a4446f4d7327c471129d5f54c938dc1f/model/c2d.py

```
# Complete Pipeline: Optical-Guided Thermal Super-Resolution

## Overview of All Steps

```
Step 1: Dataset Preprocessing (Create LR/HR pairs as .npy)
Step 2: Train/Val/Test Split (80/10/10)
Step 3: Dataset Statistics (Global min/max/mean/std for normalization)
Step 4: PyTorch Dataset & DataLoader
Step 5: Model Architecture (Optical-Guided Fusion SR Network)
Step 6: Training Loop (with PSNR/SSIM/RMSE tracking)
Step 7: Evaluation on Test Set
Step 8: Demo Script (Single image inference + visualization)
```

---

## Step 1: Dataset Preprocessing

This reads raw GeoTIFFs, crops to 256×256, creates LR thermal (64×64), and saves as `.npy` + preview `.png`.

````python
# filepath: c:\Users\Admin\Downloads\ssl4eo_l_oli_tirs_toa_benchmark\step1_preprocess.py
import os
import glob
import numpy as np
import rasterio
from PIL import Image
from scipy.ndimage import zoom

# === CONFIGURATION ===
DATASET_ROOT = r"C:\Users\Admin\Downloads\ssl4eo_l_oli_tirs_toa_benchmark"
OUTPUT_ROOT = os.path.join(DATASET_ROOT, "dataset")
SCALE_FACTOR = 4          # 4× super-resolution
CROP_SIZE = 256           # Target size (power of 2)
LR_SIZE = CROP_SIZE // SCALE_FACTOR  # 64×64

OPTICAL_BANDS = (4, 3, 2)  # Red, Green, Blue
THERMAL_BAND = 10           # TIRS-1


def center_crop(arr, size):
    """Center crop a 2D or 3D array to (size, size)."""
    if arr.ndim == 2:
        h, w = arr.shape
        top = (h - size) // 2
        left = (w - size) // 2
        return arr[top:top+size, left:left+size]
    elif arr.ndim == 3:
        h, w = arr.shape[1], arr.shape[2]
        top = (h - size) // 2
        left = (w - size) // 2
        return arr[:, top:top+size, left:left+size]


def normalize_for_preview(arr, percentile_low=2, percentile_high=98):
    """Normalize to uint8 for PNG preview only."""
    arr = arr.astype(np.float32)
    low = np.percentile(arr, percentile_low)
    high = np.percentile(arr, percentile_high)
    arr = np.clip(arr, low, high)
    if high - low == 0:
        return np.zeros_like(arr, dtype=np.uint8)
    return ((arr - low) / (high - low) * 255.0).astype(np.uint8)


def create_lr_thermal(hr_thermal, scale_factor):
    """Downsample HR thermal using bicubic interpolation."""
    lr = zoom(hr_thermal, 1.0 / scale_factor, order=3)  # order=3 = bicubic
    return lr


def process_scene(tif_path, output_dir):
    """Process one GeoTIFF into optical_hr, thermal_hr, thermal_lr."""
    with rasterio.open(tif_path) as src:
        if src.count < THERMAL_BAND:
            return False, "Not enough bands"

        # Read optical bands (4, 3, 2) as float32
        red = src.read(OPTICAL_BANDS[0]).astype(np.float32)
        green = src.read(OPTICAL_BANDS[1]).astype(np.float32)
        blue = src.read(OPTICAL_BANDS[2]).astype(np.float32)
        optical = np.stack([red, green, blue], axis=0)  # (3, H, W)

        # Read thermal band 10 as float32
        thermal = src.read(THERMAL_BAND).astype(np.float32)

    # Check for blank thermal
    if thermal.max() == 0 or np.mean(thermal == 0) > 0.5:
        return False, "Blank thermal"

    # Check for blank optical
    if optical.max() == 0:
        return False, "Blank optical"

    # Center crop to 256×256
    optical = center_crop(optical, CROP_SIZE)      # (3, 256, 256)
    thermal_hr = center_crop(thermal, CROP_SIZE)    # (256, 256)

    # Create LR thermal (64×64)
    thermal_lr = create_lr_thermal(thermal_hr, SCALE_FACTOR)  # (64, 64)

    # Convert optical to (H, W, C) for saving
    optical_hwc = np.transpose(optical, (1, 2, 0))  # (256, 256, 3)

    # Save as .npy (float32 — preserves raw values)
    os.makedirs(output_dir, exist_ok=True)
    np.save(os.path.join(output_dir, "optical_hr.npy"), optical_hwc)
    np.save(os.path.join(output_dir, "thermal_hr.npy"), thermal_hr)
    np.save(os.path.join(output_dir, "thermal_lr.npy"), thermal_lr)

    # Save preview PNGs
    # Optical preview
    opt_preview = np.stack([
        normalize_for_preview(optical_hwc[:, :, 0]),
        normalize_for_preview(optical_hwc[:, :, 1]),
        normalize_for_preview(optical_hwc[:, :, 2])
    ], axis=-1)
    Image.fromarray(opt_preview).save(os.path.join(output_dir, "optical_hr_preview.png"))

    # Thermal HR preview
    th_hr_preview = normalize_for_preview(thermal_hr, 1, 99)
    Image.fromarray(th_hr_preview, mode="L").save(os.path.join(output_dir, "thermal_hr_preview.png"))

    # Thermal LR preview
    th_lr_preview = normalize_for_preview(thermal_lr, 1, 99)
    Image.fromarray(th_lr_preview, mode="L").save(os.path.join(output_dir, "thermal_lr_preview.png"))

    return True, "OK"


def main():
    print("=" * 60)
    print("STEP 1: Dataset Preprocessing")
    print("=" * 60)
    print(f"Scale factor: {SCALE_FACTOR}×")
    print(f"HR size: {CROP_SIZE}×{CROP_SIZE}")
    print(f"LR size: {LR_SIZE}×{LR_SIZE}")

    search_pattern = os.path.join(DATASET_ROOT, "*", "*", "all_bands.tif")
    tif_files = sorted(glob.glob(search_pattern))

    # Exclude 'output' and 'dataset' folders
    tif_files = [f for f in tif_files if "\\output\\" not in f and "\\dataset\\" not in f]

    if not tif_files:
        print("[ERROR] No GeoTIFF files found!")
        return

    print(f"Found {len(tif_files)} scenes.")
    print(f"Output: {OUTPUT_ROOT}\n")

    os.makedirs(OUTPUT_ROOT, exist_ok=True)

    success = 0
    skipped = 0
    errors = []

    for i, tif_path in enumerate(tif_files, 1):
        parts = tif_path.split(os.sep)
        scene_id = parts[-2]
        location_id = parts[-3]
        folder_name = f"{location_id}_{scene_id}"
        output_dir = os.path.join(OUTPUT_ROOT, folder_name)

        if i % 500 == 0 or i == 1:
            print(f"[{i}/{len(tif_files)}] Processing: {folder_name}")

        try:
            ok, msg = process_scene(tif_path, output_dir)
            if ok:
                success += 1
            else:
                skipped += 1
                if os.path.exists(output_dir):
                    import shutil
                    shutil.rmtree(output_dir)
        except Exception as e:
            errors.append((folder_name, str(e)))
            skipped += 1

    print("\n" + "=" * 60)
    print("STEP 1 COMPLETE")
    print(f"  Valid scenes:   {success}")
    print(f"  Skipped:        {skipped}")
    print(f"  Errors:         {len(errors)}")
    print(f"  Output folder:  {OUTPUT_ROOT}")
    print("=" * 60)

    # Save scene list
    scenes = sorted([d for d in os.listdir(OUTPUT_ROOT) if os.path.isdir(os.path.join(OUTPUT_ROOT, d))])
    with open(os.path.join(OUTPUT_ROOT, "all_scenes.txt"), "w") as f:
        for s in scenes:
            f.write(s + "\n")
    print(f"Scene list saved: {len(scenes)} scenes in all_scenes.txt")


if __name__ == "__main__":
    main()
````

---

## Step 2: Train/Val/Test Split

````python
# filepath: c:\Users\Admin\Downloads\ssl4eo_l_oli_tirs_toa_benchmark\step2_split.py
import os
import random

DATASET_ROOT = r"C:\Users\Admin\Downloads\ssl4eo_l_oli_tirs_toa_benchmark\dataset"
SEED = 42
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1


def main():
    print("=" * 60)
    print("STEP 2: Train/Val/Test Split")
    print("=" * 60)

    scenes_file = os.path.join(DATASET_ROOT, "all_scenes.txt")
    with open(scenes_file, "r") as f:
        scenes = [line.strip() for line in f if line.strip()]

    print(f"Total scenes: {len(scenes)}")

    random.seed(SEED)
    random.shuffle(scenes)

    n = len(scenes)
    train_end = int(n * TRAIN_RATIO)
    val_end = train_end + int(n * VAL_RATIO)

    train_scenes = scenes[:train_end]
    val_scenes = scenes[train_end:val_end]
    test_scenes = scenes[val_end:]

    print(f"  Train: {len(train_scenes)}")
    print(f"  Val:   {len(val_scenes)}")
    print(f"  Test:  {len(test_scenes)}")

    # Save split files
    for split_name, split_scenes in [("train", train_scenes), ("val", val_scenes), ("test", test_scenes)]:
        split_file = os.path.join(DATASET_ROOT, f"{split_name}.txt")
        with open(split_file, "w") as f:
            for s in split_scenes:
                f.write(s + "\n")
        print(f"  Saved: {split_file}")

    print("\nDone!")


if __name__ == "__main__":
    main()
````

---

## Step 3: Compute Dataset Statistics

````python
# filepath: c:\Users\Admin\Downloads\ssl4eo_l_oli_tirs_toa_benchmark\step3_stats.py
import os
import numpy as np

DATASET_ROOT = r"C:\Users\Admin\Downloads\ssl4eo_l_oli_tirs_toa_benchmark\dataset"


def main():
    print("=" * 60)
    print("STEP 3: Compute Dataset Statistics")
    print("=" * 60)

    # Only use training set for computing stats
    train_file = os.path.join(DATASET_ROOT, "train.txt")
    with open(train_file, "r") as f:
        train_scenes = [line.strip() for line in f if line.strip()]

    print(f"Computing stats from {len(train_scenes)} training scenes...")

    # Running stats
    optical_sum = np.zeros(3, dtype=np.float64)
    optical_sq_sum = np.zeros(3, dtype=np.float64)
    thermal_sum = 0.0
    thermal_sq_sum = 0.0
    thermal_min = float('inf')
    thermal_max = float('-inf')
    optical_min = np.full(3, float('inf'))
    optical_max = np.full(3, float('-inf'))
    pixel_count = 0
    thermal_pixel_count = 0

    for i, scene in enumerate(train_scenes):
        if i % 1000 == 0:
            print(f"  [{i}/{len(train_scenes)}]")

        scene_dir = os.path.join(DATASET_ROOT, scene)
        optical = np.load(os.path.join(scene_dir, "optical_hr.npy"))      # (256, 256, 3)
        thermal = np.load(os.path.join(scene_dir, "thermal_hr.npy"))      # (256, 256)

        # Optical stats per channel
        for c in range(3):
            ch = optical[:, :, c].astype(np.float64)
            optical_sum[c] += ch.sum()
            optical_sq_sum[c] += (ch ** 2).sum()
            optical_min[c] = min(optical_min[c], ch.min())
            optical_max[c] = max(optical_max[c], ch.max())

        pixel_count += optical.shape[0] * optical.shape[1]

        # Thermal stats
        th = thermal.astype(np.float64)
        thermal_sum += th.sum()
        thermal_sq_sum += (th ** 2).sum()
        thermal_min = min(thermal_min, th.min())
        thermal_max = max(thermal_max, th.max())
        thermal_pixel_count += th.size

    # Compute mean and std
    optical_mean = optical_sum / pixel_count
    optical_std = np.sqrt(optical_sq_sum / pixel_count - optical_mean ** 2)
    thermal_mean = thermal_sum / thermal_pixel_count
    thermal_std = np.sqrt(thermal_sq_sum / thermal_pixel_count - thermal_mean ** 2)

    print("\n" + "=" * 60)
    print("DATASET STATISTICS (Training Set)")
    print("=" * 60)
    print(f"\nOptical (Bands 4, 3, 2):")
    print(f"  Mean: [{optical_mean[0]:.4f}, {optical_mean[1]:.4f}, {optical_mean[2]:.4f}]")
    print(f"  Std:  [{optical_std[0]:.4f}, {optical_std[1]:.4f}, {optical_std[2]:.4f}]")
    print(f"  Min:  [{optical_min[0]:.4f}, {optical_min[1]:.4f}, {optical_min[2]:.4f}]")
    print(f"  Max:  [{optical_max[0]:.4f}, {optical_max[1]:.4f}, {optical_max[2]:.4f}]")
    print(f"\nThermal (Band 10):")
    print(f"  Mean: {thermal_mean:.4f}")
    print(f"  Std:  {thermal_std:.4f}")
    print(f"  Min:  {thermal_min:.4f}")
    print(f"  Max:  {thermal_max:.4f}")

    # Save stats
    stats = {
        "optical_mean": optical_mean,
        "optical_std": optical_std,
        "optical_min": optical_min,
        "optical_max": optical_max,
        "thermal_mean": thermal_mean,
        "thermal_std": thermal_std,
        "thermal_min": thermal_min,
        "thermal_max": thermal_max,
    }
    np.save(os.path.join(DATASET_ROOT, "dataset_stats.npy"), stats)
    print(f"\nStats saved to: {os.path.join(DATASET_ROOT, 'dataset_stats.npy')}")


if __name__ == "__main__":
    main()
````

---

## Step 4: PyTorch Dataset & DataLoader

````python
# filepath: c:\Users\Admin\Downloads\ssl4eo_l_oli_tirs_toa_benchmark\step4_dataloader.py
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
````

---

## Step 5: Model Architecture

````python
# filepath: c:\Users\Admin\Downloads\ssl4eo_l_oli_tirs_toa_benchmark\step5_model.py
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
```


## License: unknown
https://github.com/Sumeet2807/Vid2Sound/blob/17e556d8a4446f4d7327c471129d5f54c938dc1f/model/c2d.py

```
# Complete Pipeline: Optical-Guided Thermal Super-Resolution

## Overview of All Steps

```
Step 1: Dataset Preprocessing (Create LR/HR pairs as .npy)
Step 2: Train/Val/Test Split (80/10/10)
Step 3: Dataset Statistics (Global min/max/mean/std for normalization)
Step 4: PyTorch Dataset & DataLoader
Step 5: Model Architecture (Optical-Guided Fusion SR Network)
Step 6: Training Loop (with PSNR/SSIM/RMSE tracking)
Step 7: Evaluation on Test Set
Step 8: Demo Script (Single image inference + visualization)
```

---

## Step 1: Dataset Preprocessing

This reads raw GeoTIFFs, crops to 256×256, creates LR thermal (64×64), and saves as `.npy` + preview `.png`.

````python
# filepath: c:\Users\Admin\Downloads\ssl4eo_l_oli_tirs_toa_benchmark\step1_preprocess.py
import os
import glob
import numpy as np
import rasterio
from PIL import Image
from scipy.ndimage import zoom

# === CONFIGURATION ===
DATASET_ROOT = r"C:\Users\Admin\Downloads\ssl4eo_l_oli_tirs_toa_benchmark"
OUTPUT_ROOT = os.path.join(DATASET_ROOT, "dataset")
SCALE_FACTOR = 4          # 4× super-resolution
CROP_SIZE = 256           # Target size (power of 2)
LR_SIZE = CROP_SIZE // SCALE_FACTOR  # 64×64

OPTICAL_BANDS = (4, 3, 2)  # Red, Green, Blue
THERMAL_BAND = 10           # TIRS-1


def center_crop(arr, size):
    """Center crop a 2D or 3D array to (size, size)."""
    if arr.ndim == 2:
        h, w = arr.shape
        top = (h - size) // 2
        left = (w - size) // 2
        return arr[top:top+size, left:left+size]
    elif arr.ndim == 3:
        h, w = arr.shape[1], arr.shape[2]
        top = (h - size) // 2
        left = (w - size) // 2
        return arr[:, top:top+size, left:left+size]


def normalize_for_preview(arr, percentile_low=2, percentile_high=98):
    """Normalize to uint8 for PNG preview only."""
    arr = arr.astype(np.float32)
    low = np.percentile(arr, percentile_low)
    high = np.percentile(arr, percentile_high)
    arr = np.clip(arr, low, high)
    if high - low == 0:
        return np.zeros_like(arr, dtype=np.uint8)
    return ((arr - low) / (high - low) * 255.0).astype(np.uint8)


def create_lr_thermal(hr_thermal, scale_factor):
    """Downsample HR thermal using bicubic interpolation."""
    lr = zoom(hr_thermal, 1.0 / scale_factor, order=3)  # order=3 = bicubic
    return lr


def process_scene(tif_path, output_dir):
    """Process one GeoTIFF into optical_hr, thermal_hr, thermal_lr."""
    with rasterio.open(tif_path) as src:
        if src.count < THERMAL_BAND:
            return False, "Not enough bands"

        # Read optical bands (4, 3, 2) as float32
        red = src.read(OPTICAL_BANDS[0]).astype(np.float32)
        green = src.read(OPTICAL_BANDS[1]).astype(np.float32)
        blue = src.read(OPTICAL_BANDS[2]).astype(np.float32)
        optical = np.stack([red, green, blue], axis=0)  # (3, H, W)

        # Read thermal band 10 as float32
        thermal = src.read(THERMAL_BAND).astype(np.float32)

    # Check for blank thermal
    if thermal.max() == 0 or np.mean(thermal == 0) > 0.5:
        return False, "Blank thermal"

    # Check for blank optical
    if optical.max() == 0:
        return False, "Blank optical"

    # Center crop to 256×256
    optical = center_crop(optical, CROP_SIZE)      # (3, 256, 256)
    thermal_hr = center_crop(thermal, CROP_SIZE)    # (256, 256)

    # Create LR thermal (64×64)
    thermal_lr = create_lr_thermal(thermal_hr, SCALE_FACTOR)  # (64, 64)

    # Convert optical to (H, W, C) for saving
    optical_hwc = np.transpose(optical, (1, 2, 0))  # (256, 256, 3)

    # Save as .npy (float32 — preserves raw values)
    os.makedirs(output_dir, exist_ok=True)
    np.save(os.path.join(output_dir, "optical_hr.npy"), optical_hwc)
    np.save(os.path.join(output_dir, "thermal_hr.npy"), thermal_hr)
    np.save(os.path.join(output_dir, "thermal_lr.npy"), thermal_lr)

    # Save preview PNGs
    # Optical preview
    opt_preview = np.stack([
        normalize_for_preview(optical_hwc[:, :, 0]),
        normalize_for_preview(optical_hwc[:, :, 1]),
        normalize_for_preview(optical_hwc[:, :, 2])
    ], axis=-1)
    Image.fromarray(opt_preview).save(os.path.join(output_dir, "optical_hr_preview.png"))

    # Thermal HR preview
    th_hr_preview = normalize_for_preview(thermal_hr, 1, 99)
    Image.fromarray(th_hr_preview, mode="L").save(os.path.join(output_dir, "thermal_hr_preview.png"))

    # Thermal LR preview
    th_lr_preview = normalize_for_preview(thermal_lr, 1, 99)
    Image.fromarray(th_lr_preview, mode="L").save(os.path.join(output_dir, "thermal_lr_preview.png"))

    return True, "OK"


def main():
    print("=" * 60)
    print("STEP 1: Dataset Preprocessing")
    print("=" * 60)
    print(f"Scale factor: {SCALE_FACTOR}×")
    print(f"HR size: {CROP_SIZE}×{CROP_SIZE}")
    print(f"LR size: {LR_SIZE}×{LR_SIZE}")

    search_pattern = os.path.join(DATASET_ROOT, "*", "*", "all_bands.tif")
    tif_files = sorted(glob.glob(search_pattern))

    # Exclude 'output' and 'dataset' folders
    tif_files = [f for f in tif_files if "\\output\\" not in f and "\\dataset\\" not in f]

    if not tif_files:
        print("[ERROR] No GeoTIFF files found!")
        return

    print(f"Found {len(tif_files)} scenes.")
    print(f"Output: {OUTPUT_ROOT}\n")

    os.makedirs(OUTPUT_ROOT, exist_ok=True)

    success = 0
    skipped = 0
    errors = []

    for i, tif_path in enumerate(tif_files, 1):
        parts = tif_path.split(os.sep)
        scene_id = parts[-2]
        location_id = parts[-3]
        folder_name = f"{location_id}_{scene_id}"
        output_dir = os.path.join(OUTPUT_ROOT, folder_name)

        if i % 500 == 0 or i == 1:
            print(f"[{i}/{len(tif_files)}] Processing: {folder_name}")

        try:
            ok, msg = process_scene(tif_path, output_dir)
            if ok:
                success += 1
            else:
                skipped += 1
                if os.path.exists(output_dir):
                    import shutil
                    shutil.rmtree(output_dir)
        except Exception as e:
            errors.append((folder_name, str(e)))
            skipped += 1

    print("\n" + "=" * 60)
    print("STEP 1 COMPLETE")
    print(f"  Valid scenes:   {success}")
    print(f"  Skipped:        {skipped}")
    print(f"  Errors:         {len(errors)}")
    print(f"  Output folder:  {OUTPUT_ROOT}")
    print("=" * 60)

    # Save scene list
    scenes = sorted([d for d in os.listdir(OUTPUT_ROOT) if os.path.isdir(os.path.join(OUTPUT_ROOT, d))])
    with open(os.path.join(OUTPUT_ROOT, "all_scenes.txt"), "w") as f:
        for s in scenes:
            f.write(s + "\n")
    print(f"Scene list saved: {len(scenes)} scenes in all_scenes.txt")


if __name__ == "__main__":
    main()
````

---

## Step 2: Train/Val/Test Split

````python
# filepath: c:\Users\Admin\Downloads\ssl4eo_l_oli_tirs_toa_benchmark\step2_split.py
import os
import random

DATASET_ROOT = r"C:\Users\Admin\Downloads\ssl4eo_l_oli_tirs_toa_benchmark\dataset"
SEED = 42
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1


def main():
    print("=" * 60)
    print("STEP 2: Train/Val/Test Split")
    print("=" * 60)

    scenes_file = os.path.join(DATASET_ROOT, "all_scenes.txt")
    with open(scenes_file, "r") as f:
        scenes = [line.strip() for line in f if line.strip()]

    print(f"Total scenes: {len(scenes)}")

    random.seed(SEED)
    random.shuffle(scenes)

    n = len(scenes)
    train_end = int(n * TRAIN_RATIO)
    val_end = train_end + int(n * VAL_RATIO)

    train_scenes = scenes[:train_end]
    val_scenes = scenes[train_end:val_end]
    test_scenes = scenes[val_end:]

    print(f"  Train: {len(train_scenes)}")
    print(f"  Val:   {len(val_scenes)}")
    print(f"  Test:  {len(test_scenes)}")

    # Save split files
    for split_name, split_scenes in [("train", train_scenes), ("val", val_scenes), ("test", test_scenes)]:
        split_file = os.path.join(DATASET_ROOT, f"{split_name}.txt")
        with open(split_file, "w") as f:
            for s in split_scenes:
                f.write(s + "\n")
        print(f"  Saved: {split_file}")

    print("\nDone!")


if __name__ == "__main__":
    main()
````

---

## Step 3: Compute Dataset Statistics

````python
# filepath: c:\Users\Admin\Downloads\ssl4eo_l_oli_tirs_toa_benchmark\step3_stats.py
import os
import numpy as np

DATASET_ROOT = r"C:\Users\Admin\Downloads\ssl4eo_l_oli_tirs_toa_benchmark\dataset"


def main():
    print("=" * 60)
    print("STEP 3: Compute Dataset Statistics")
    print("=" * 60)

    # Only use training set for computing stats
    train_file = os.path.join(DATASET_ROOT, "train.txt")
    with open(train_file, "r") as f:
        train_scenes = [line.strip() for line in f if line.strip()]

    print(f"Computing stats from {len(train_scenes)} training scenes...")

    # Running stats
    optical_sum = np.zeros(3, dtype=np.float64)
    optical_sq_sum = np.zeros(3, dtype=np.float64)
    thermal_sum = 0.0
    thermal_sq_sum = 0.0
    thermal_min = float('inf')
    thermal_max = float('-inf')
    optical_min = np.full(3, float('inf'))
    optical_max = np.full(3, float('-inf'))
    pixel_count = 0
    thermal_pixel_count = 0

    for i, scene in enumerate(train_scenes):
        if i % 1000 == 0:
            print(f"  [{i}/{len(train_scenes)}]")

        scene_dir = os.path.join(DATASET_ROOT, scene)
        optical = np.load(os.path.join(scene_dir, "optical_hr.npy"))      # (256, 256, 3)
        thermal = np.load(os.path.join(scene_dir, "thermal_hr.npy"))      # (256, 256)

        # Optical stats per channel
        for c in range(3):
            ch = optical[:, :, c].astype(np.float64)
            optical_sum[c] += ch.sum()
            optical_sq_sum[c] += (ch ** 2).sum()
            optical_min[c] = min(optical_min[c], ch.min())
            optical_max[c] = max(optical_max[c], ch.max())

        pixel_count += optical.shape[0] * optical.shape[1]

        # Thermal stats
        th = thermal.astype(np.float64)
        thermal_sum += th.sum()
        thermal_sq_sum += (th ** 2).sum()
        thermal_min = min(thermal_min, th.min())
        thermal_max = max(thermal_max, th.max())
        thermal_pixel_count += th.size

    # Compute mean and std
    optical_mean = optical_sum / pixel_count
    optical_std = np.sqrt(optical_sq_sum / pixel_count - optical_mean ** 2)
    thermal_mean = thermal_sum / thermal_pixel_count
    thermal_std = np.sqrt(thermal_sq_sum / thermal_pixel_count - thermal_mean ** 2)

    print("\n" + "=" * 60)
    print("DATASET STATISTICS (Training Set)")
    print("=" * 60)
    print(f"\nOptical (Bands 4, 3, 2):")
    print(f"  Mean: [{optical_mean[0]:.4f}, {optical_mean[1]:.4f}, {optical_mean[2]:.4f}]")
    print(f"  Std:  [{optical_std[0]:.4f}, {optical_std[1]:.4f}, {optical_std[2]:.4f}]")
    print(f"  Min:  [{optical_min[0]:.4f}, {optical_min[1]:.4f}, {optical_min[2]:.4f}]")
    print(f"  Max:  [{optical_max[0]:.4f}, {optical_max[1]:.4f}, {optical_max[2]:.4f}]")
    print(f"\nThermal (Band 10):")
    print(f"  Mean: {thermal_mean:.4f}")
    print(f"  Std:  {thermal_std:.4f}")
    print(f"  Min:  {thermal_min:.4f}")
    print(f"  Max:  {thermal_max:.4f}")

    # Save stats
    stats = {
        "optical_mean": optical_mean,
        "optical_std": optical_std,
        "optical_min": optical_min,
        "optical_max": optical_max,
        "thermal_mean": thermal_mean,
        "thermal_std": thermal_std,
        "thermal_min": thermal_min,
        "thermal_max": thermal_max,
    }
    np.save(os.path.join(DATASET_ROOT, "dataset_stats.npy"), stats)
    print(f"\nStats saved to: {os.path.join(DATASET_ROOT, 'dataset_stats.npy')}")


if __name__ == "__main__":
    main()
````

---

## Step 4: PyTorch Dataset & DataLoader

````python
# filepath: c:\Users\Admin\Downloads\ssl4eo_l_oli_tirs_toa_benchmark\step4_dataloader.py
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
````

---

## Step 5: Model Architecture

````python
# filepath: c:\Users\Admin\Downloads\ssl4eo_l_oli_tirs_toa_benchmark\step5_model.py
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
```


## License: unknown
https://github.com/Sumeet2807/Vid2Sound/blob/17e556d8a4446f4d7327c471129d5f54c938dc1f/model/c2d.py

```
# Complete Pipeline: Optical-Guided Thermal Super-Resolution

## Overview of All Steps

```
Step 1: Dataset Preprocessing (Create LR/HR pairs as .npy)
Step 2: Train/Val/Test Split (80/10/10)
Step 3: Dataset Statistics (Global min/max/mean/std for normalization)
Step 4: PyTorch Dataset & DataLoader
Step 5: Model Architecture (Optical-Guided Fusion SR Network)
Step 6: Training Loop (with PSNR/SSIM/RMSE tracking)
Step 7: Evaluation on Test Set
Step 8: Demo Script (Single image inference + visualization)
```

---

## Step 1: Dataset Preprocessing

This reads raw GeoTIFFs, crops to 256×256, creates LR thermal (64×64), and saves as `.npy` + preview `.png`.

````python
# filepath: c:\Users\Admin\Downloads\ssl4eo_l_oli_tirs_toa_benchmark\step1_preprocess.py
import os
import glob
import numpy as np
import rasterio
from PIL import Image
from scipy.ndimage import zoom

# === CONFIGURATION ===
DATASET_ROOT = r"C:\Users\Admin\Downloads\ssl4eo_l_oli_tirs_toa_benchmark"
OUTPUT_ROOT = os.path.join(DATASET_ROOT, "dataset")
SCALE_FACTOR = 4          # 4× super-resolution
CROP_SIZE = 256           # Target size (power of 2)
LR_SIZE = CROP_SIZE // SCALE_FACTOR  # 64×64

OPTICAL_BANDS = (4, 3, 2)  # Red, Green, Blue
THERMAL_BAND = 10           # TIRS-1


def center_crop(arr, size):
    """Center crop a 2D or 3D array to (size, size)."""
    if arr.ndim == 2:
        h, w = arr.shape
        top = (h - size) // 2
        left = (w - size) // 2
        return arr[top:top+size, left:left+size]
    elif arr.ndim == 3:
        h, w = arr.shape[1], arr.shape[2]
        top = (h - size) // 2
        left = (w - size) // 2
        return arr[:, top:top+size, left:left+size]


def normalize_for_preview(arr, percentile_low=2, percentile_high=98):
    """Normalize to uint8 for PNG preview only."""
    arr = arr.astype(np.float32)
    low = np.percentile(arr, percentile_low)
    high = np.percentile(arr, percentile_high)
    arr = np.clip(arr, low, high)
    if high - low == 0:
        return np.zeros_like(arr, dtype=np.uint8)
    return ((arr - low) / (high - low) * 255.0).astype(np.uint8)


def create_lr_thermal(hr_thermal, scale_factor):
    """Downsample HR thermal using bicubic interpolation."""
    lr = zoom(hr_thermal, 1.0 / scale_factor, order=3)  # order=3 = bicubic
    return lr


def process_scene(tif_path, output_dir):
    """Process one GeoTIFF into optical_hr, thermal_hr, thermal_lr."""
    with rasterio.open(tif_path) as src:
        if src.count < THERMAL_BAND:
            return False, "Not enough bands"

        # Read optical bands (4, 3, 2) as float32
        red = src.read(OPTICAL_BANDS[0]).astype(np.float32)
        green = src.read(OPTICAL_BANDS[1]).astype(np.float32)
        blue = src.read(OPTICAL_BANDS[2]).astype(np.float32)
        optical = np.stack([red, green, blue], axis=0)  # (3, H, W)

        # Read thermal band 10 as float32
        thermal = src.read(THERMAL_BAND).astype(np.float32)

    # Check for blank thermal
    if thermal.max() == 0 or np.mean(thermal == 0) > 0.5:
        return False, "Blank thermal"

    # Check for blank optical
    if optical.max() == 0:
        return False, "Blank optical"

    # Center crop to 256×256
    optical = center_crop(optical, CROP_SIZE)      # (3, 256, 256)
    thermal_hr = center_crop(thermal, CROP_SIZE)    # (256, 256)

    # Create LR thermal (64×64)
    thermal_lr = create_lr_thermal(thermal_hr, SCALE_FACTOR)  # (64, 64)

    # Convert optical to (H, W, C) for saving
    optical_hwc = np.transpose(optical, (1, 2, 0))  # (256, 256, 3)

    # Save as .npy (float32 — preserves raw values)
    os.makedirs(output_dir, exist_ok=True)
    np.save(os.path.join(output_dir, "optical_hr.npy"), optical_hwc)
    np.save(os.path.join(output_dir, "thermal_hr.npy"), thermal_hr)
    np.save(os.path.join(output_dir, "thermal_lr.npy"), thermal_lr)

    # Save preview PNGs
    # Optical preview
    opt_preview = np.stack([
        normalize_for_preview(optical_hwc[:, :, 0]),
        normalize_for_preview(optical_hwc[:, :, 1]),
        normalize_for_preview(optical_hwc[:, :, 2])
    ], axis=-1)
    Image.fromarray(opt_preview).save(os.path.join(output_dir, "optical_hr_preview.png"))

    # Thermal HR preview
    th_hr_preview = normalize_for_preview(thermal_hr, 1, 99)
    Image.fromarray(th_hr_preview, mode="L").save(os.path.join(output_dir, "thermal_hr_preview.png"))

    # Thermal LR preview
    th_lr_preview = normalize_for_preview(thermal_lr, 1, 99)
    Image.fromarray(th_lr_preview, mode="L").save(os.path.join(output_dir, "thermal_lr_preview.png"))

    return True, "OK"


def main():
    print("=" * 60)
    print("STEP 1: Dataset Preprocessing")
    print("=" * 60)
    print(f"Scale factor: {SCALE_FACTOR}×")
    print(f"HR size: {CROP_SIZE}×{CROP_SIZE}")
    print(f"LR size: {LR_SIZE}×{LR_SIZE}")

    search_pattern = os.path.join(DATASET_ROOT, "*", "*", "all_bands.tif")
    tif_files = sorted(glob.glob(search_pattern))

    # Exclude 'output' and 'dataset' folders
    tif_files = [f for f in tif_files if "\\output\\" not in f and "\\dataset\\" not in f]

    if not tif_files:
        print("[ERROR] No GeoTIFF files found!")
        return

    print(f"Found {len(tif_files)} scenes.")
    print(f"Output: {OUTPUT_ROOT}\n")

    os.makedirs(OUTPUT_ROOT, exist_ok=True)

    success = 0
    skipped = 0
    errors = []

    for i, tif_path in enumerate(tif_files, 1):
        parts = tif_path.split(os.sep)
        scene_id = parts[-2]
        location_id = parts[-3]
        folder_name = f"{location_id}_{scene_id}"
        output_dir = os.path.join(OUTPUT_ROOT, folder_name)

        if i % 500 == 0 or i == 1:
            print(f"[{i}/{len(tif_files)}] Processing: {folder_name}")

        try:
            ok, msg = process_scene(tif_path, output_dir)
            if ok:
                success += 1
            else:
                skipped += 1
                if os.path.exists(output_dir):
                    import shutil
                    shutil.rmtree(output_dir)
        except Exception as e:
            errors.append((folder_name, str(e)))
            skipped += 1

    print("\n" + "=" * 60)
    print("STEP 1 COMPLETE")
    print(f"  Valid scenes:   {success}")
    print(f"  Skipped:        {skipped}")
    print(f"  Errors:         {len(errors)}")
    print(f"  Output folder:  {OUTPUT_ROOT}")
    print("=" * 60)

    # Save scene list
    scenes = sorted([d for d in os.listdir(OUTPUT_ROOT) if os.path.isdir(os.path.join(OUTPUT_ROOT, d))])
    with open(os.path.join(OUTPUT_ROOT, "all_scenes.txt"), "w") as f:
        for s in scenes:
            f.write(s + "\n")
    print(f"Scene list saved: {len(scenes)} scenes in all_scenes.txt")


if __name__ == "__main__":
    main()
````

---

## Step 2: Train/Val/Test Split

````python
# filepath: c:\Users\Admin\Downloads\ssl4eo_l_oli_tirs_toa_benchmark\step2_split.py
import os
import random

DATASET_ROOT = r"C:\Users\Admin\Downloads\ssl4eo_l_oli_tirs_toa_benchmark\dataset"
SEED = 42
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1


def main():
    print("=" * 60)
    print("STEP 2: Train/Val/Test Split")
    print("=" * 60)

    scenes_file = os.path.join(DATASET_ROOT, "all_scenes.txt")
    with open(scenes_file, "r") as f:
        scenes = [line.strip() for line in f if line.strip()]

    print(f"Total scenes: {len(scenes)}")

    random.seed(SEED)
    random.shuffle(scenes)

    n = len(scenes)
    train_end = int(n * TRAIN_RATIO)
    val_end = train_end + int(n * VAL_RATIO)

    train_scenes = scenes[:train_end]
    val_scenes = scenes[train_end:val_end]
    test_scenes = scenes[val_end:]

    print(f"  Train: {len(train_scenes)}")
    print(f"  Val:   {len(val_scenes)}")
    print(f"  Test:  {len(test_scenes)}")

    # Save split files
    for split_name, split_scenes in [("train", train_scenes), ("val", val_scenes), ("test", test_scenes)]:
        split_file = os.path.join(DATASET_ROOT, f"{split_name}.txt")
        with open(split_file, "w") as f:
            for s in split_scenes:
                f.write(s + "\n")
        print(f"  Saved: {split_file}")

    print("\nDone!")


if __name__ == "__main__":
    main()
````

---

## Step 3: Compute Dataset Statistics

````python
# filepath: c:\Users\Admin\Downloads\ssl4eo_l_oli_tirs_toa_benchmark\step3_stats.py
import os
import numpy as np

DATASET_ROOT = r"C:\Users\Admin\Downloads\ssl4eo_l_oli_tirs_toa_benchmark\dataset"


def main():
    print("=" * 60)
    print("STEP 3: Compute Dataset Statistics")
    print("=" * 60)

    # Only use training set for computing stats
    train_file = os.path.join(DATASET_ROOT, "train.txt")
    with open(train_file, "r") as f:
        train_scenes = [line.strip() for line in f if line.strip()]

    print(f"Computing stats from {len(train_scenes)} training scenes...")

    # Running stats
    optical_sum = np.zeros(3, dtype=np.float64)
    optical_sq_sum = np.zeros(3, dtype=np.float64)
    thermal_sum = 0.0
    thermal_sq_sum = 0.0
    thermal_min = float('inf')
    thermal_max = float('-inf')
    optical_min = np.full(3, float('inf'))
    optical_max = np.full(3, float('-inf'))
    pixel_count = 0
    thermal_pixel_count = 0

    for i, scene in enumerate(train_scenes):
        if i % 1000 == 0:
            print(f"  [{i}/{len(train_scenes)}]")

        scene_dir = os.path.join(DATASET_ROOT, scene)
        optical = np.load(os.path.join(scene_dir, "optical_hr.npy"))      # (256, 256, 3)
        thermal = np.load(os.path.join(scene_dir, "thermal_hr.npy"))      # (256, 256)

        # Optical stats per channel
        for c in range(3):
            ch = optical[:, :, c].astype(np.float64)
            optical_sum[c] += ch.sum()
            optical_sq_sum[c] += (ch ** 2).sum()
            optical_min[c] = min(optical_min[c], ch.min())
            optical_max[c] = max(optical_max[c], ch.max())

        pixel_count += optical.shape[0] * optical.shape[1]

        # Thermal stats
        th = thermal.astype(np.float64)
        thermal_sum += th.sum()
        thermal_sq_sum += (th ** 2).sum()
        thermal_min = min(thermal_min, th.min())
        thermal_max = max(thermal_max, th.max())
        thermal_pixel_count += th.size

    # Compute mean and std
    optical_mean = optical_sum / pixel_count
    optical_std = np.sqrt(optical_sq_sum / pixel_count - optical_mean ** 2)
    thermal_mean = thermal_sum / thermal_pixel_count
    thermal_std = np.sqrt(thermal_sq_sum / thermal_pixel_count - thermal_mean ** 2)

    print("\n" + "=" * 60)
    print("DATASET STATISTICS (Training Set)")
    print("=" * 60)
    print(f"\nOptical (Bands 4, 3, 2):")
    print(f"  Mean: [{optical_mean[0]:.4f}, {optical_mean[1]:.4f}, {optical_mean[2]:.4f}]")
    print(f"  Std:  [{optical_std[0]:.4f}, {optical_std[1]:.4f}, {optical_std[2]:.4f}]")
    print(f"  Min:  [{optical_min[0]:.4f}, {optical_min[1]:.4f}, {optical_min[2]:.4f}]")
    print(f"  Max:  [{optical_max[0]:.4f}, {optical_max[1]:.4f}, {optical_max[2]:.4f}]")
    print(f"\nThermal (Band 10):")
    print(f"  Mean: {thermal_mean:.4f}")
    print(f"  Std:  {thermal_std:.4f}")
    print(f"  Min:  {thermal_min:.4f}")
    print(f"  Max:  {thermal_max:.4f}")

    # Save stats
    stats = {
        "optical_mean": optical_mean,
        "optical_std": optical_std,
        "optical_min": optical_min,
        "optical_max": optical_max,
        "thermal_mean": thermal_mean,
        "thermal_std": thermal_std,
        "thermal_min": thermal_min,
        "thermal_max": thermal_max,
    }
    np.save(os.path.join(DATASET_ROOT, "dataset_stats.npy"), stats)
    print(f"\nStats saved to: {os.path.join(DATASET_ROOT, 'dataset_stats.npy')}")


if __name__ == "__main__":
    main()
````

---

## Step 4: PyTorch Dataset & DataLoader

````python
# filepath: c:\Users\Admin\Downloads\ssl4eo_l_oli_tirs_toa_benchmark\step4_dataloader.py
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
````

---

## Step 5: Model Architecture

````python
# filepath: c:\Users\Admin\Downloads\ssl4eo_l_oli_tirs_toa_benchmark\step5_model.py
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
```


## License: unknown
https://github.com/Sumeet2807/Vid2Sound/blob/17e556d8a4446f4d7327c471129d5f54c938dc1f/model/c2d.py

```
# Complete Pipeline: Optical-Guided Thermal Super-Resolution

## Overview of All Steps

```
Step 1: Dataset Preprocessing (Create LR/HR pairs as .npy)
Step 2: Train/Val/Test Split (80/10/10)
Step 3: Dataset Statistics (Global min/max/mean/std for normalization)
Step 4: PyTorch Dataset & DataLoader
Step 5: Model Architecture (Optical-Guided Fusion SR Network)
Step 6: Training Loop (with PSNR/SSIM/RMSE tracking)
Step 7: Evaluation on Test Set
Step 8: Demo Script (Single image inference + visualization)
```

---

## Step 1: Dataset Preprocessing

This reads raw GeoTIFFs, crops to 256×256, creates LR thermal (64×64), and saves as `.npy` + preview `.png`.

````python
# filepath: c:\Users\Admin\Downloads\ssl4eo_l_oli_tirs_toa_benchmark\step1_preprocess.py
import os
import glob
import numpy as np
import rasterio
from PIL import Image
from scipy.ndimage import zoom

# === CONFIGURATION ===
DATASET_ROOT = r"C:\Users\Admin\Downloads\ssl4eo_l_oli_tirs_toa_benchmark"
OUTPUT_ROOT = os.path.join(DATASET_ROOT, "dataset")
SCALE_FACTOR = 4          # 4× super-resolution
CROP_SIZE = 256           # Target size (power of 2)
LR_SIZE = CROP_SIZE // SCALE_FACTOR  # 64×64

OPTICAL_BANDS = (4, 3, 2)  # Red, Green, Blue
THERMAL_BAND = 10           # TIRS-1


def center_crop(arr, size):
    """Center crop a 2D or 3D array to (size, size)."""
    if arr.ndim == 2:
        h, w = arr.shape
        top = (h - size) // 2
        left = (w - size) // 2
        return arr[top:top+size, left:left+size]
    elif arr.ndim == 3:
        h, w = arr.shape[1], arr.shape[2]
        top = (h - size) // 2
        left = (w - size) // 2
        return arr[:, top:top+size, left:left+size]


def normalize_for_preview(arr, percentile_low=2, percentile_high=98):
    """Normalize to uint8 for PNG preview only."""
    arr = arr.astype(np.float32)
    low = np.percentile(arr, percentile_low)
    high = np.percentile(arr, percentile_high)
    arr = np.clip(arr, low, high)
    if high - low == 0:
        return np.zeros_like(arr, dtype=np.uint8)
    return ((arr - low) / (high - low) * 255.0).astype(np.uint8)


def create_lr_thermal(hr_thermal, scale_factor):
    """Downsample HR thermal using bicubic interpolation."""
    lr = zoom(hr_thermal, 1.0 / scale_factor, order=3)  # order=3 = bicubic
    return lr


def process_scene(tif_path, output_dir):
    """Process one GeoTIFF into optical_hr, thermal_hr, thermal_lr."""
    with rasterio.open(tif_path) as src:
        if src.count < THERMAL_BAND:
            return False, "Not enough bands"

        # Read optical bands (4, 3, 2) as float32
        red = src.read(OPTICAL_BANDS[0]).astype(np.float32)
        green = src.read(OPTICAL_BANDS[1]).astype(np.float32)
        blue = src.read(OPTICAL_BANDS[2]).astype(np.float32)
        optical = np.stack([red, green, blue], axis=0)  # (3, H, W)

        # Read thermal band 10 as float32
        thermal = src.read(THERMAL_BAND).astype(np.float32)

    # Check for blank thermal
    if thermal.max() == 0 or np.mean(thermal == 0) > 0.5:
        return False, "Blank thermal"

    # Check for blank optical
    if optical.max() == 0:
        return False, "Blank optical"

    # Center crop to 256×256
    optical = center_crop(optical, CROP_SIZE)      # (3, 256, 256)
    thermal_hr = center_crop(thermal, CROP_SIZE)    # (256, 256)

    # Create LR thermal (64×64)
    thermal_lr = create_lr_thermal(thermal_hr, SCALE_FACTOR)  # (64, 64)

    # Convert optical to (H, W, C) for saving
    optical_hwc = np.transpose(optical, (1, 2, 0))  # (256, 256, 3)

    # Save as .npy (float32 — preserves raw values)
    os.makedirs(output_dir, exist_ok=True)
    np.save(os.path.join(output_dir, "optical_hr.npy"), optical_hwc)
    np.save(os.path.join(output_dir, "thermal_hr.npy"), thermal_hr)
    np.save(os.path.join(output_dir, "thermal_lr.npy"), thermal_lr)

    # Save preview PNGs
    # Optical preview
    opt_preview = np.stack([
        normalize_for_preview(optical_hwc[:, :, 0]),
        normalize_for_preview(optical_hwc[:, :, 1]),
        normalize_for_preview(optical_hwc[:, :, 2])
    ], axis=-1)
    Image.fromarray(opt_preview).save(os.path.join(output_dir, "optical_hr_preview.png"))

    # Thermal HR preview
    th_hr_preview = normalize_for_preview(thermal_hr, 1, 99)
    Image.fromarray(th_hr_preview, mode="L").save(os.path.join(output_dir, "thermal_hr_preview.png"))

    # Thermal LR preview
    th_lr_preview = normalize_for_preview(thermal_lr, 1, 99)
    Image.fromarray(th_lr_preview, mode="L").save(os.path.join(output_dir, "thermal_lr_preview.png"))

    return True, "OK"


def main():
    print("=" * 60)
    print("STEP 1: Dataset Preprocessing")
    print("=" * 60)
    print(f"Scale factor: {SCALE_FACTOR}×")
    print(f"HR size: {CROP_SIZE}×{CROP_SIZE}")
    print(f"LR size: {LR_SIZE}×{LR_SIZE}")

    search_pattern = os.path.join(DATASET_ROOT, "*", "*", "all_bands.tif")
    tif_files = sorted(glob.glob(search_pattern))

    # Exclude 'output' and 'dataset' folders
    tif_files = [f for f in tif_files if "\\output\\" not in f and "\\dataset\\" not in f]

    if not tif_files:
        print("[ERROR] No GeoTIFF files found!")
        return

    print(f"Found {len(tif_files)} scenes.")
    print(f"Output: {OUTPUT_ROOT}\n")

    os.makedirs(OUTPUT_ROOT, exist_ok=True)

    success = 0
    skipped = 0
    errors = []

    for i, tif_path in enumerate(tif_files, 1):
        parts = tif_path.split(os.sep)
        scene_id = parts[-2]
        location_id = parts[-3]
        folder_name = f"{location_id}_{scene_id}"
        output_dir = os.path.join(OUTPUT_ROOT, folder_name)

        if i % 500 == 0 or i == 1:
            print(f"[{i}/{len(tif_files)}] Processing: {folder_name}")

        try:
            ok, msg = process_scene(tif_path, output_dir)
            if ok:
                success += 1
            else:
                skipped += 1
                if os.path.exists(output_dir):
                    import shutil
                    shutil.rmtree(output_dir)
        except Exception as e:
            errors.append((folder_name, str(e)))
            skipped += 1

    print("\n" + "=" * 60)
    print("STEP 1 COMPLETE")
    print(f"  Valid scenes:   {success}")
    print(f"  Skipped:        {skipped}")
    print(f"  Errors:         {len(errors)}")
    print(f"  Output folder:  {OUTPUT_ROOT}")
    print("=" * 60)

    # Save scene list
    scenes = sorted([d for d in os.listdir(OUTPUT_ROOT) if os.path.isdir(os.path.join(OUTPUT_ROOT, d))])
    with open(os.path.join(OUTPUT_ROOT, "all_scenes.txt"), "w") as f:
        for s in scenes:
            f.write(s + "\n")
    print(f"Scene list saved: {len(scenes)} scenes in all_scenes.txt")


if __name__ == "__main__":
    main()
````

---

## Step 2: Train/Val/Test Split

````python
# filepath: c:\Users\Admin\Downloads\ssl4eo_l_oli_tirs_toa_benchmark\step2_split.py
import os
import random

DATASET_ROOT = r"C:\Users\Admin\Downloads\ssl4eo_l_oli_tirs_toa_benchmark\dataset"
SEED = 42
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1


def main():
    print("=" * 60)
    print("STEP 2: Train/Val/Test Split")
    print("=" * 60)

    scenes_file = os.path.join(DATASET_ROOT, "all_scenes.txt")
    with open(scenes_file, "r") as f:
        scenes = [line.strip() for line in f if line.strip()]

    print(f"Total scenes: {len(scenes)}")

    random.seed(SEED)
    random.shuffle(scenes)

    n = len(scenes)
    train_end = int(n * TRAIN_RATIO)
    val_end = train_end + int(n * VAL_RATIO)

    train_scenes = scenes[:train_end]
    val_scenes = scenes[train_end:val_end]
    test_scenes = scenes[val_end:]

    print(f"  Train: {len(train_scenes)}")
    print(f"  Val:   {len(val_scenes)}")
    print(f"  Test:  {len(test_scenes)}")

    # Save split files
    for split_name, split_scenes in [("train", train_scenes), ("val", val_scenes), ("test", test_scenes)]:
        split_file = os.path.join(DATASET_ROOT, f"{split_name}.txt")
        with open(split_file, "w") as f:
            for s in split_scenes:
                f.write(s + "\n")
        print(f"  Saved: {split_file}")

    print("\nDone!")


if __name__ == "__main__":
    main()
````

---

## Step 3: Compute Dataset Statistics

````python
# filepath: c:\Users\Admin\Downloads\ssl4eo_l_oli_tirs_toa_benchmark\step3_stats.py
import os
import numpy as np

DATASET_ROOT = r"C:\Users\Admin\Downloads\ssl4eo_l_oli_tirs_toa_benchmark\dataset"


def main():
    print("=" * 60)
    print("STEP 3: Compute Dataset Statistics")
    print("=" * 60)

    # Only use training set for computing stats
    train_file = os.path.join(DATASET_ROOT, "train.txt")
    with open(train_file, "r") as f:
        train_scenes = [line.strip() for line in f if line.strip()]

    print(f"Computing stats from {len(train_scenes)} training scenes...")

    # Running stats
    optical_sum = np.zeros(3, dtype=np.float64)
    optical_sq_sum = np.zeros(3, dtype=np.float64)
    thermal_sum = 0.0
    thermal_sq_sum = 0.0
    thermal_min = float('inf')
    thermal_max = float('-inf')
    optical_min = np.full(3, float('inf'))
    optical_max = np.full(3, float('-inf'))
    pixel_count = 0
    thermal_pixel_count = 0

    for i, scene in enumerate(train_scenes):
        if i % 1000 == 0:
            print(f"  [{i}/{len(train_scenes)}]")

        scene_dir = os.path.join(DATASET_ROOT, scene)
        optical = np.load(os.path.join(scene_dir, "optical_hr.npy"))      # (256, 256, 3)
        thermal = np.load(os.path.join(scene_dir, "thermal_hr.npy"))      # (256, 256)

        # Optical stats per channel
        for c in range(3):
            ch = optical[:, :, c].astype(np.float64)
            optical_sum[c] += ch.sum()
            optical_sq_sum[c] += (ch ** 2).sum()
            optical_min[c] = min(optical_min[c], ch.min())
            optical_max[c] = max(optical_max[c], ch.max())

        pixel_count += optical.shape[0] * optical.shape[1]

        # Thermal stats
        th = thermal.astype(np.float64)
        thermal_sum += th.sum()
        thermal_sq_sum += (th ** 2).sum()
        thermal_min = min(thermal_min, th.min())
        thermal_max = max(thermal_max, th.max())
        thermal_pixel_count += th.size

    # Compute mean and std
    optical_mean = optical_sum / pixel_count
    optical_std = np.sqrt(optical_sq_sum / pixel_count - optical_mean ** 2)
    thermal_mean = thermal_sum / thermal_pixel_count
    thermal_std = np.sqrt(thermal_sq_sum / thermal_pixel_count - thermal_mean ** 2)

    print("\n" + "=" * 60)
    print("DATASET STATISTICS (Training Set)")
    print("=" * 60)
    print(f"\nOptical (Bands 4, 3, 2):")
    print(f"  Mean: [{optical_mean[0]:.4f}, {optical_mean[1]:.4f}, {optical_mean[2]:.4f}]")
    print(f"  Std:  [{optical_std[0]:.4f}, {optical_std[1]:.4f}, {optical_std[2]:.4f}]")
    print(f"  Min:  [{optical_min[0]:.4f}, {optical_min[1]:.4f}, {optical_min[2]:.4f}]")
    print(f"  Max:  [{optical_max[0]:.4f}, {optical_max[1]:.4f}, {optical_max[2]:.4f}]")
    print(f"\nThermal (Band 10):")
    print(f"  Mean: {thermal_mean:.4f}")
    print(f"  Std:  {thermal_std:.4f}")
    print(f"  Min:  {thermal_min:.4f}")
    print(f"  Max:  {thermal_max:.4f}")

    # Save stats
    stats = {
        "optical_mean": optical_mean,
        "optical_std": optical_std,
        "optical_min": optical_min,
        "optical_max": optical_max,
        "thermal_mean": thermal_mean,
        "thermal_std": thermal_std,
        "thermal_min": thermal_min,
        "thermal_max": thermal_max,
    }
    np.save(os.path.join(DATASET_ROOT, "dataset_stats.npy"), stats)
    print(f"\nStats saved to: {os.path.join(DATASET_ROOT, 'dataset_stats.npy')}")


if __name__ == "__main__":
    main()
````

---

## Step 4: PyTorch Dataset & DataLoader

````python
# filepath: c:\Users\Admin\Downloads\ssl4eo_l_oli_tirs_toa_benchmark\step4_dataloader.py
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
````

---

## Step 5: Model Architecture

````python
# filepath: c:\Users\Admin\Downloads\ssl4eo_l_oli_tirs_toa_benchmark\step5_model.py
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
```


## License: unknown
https://github.com/Sumeet2807/Vid2Sound/blob/17e556d8a4446f4d7327c471129d5f54c938dc1f/model/c2d.py

```
# Complete Pipeline: Optical-Guided Thermal Super-Resolution

## Overview of All Steps

```
Step 1: Dataset Preprocessing (Create LR/HR pairs as .npy)
Step 2: Train/Val/Test Split (80/10/10)
Step 3: Dataset Statistics (Global min/max/mean/std for normalization)
Step 4: PyTorch Dataset & DataLoader
Step 5: Model Architecture (Optical-Guided Fusion SR Network)
Step 6: Training Loop (with PSNR/SSIM/RMSE tracking)
Step 7: Evaluation on Test Set
Step 8: Demo Script (Single image inference + visualization)
```

---

## Step 1: Dataset Preprocessing

This reads raw GeoTIFFs, crops to 256×256, creates LR thermal (64×64), and saves as `.npy` + preview `.png`.

````python
# filepath: c:\Users\Admin\Downloads\ssl4eo_l_oli_tirs_toa_benchmark\step1_preprocess.py
import os
import glob
import numpy as np
import rasterio
from PIL import Image
from scipy.ndimage import zoom

# === CONFIGURATION ===
DATASET_ROOT = r"C:\Users\Admin\Downloads\ssl4eo_l_oli_tirs_toa_benchmark"
OUTPUT_ROOT = os.path.join(DATASET_ROOT, "dataset")
SCALE_FACTOR = 4          # 4× super-resolution
CROP_SIZE = 256           # Target size (power of 2)
LR_SIZE = CROP_SIZE // SCALE_FACTOR  # 64×64

OPTICAL_BANDS = (4, 3, 2)  # Red, Green, Blue
THERMAL_BAND = 10           # TIRS-1


def center_crop(arr, size):
    """Center crop a 2D or 3D array to (size, size)."""
    if arr.ndim == 2:
        h, w = arr.shape
        top = (h - size) // 2
        left = (w - size) // 2
        return arr[top:top+size, left:left+size]
    elif arr.ndim == 3:
        h, w = arr.shape[1], arr.shape[2]
        top = (h - size) // 2
        left = (w - size) // 2
        return arr[:, top:top+size, left:left+size]


def normalize_for_preview(arr, percentile_low=2, percentile_high=98):
    """Normalize to uint8 for PNG preview only."""
    arr = arr.astype(np.float32)
    low = np.percentile(arr, percentile_low)
    high = np.percentile(arr, percentile_high)
    arr = np.clip(arr, low, high)
    if high - low == 0:
        return np.zeros_like(arr, dtype=np.uint8)
    return ((arr - low) / (high - low) * 255.0).astype(np.uint8)


def create_lr_thermal(hr_thermal, scale_factor):
    """Downsample HR thermal using bicubic interpolation."""
    lr = zoom(hr_thermal, 1.0 / scale_factor, order=3)  # order=3 = bicubic
    return lr


def process_scene(tif_path, output_dir):
    """Process one GeoTIFF into optical_hr, thermal_hr, thermal_lr."""
    with rasterio.open(tif_path) as src:
        if src.count < THERMAL_BAND:
            return False, "Not enough bands"

        # Read optical bands (4, 3, 2) as float32
        red = src.read(OPTICAL_BANDS[0]).astype(np.float32)
        green = src.read(OPTICAL_BANDS[1]).astype(np.float32)
        blue = src.read(OPTICAL_BANDS[2]).astype(np.float32)
        optical = np.stack([red, green, blue], axis=0)  # (3, H, W)

        # Read thermal band 10 as float32
        thermal = src.read(THERMAL_BAND).astype(np.float32)

    # Check for blank thermal
    if thermal.max() == 0 or np.mean(thermal == 0) > 0.5:
        return False, "Blank thermal"

    # Check for blank optical
    if optical.max() == 0:
        return False, "Blank optical"

    # Center crop to 256×256
    optical = center_crop(optical, CROP_SIZE)      # (3, 256, 256)
    thermal_hr = center_crop(thermal, CROP_SIZE)    # (256, 256)

    # Create LR thermal (64×64)
    thermal_lr = create_lr_thermal(thermal_hr, SCALE_FACTOR)  # (64, 64)

    # Convert optical to (H, W, C) for saving
    optical_hwc = np.transpose(optical, (1, 2, 0))  # (256, 256, 3)

    # Save as .npy (float32 — preserves raw values)
    os.makedirs(output_dir, exist_ok=True)
    np.save(os.path.join(output_dir, "optical_hr.npy"), optical_hwc)
    np.save(os.path.join(output_dir, "thermal_hr.npy"), thermal_hr)
    np.save(os.path.join(output_dir, "thermal_lr.npy"), thermal_lr)

    # Save preview PNGs
    # Optical preview
    opt_preview = np.stack([
        normalize_for_preview(optical_hwc[:, :, 0]),
        normalize_for_preview(optical_hwc[:, :, 1]),
        normalize_for_preview(optical_hwc[:, :, 2])
    ], axis=-1)
    Image.fromarray(opt_preview).save(os.path.join(output_dir, "optical_hr_preview.png"))

    # Thermal HR preview
    th_hr_preview = normalize_for_preview(thermal_hr, 1, 99)
    Image.fromarray(th_hr_preview, mode="L").save(os.path.join(output_dir, "thermal_hr_preview.png"))

    # Thermal LR preview
    th_lr_preview = normalize_for_preview(thermal_lr, 1, 99)
    Image.fromarray(th_lr_preview, mode="L").save(os.path.join(output_dir, "thermal_lr_preview.png"))

    return True, "OK"


def main():
    print("=" * 60)
    print("STEP 1: Dataset Preprocessing")
    print("=" * 60)
    print(f"Scale factor: {SCALE_FACTOR}×")
    print(f"HR size: {CROP_SIZE}×{CROP_SIZE}")
    print(f"LR size: {LR_SIZE}×{LR_SIZE}")

    search_pattern = os.path.join(DATASET_ROOT, "*", "*", "all_bands.tif")
    tif_files = sorted(glob.glob(search_pattern))

    # Exclude 'output' and 'dataset' folders
    tif_files = [f for f in tif_files if "\\output\\" not in f and "\\dataset\\" not in f]

    if not tif_files:
        print("[ERROR] No GeoTIFF files found!")
        return

    print(f"Found {len(tif_files)} scenes.")
    print(f"Output: {OUTPUT_ROOT}\n")

    os.makedirs(OUTPUT_ROOT, exist_ok=True)

    success = 0
    skipped = 0
    errors = []

    for i, tif_path in enumerate(tif_files, 1):
        parts = tif_path.split(os.sep)
        scene_id = parts[-2]
        location_id = parts[-3]
        folder_name = f"{location_id}_{scene_id}"
        output_dir = os.path.join(OUTPUT_ROOT, folder_name)

        if i % 500 == 0 or i == 1:
            print(f"[{i}/{len(tif_files)}] Processing: {folder_name}")

        try:
            ok, msg = process_scene(tif_path, output_dir)
            if ok:
                success += 1
            else:
                skipped += 1
                if os.path.exists(output_dir):
                    import shutil
                    shutil.rmtree(output_dir)
        except Exception as e:
            errors.append((folder_name, str(e)))
            skipped += 1

    print("\n" + "=" * 60)
    print("STEP 1 COMPLETE")
    print(f"  Valid scenes:   {success}")
    print(f"  Skipped:        {skipped}")
    print(f"  Errors:         {len(errors)}")
    print(f"  Output folder:  {OUTPUT_ROOT}")
    print("=" * 60)

    # Save scene list
    scenes = sorted([d for d in os.listdir(OUTPUT_ROOT) if os.path.isdir(os.path.join(OUTPUT_ROOT, d))])
    with open(os.path.join(OUTPUT_ROOT, "all_scenes.txt"), "w") as f:
        for s in scenes:
            f.write(s + "\n")
    print(f"Scene list saved: {len(scenes)} scenes in all_scenes.txt")


if __name__ == "__main__":
    main()
````

---

## Step 2: Train/Val/Test Split

````python
# filepath: c:\Users\Admin\Downloads\ssl4eo_l_oli_tirs_toa_benchmark\step2_split.py
import os
import random

DATASET_ROOT = r"C:\Users\Admin\Downloads\ssl4eo_l_oli_tirs_toa_benchmark\dataset"
SEED = 42
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1


def main():
    print("=" * 60)
    print("STEP 2: Train/Val/Test Split")
    print("=" * 60)

    scenes_file = os.path.join(DATASET_ROOT, "all_scenes.txt")
    with open(scenes_file, "r") as f:
        scenes = [line.strip() for line in f if line.strip()]

    print(f"Total scenes: {len(scenes)}")

    random.seed(SEED)
    random.shuffle(scenes)

    n = len(scenes)
    train_end = int(n * TRAIN_RATIO)
    val_end = train_end + int(n * VAL_RATIO)

    train_scenes = scenes[:train_end]
    val_scenes = scenes[train_end:val_end]
    test_scenes = scenes[val_end:]

    print(f"  Train: {len(train_scenes)}")
    print(f"  Val:   {len(val_scenes)}")
    print(f"  Test:  {len(test_scenes)}")

    # Save split files
    for split_name, split_scenes in [("train", train_scenes), ("val", val_scenes), ("test", test_scenes)]:
        split_file = os.path.join(DATASET_ROOT, f"{split_name}.txt")
        with open(split_file, "w") as f:
            for s in split_scenes:
                f.write(s + "\n")
        print(f"  Saved: {split_file}")

    print("\nDone!")


if __name__ == "__main__":
    main()
````

---

## Step 3: Compute Dataset Statistics

````python
# filepath: c:\Users\Admin\Downloads\ssl4eo_l_oli_tirs_toa_benchmark\step3_stats.py
import os
import numpy as np

DATASET_ROOT = r"C:\Users\Admin\Downloads\ssl4eo_l_oli_tirs_toa_benchmark\dataset"


def main():
    print("=" * 60)
    print("STEP 3: Compute Dataset Statistics")
    print("=" * 60)

    # Only use training set for computing stats
    train_file = os.path.join(DATASET_ROOT, "train.txt")
    with open(train_file, "r") as f:
        train_scenes = [line.strip() for line in f if line.strip()]

    print(f"Computing stats from {len(train_scenes)} training scenes...")

    # Running stats
    optical_sum = np.zeros(3, dtype=np.float64)
    optical_sq_sum = np.zeros(3, dtype=np.float64)
    thermal_sum = 0.0
    thermal_sq_sum = 0.0
    thermal_min = float('inf')
    thermal_max = float('-inf')
    optical_min = np.full(3, float('inf'))
    optical_max = np.full(3, float('-inf'))
    pixel_count = 0
    thermal_pixel_count = 0

    for i, scene in enumerate(train_scenes):
        if i % 1000 == 0:
            print(f"  [{i}/{len(train_scenes)}]")

        scene_dir = os.path.join(DATASET_ROOT, scene)
        optical = np.load(os.path.join(scene_dir, "optical_hr.npy"))      # (256, 256, 3)
        thermal = np.load(os.path.join(scene_dir, "thermal_hr.npy"))      # (256, 256)

        # Optical stats per channel
        for c in range(3):
            ch = optical[:, :, c].astype(np.float64)
            optical_sum[c] += ch.sum()
            optical_sq_sum[c] += (ch ** 2).sum()
            optical_min[c] = min(optical_min[c], ch.min())
            optical_max[c] = max(optical_max[c], ch.max())

        pixel_count += optical.shape[0] * optical.shape[1]

        # Thermal stats
        th = thermal.astype(np.float64)
        thermal_sum += th.sum()
        thermal_sq_sum += (th ** 2).sum()
        thermal_min = min(thermal_min, th.min())
        thermal_max = max(thermal_max, th.max())
        thermal_pixel_count += th.size

    # Compute mean and std
    optical_mean = optical_sum / pixel_count
    optical_std = np.sqrt(optical_sq_sum / pixel_count - optical_mean ** 2)
    thermal_mean = thermal_sum / thermal_pixel_count
    thermal_std = np.sqrt(thermal_sq_sum / thermal_pixel_count - thermal_mean ** 2)

    print("\n" + "=" * 60)
    print("DATASET STATISTICS (Training Set)")
    print("=" * 60)
    print(f"\nOptical (Bands 4, 3, 2):")
    print(f"  Mean: [{optical_mean[0]:.4f}, {optical_mean[1]:.4f}, {optical_mean[2]:.4f}]")
    print(f"  Std:  [{optical_std[0]:.4f}, {optical_std[1]:.4f}, {optical_std[2]:.4f}]")
    print(f"  Min:  [{optical_min[0]:.4f}, {optical_min[1]:.4f}, {optical_min[2]:.4f}]")
    print(f"  Max:  [{optical_max[0]:.4f}, {optical_max[1]:.4f}, {optical_max[2]:.4f}]")
    print(f"\nThermal (Band 10):")
    print(f"  Mean: {thermal_mean:.4f}")
    print(f"  Std:  {thermal_std:.4f}")
    print(f"  Min:  {thermal_min:.4f}")
    print(f"  Max:  {thermal_max:.4f}")

    # Save stats
    stats = {
        "optical_mean": optical_mean,
        "optical_std": optical_std,
        "optical_min": optical_min,
        "optical_max": optical_max,
        "thermal_mean": thermal_mean,
        "thermal_std": thermal_std,
        "thermal_min": thermal_min,
        "thermal_max": thermal_max,
    }
    np.save(os.path.join(DATASET_ROOT, "dataset_stats.npy"), stats)
    print(f"\nStats saved to: {os.path.join(DATASET_ROOT, 'dataset_stats.npy')}")


if __name__ == "__main__":
    main()
````

---

## Step 4: PyTorch Dataset & DataLoader

````python
# filepath: c:\Users\Admin\Downloads\ssl4eo_l_oli_tirs_toa_benchmark\step4_dataloader.py
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
````

---

## Step 5: Model Architecture

````python
# filepath: c:\Users\Admin\Downloads\ssl4eo_l_oli_tirs_toa_benchmark\step5_model.py
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
```


## License: unknown
https://github.com/Sumeet2807/Vid2Sound/blob/17e556d8a4446f4d7327c471129d5f54c938dc1f/model/c2d.py

```
# Complete Pipeline: Optical-Guided Thermal Super-Resolution

## Overview of All Steps

```
Step 1: Dataset Preprocessing (Create LR/HR pairs as .npy)
Step 2: Train/Val/Test Split (80/10/10)
Step 3: Dataset Statistics (Global min/max/mean/std for normalization)
Step 4: PyTorch Dataset & DataLoader
Step 5: Model Architecture (Optical-Guided Fusion SR Network)
Step 6: Training Loop (with PSNR/SSIM/RMSE tracking)
Step 7: Evaluation on Test Set
Step 8: Demo Script (Single image inference + visualization)
```

---

## Step 1: Dataset Preprocessing

This reads raw GeoTIFFs, crops to 256×256, creates LR thermal (64×64), and saves as `.npy` + preview `.png`.

````python
# filepath: c:\Users\Admin\Downloads\ssl4eo_l_oli_tirs_toa_benchmark\step1_preprocess.py
import os
import glob
import numpy as np
import rasterio
from PIL import Image
from scipy.ndimage import zoom

# === CONFIGURATION ===
DATASET_ROOT = r"C:\Users\Admin\Downloads\ssl4eo_l_oli_tirs_toa_benchmark"
OUTPUT_ROOT = os.path.join(DATASET_ROOT, "dataset")
SCALE_FACTOR = 4          # 4× super-resolution
CROP_SIZE = 256           # Target size (power of 2)
LR_SIZE = CROP_SIZE // SCALE_FACTOR  # 64×64

OPTICAL_BANDS = (4, 3, 2)  # Red, Green, Blue
THERMAL_BAND = 10           # TIRS-1


def center_crop(arr, size):
    """Center crop a 2D or 3D array to (size, size)."""
    if arr.ndim == 2:
        h, w = arr.shape
        top = (h - size) // 2
        left = (w - size) // 2
        return arr[top:top+size, left:left+size]
    elif arr.ndim == 3:
        h, w = arr.shape[1], arr.shape[2]
        top = (h - size) // 2
        left = (w - size) // 2
        return arr[:, top:top+size, left:left+size]


def normalize_for_preview(arr, percentile_low=2, percentile_high=98):
    """Normalize to uint8 for PNG preview only."""
    arr = arr.astype(np.float32)
    low = np.percentile(arr, percentile_low)
    high = np.percentile(arr, percentile_high)
    arr = np.clip(arr, low, high)
    if high - low == 0:
        return np.zeros_like(arr, dtype=np.uint8)
    return ((arr - low) / (high - low) * 255.0).astype(np.uint8)


def create_lr_thermal(hr_thermal, scale_factor):
    """Downsample HR thermal using bicubic interpolation."""
    lr = zoom(hr_thermal, 1.0 / scale_factor, order=3)  # order=3 = bicubic
    return lr


def process_scene(tif_path, output_dir):
    """Process one GeoTIFF into optical_hr, thermal_hr, thermal_lr."""
    with rasterio.open(tif_path) as src:
        if src.count < THERMAL_BAND:
            return False, "Not enough bands"

        # Read optical bands (4, 3, 2) as float32
        red = src.read(OPTICAL_BANDS[0]).astype(np.float32)
        green = src.read(OPTICAL_BANDS[1]).astype(np.float32)
        blue = src.read(OPTICAL_BANDS[2]).astype(np.float32)
        optical = np.stack([red, green, blue], axis=0)  # (3, H, W)

        # Read thermal band 10 as float32
        thermal = src.read(THERMAL_BAND).astype(np.float32)

    # Check for blank thermal
    if thermal.max() == 0 or np.mean(thermal == 0) > 0.5:
        return False, "Blank thermal"

    # Check for blank optical
    if optical.max() == 0:
        return False, "Blank optical"

    # Center crop to 256×256
    optical = center_crop(optical, CROP_SIZE)      # (3, 256, 256)
    thermal_hr = center_crop(thermal, CROP_SIZE)    # (256, 256)

    # Create LR thermal (64×64)
    thermal_lr = create_lr_thermal(thermal_hr, SCALE_FACTOR)  # (64, 64)

    # Convert optical to (H, W, C) for saving
    optical_hwc = np.transpose(optical, (1, 2, 0))  # (256, 256, 3)

    # Save as .npy (float32 — preserves raw values)
    os.makedirs(output_dir, exist_ok=True)
    np.save(os.path.join(output_dir, "optical_hr.npy"), optical_hwc)
    np.save(os.path.join(output_dir, "thermal_hr.npy"), thermal_hr)
    np.save(os.path.join(output_dir, "thermal_lr.npy"), thermal_lr)

    # Save preview PNGs
    # Optical preview
    opt_preview = np.stack([
        normalize_for_preview(optical_hwc[:, :, 0]),
        normalize_for_preview(optical_hwc[:, :, 1]),
        normalize_for_preview(optical_hwc[:, :, 2])
    ], axis=-1)
    Image.fromarray(opt_preview).save(os.path.join(output_dir, "optical_hr_preview.png"))

    # Thermal HR preview
    th_hr_preview = normalize_for_preview(thermal_hr, 1, 99)
    Image.fromarray(th_hr_preview, mode="L").save(os.path.join(output_dir, "thermal_hr_preview.png"))

    # Thermal LR preview
    th_lr_preview = normalize_for_preview(thermal_lr, 1, 99)
    Image.fromarray(th_lr_preview, mode="L").save(os.path.join(output_dir, "thermal_lr_preview.png"))

    return True, "OK"


def main():
    print("=" * 60)
    print("STEP 1: Dataset Preprocessing")
    print("=" * 60)
    print(f"Scale factor: {SCALE_FACTOR}×")
    print(f"HR size: {CROP_SIZE}×{CROP_SIZE}")
    print(f"LR size: {LR_SIZE}×{LR_SIZE}")

    search_pattern = os.path.join(DATASET_ROOT, "*", "*", "all_bands.tif")
    tif_files = sorted(glob.glob(search_pattern))

    # Exclude 'output' and 'dataset' folders
    tif_files = [f for f in tif_files if "\\output\\" not in f and "\\dataset\\" not in f]

    if not tif_files:
        print("[ERROR] No GeoTIFF files found!")
        return

    print(f"Found {len(tif_files)} scenes.")
    print(f"Output: {OUTPUT_ROOT}\n")

    os.makedirs(OUTPUT_ROOT, exist_ok=True)

    success = 0
    skipped = 0
    errors = []

    for i, tif_path in enumerate(tif_files, 1):
        parts = tif_path.split(os.sep)
        scene_id = parts[-2]
        location_id = parts[-3]
        folder_name = f"{location_id}_{scene_id}"
        output_dir = os.path.join(OUTPUT_ROOT, folder_name)

        if i % 500 == 0 or i == 1:
            print(f"[{i}/{len(tif_files)}] Processing: {folder_name}")

        try:
            ok, msg = process_scene(tif_path, output_dir)
            if ok:
                success += 1
            else:
                skipped += 1
                if os.path.exists(output_dir):
                    import shutil
                    shutil.rmtree(output_dir)
        except Exception as e:
            errors.append((folder_name, str(e)))
            skipped += 1

    print("\n" + "=" * 60)
    print("STEP 1 COMPLETE")
    print(f"  Valid scenes:   {success}")
    print(f"  Skipped:        {skipped}")
    print(f"  Errors:         {len(errors)}")
    print(f"  Output folder:  {OUTPUT_ROOT}")
    print("=" * 60)

    # Save scene list
    scenes = sorted([d for d in os.listdir(OUTPUT_ROOT) if os.path.isdir(os.path.join(OUTPUT_ROOT, d))])
    with open(os.path.join(OUTPUT_ROOT, "all_scenes.txt"), "w") as f:
        for s in scenes:
            f.write(s + "\n")
    print(f"Scene list saved: {len(scenes)} scenes in all_scenes.txt")


if __name__ == "__main__":
    main()
````

---

## Step 2: Train/Val/Test Split

````python
# filepath: c:\Users\Admin\Downloads\ssl4eo_l_oli_tirs_toa_benchmark\step2_split.py
import os
import random

DATASET_ROOT = r"C:\Users\Admin\Downloads\ssl4eo_l_oli_tirs_toa_benchmark\dataset"
SEED = 42
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1


def main():
    print("=" * 60)
    print("STEP 2: Train/Val/Test Split")
    print("=" * 60)

    scenes_file = os.path.join(DATASET_ROOT, "all_scenes.txt")
    with open(scenes_file, "r") as f:
        scenes = [line.strip() for line in f if line.strip()]

    print(f"Total scenes: {len(scenes)}")

    random.seed(SEED)
    random.shuffle(scenes)

    n = len(scenes)
    train_end = int(n * TRAIN_RATIO)
    val_end = train_end + int(n * VAL_RATIO)

    train_scenes = scenes[:train_end]
    val_scenes = scenes[train_end:val_end]
    test_scenes = scenes[val_end:]

    print(f"  Train: {len(train_scenes)}")
    print(f"  Val:   {len(val_scenes)}")
    print(f"  Test:  {len(test_scenes)}")

    # Save split files
    for split_name, split_scenes in [("train", train_scenes), ("val", val_scenes), ("test", test_scenes)]:
        split_file = os.path.join(DATASET_ROOT, f"{split_name}.txt")
        with open(split_file, "w") as f:
            for s in split_scenes:
                f.write(s + "\n")
        print(f"  Saved: {split_file}")

    print("\nDone!")


if __name__ == "__main__":
    main()
````

---

## Step 3: Compute Dataset Statistics

````python
# filepath: c:\Users\Admin\Downloads\ssl4eo_l_oli_tirs_toa_benchmark\step3_stats.py
import os
import numpy as np

DATASET_ROOT = r"C:\Users\Admin\Downloads\ssl4eo_l_oli_tirs_toa_benchmark\dataset"


def main():
    print("=" * 60)
    print("STEP 3: Compute Dataset Statistics")
    print("=" * 60)

    # Only use training set for computing stats
    train_file = os.path.join(DATASET_ROOT, "train.txt")
    with open(train_file, "r") as f:
        train_scenes = [line.strip() for line in f if line.strip()]

    print(f"Computing stats from {len(train_scenes)} training scenes...")

    # Running stats
    optical_sum = np.zeros(3, dtype=np.float64)
    optical_sq_sum = np.zeros(3, dtype=np.float64)
    thermal_sum = 0.0
    thermal_sq_sum = 0.0
    thermal_min = float('inf')
    thermal_max = float('-inf')
    optical_min = np.full(3, float('inf'))
    optical_max = np.full(3, float('-inf'))
    pixel_count = 0
    thermal_pixel_count = 0

    for i, scene in enumerate(train_scenes):
        if i % 1000 == 0:
            print(f"  [{i}/{len(train_scenes)}]")

        scene_dir = os.path.join(DATASET_ROOT, scene)
        optical = np.load(os.path.join(scene_dir, "optical_hr.npy"))      # (256, 256, 3)
        thermal = np.load(os.path.join(scene_dir, "thermal_hr.npy"))      # (256, 256)

        # Optical stats per channel
        for c in range(3):
            ch = optical[:, :, c].astype(np.float64)
            optical_sum[c] += ch.sum()
            optical_sq_sum[c] += (ch ** 2).sum()
            optical_min[c] = min(optical_min[c], ch.min())
            optical_max[c] = max(optical_max[c], ch.max())

        pixel_count += optical.shape[0] * optical.shape[1]

        # Thermal stats
        th = thermal.astype(np.float64)
        thermal_sum += th.sum()
        thermal_sq_sum += (th ** 2).sum()
        thermal_min = min(thermal_min, th.min())
        thermal_max = max(thermal_max, th.max())
        thermal_pixel_count += th.size

    # Compute mean and std
    optical_mean = optical_sum / pixel_count
    optical_std = np.sqrt(optical_sq_sum / pixel_count - optical_mean ** 2)
    thermal_mean = thermal_sum / thermal_pixel_count
    thermal_std = np.sqrt(thermal_sq_sum / thermal_pixel_count - thermal_mean ** 2)

    print("\n" + "=" * 60)
    print("DATASET STATISTICS (Training Set)")
    print("=" * 60)
    print(f"\nOptical (Bands 4, 3, 2):")
    print(f"  Mean: [{optical_mean[0]:.4f}, {optical_mean[1]:.4f}, {optical_mean[2]:.4f}]")
    print(f"  Std:  [{optical_std[0]:.4f}, {optical_std[1]:.4f}, {optical_std[2]:.4f}]")
    print(f"  Min:  [{optical_min[0]:.4f}, {optical_min[1]:.4f}, {optical_min[2]:.4f}]")
    print(f"  Max:  [{optical_max[0]:.4f}, {optical_max[1]:.4f}, {optical_max[2]:.4f}]")
    print(f"\nThermal (Band 10):")
    print(f"  Mean: {thermal_mean:.4f}")
    print(f"  Std:  {thermal_std:.4f}")
    print(f"  Min:  {thermal_min:.4f}")
    print(f"  Max:  {thermal_max:.4f}")

    # Save stats
    stats = {
        "optical_mean": optical_mean,
        "optical_std": optical_std,
        "optical_min": optical_min,
        "optical_max": optical_max,
        "thermal_mean": thermal_mean,
        "thermal_std": thermal_std,
        "thermal_min": thermal_min,
        "thermal_max": thermal_max,
    }
    np.save(os.path.join(DATASET_ROOT, "dataset_stats.npy"), stats)
    print(f"\nStats saved to: {os.path.join(DATASET_ROOT, 'dataset_stats.npy')}")


if __name__ == "__main__":
    main()
````

---

## Step 4: PyTorch Dataset & DataLoader

````python
# filepath: c:\Users\Admin\Downloads\ssl4eo_l_oli_tirs_toa_benchmark\step4_dataloader.py
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
````

---

## Step 5: Model Architecture

````python
# filepath: c:\Users\Admin\Downloads\ssl4eo_l_oli_tirs_toa_benchmark\step5_model.py
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
```


## License: unknown
https://github.com/Sumeet2807/Vid2Sound/blob/17e556d8a4446f4d7327c471129d5f54c938dc1f/model/c2d.py

```
# Complete Pipeline: Optical-Guided Thermal Super-Resolution

## Overview of All Steps

```
Step 1: Dataset Preprocessing (Create LR/HR pairs as .npy)
Step 2: Train/Val/Test Split (80/10/10)
Step 3: Dataset Statistics (Global min/max/mean/std for normalization)
Step 4: PyTorch Dataset & DataLoader
Step 5: Model Architecture (Optical-Guided Fusion SR Network)
Step 6: Training Loop (with PSNR/SSIM/RMSE tracking)
Step 7: Evaluation on Test Set
Step 8: Demo Script (Single image inference + visualization)
```

---

## Step 1: Dataset Preprocessing

This reads raw GeoTIFFs, crops to 256×256, creates LR thermal (64×64), and saves as `.npy` + preview `.png`.

````python
# filepath: c:\Users\Admin\Downloads\ssl4eo_l_oli_tirs_toa_benchmark\step1_preprocess.py
import os
import glob
import numpy as np
import rasterio
from PIL import Image
from scipy.ndimage import zoom

# === CONFIGURATION ===
DATASET_ROOT = r"C:\Users\Admin\Downloads\ssl4eo_l_oli_tirs_toa_benchmark"
OUTPUT_ROOT = os.path.join(DATASET_ROOT, "dataset")
SCALE_FACTOR = 4          # 4× super-resolution
CROP_SIZE = 256           # Target size (power of 2)
LR_SIZE = CROP_SIZE // SCALE_FACTOR  # 64×64

OPTICAL_BANDS = (4, 3, 2)  # Red, Green, Blue
THERMAL_BAND = 10           # TIRS-1


def center_crop(arr, size):
    """Center crop a 2D or 3D array to (size, size)."""
    if arr.ndim == 2:
        h, w = arr.shape
        top = (h - size) // 2
        left = (w - size) // 2
        return arr[top:top+size, left:left+size]
    elif arr.ndim == 3:
        h, w = arr.shape[1], arr.shape[2]
        top = (h - size) // 2
        left = (w - size) // 2
        return arr[:, top:top+size, left:left+size]


def normalize_for_preview(arr, percentile_low=2, percentile_high=98):
    """Normalize to uint8 for PNG preview only."""
    arr = arr.astype(np.float32)
    low = np.percentile(arr, percentile_low)
    high = np.percentile(arr, percentile_high)
    arr = np.clip(arr, low, high)
    if high - low == 0:
        return np.zeros_like(arr, dtype=np.uint8)
    return ((arr - low) / (high - low) * 255.0).astype(np.uint8)


def create_lr_thermal(hr_thermal, scale_factor):
    """Downsample HR thermal using bicubic interpolation."""
    lr = zoom(hr_thermal, 1.0 / scale_factor, order=3)  # order=3 = bicubic
    return lr


def process_scene(tif_path, output_dir):
    """Process one GeoTIFF into optical_hr, thermal_hr, thermal_lr."""
    with rasterio.open(tif_path) as src:
        if src.count < THERMAL_BAND:
            return False, "Not enough bands"

        # Read optical bands (4, 3, 2) as float32
        red = src.read(OPTICAL_BANDS[0]).astype(np.float32)
        green = src.read(OPTICAL_BANDS[1]).astype(np.float32)
        blue = src.read(OPTICAL_BANDS[2]).astype(np.float32)
        optical = np.stack([red, green, blue], axis=0)  # (3, H, W)

        # Read thermal band 10 as float32
        thermal = src.read(THERMAL_BAND).astype(np.float32)

    # Check for blank thermal
    if thermal.max() == 0 or np.mean(thermal == 0) > 0.5:
        return False, "Blank thermal"

    # Check for blank optical
    if optical.max() == 0:
        return False, "Blank optical"

    # Center crop to 256×256
    optical = center_crop(optical, CROP_SIZE)      # (3, 256, 256)
    thermal_hr = center_crop(thermal, CROP_SIZE)    # (256, 256)

    # Create LR thermal (64×64)
    thermal_lr = create_lr_thermal(thermal_hr, SCALE_FACTOR)  # (64, 64)

    # Convert optical to (H, W, C) for saving
    optical_hwc = np.transpose(optical, (1, 2, 0))  # (256, 256, 3)

    # Save as .npy (float32 — preserves raw values)
    os.makedirs(output_dir, exist_ok=True)
    np.save(os.path.join(output_dir, "optical_hr.npy"), optical_hwc)
    np.save(os.path.join(output_dir, "thermal_hr.npy"), thermal_hr)
    np.save(os.path.join(output_dir, "thermal_lr.npy"), thermal_lr)

    # Save preview PNGs
    # Optical preview
    opt_preview = np.stack([
        normalize_for_preview(optical_hwc[:, :, 0]),
        normalize_for_preview(optical_hwc[:, :, 1]),
        normalize_for_preview(optical_hwc[:, :, 2])
    ], axis=-1)
    Image.fromarray(opt_preview).save(os.path.join(output_dir, "optical_hr_preview.png"))

    # Thermal HR preview
    th_hr_preview = normalize_for_preview(thermal_hr, 1, 99)
    Image.fromarray(th_hr_preview, mode="L").save(os.path.join(output_dir, "thermal_hr_preview.png"))

    # Thermal LR preview
    th_lr_preview = normalize_for_preview(thermal_lr, 1, 99)
    Image.fromarray(th_lr_preview, mode="L").save(os.path.join(output_dir, "thermal_lr_preview.png"))

    return True, "OK"


def main():
    print("=" * 60)
    print("STEP 1: Dataset Preprocessing")
    print("=" * 60)
    print(f"Scale factor: {SCALE_FACTOR}×")
    print(f"HR size: {CROP_SIZE}×{CROP_SIZE}")
    print(f"LR size: {LR_SIZE}×{LR_SIZE}")

    search_pattern = os.path.join(DATASET_ROOT, "*", "*", "all_bands.tif")
    tif_files = sorted(glob.glob(search_pattern))

    # Exclude 'output' and 'dataset' folders
    tif_files = [f for f in tif_files if "\\output\\" not in f and "\\dataset\\" not in f]

    if not tif_files:
        print("[ERROR] No GeoTIFF files found!")
        return

    print(f"Found {len(tif_files)} scenes.")
    print(f"Output: {OUTPUT_ROOT}\n")

    os.makedirs(OUTPUT_ROOT, exist_ok=True)

    success = 0
    skipped = 0
    errors = []

    for i, tif_path in enumerate(tif_files, 1):
        parts = tif_path.split(os.sep)
        scene_id = parts[-2]
        location_id = parts[-3]
        folder_name = f"{location_id}_{scene_id}"
        output_dir = os.path.join(OUTPUT_ROOT, folder_name)

        if i % 500 == 0 or i == 1:
            print(f"[{i}/{len(tif_files)}] Processing: {folder_name}")

        try:
            ok, msg = process_scene(tif_path, output_dir)
            if ok:
                success += 1
            else:
                skipped += 1
                if os.path.exists(output_dir):
                    import shutil
                    shutil.rmtree(output_dir)
        except Exception as e:
            errors.append((folder_name, str(e)))
            skipped += 1

    print("\n" + "=" * 60)
    print("STEP 1 COMPLETE")
    print(f"  Valid scenes:   {success}")
    print(f"  Skipped:        {skipped}")
    print(f"  Errors:         {len(errors)}")
    print(f"  Output folder:  {OUTPUT_ROOT}")
    print("=" * 60)

    # Save scene list
    scenes = sorted([d for d in os.listdir(OUTPUT_ROOT) if os.path.isdir(os.path.join(OUTPUT_ROOT, d))])
    with open(os.path.join(OUTPUT_ROOT, "all_scenes.txt"), "w") as f:
        for s in scenes:
            f.write(s + "\n")
    print(f"Scene list saved: {len(scenes)} scenes in all_scenes.txt")


if __name__ == "__main__":
    main()
````

---

## Step 2: Train/Val/Test Split

````python
# filepath: c:\Users\Admin\Downloads\ssl4eo_l_oli_tirs_toa_benchmark\step2_split.py
import os
import random

DATASET_ROOT = r"C:\Users\Admin\Downloads\ssl4eo_l_oli_tirs_toa_benchmark\dataset"
SEED = 42
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1


def main():
    print("=" * 60)
    print("STEP 2: Train/Val/Test Split")
    print("=" * 60)

    scenes_file = os.path.join(DATASET_ROOT, "all_scenes.txt")
    with open(scenes_file, "r") as f:
        scenes = [line.strip() for line in f if line.strip()]

    print(f"Total scenes: {len(scenes)}")

    random.seed(SEED)
    random.shuffle(scenes)

    n = len(scenes)
    train_end = int(n * TRAIN_RATIO)
    val_end = train_end + int(n * VAL_RATIO)

    train_scenes = scenes[:train_end]
    val_scenes = scenes[train_end:val_end]
    test_scenes = scenes[val_end:]

    print(f"  Train: {len(train_scenes)}")
    print(f"  Val:   {len(val_scenes)}")
    print(f"  Test:  {len(test_scenes)}")

    # Save split files
    for split_name, split_scenes in [("train", train_scenes), ("val", val_scenes), ("test", test_scenes)]:
        split_file = os.path.join(DATASET_ROOT, f"{split_name}.txt")
        with open(split_file, "w") as f:
            for s in split_scenes:
                f.write(s + "\n")
        print(f"  Saved: {split_file}")

    print("\nDone!")


if __name__ == "__main__":
    main()
````

---

## Step 3: Compute Dataset Statistics

````python
# filepath: c:\Users\Admin\Downloads\ssl4eo_l_oli_tirs_toa_benchmark\step3_stats.py
import os
import numpy as np

DATASET_ROOT = r"C:\Users\Admin\Downloads\ssl4eo_l_oli_tirs_toa_benchmark\dataset"


def main():
    print("=" * 60)
    print("STEP 3: Compute Dataset Statistics")
    print("=" * 60)

    # Only use training set for computing stats
    train_file = os.path.join(DATASET_ROOT, "train.txt")
    with open(train_file, "r") as f:
        train_scenes = [line.strip() for line in f if line.strip()]

    print(f"Computing stats from {len(train_scenes)} training scenes...")

    # Running stats
    optical_sum = np.zeros(3, dtype=np.float64)
    optical_sq_sum = np.zeros(3, dtype=np.float64)
    thermal_sum = 0.0
    thermal_sq_sum = 0.0
    thermal_min = float('inf')
    thermal_max = float('-inf')
    optical_min = np.full(3, float('inf'))
    optical_max = np.full(3, float('-inf'))
    pixel_count = 0
    thermal_pixel_count = 0

    for i, scene in enumerate(train_scenes):
        if i % 1000 == 0:
            print(f"  [{i}/{len(train_scenes)}]")

        scene_dir = os.path.join(DATASET_ROOT, scene)
        optical = np.load(os.path.join(scene_dir, "optical_hr.npy"))      # (256, 256, 3)
        thermal = np.load(os.path.join(scene_dir, "thermal_hr.npy"))      # (256, 256)

        # Optical stats per channel
        for c in range(3):
            ch = optical[:, :, c].astype(np.float64)
            optical_sum[c] += ch.sum()
            optical_sq_sum[c] += (ch ** 2).sum()
            optical_min[c] = min(optical_min[c], ch.min())
            optical_max[c] = max(optical_max[c], ch.max())

        pixel_count += optical.shape[0] * optical.shape[1]

        # Thermal stats
        th = thermal.astype(np.float64)
        thermal_sum += th.sum()
        thermal_sq_sum += (th ** 2).sum()
        thermal_min = min(thermal_min, th.min())
        thermal_max = max(thermal_max, th.max())
        thermal_pixel_count += th.size

    # Compute mean and std
    optical_mean = optical_sum / pixel_count
    optical_std = np.sqrt(optical_sq_sum / pixel_count - optical_mean ** 2)
    thermal_mean = thermal_sum / thermal_pixel_count
    thermal_std = np.sqrt(thermal_sq_sum / thermal_pixel_count - thermal_mean ** 2)

    print("\n" + "=" * 60)
    print("DATASET STATISTICS (Training Set)")
    print("=" * 60)
    print(f"\nOptical (Bands 4, 3, 2):")
    print(f"  Mean: [{optical_mean[0]:.4f}, {optical_mean[1]:.4f}, {optical_mean[2]:.4f}]")
    print(f"  Std:  [{optical_std[0]:.4f}, {optical_std[1]:.4f}, {optical_std[2]:.4f}]")
    print(f"  Min:  [{optical_min[0]:.4f}, {optical_min[1]:.4f}, {optical_min[2]:.4f}]")
    print(f"  Max:  [{optical_max[0]:.4f}, {optical_max[1]:.4f}, {optical_max[2]:.4f}]")
    print(f"\nThermal (Band 10):")
    print(f"  Mean: {thermal_mean:.4f}")
    print(f"  Std:  {thermal_std:.4f}")
    print(f"  Min:  {thermal_min:.4f}")
    print(f"  Max:  {thermal_max:.4f}")

    # Save stats
    stats = {
        "optical_mean": optical_mean,
        "optical_std": optical_std,
        "optical_min": optical_min,
        "optical_max": optical_max,
        "thermal_mean": thermal_mean,
        "thermal_std": thermal_std,
        "thermal_min": thermal_min,
        "thermal_max": thermal_max,
    }
    np.save(os.path.join(DATASET_ROOT, "dataset_stats.npy"), stats)
    print(f"\nStats saved to: {os.path.join(DATASET_ROOT, 'dataset_stats.npy')}")


if __name__ == "__main__":
    main()
````

---

## Step 4: PyTorch Dataset & DataLoader

````python
# filepath: c:\Users\Admin\Downloads\ssl4eo_l_oli_tirs_toa_benchmark\step4_dataloader.py
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
````

---

## Step 5: Model Architecture

````python
# filepath: c:\Users\Admin\Downloads\ssl4eo_l_oli_tirs_toa_benchmark\step5_model.py
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
```


## License: unknown
https://github.com/Sumeet2807/Vid2Sound/blob/17e556d8a4446f4d7327c471129d5f54c938dc1f/model/c2d.py

```
# Complete Pipeline: Optical-Guided Thermal Super-Resolution

## Overview of All Steps

```
Step 1: Dataset Preprocessing (Create LR/HR pairs as .npy)
Step 2: Train/Val/Test Split (80/10/10)
Step 3: Dataset Statistics (Global min/max/mean/std for normalization)
Step 4: PyTorch Dataset & DataLoader
Step 5: Model Architecture (Optical-Guided Fusion SR Network)
Step 6: Training Loop (with PSNR/SSIM/RMSE tracking)
Step 7: Evaluation on Test Set
Step 8: Demo Script (Single image inference + visualization)
```

---

## Step 1: Dataset Preprocessing

This reads raw GeoTIFFs, crops to 256×256, creates LR thermal (64×64), and saves as `.npy` + preview `.png`.

````python
# filepath: c:\Users\Admin\Downloads\ssl4eo_l_oli_tirs_toa_benchmark\step1_preprocess.py
import os
import glob
import numpy as np
import rasterio
from PIL import Image
from scipy.ndimage import zoom

# === CONFIGURATION ===
DATASET_ROOT = r"C:\Users\Admin\Downloads\ssl4eo_l_oli_tirs_toa_benchmark"
OUTPUT_ROOT = os.path.join(DATASET_ROOT, "dataset")
SCALE_FACTOR = 4          # 4× super-resolution
CROP_SIZE = 256           # Target size (power of 2)
LR_SIZE = CROP_SIZE // SCALE_FACTOR  # 64×64

OPTICAL_BANDS = (4, 3, 2)  # Red, Green, Blue
THERMAL_BAND = 10           # TIRS-1


def center_crop(arr, size):
    """Center crop a 2D or 3D array to (size, size)."""
    if arr.ndim == 2:
        h, w = arr.shape
        top = (h - size) // 2
        left = (w - size) // 2
        return arr[top:top+size, left:left+size]
    elif arr.ndim == 3:
        h, w = arr.shape[1], arr.shape[2]
        top = (h - size) // 2
        left = (w - size) // 2
        return arr[:, top:top+size, left:left+size]


def normalize_for_preview(arr, percentile_low=2, percentile_high=98):
    """Normalize to uint8 for PNG preview only."""
    arr = arr.astype(np.float32)
    low = np.percentile(arr, percentile_low)
    high = np.percentile(arr, percentile_high)
    arr = np.clip(arr, low, high)
    if high - low == 0:
        return np.zeros_like(arr, dtype=np.uint8)
    return ((arr - low) / (high - low) * 255.0).astype(np.uint8)


def create_lr_thermal(hr_thermal, scale_factor):
    """Downsample HR thermal using bicubic interpolation."""
    lr = zoom(hr_thermal, 1.0 / scale_factor, order=3)  # order=3 = bicubic
    return lr


def process_scene(tif_path, output_dir):
    """Process one GeoTIFF into optical_hr, thermal_hr, thermal_lr."""
    with rasterio.open(tif_path) as src:
        if src.count < THERMAL_BAND:
            return False, "Not enough bands"

        # Read optical bands (4, 3, 2) as float32
        red = src.read(OPTICAL_BANDS[0]).astype(np.float32)
        green = src.read(OPTICAL_BANDS[1]).astype(np.float32)
        blue = src.read(OPTICAL_BANDS[2]).astype(np.float32)
        optical = np.stack([red, green, blue], axis=0)  # (3, H, W)

        # Read thermal band 10 as float32
        thermal = src.read(THERMAL_BAND).astype(np.float32)

    # Check for blank thermal
    if thermal.max() == 0 or np.mean(thermal == 0) > 0.5:
        return False, "Blank thermal"

    # Check for blank optical
    if optical.max() == 0:
        return False, "Blank optical"

    # Center crop to 256×256
    optical = center_crop(optical, CROP_SIZE)      # (3, 256, 256)
    thermal_hr = center_crop(thermal, CROP_SIZE)    # (256, 256)

    # Create LR thermal (64×64)
    thermal_lr = create_lr_thermal(thermal_hr, SCALE_FACTOR)  # (64, 64)

    # Convert optical to (H, W, C) for saving
    optical_hwc = np.transpose(optical, (1, 2, 0))  # (256, 256, 3)

    # Save as .npy (float32 — preserves raw values)
    os.makedirs(output_dir, exist_ok=True)
    np.save(os.path.join(output_dir, "optical_hr.npy"), optical_hwc)
    np.save(os.path.join(output_dir, "thermal_hr.npy"), thermal_hr)
    np.save(os.path.join(output_dir, "thermal_lr.npy"), thermal_lr)

    # Save preview PNGs
    # Optical preview
    opt_preview = np.stack([
        normalize_for_preview(optical_hwc[:, :, 0]),
        normalize_for_preview(optical_hwc[:, :, 1]),
        normalize_for_preview(optical_hwc[:, :, 2])
    ], axis=-1)
    Image.fromarray(opt_preview).save(os.path.join(output_dir, "optical_hr_preview.png"))

    # Thermal HR preview
    th_hr_preview = normalize_for_preview(thermal_hr, 1, 99)
    Image.fromarray(th_hr_preview, mode="L").save(os.path.join(output_dir, "thermal_hr_preview.png"))

    # Thermal LR preview
    th_lr_preview = normalize_for_preview(thermal_lr, 1, 99)
    Image.fromarray(th_lr_preview, mode="L").save(os.path.join(output_dir, "thermal_lr_preview.png"))

    return True, "OK"


def main():
    print("=" * 60)
    print("STEP 1: Dataset Preprocessing")
    print("=" * 60)
    print(f"Scale factor: {SCALE_FACTOR}×")
    print(f"HR size: {CROP_SIZE}×{CROP_SIZE}")
    print(f"LR size: {LR_SIZE}×{LR_SIZE}")

    search_pattern = os.path.join(DATASET_ROOT, "*", "*", "all_bands.tif")
    tif_files = sorted(glob.glob(search_pattern))

    # Exclude 'output' and 'dataset' folders
    tif_files = [f for f in tif_files if "\\output\\" not in f and "\\dataset\\" not in f]

    if not tif_files:
        print("[ERROR] No GeoTIFF files found!")
        return

    print(f"Found {len(tif_files)} scenes.")
    print(f"Output: {OUTPUT_ROOT}\n")

    os.makedirs(OUTPUT_ROOT, exist_ok=True)

    success = 0
    skipped = 0
    errors = []

    for i, tif_path in enumerate(tif_files, 1):
        parts = tif_path.split(os.sep)
        scene_id = parts[-2]
        location_id = parts[-3]
        folder_name = f"{location_id}_{scene_id}"
        output_dir = os.path.join(OUTPUT_ROOT, folder_name)

        if i % 500 == 0 or i == 1:
            print(f"[{i}/{len(tif_files)}] Processing: {folder_name}")

        try:
            ok, msg = process_scene(tif_path, output_dir)
            if ok:
                success += 1
            else:
                skipped += 1
                if os.path.exists(output_dir):
                    import shutil
                    shutil.rmtree(output_dir)
        except Exception as e:
            errors.append((folder_name, str(e)))
            skipped += 1

    print("\n" + "=" * 60)
    print("STEP 1 COMPLETE")
    print(f"  Valid scenes:   {success}")
    print(f"  Skipped:        {skipped}")
    print(f"  Errors:         {len(errors)}")
    print(f"  Output folder:  {OUTPUT_ROOT}")
    print("=" * 60)

    # Save scene list
    scenes = sorted([d for d in os.listdir(OUTPUT_ROOT) if os.path.isdir(os.path.join(OUTPUT_ROOT, d))])
    with open(os.path.join(OUTPUT_ROOT, "all_scenes.txt"), "w") as f:
        for s in scenes:
            f.write(s + "\n")
    print(f"Scene list saved: {len(scenes)} scenes in all_scenes.txt")


if __name__ == "__main__":
    main()
````

---

## Step 2: Train/Val/Test Split

````python
# filepath: c:\Users\Admin\Downloads\ssl4eo_l_oli_tirs_toa_benchmark\step2_split.py
import os
import random

DATASET_ROOT = r"C:\Users\Admin\Downloads\ssl4eo_l_oli_tirs_toa_benchmark\dataset"
SEED = 42
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1


def main():
    print("=" * 60)
    print("STEP 2: Train/Val/Test Split")
    print("=" * 60)

    scenes_file = os.path.join(DATASET_ROOT, "all_scenes.txt")
    with open(scenes_file, "r") as f:
        scenes = [line.strip() for line in f if line.strip()]

    print(f"Total scenes: {len(scenes)}")

    random.seed(SEED)
    random.shuffle(scenes)

    n = len(scenes)
    train_end = int(n * TRAIN_RATIO)
    val_end = train_end + int(n * VAL_RATIO)

    train_scenes = scenes[:train_end]
    val_scenes = scenes[train_end:val_end]
    test_scenes = scenes[val_end:]

    print(f"  Train: {len(train_scenes)}")
    print(f"  Val:   {len(val_scenes)}")
    print(f"  Test:  {len(test_scenes)}")

    # Save split files
    for split_name, split_scenes in [("train", train_scenes), ("val", val_scenes), ("test", test_scenes)]:
        split_file = os.path.join(DATASET_ROOT, f"{split_name}.txt")
        with open(split_file, "w") as f:
            for s in split_scenes:
                f.write(s + "\n")
        print(f"  Saved: {split_file}")

    print("\nDone!")


if __name__ == "__main__":
    main()
````

---

## Step 3: Compute Dataset Statistics

````python
# filepath: c:\Users\Admin\Downloads\ssl4eo_l_oli_tirs_toa_benchmark\step3_stats.py
import os
import numpy as np

DATASET_ROOT = r"C:\Users\Admin\Downloads\ssl4eo_l_oli_tirs_toa_benchmark\dataset"


def main():
    print("=" * 60)
    print("STEP 3: Compute Dataset Statistics")
    print("=" * 60)

    # Only use training set for computing stats
    train_file = os.path.join(DATASET_ROOT, "train.txt")
    with open(train_file, "r") as f:
        train_scenes = [line.strip() for line in f if line.strip()]

    print(f"Computing stats from {len(train_scenes)} training scenes...")

    # Running stats
    optical_sum = np.zeros(3, dtype=np.float64)
    optical_sq_sum = np.zeros(3, dtype=np.float64)
    thermal_sum = 0.0
    thermal_sq_sum = 0.0
    thermal_min = float('inf')
    thermal_max = float('-inf')
    optical_min = np.full(3, float('inf'))
    optical_max = np.full(3, float('-inf'))
    pixel_count = 0
    thermal_pixel_count = 0

    for i, scene in enumerate(train_scenes):
        if i % 1000 == 0:
            print(f"  [{i}/{len(train_scenes)}]")

        scene_dir = os.path.join(DATASET_ROOT, scene)
        optical = np.load(os.path.join(scene_dir, "optical_hr.npy"))      # (256, 256, 3)
        thermal = np.load(os.path.join(scene_dir, "thermal_hr.npy"))      # (256, 256)

        # Optical stats per channel
        for c in range(3):
            ch = optical[:, :, c].astype(np.float64)
            optical_sum[c] += ch.sum()
            optical_sq_sum[c] += (ch ** 2).sum()
            optical_min[c] = min(optical_min[c], ch.min())
            optical_max[c] = max(optical_max[c], ch.max())

        pixel_count += optical.shape[0] * optical.shape[1]

        # Thermal stats
        th = thermal.astype(np.float64)
        thermal_sum += th.sum()
        thermal_sq_sum += (th ** 2).sum()
        thermal_min = min(thermal_min, th.min())
        thermal_max = max(thermal_max, th.max())
        thermal_pixel_count += th.size

    # Compute mean and std
    optical_mean = optical_sum / pixel_count
    optical_std = np.sqrt(optical_sq_sum / pixel_count - optical_mean ** 2)
    thermal_mean = thermal_sum / thermal_pixel_count
    thermal_std = np.sqrt(thermal_sq_sum / thermal_pixel_count - thermal_mean ** 2)

    print("\n" + "=" * 60)
    print("DATASET STATISTICS (Training Set)")
    print("=" * 60)
    print(f"\nOptical (Bands 4, 3, 2):")
    print(f"  Mean: [{optical_mean[0]:.4f}, {optical_mean[1]:.4f}, {optical_mean[2]:.4f}]")
    print(f"  Std:  [{optical_std[0]:.4f}, {optical_std[1]:.4f}, {optical_std[2]:.4f}]")
    print(f"  Min:  [{optical_min[0]:.4f}, {optical_min[1]:.4f}, {optical_min[2]:.4f}]")
    print(f"  Max:  [{optical_max[0]:.4f}, {optical_max[1]:.4f}, {optical_max[2]:.4f}]")
    print(f"\nThermal (Band 10):")
    print(f"  Mean: {thermal_mean:.4f}")
    print(f"  Std:  {thermal_std:.4f}")
    print(f"  Min:  {thermal_min:.4f}")
    print(f"  Max:  {thermal_max:.4f}")

    # Save stats
    stats = {
        "optical_mean": optical_mean,
        "optical_std": optical_std,
        "optical_min": optical_min,
        "optical_max": optical_max,
        "thermal_mean": thermal_mean,
        "thermal_std": thermal_std,
        "thermal_min": thermal_min,
        "thermal_max": thermal_max,
    }
    np.save(os.path.join(DATASET_ROOT, "dataset_stats.npy"), stats)
    print(f"\nStats saved to: {os.path.join(DATASET_ROOT, 'dataset_stats.npy')}")


if __name__ == "__main__":
    main()
````

---

## Step 4: PyTorch Dataset & DataLoader

````python
# filepath: c:\Users\Admin\Downloads\ssl4eo_l_oli_tirs_toa_benchmark\step4_dataloader.py
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
````

---

## Step 5: Model Architecture

````python
# filepath: c:\Users\Admin\Downloads\ssl4eo_l_oli_tirs_toa_benchmark\step5_model.py
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
```


## License: unknown
https://github.com/Sumeet2807/Vid2Sound/blob/17e556d8a4446f4d7327c471129d5f54c938dc1f/model/c2d.py

```
# Complete Pipeline: Optical-Guided Thermal Super-Resolution

## Overview of All Steps

```
Step 1: Dataset Preprocessing (Create LR/HR pairs as .npy)
Step 2: Train/Val/Test Split (80/10/10)
Step 3: Dataset Statistics (Global min/max/mean/std for normalization)
Step 4: PyTorch Dataset & DataLoader
Step 5: Model Architecture (Optical-Guided Fusion SR Network)
Step 6: Training Loop (with PSNR/SSIM/RMSE tracking)
Step 7: Evaluation on Test Set
Step 8: Demo Script (Single image inference + visualization)
```

---

## Step 1: Dataset Preprocessing

This reads raw GeoTIFFs, crops to 256×256, creates LR thermal (64×64), and saves as `.npy` + preview `.png`.

````python
# filepath: c:\Users\Admin\Downloads\ssl4eo_l_oli_tirs_toa_benchmark\step1_preprocess.py
import os
import glob
import numpy as np
import rasterio
from PIL import Image
from scipy.ndimage import zoom

# === CONFIGURATION ===
DATASET_ROOT = r"C:\Users\Admin\Downloads\ssl4eo_l_oli_tirs_toa_benchmark"
OUTPUT_ROOT = os.path.join(DATASET_ROOT, "dataset")
SCALE_FACTOR = 4          # 4× super-resolution
CROP_SIZE = 256           # Target size (power of 2)
LR_SIZE = CROP_SIZE // SCALE_FACTOR  # 64×64

OPTICAL_BANDS = (4, 3, 2)  # Red, Green, Blue
THERMAL_BAND = 10           # TIRS-1


def center_crop(arr, size):
    """Center crop a 2D or 3D array to (size, size)."""
    if arr.ndim == 2:
        h, w = arr.shape
        top = (h - size) // 2
        left = (w - size) // 2
        return arr[top:top+size, left:left+size]
    elif arr.ndim == 3:
        h, w = arr.shape[1], arr.shape[2]
        top = (h - size) // 2
        left = (w - size) // 2
        return arr[:, top:top+size, left:left+size]


def normalize_for_preview(arr, percentile_low=2, percentile_high=98):
    """Normalize to uint8 for PNG preview only."""
    arr = arr.astype(np.float32)
    low = np.percentile(arr, percentile_low)
    high = np.percentile(arr, percentile_high)
    arr = np.clip(arr, low, high)
    if high - low == 0:
        return np.zeros_like(arr, dtype=np.uint8)
    return ((arr - low) / (high - low) * 255.0).astype(np.uint8)


def create_lr_thermal(hr_thermal, scale_factor):
    """Downsample HR thermal using bicubic interpolation."""
    lr = zoom(hr_thermal, 1.0 / scale_factor, order=3)  # order=3 = bicubic
    return lr


def process_scene(tif_path, output_dir):
    """Process one GeoTIFF into optical_hr, thermal_hr, thermal_lr."""
    with rasterio.open(tif_path) as src:
        if src.count < THERMAL_BAND:
            return False, "Not enough bands"

        # Read optical bands (4, 3, 2) as float32
        red = src.read(OPTICAL_BANDS[0]).astype(np.float32)
        green = src.read(OPTICAL_BANDS[1]).astype(np.float32)
        blue = src.read(OPTICAL_BANDS[2]).astype(np.float32)
        optical = np.stack([red, green, blue], axis=0)  # (3, H, W)

        # Read thermal band 10 as float32
        thermal = src.read(THERMAL_BAND).astype(np.float32)

    # Check for blank thermal
    if thermal.max() == 0 or np.mean(thermal == 0) > 0.5:
        return False, "Blank thermal"

    # Check for blank optical
    if optical.max() == 0:
        return False, "Blank optical"

    # Center crop to 256×256
    optical = center_crop(optical, CROP_SIZE)      # (3, 256, 256)
    thermal_hr = center_crop(thermal, CROP_SIZE)    # (256, 256)

    # Create LR thermal (64×64)
    thermal_lr = create_lr_thermal(thermal_hr, SCALE_FACTOR)  # (64, 64)

    # Convert optical to (H, W, C) for saving
    optical_hwc = np.transpose(optical, (1, 2, 0))  # (256, 256, 3)

    # Save as .npy (float32 — preserves raw values)
    os.makedirs(output_dir, exist_ok=True)
    np.save(os.path.join(output_dir, "optical_hr.npy"), optical_hwc)
    np.save(os.path.join(output_dir, "thermal_hr.npy"), thermal_hr)
    np.save(os.path.join(output_dir, "thermal_lr.npy"), thermal_lr)

    # Save preview PNGs
    # Optical preview
    opt_preview = np.stack([
        normalize_for_preview(optical_hwc[:, :, 0]),
        normalize_for_preview(optical_hwc[:, :, 1]),
        normalize_for_preview(optical_hwc[:, :, 2])
    ], axis=-1)
    Image.fromarray(opt_preview).save(os.path.join(output_dir, "optical_hr_preview.png"))

    # Thermal HR preview
    th_hr_preview = normalize_for_preview(thermal_hr, 1, 99)
    Image.fromarray(th_hr_preview, mode="L").save(os.path.join(output_dir, "thermal_hr_preview.png"))

    # Thermal LR preview
    th_lr_preview = normalize_for_preview(thermal_lr, 1, 99)
    Image.fromarray(th_lr_preview, mode="L").save(os.path.join(output_dir, "thermal_lr_preview.png"))

    return True, "OK"


def main():
    print("=" * 60)
    print("STEP 1: Dataset Preprocessing")
    print("=" * 60)
    print(f"Scale factor: {SCALE_FACTOR}×")
    print(f"HR size: {CROP_SIZE}×{CROP_SIZE}")
    print(f"LR size: {LR_SIZE}×{LR_SIZE}")

    search_pattern = os.path.join(DATASET_ROOT, "*", "*", "all_bands.tif")
    tif_files = sorted(glob.glob(search_pattern))

    # Exclude 'output' and 'dataset' folders
    tif_files = [f for f in tif_files if "\\output\\" not in f and "\\dataset\\" not in f]

    if not tif_files:
        print("[ERROR] No GeoTIFF files found!")
        return

    print(f"Found {len(tif_files)} scenes.")
    print(f"Output: {OUTPUT_ROOT}\n")

    os.makedirs(OUTPUT_ROOT, exist_ok=True)

    success = 0
    skipped = 0
    errors = []

    for i, tif_path in enumerate(tif_files, 1):
        parts = tif_path.split(os.sep)
        scene_id = parts[-2]
        location_id = parts[-3]
        folder_name = f"{location_id}_{scene_id}"
        output_dir = os.path.join(OUTPUT_ROOT, folder_name)

        if i % 500 == 0 or i == 1:
            print(f"[{i}/{len(tif_files)}] Processing: {folder_name}")

        try:
            ok, msg = process_scene(tif_path, output_dir)
            if ok:
                success += 1
            else:
                skipped += 1
                if os.path.exists(output_dir):
                    import shutil
                    shutil.rmtree(output_dir)
        except Exception as e:
            errors.append((folder_name, str(e)))
            skipped += 1

    print("\n" + "=" * 60)
    print("STEP 1 COMPLETE")
    print(f"  Valid scenes:   {success}")
    print(f"  Skipped:        {skipped}")
    print(f"  Errors:         {len(errors)}")
    print(f"  Output folder:  {OUTPUT_ROOT}")
    print("=" * 60)

    # Save scene list
    scenes = sorted([d for d in os.listdir(OUTPUT_ROOT) if os.path.isdir(os.path.join(OUTPUT_ROOT, d))])
    with open(os.path.join(OUTPUT_ROOT, "all_scenes.txt"), "w") as f:
        for s in scenes:
            f.write(s + "\n")
    print(f"Scene list saved: {len(scenes)} scenes in all_scenes.txt")


if __name__ == "__main__":
    main()
````

---

## Step 2: Train/Val/Test Split

````python
# filepath: c:\Users\Admin\Downloads\ssl4eo_l_oli_tirs_toa_benchmark\step2_split.py
import os
import random

DATASET_ROOT = r"C:\Users\Admin\Downloads\ssl4eo_l_oli_tirs_toa_benchmark\dataset"
SEED = 42
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1


def main():
    print("=" * 60)
    print("STEP 2: Train/Val/Test Split")
    print("=" * 60)

    scenes_file = os.path.join(DATASET_ROOT, "all_scenes.txt")
    with open(scenes_file, "r") as f:
        scenes = [line.strip() for line in f if line.strip()]

    print(f"Total scenes: {len(scenes)}")

    random.seed(SEED)
    random.shuffle(scenes)

    n = len(scenes)
    train_end = int(n * TRAIN_RATIO)
    val_end = train_end + int(n * VAL_RATIO)

    train_scenes = scenes[:train_end]
    val_scenes = scenes[train_end:val_end]
    test_scenes = scenes[val_end:]

    print(f"  Train: {len(train_scenes)}")
    print(f"  Val:   {len(val_scenes)}")
    print(f"  Test:  {len(test_scenes)}")

    # Save split files
    for split_name, split_scenes in [("train", train_scenes), ("val", val_scenes), ("test", test_scenes)]:
        split_file = os.path.join(DATASET_ROOT, f"{split_name}.txt")
        with open(split_file, "w") as f:
            for s in split_scenes:
                f.write(s + "\n")
        print(f"  Saved: {split_file}")

    print("\nDone!")


if __name__ == "__main__":
    main()
````

---

## Step 3: Compute Dataset Statistics

````python
# filepath: c:\Users\Admin\Downloads\ssl4eo_l_oli_tirs_toa_benchmark\step3_stats.py
import os
import numpy as np

DATASET_ROOT = r"C:\Users\Admin\Downloads\ssl4eo_l_oli_tirs_toa_benchmark\dataset"


def main():
    print("=" * 60)
    print("STEP 3: Compute Dataset Statistics")
    print("=" * 60)

    # Only use training set for computing stats
    train_file = os.path.join(DATASET_ROOT, "train.txt")
    with open(train_file, "r") as f:
        train_scenes = [line.strip() for line in f if line.strip()]

    print(f"Computing stats from {len(train_scenes)} training scenes...")

    # Running stats
    optical_sum = np.zeros(3, dtype=np.float64)
    optical_sq_sum = np.zeros(3, dtype=np.float64)
    thermal_sum = 0.0
    thermal_sq_sum = 0.0
    thermal_min = float('inf')
    thermal_max = float('-inf')
    optical_min = np.full(3, float('inf'))
    optical_max = np.full(3, float('-inf'))
    pixel_count = 0
    thermal_pixel_count = 0

    for i, scene in enumerate(train_scenes):
        if i % 1000 == 0:
            print(f"  [{i}/{len(train_scenes)}]")

        scene_dir = os.path.join(DATASET_ROOT, scene)
        optical = np.load(os.path.join(scene_dir, "optical_hr.npy"))      # (256, 256, 3)
        thermal = np.load(os.path.join(scene_dir, "thermal_hr.npy"))      # (256, 256)

        # Optical stats per channel
        for c in range(3):
            ch = optical[:, :, c].astype(np.float64)
            optical_sum[c] += ch.sum()
            optical_sq_sum[c] += (ch ** 2).sum()
            optical_min[c] = min(optical_min[c], ch.min())
            optical_max[c] = max(optical_max[c], ch.max())

        pixel_count += optical.shape[0] * optical.shape[1]

        # Thermal stats
        th = thermal.astype(np.float64)
        thermal_sum += th.sum()
        thermal_sq_sum += (th ** 2).sum()
        thermal_min = min(thermal_min, th.min())
        thermal_max = max(thermal_max, th.max())
        thermal_pixel_count += th.size

    # Compute mean and std
    optical_mean = optical_sum / pixel_count
    optical_std = np.sqrt(optical_sq_sum / pixel_count - optical_mean ** 2)
    thermal_mean = thermal_sum / thermal_pixel_count
    thermal_std = np.sqrt(thermal_sq_sum / thermal_pixel_count - thermal_mean ** 2)

    print("\n" + "=" * 60)
    print("DATASET STATISTICS (Training Set)")
    print("=" * 60)
    print(f"\nOptical (Bands 4, 3, 2):")
    print(f"  Mean: [{optical_mean[0]:.4f}, {optical_mean[1]:.4f}, {optical_mean[2]:.4f}]")
    print(f"  Std:  [{optical_std[0]:.4f}, {optical_std[1]:.4f}, {optical_std[2]:.4f}]")
    print(f"  Min:  [{optical_min[0]:.4f}, {optical_min[1]:.4f}, {optical_min[2]:.4f}]")
    print(f"  Max:  [{optical_max[0]:.4f}, {optical_max[1]:.4f}, {optical_max[2]:.4f}]")
    print(f"\nThermal (Band 10):")
    print(f"  Mean: {thermal_mean:.4f}")
    print(f"  Std:  {thermal_std:.4f}")
    print(f"  Min:  {thermal_min:.4f}")
    print(f"  Max:  {thermal_max:.4f}")

    # Save stats
    stats = {
        "optical_mean": optical_mean,
        "optical_std": optical_std,
        "optical_min": optical_min,
        "optical_max": optical_max,
        "thermal_mean": thermal_mean,
        "thermal_std": thermal_std,
        "thermal_min": thermal_min,
        "thermal_max": thermal_max,
    }
    np.save(os.path.join(DATASET_ROOT, "dataset_stats.npy"), stats)
    print(f"\nStats saved to: {os.path.join(DATASET_ROOT, 'dataset_stats.npy')}")


if __name__ == "__main__":
    main()
````

---

## Step 4: PyTorch Dataset & DataLoader

````python
# filepath: c:\Users\Admin\Downloads\ssl4eo_l_oli_tirs_toa_benchmark\step4_dataloader.py
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
````

---

## Step 5: Model Architecture

````python
# filepath: c:\Users\Admin\Downloads\ssl4eo_l_oli_tirs_toa_benchmark\step5_model.py
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
```


## License: unknown
https://github.com/Sumeet2807/Vid2Sound/blob/17e556d8a4446f4d7327c471129d5f54c938dc1f/model/c2d.py

```
# Complete Pipeline: Optical-Guided Thermal Super-Resolution

## Overview of All Steps

```
Step 1: Dataset Preprocessing (Create LR/HR pairs as .npy)
Step 2: Train/Val/Test Split (80/10/10)
Step 3: Dataset Statistics (Global min/max/mean/std for normalization)
Step 4: PyTorch Dataset & DataLoader
Step 5: Model Architecture (Optical-Guided Fusion SR Network)
Step 6: Training Loop (with PSNR/SSIM/RMSE tracking)
Step 7: Evaluation on Test Set
Step 8: Demo Script (Single image inference + visualization)
```

---

## Step 1: Dataset Preprocessing

This reads raw GeoTIFFs, crops to 256×256, creates LR thermal (64×64), and saves as `.npy` + preview `.png`.

````python
# filepath: c:\Users\Admin\Downloads\ssl4eo_l_oli_tirs_toa_benchmark\step1_preprocess.py
import os
import glob
import numpy as np
import rasterio
from PIL import Image
from scipy.ndimage import zoom

# === CONFIGURATION ===
DATASET_ROOT = r"C:\Users\Admin\Downloads\ssl4eo_l_oli_tirs_toa_benchmark"
OUTPUT_ROOT = os.path.join(DATASET_ROOT, "dataset")
SCALE_FACTOR = 4          # 4× super-resolution
CROP_SIZE = 256           # Target size (power of 2)
LR_SIZE = CROP_SIZE // SCALE_FACTOR  # 64×64

OPTICAL_BANDS = (4, 3, 2)  # Red, Green, Blue
THERMAL_BAND = 10           # TIRS-1


def center_crop(arr, size):
    """Center crop a 2D or 3D array to (size, size)."""
    if arr.ndim == 2:
        h, w = arr.shape
        top = (h - size) // 2
        left = (w - size) // 2
        return arr[top:top+size, left:left+size]
    elif arr.ndim == 3:
        h, w = arr.shape[1], arr.shape[2]
        top = (h - size) // 2
        left = (w - size) // 2
        return arr[:, top:top+size, left:left+size]


def normalize_for_preview(arr, percentile_low=2, percentile_high=98):
    """Normalize to uint8 for PNG preview only."""
    arr = arr.astype(np.float32)
    low = np.percentile(arr, percentile_low)
    high = np.percentile(arr, percentile_high)
    arr = np.clip(arr, low, high)
    if high - low == 0:
        return np.zeros_like(arr, dtype=np.uint8)
    return ((arr - low) / (high - low) * 255.0).astype(np.uint8)


def create_lr_thermal(hr_thermal, scale_factor):
    """Downsample HR thermal using bicubic interpolation."""
    lr = zoom(hr_thermal, 1.0 / scale_factor, order=3)  # order=3 = bicubic
    return lr


def process_scene(tif_path, output_dir):
    """Process one GeoTIFF into optical_hr, thermal_hr, thermal_lr."""
    with rasterio.open(tif_path) as src:
        if src.count < THERMAL_BAND:
            return False, "Not enough bands"

        # Read optical bands (4, 3, 2) as float32
        red = src.read(OPTICAL_BANDS[0]).astype(np.float32)
        green = src.read(OPTICAL_BANDS[1]).astype(np.float32)
        blue = src.read(OPTICAL_BANDS[2]).astype(np.float32)
        optical = np.stack([red, green, blue], axis=0)  # (3, H, W)

        # Read thermal band 10 as float32
        thermal = src.read(THERMAL_BAND).astype(np.float32)

    # Check for blank thermal
    if thermal.max() == 0 or np.mean(thermal == 0) > 0.5:
        return False, "Blank thermal"

    # Check for blank optical
    if optical.max() == 0:
        return False, "Blank optical"

    # Center crop to 256×256
    optical = center_crop(optical, CROP_SIZE)      # (3, 256, 256)
    thermal_hr = center_crop(thermal, CROP_SIZE)    # (256, 256)

    # Create LR thermal (64×64)
    thermal_lr = create_lr_thermal(thermal_hr, SCALE_FACTOR)  # (64, 64)

    # Convert optical to (H, W, C) for saving
    optical_hwc = np.transpose(optical, (1, 2, 0))  # (256, 256, 3)

    # Save as .npy (float32 — preserves raw values)
    os.makedirs(output_dir, exist_ok=True)
    np.save(os.path.join(output_dir, "optical_hr.npy"), optical_hwc)
    np.save(os.path.join(output_dir, "thermal_hr.npy"), thermal_hr)
    np.save(os.path.join(output_dir, "thermal_lr.npy"), thermal_lr)

    # Save preview PNGs
    # Optical preview
    opt_preview = np.stack([
        normalize_for_preview(optical_hwc[:, :, 0]),
        normalize_for_preview(optical_hwc[:, :, 1]),
        normalize_for_preview(optical_hwc[:, :, 2])
    ], axis=-1)
    Image.fromarray(opt_preview).save(os.path.join(output_dir, "optical_hr_preview.png"))

    # Thermal HR preview
    th_hr_preview = normalize_for_preview(thermal_hr, 1, 99)
    Image.fromarray(th_hr_preview, mode="L").save(os.path.join(output_dir, "thermal_hr_preview.png"))

    # Thermal LR preview
    th_lr_preview = normalize_for_preview(thermal_lr, 1, 99)
    Image.fromarray(th_lr_preview, mode="L").save(os.path.join(output_dir, "thermal_lr_preview.png"))

    return True, "OK"


def main():
    print("=" * 60)
    print("STEP 1: Dataset Preprocessing")
    print("=" * 60)
    print(f"Scale factor: {SCALE_FACTOR}×")
    print(f"HR size: {CROP_SIZE}×{CROP_SIZE}")
    print(f"LR size: {LR_SIZE}×{LR_SIZE}")

    search_pattern = os.path.join(DATASET_ROOT, "*", "*", "all_bands.tif")
    tif_files = sorted(glob.glob(search_pattern))

    # Exclude 'output' and 'dataset' folders
    tif_files = [f for f in tif_files if "\\output\\" not in f and "\\dataset\\" not in f]

    if not tif_files:
        print("[ERROR] No GeoTIFF files found!")
        return

    print(f"Found {len(tif_files)} scenes.")
    print(f"Output: {OUTPUT_ROOT}\n")

    os.makedirs(OUTPUT_ROOT, exist_ok=True)

    success = 0
    skipped = 0
    errors = []

    for i, tif_path in enumerate(tif_files, 1):
        parts = tif_path.split(os.sep)
        scene_id = parts[-2]
        location_id = parts[-3]
        folder_name = f"{location_id}_{scene_id}"
        output_dir = os.path.join(OUTPUT_ROOT, folder_name)

        if i % 500 == 0 or i == 1:
            print(f"[{i}/{len(tif_files)}] Processing: {folder_name}")

        try:
            ok, msg = process_scene(tif_path, output_dir)
            if ok:
                success += 1
            else:
                skipped += 1
                if os.path.exists(output_dir):
                    import shutil
                    shutil.rmtree(output_dir)
        except Exception as e:
            errors.append((folder_name, str(e)))
            skipped += 1

    print("\n" + "=" * 60)
    print("STEP 1 COMPLETE")
    print(f"  Valid scenes:   {success}")
    print(f"  Skipped:        {skipped}")
    print(f"  Errors:         {len(errors)}")
    print(f"  Output folder:  {OUTPUT_ROOT}")
    print("=" * 60)

    # Save scene list
    scenes = sorted([d for d in os.listdir(OUTPUT_ROOT) if os.path.isdir(os.path.join(OUTPUT_ROOT, d))])
    with open(os.path.join(OUTPUT_ROOT, "all_scenes.txt"), "w") as f:
        for s in scenes:
            f.write(s + "\n")
    print(f"Scene list saved: {len(scenes)} scenes in all_scenes.txt")


if __name__ == "__main__":
    main()
````

---

## Step 2: Train/Val/Test Split

````python
# filepath: c:\Users\Admin\Downloads\ssl4eo_l_oli_tirs_toa_benchmark\step2_split.py
import os
import random

DATASET_ROOT = r"C:\Users\Admin\Downloads\ssl4eo_l_oli_tirs_toa_benchmark\dataset"
SEED = 42
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1


def main():
    print("=" * 60)
    print("STEP 2: Train/Val/Test Split")
    print("=" * 60)

    scenes_file = os.path.join(DATASET_ROOT, "all_scenes.txt")
    with open(scenes_file, "r") as f:
        scenes = [line.strip() for line in f if line.strip()]

    print(f"Total scenes: {len(scenes)}")

    random.seed(SEED)
    random.shuffle(scenes)

    n = len(scenes)
    train_end = int(n * TRAIN_RATIO)
    val_end = train_end + int(n * VAL_RATIO)

    train_scenes = scenes[:train_end]
    val_scenes = scenes[train_end:val_end]
    test_scenes = scenes[val_end:]

    print(f"  Train: {len(train_scenes)}")
    print(f"  Val:   {len(val_scenes)}")
    print(f"  Test:  {len(test_scenes)}")

    # Save split files
    for split_name, split_scenes in [("train", train_scenes), ("val", val_scenes), ("test", test_scenes)]:
        split_file = os.path.join(DATASET_ROOT, f"{split_name}.txt")
        with open(split_file, "w") as f:
            for s in split_scenes:
                f.write(s + "\n")
        print(f"  Saved: {split_file}")

    print("\nDone!")


if __name__ == "__main__":
    main()
````

---

## Step 3: Compute Dataset Statistics

````python
# filepath: c:\Users\Admin\Downloads\ssl4eo_l_oli_tirs_toa_benchmark\step3_stats.py
import os
import numpy as np

DATASET_ROOT = r"C:\Users\Admin\Downloads\ssl4eo_l_oli_tirs_toa_benchmark\dataset"


def main():
    print("=" * 60)
    print("STEP 3: Compute Dataset Statistics")
    print("=" * 60)

    # Only use training set for computing stats
    train_file = os.path.join(DATASET_ROOT, "train.txt")
    with open(train_file, "r") as f:
        train_scenes = [line.strip() for line in f if line.strip()]

    print(f"Computing stats from {len(train_scenes)} training scenes...")

    # Running stats
    optical_sum = np.zeros(3, dtype=np.float64)
    optical_sq_sum = np.zeros(3, dtype=np.float64)
    thermal_sum = 0.0
    thermal_sq_sum = 0.0
    thermal_min = float('inf')
    thermal_max = float('-inf')
    optical_min = np.full(3, float('inf'))
    optical_max = np.full(3, float('-inf'))
    pixel_count = 0
    thermal_pixel_count = 0

    for i, scene in enumerate(train_scenes):
        if i % 1000 == 0:
            print(f"  [{i}/{len(train_scenes)}]")

        scene_dir = os.path.join(DATASET_ROOT, scene)
        optical = np.load(os.path.join(scene_dir, "optical_hr.npy"))      # (256, 256, 3)
        thermal = np.load(os.path.join(scene_dir, "thermal_hr.npy"))      # (256, 256)

        # Optical stats per channel
        for c in range(3):
            ch = optical[:, :, c].astype(np.float64)
            optical_sum[c] += ch.sum()
            optical_sq_sum[c] += (ch ** 2).sum()
            optical_min[c] = min(optical_min[c], ch.min())
            optical_max[c] = max(optical_max[c], ch.max())

        pixel_count += optical.shape[0] * optical.shape[1]

        # Thermal stats
        th = thermal.astype(np.float64)
        thermal_sum += th.sum()
        thermal_sq_sum += (th ** 2).sum()
        thermal_min = min(thermal_min, th.min())
        thermal_max = max(thermal_max, th.max())
        thermal_pixel_count += th.size

    # Compute mean and std
    optical_mean = optical_sum / pixel_count
    optical_std = np.sqrt(optical_sq_sum / pixel_count - optical_mean ** 2)
    thermal_mean = thermal_sum / thermal_pixel_count
    thermal_std = np.sqrt(thermal_sq_sum / thermal_pixel_count - thermal_mean ** 2)

    print("\n" + "=" * 60)
    print("DATASET STATISTICS (Training Set)")
    print("=" * 60)
    print(f"\nOptical (Bands 4, 3, 2):")
    print(f"  Mean: [{optical_mean[0]:.4f}, {optical_mean[1]:.4f}, {optical_mean[2]:.4f}]")
    print(f"  Std:  [{optical_std[0]:.4f}, {optical_std[1]:.4f}, {optical_std[2]:.4f}]")
    print(f"  Min:  [{optical_min[0]:.4f}, {optical_min[1]:.4f}, {optical_min[2]:.4f}]")
    print(f"  Max:  [{optical_max[0]:.4f}, {optical_max[1]:.4f}, {optical_max[2]:.4f}]")
    print(f"\nThermal (Band 10):")
    print(f"  Mean: {thermal_mean:.4f}")
    print(f"  Std:  {thermal_std:.4f}")
    print(f"  Min:  {thermal_min:.4f}")
    print(f"  Max:  {thermal_max:.4f}")

    # Save stats
    stats = {
        "optical_mean": optical_mean,
        "optical_std": optical_std,
        "optical_min": optical_min,
        "optical_max": optical_max,
        "thermal_mean": thermal_mean,
        "thermal_std": thermal_std,
        "thermal_min": thermal_min,
        "thermal_max": thermal_max,
    }
    np.save(os.path.join(DATASET_ROOT, "dataset_stats.npy"), stats)
    print(f"\nStats saved to: {os.path.join(DATASET_ROOT, 'dataset_stats.npy')}")


if __name__ == "__main__":
    main()
````

---

## Step 4: PyTorch Dataset & DataLoader

````python
# filepath: c:\Users\Admin\Downloads\ssl4eo_l_oli_tirs_toa_benchmark\step4_dataloader.py
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
````

---

## Step 5: Model Architecture

````python
# filepath: c:\Users\Admin\Downloads\ssl4eo_l_oli_tirs_toa_benchmark\step5_model.py
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

    def forward(self, x
```

