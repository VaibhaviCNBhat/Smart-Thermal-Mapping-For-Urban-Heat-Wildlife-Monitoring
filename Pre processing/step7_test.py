import os
import torch
import torch.serialization
import numpy as np
import random
import time
from step4_dataloader import ThermalSRDataset
from step5_model import OpticalGuidedThermalSR
import rasterio
from rasterio.transform import from_origin

# Paths
DATASET_ROOT = r"C:\Users\Admin\Downloads\ssl4eo_l_oli_tirs_toa_benchmark\dataset"
CHECKPOINT_PATH = r"C:\Users\Admin\Downloads\ssl4eo_l_oli_tirs_toa_benchmark\checkpoints\best_model.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load test dataset
print("Loading test dataset...")
test_ds = ThermalSRDataset(DATASET_ROOT, split="test")

# Target specific IDs requested by user
target_ids = [
    "0003342_LC08_044034_20190706",
    "0027735_LC08_031027_20190828",
    "0017390_LC08_046029_20190720"
]

# Find indices for target IDs
target_indices = []
for i, scene in enumerate(test_ds.scenes):
    if scene in target_ids:
        target_indices.append(i)

if not target_indices:
    print("Warning: None of the target IDs were found in the test dataset.")
else:
    print(f"Found {len(target_indices)} target samples: {[test_ds.scenes[i] for i in target_indices]}")

# Use specific indices instead of random sampling
random_indices = target_indices

# Load model
print("Loading model from checkpoint...")
model = OpticalGuidedThermalSR().to(DEVICE)
# Fix for PyTorch 2.6 weights loading error
import numpy
with torch.serialization.safe_globals([numpy.dtype, numpy._core.multiarray.scalar, numpy.dtypes.Float64DType]):
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE, weights_only=False)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Test loop
psnr_list = []

def compute_psnr(pred, target, max_val=None):
    mse = torch.mean((pred - target) ** 2).item()
    if mse == 0:
        return float('inf')
    if max_val is None:
        max_val = target.max().item() - target.min().item()
    return 10 * np.log10(max_val ** 2 / mse)

print("Running test evaluation...")


import matplotlib.pyplot as plt
os.makedirs("predictions", exist_ok=True)

timestamp = int(time.time())

for i, idx in enumerate(random_indices):
    batch = test_ds[idx]
    optical = torch.tensor(batch["optical_hr"]).unsqueeze(0).to(DEVICE)
    thermal_lr = torch.tensor(batch["thermal_lr"]).unsqueeze(0).to(DEVICE)
    thermal_hr = torch.tensor(batch["thermal_hr"]).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        pred = model(optical, thermal_lr)

    mse = torch.mean((pred - thermal_hr) ** 2).item()
    psnr = compute_psnr(pred, thermal_hr)
    psnr_list.append(psnr)

    pred_img = pred.cpu().squeeze().numpy()
    gt_img = thermal_hr.cpu().squeeze().numpy()
    ip_img = thermal_lr.cpu().squeeze().numpy()
    # Ensure 2D for PNG
    if ip_img.ndim > 2:
        ip_img = ip_img[0]
    pred_img = (pred_img - pred_img.min()) / (pred_img.max() - pred_img.min() + 1e-8)
    gt_img = (gt_img - gt_img.min()) / (gt_img.max() - gt_img.min() + 1e-8)
    ip_img = (ip_img - ip_img.min()) / (ip_img.max() - ip_img.min() + 1e-8)
    # Use scene name for unique filenames
    scene_name = test_ds.scenes[idx]
    
    # Save input image (thermal_lr)
    plt.imsave(f"predictions/ip_{scene_name}.png", ip_img, cmap="gray")
    
    # Save predicted image
    plt.imsave(f"predictions/pred_{scene_name}.png", pred_img, cmap="gray")

    # Save ground truth image (thermal_hr)
    plt.imsave(f"predictions/gt_{scene_name}.png", gt_img, cmap="gray")

    print(f"Saved results for {scene_name} (PSNR: {psnr:.2f} dB)")

print(f"Test PSNR (mean): {np.mean(psnr_list):.2f} dB")

# Load validation dataset
print("Loading validation dataset...")
val_ds = ThermalSRDataset(DATASET_ROOT, split="val")
val_loader = torch.utils.data.DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=0)

# Test loop
psnr_list = []

print("Running validation evaluation...")

for batch_idx, batch in enumerate(val_loader):
    optical = batch["optical_hr"].to(DEVICE)
    thermal_lr = batch["thermal_lr"].to(DEVICE)
    thermal_hr = batch["thermal_hr"].to(DEVICE)

    with torch.no_grad():
        pred = model(optical, thermal_lr)

    # Compute PSNR
    psnr = compute_psnr(pred, thermal_hr)
    psnr_list.append(psnr)

    # Save first 5 predictions for inspection as PNG files
    if batch_idx < 5:
        pred_img = pred.cpu().squeeze().numpy()
        gt_img = thermal_hr.cpu().squeeze().numpy()
        ip_img = thermal_lr.cpu().squeeze().numpy()
        # Ensure 2D for PNG
        if ip_img.ndim > 2:
            ip_img = ip_img[0]
        pred_img = (pred_img - pred_img.min()) / (pred_img.max() - pred_img.min() + 1e-8)
        gt_img = (gt_img - gt_img.min()) / (gt_img.max() - gt_img.min() + 1e-8)
        ip_img = (ip_img - ip_img.min()) / (ip_img.max() - ip_img.min() + 1e-8)
        plt.imsave(f"predictions/pred_{batch_idx}.png", pred_img, cmap="gray")
        plt.imsave(f"predictions/gt_{batch_idx}.png", gt_img, cmap="gray")
        plt.imsave(f"predictions/ip_{batch_idx}.png", ip_img, cmap="gray")
        # Save prediction as GeoTIFF (.tif)
        tif_path = f"predictions/pred_{batch_idx}.tif"
        transform = from_origin(0, 0, 1, 1)
        with rasterio.open(
            tif_path,
            'w',
            driver='GTiff',
            height=pred_img.shape[0],
            width=pred_img.shape[1],
            count=1,
            dtype=pred_img.dtype,
            transform=transform,
            crs=None
        ) as dst:
            dst.write(pred_img, 1)

print(f"Validation PSNR (mean): {np.mean(psnr_list):.2f} dB")
