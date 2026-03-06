import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader
from step4_dataloader import ThermalSRDataset
from step5_model import OpticalGuidedThermalSR

# === CONFIGURATION ===
DATASET_ROOT = r"C:\Users\Admin\Downloads\ssl4eo_l_oli_tirs_toa_benchmark\dataset"
CHECKPOINT_DIR = r"C:\Users\Admin\Downloads\ssl4eo_l_oli_tirs_toa_benchmark\checkpoints"
BATCH_SIZE = 2
# Set to 10 for initial training. After resuming, set to 15.
NUM_EPOCHS = 10
LEARNING_RATE = 1e-4
SCALE_FACTOR = 4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === RESUME TRAINING CONFIG ===
RESUME = True  # Set to True to resume from checkpoint
RESUME_CHECKPOINT = os.path.join(CHECKPOINT_DIR, "best_model.pth")


def compute_psnr(pred, target, max_val=None):
    mse = torch.mean((pred - target) ** 2).item()
    if mse == 0:
        return float('inf')
    if max_val is None:
        max_val = target.max().item() - target.min().item()
    return 10 * np.log10(max_val ** 2 / mse)


def compute_ssim_simple(pred, target, window_size=11):
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    mu_pred = torch.nn.functional.avg_pool2d(pred, window_size, stride=1, padding=window_size // 2)
    mu_target = torch.nn.functional.avg_pool2d(target, window_size, stride=1, padding=window_size // 2)

    mu_pred_sq = mu_pred ** 2
    mu_target_sq = mu_target ** 2
    mu_pred_target = mu_pred * mu_target

    sigma_pred_sq = torch.nn.functional.avg_pool2d(pred ** 2, window_size, stride=1, padding=window_size // 2) - mu_pred_sq
    sigma_target_sq = torch.nn.functional.avg_pool2d(target ** 2, window_size, stride=1, padding=window_size // 2) - mu_target_sq
    sigma_pred_target = torch.nn.functional.avg_pool2d(pred * target, window_size, stride=1, padding=window_size // 2) - mu_pred_target

    numerator = (2 * mu_pred_target + C1) * (2 * sigma_pred_target + C2)
    denominator = (mu_pred_sq + mu_target_sq + C1) * (sigma_pred_sq + sigma_target_sq + C2)

    ssim_map = numerator / denominator
    return ssim_map.mean().item()


class ThermalSRLoss(nn.Module):
    def __init__(self, edge_weight=0.1):
        super().__init__()
        self.l1_loss = nn.L1Loss()
        self.edge_weight = edge_weight
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).reshape(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).reshape(1, 1, 3, 3)
        self.register_buffer('sobel_x', sobel_x)
        self.register_buffer('sobel_y', sobel_y)

    def edge_loss(self, pred, target):
        pred_edge_x = torch.nn.functional.conv2d(pred, self.sobel_x, padding=1)
        pred_edge_y = torch.nn.functional.conv2d(pred, self.sobel_y, padding=1)
        target_edge_x = torch.nn.functional.conv2d(target, self.sobel_x, padding=1)
        target_edge_y = torch.nn.functional.conv2d(target, self.sobel_y, padding=1)
        pred_edges = torch.sqrt(pred_edge_x ** 2 + pred_edge_y ** 2 + 1e-8)
        target_edges = torch.sqrt(target_edge_x ** 2 + target_edge_y ** 2 + 1e-8)
        return self.l1_loss(pred_edges, target_edges)

    def forward(self, pred, target):
        pixel_loss = self.l1_loss(pred, target)
        e_loss = self.edge_loss(pred, target)
        return pixel_loss + self.edge_weight * e_loss


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    total_psnr = 0
    count = 0
    for batch_idx, batch in enumerate(loader):
        optical = batch["optical_hr"].to(device)
        thermal_lr = batch["thermal_lr"].to(device)
        thermal_hr = batch["thermal_hr"].to(device)
        optimizer.zero_grad()
        pred = model(optical, thermal_lr)
        loss = criterion(pred, thermal_hr)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        total_psnr += compute_psnr(pred.detach(), thermal_hr)
        count += 1
        if batch_idx % 10 == 0:
            print(f"  Training batch {batch_idx}/{len(loader)}")
    return total_loss / count, total_psnr / count


def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    total_psnr = 0
    total_ssim = 0
    count = 0
    with torch.no_grad():
        for batch in loader:
            optical = batch["optical_hr"].to(device)
            thermal_lr = batch["thermal_lr"].to(device)
            thermal_hr = batch["thermal_hr"].to(device)
            pred = model(optical, thermal_lr)
            loss = criterion(pred, thermal_hr)
            total_loss += loss.item()
            total_psnr += compute_psnr(pred, thermal_hr)
            total_ssim += compute_ssim_simple(pred, thermal_hr)
            count += 1
    return total_loss / count, total_psnr / count, total_ssim / count


def main():
    print("=" * 60)
    print("STEP 6: Training")
    print("=" * 60)
    print("Training script started, initializing...")
    print(f"Device: {DEVICE}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Epochs: {NUM_EPOCHS}")
    print(f"Learning rate: {LEARNING_RATE}")
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    print("Loading datasets...")
    train_ds = ThermalSRDataset(DATASET_ROOT, split="train", scale_factor=SCALE_FACTOR)
    val_ds = ThermalSRDataset(DATASET_ROOT, split="val", scale_factor=SCALE_FACTOR)
    print("Creating dataloaders...")
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=1, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=1, pin_memory=True)
    print("Building model...")
    model = OpticalGuidedThermalSR(scale_factor=SCALE_FACTOR).to(DEVICE)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")
    criterion = ThermalSRLoss(edge_weight=0.1).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    start_epoch = 1
    best_val_loss = float('inf')

    # Resume logic
    if RESUME and os.path.exists(RESUME_CHECKPOINT):
        print(f"Resuming from checkpoint: {RESUME_CHECKPOINT}")
        checkpoint = torch.load(RESUME_CHECKPOINT, map_location=DEVICE, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_loss = checkpoint.get('val_loss', float('inf'))
        print(f"Resumed from epoch {checkpoint['epoch']}")

    train_log = []
    print("Starting training loop...")
    for epoch in range(start_epoch, NUM_EPOCHS + 1):
        print(f"Starting epoch {epoch}...")
        start = time.time()
        train_loss, train_psnr = train_one_epoch(model, train_loader, criterion, optimizer, DEVICE)
        val_loss, val_psnr, val_ssim = validate(model, val_loader, criterion, DEVICE)
        scheduler.step(val_loss)
        elapsed = time.time() - start
        log_entry = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_psnr": train_psnr,
            "val_loss": val_loss,
            "val_psnr": val_psnr,
            "val_ssim": val_ssim,
            "lr": optimizer.param_groups[0]['lr'],
            "time": elapsed
        }
        train_log.append(log_entry)
        print(f"Epoch [{epoch}/{NUM_EPOCHS}] "
              f"Train Loss: {train_loss:.6f} | Train PSNR: {train_psnr:.2f} dB | "
              f"Val Loss: {val_loss:.6f} | Val PSNR: {val_psnr:.2f} dB | "
              f"Val SSIM: {val_ssim:.4f} | "
              f"LR: {optimizer.param_groups[0]['lr']:.6f} | "
              f"Time: {elapsed:.1f}s")
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_psnr': val_psnr,
                'val_ssim': val_ssim,
            }, os.path.join(CHECKPOINT_DIR, "best_model.pth"))
            print(f"  ★ Best model saved! (Val Loss: {val_loss:.6f})")
        if epoch % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, os.path.join(CHECKPOINT_DIR, f"checkpoint_epoch{epoch}.pth"))
    np.save(os.path.join(CHECKPOINT_DIR, "training_log.npy"), train_log)
    print(f"\nTraining complete! Best Val Loss: {best_val_loss:.6f}")

if __name__ == "__main__":
    main()
