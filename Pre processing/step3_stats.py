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
        if i % 2000 == 0:
            print(f"  [{i}/{len(train_scenes)}]...")

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
