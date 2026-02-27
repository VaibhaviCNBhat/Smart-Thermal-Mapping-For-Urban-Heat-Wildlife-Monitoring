import numpy as np

# Load the .npy files
base = r"C:\Users\Admin\Downloads\ssl4eo_l_oli_tirs_toa_benchmark\dataset\0000000_LC08_045030_20190814"

optical = np.load(base + r"\optical_hr.npy")

print("=" * 60)
print("optical_hr.npy Contents")
print("=" * 60)
print(f"Shape:  {optical.shape}")
print(f"Dtype:  {optical.dtype}")
print(f"Size:   {optical.nbytes / 1024:.1f} KB")

channel_names = ["Band 4 (Red)", "Band 3 (Green)", "Band 2 (Blue)"]
for c in range(3):
    ch = optical[:, :, c]
    print(f"\nChannel {c} - {channel_names[c]}:")
    print(f"  Min:    {ch.min():.6f}")
    print(f"  Max:    {ch.max():.6f}")
    print(f"  Mean:   {ch.mean():.6f}")
    print(f"  Std:    {ch.std():.6f}")

if optical.max() <= 1.5:
    print("\nOptical: TOA REFLECTANCE (0 to ~1.0)")
elif optical.max() > 200:
    print("\nOptical: Raw DN (Digital Numbers)")
else:
    print(f"\nOptical range: {optical.min():.4f} to {optical.max():.4f}")

# Thermal
thermal_hr = np.load(base + r"\thermal_hr.npy")
thermal_lr = np.load(base + r"\thermal_lr.npy")

print("\n" + "=" * 60)
print("thermal_hr.npy")
print("=" * 60)
print(f"Shape:  {thermal_hr.shape}")
print(f"Dtype:  {thermal_hr.dtype}")
print(f"Min:    {thermal_hr.min():.6f}")
print(f"Max:    {thermal_hr.max():.6f}")
print(f"Mean:   {thermal_hr.mean():.6f}")

print(f"\nthermal_lr.npy")
print("=" * 60)
print(f"Shape:  {thermal_lr.shape}")
print(f"Dtype:  {thermal_lr.dtype}")
print(f"Min:    {thermal_lr.min():.6f}")
print(f"Max:    {thermal_lr.max():.6f}")
print(f"Mean:   {thermal_lr.mean():.6f}")

print(f"\n--- Thermal Value Interpretation ---")
if thermal_hr.max() > 200:
    print("Values are in KELVIN range (200-350K)")
    print("Can directly compute RMSE in Kelvin. No conversion needed!")
elif thermal_hr.max() < 1.5:
    print("Values are TOA REFLECTANCE/RADIANCE (0-1 range)")
    print("Need Planck equation: T = K2 / ln(K1/L + 1)")
    print("K1=774.8853, K2=1321.0789 for Band 10")
elif thermal_hr.max() < 50:
    print("Values likely TOA RADIANCE (W/m2/sr/um)")
    print("Need Planck equation to convert to Kelvin")
else:
    print(f"Values range: {thermal_hr.min():.4f} to {thermal_hr.max():.4f}")
