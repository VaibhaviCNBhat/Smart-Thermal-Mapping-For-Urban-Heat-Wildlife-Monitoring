import rasterio
import matplotlib.pyplot as plt
import numpy as np

file_path = r"C:\Users\Admin\Downloads\ssl4eo_l_oli_tirs_toa_benchmark\0000291\LC08_029030_20190729\all_bands.tif"

with rasterio.open(file_path) as src:
    print(f"Total bands: {src.count}")
    print(f"Image size: {src.width} x {src.height}")

    # Check if band 10 exists
    if src.count >= 10:
        band10 = src.read(10)
    else:
        print(f"Only {src.count} bands available. Reading last band instead.")
        band10 = src.read(src.count)

# Normalize for display
band10_float = band10.astype(np.float32)
band10_norm = (band10_float - band10_float.min()) / (band10_float.max() - band10_float.min() + 1e-10)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Thermal colormap (good for thermal IR data)
axes[0].imshow(band10_norm, cmap='inferno')
axes[0].set_title("Band 10 - Thermal IR (inferno)")
axes[0].axis('off')

# Grayscale version
axes[1].imshow(band10_norm, cmap='gray')
axes[1].set_title("Band 10 - Thermal IR (grayscale)")
axes[1].axis('off')

plt.tight_layout()
plt.show()

print(f"Band 10 min: {band10.min()}, max: {band10.max()}, mean: {band10.mean():.2f}")