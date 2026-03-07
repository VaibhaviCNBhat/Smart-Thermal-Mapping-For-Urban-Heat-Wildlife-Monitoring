"""Extract coordinates from original GeoTIFFs and save as lookup JSON."""
import rasterio
import os
import json
import numpy as np

tif_root = r"C:\Users\Admin\Downloads\ssl4eo_l_oli_tirs_toa_benchmark"
dataset_root = "dataset"

coords = {}
scenes = sorted([s for s in os.listdir(dataset_root) 
                 if os.path.isdir(os.path.join(dataset_root, s)) and s.startswith('0')])

found = 0
for scene in scenes:
    scene_id = scene.split('_')[0]
    tif_scene = os.path.join(tif_root, scene_id)
    if not os.path.exists(tif_scene):
        continue
    subdirs = [d for d in os.listdir(tif_scene) if d.startswith('LC08')]
    if not subdirs:
        continue
    tif_path = os.path.join(tif_scene, subdirs[0], 'all_bands.tif')
    if not os.path.exists(tif_path):
        continue
    
    with rasterio.open(tif_path) as src:
        cx = (src.bounds.left + src.bounds.right) / 2
        cy = (src.bounds.bottom + src.bounds.top) / 2
        b10 = src.read(10).astype(np.float32)
        coords[scene] = {
            "lat": round(cy, 6),
            "lon": round(cx, 6),
            "bounds": [
                round(src.bounds.left, 6),
                round(src.bounds.bottom, 6),
                round(src.bounds.right, 6),
                round(src.bounds.top, 6)
            ],
            "b10_min": float(b10.min()),
            "b10_max": float(b10.max()),
            "b10_mean": float(b10.mean())
        }
        found += 1
        print(f"  [{found}] {scene}: lat={cy:.4f}, lon={cx:.4f}, B10=[{b10.min():.0f}-{b10.max():.0f}]")

print(f"\nTotal: {found} scenes with coordinates")

# Save lookup
output_path = os.path.join(dataset_root, "scene_coordinates.json")
with open(output_path, "w") as f:
    json.dump(coords, f, indent=2)
print(f"Saved to {output_path}")
