"""Extract coordinates from original GeoTIFFs and save as lookup JSON - batch version."""
import rasterio
import os
import json
import numpy as np
import sys

tif_root = r"C:\Users\Admin\Downloads\ssl4eo_l_oli_tirs_toa_benchmark"
dataset_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataset")
output_path = os.path.join(dataset_root, "scene_coordinates.json")

scenes = sorted([s for s in os.listdir(dataset_root) 
                 if os.path.isdir(os.path.join(dataset_root, s)) and s.startswith('0')])

print(f"Processing {len(scenes)} scenes...")
coords = {}
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
    
    try:
        with rasterio.open(tif_path) as src:
            cx = (src.bounds.left + src.bounds.right) / 2
            cy = (src.bounds.bottom + src.bounds.top) / 2
            coords[scene] = {
                "lat": round(cy, 6),
                "lon": round(cx, 6),
                "bounds": [
                    round(src.bounds.left, 6),
                    round(src.bounds.bottom, 6),
                    round(src.bounds.right, 6),
                    round(src.bounds.top, 6)
                ]
            }
            found += 1
    except Exception as e:
        pass

    if found % 1000 == 0 and found > 0:
        sys.stderr.write(f"  ...{found} done\n")

with open(output_path, "w") as f:
    json.dump(coords, f)

print(f"Done: {found} scenes -> {output_path}")
print(f"File size: {os.path.getsize(output_path)} bytes")
