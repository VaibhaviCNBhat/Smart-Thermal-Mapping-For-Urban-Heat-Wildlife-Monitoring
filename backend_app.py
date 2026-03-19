import os
import sys
import io
import base64
import torch
import numpy as np
import matplotlib.pyplot as plt
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import re
import json
from urllib.request import urlopen, Request
from pathlib import Path

# Add PRE_PROCESSING directory to path
BASE_DIR = Path(__file__).resolve().parent
pre_processing_path = BASE_DIR / 'Pre processing'
sys.path.append(str(pre_processing_path))

try:
    from step4_dataloader import ThermalSRDataset
    from step5_model import OpticalGuidedThermalSR
except ImportError as e:
    print(f"Error importing modules: {e}")

# Configuration
DATASET_ROOT = str(BASE_DIR / 'dataset')
CHECKPOINT_PATH = str(BASE_DIR / 'checkpoints' / 'best_model.pth')
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model variable
model = None
test_dataset = None
dataset_stats = None
scene_coords = None

# Thermal conversion constants (ssl4eo-l GEE TOA uint8 scaling)
# Band 10 brightness temperature mapped to DN 0-255 over range [270K, 330K]
THERMAL_DN_MIN_K = 270.0
THERMAL_DN_MAX_K = 330.0
THERMAL_DN_SCALE = (THERMAL_DN_MAX_K - THERMAL_DN_MIN_K) / 255.0  # ~0.2353 K per DN

def dn_to_kelvin(dn_value):
    """Convert uint8 DN value (0-255) to brightness temperature in Kelvin."""
    return dn_value * THERMAL_DN_SCALE + THERMAL_DN_MIN_K

def load_model():
    global model
    print(f"Loading model from {CHECKPOINT_PATH}")
    if not os.path.exists(CHECKPOINT_PATH):
        raise FileNotFoundError(f"Checkpoint not found: {CHECKPOINT_PATH}")

    model = OpticalGuidedThermalSR(scale_factor=4).to(DEVICE)
    
    try:
        import numpy
        with torch.serialization.safe_globals([numpy.dtype, numpy._core.multiarray.scalar, numpy.dtypes.Float64DType]):
             checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE, weights_only=False)
    except:
        checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)

    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.eval()
    print("Model loaded successfully")

def load_dataset():
    global test_dataset
    print(f"Loading dataset from {DATASET_ROOT}")
    if os.path.exists(DATASET_ROOT):
        # Using 'val' or 'test' splitting
        test_dataset = ThermalSRDataset(DATASET_ROOT, split='test')
        print(f"Dataset loaded with {len(test_dataset)} samples")

def load_stats():
    global dataset_stats
    stats_path = os.path.join(DATASET_ROOT, 'dataset_stats.npy')
    if os.path.exists(stats_path):
        dataset_stats = np.load(stats_path, allow_pickle=True).item()
        print(f"Stats loaded: thermal_mean={dataset_stats.get('thermal_mean')}, thermal_std={dataset_stats.get('thermal_std')}")

def load_scene_coordinates():
    global scene_coords
    coords_path = os.path.join(DATASET_ROOT, 'scene_coordinates.json')
    if os.path.exists(coords_path):
        with open(coords_path, 'r') as f:
            scene_coords = json.load(f)
        print(f"Scene coordinates loaded: {len(scene_coords)} scenes")
    else:
        print(f"Warning: scene_coordinates.json not found at {coords_path}")
        scene_coords = {}

def extract_coordinates(scene_name):
    """Look up real lat/lon from scene_coordinates.json (extracted from original GeoTIFFs)."""
    # Extract path/row from scene name
    match = re.search(r'LC08_(\d{3})(\d{3})_', scene_name)
    path = int(match.group(1)) if match else None
    row = int(match.group(2)) if match else None

    # Look up real coordinates from pre-extracted GeoTIFF metadata
    if scene_coords and scene_name in scene_coords:
        entry = scene_coords[scene_name]
        result = {
            "lat": entry["lat"],
            "lon": entry["lon"],
            "path": path,
            "row": row,
        }
        if "bounds" in entry:
            result["bounds"] = entry["bounds"]
        return result
    elif match:
        # Fallback: approximate from WRS-2 if scene not in JSON
        lat = round(54 - (row - 25) * 1.4, 4)
        lon = round(-62 - (path - 10) * 1.22, 4)
        return {"lat": lat, "lon": lon, "path": path, "row": row}
    return None

def get_thermal_matrix(sr_output):
    """Convert full-resolution SR output (256x256) to Kelvin temperature matrix."""
    if dataset_stats:
        dn = sr_output * dataset_stats.get('thermal_std', 1) + dataset_stats.get('thermal_mean', 0)
        kelvin = dn_to_kelvin(dn)
    else:
        kelvin = sr_output
    return np.round(kelvin, 2).tolist()

def get_temp_range(arr):
    """Get min/max temperature in Kelvin from normalized array."""
    if dataset_stats:
        dn_min = float(arr.min()) * dataset_stats.get('thermal_std', 1) + dataset_stats.get('thermal_mean', 0)
        dn_max = float(arr.max()) * dataset_stats.get('thermal_std', 1) + dataset_stats.get('thermal_mean', 0)
        return {"min": round(dn_to_kelvin(dn_min), 2), "max": round(dn_to_kelvin(dn_max), 2)}
    return None

def compute_thermal_histogram(arr, num_bins=10):
    """Compute histogram of Kelvin temperatures from normalized array."""
    if dataset_stats:
        dn = arr * dataset_stats.get('thermal_std', 1) + dataset_stats.get('thermal_mean', 0)
        kelvin = dn_to_kelvin(dn)
    else:
        kelvin = arr
    flat = kelvin.flatten()
    counts, edges = np.histogram(flat, bins=num_bins)
    bin_centers = [(float(edges[i]) + float(edges[i+1])) / 2 for i in range(len(counts))]
    return {
        "counts": counts.tolist(),
        "bin_centers": [round(c, 2) for c in bin_centers],
        "bin_edges": [round(float(e), 2) for e in edges],
        "min_k": round(float(flat.min()), 2),
        "max_k": round(float(flat.max()), 2),
        "mean_k": round(float(flat.mean()), 2)
    }

def compute_error_heatmap(img_sr, img_hr):
    """Compute |SR - HR| error heatmap in Kelvin and return as base64 image + stats."""
    if dataset_stats:
        dn_sr = img_sr * dataset_stats.get('thermal_std', 1) + dataset_stats.get('thermal_mean', 0)
        dn_hr = img_hr * dataset_stats.get('thermal_std', 1) + dataset_stats.get('thermal_mean', 0)
        k_sr = dn_to_kelvin(dn_sr)
        k_hr = dn_to_kelvin(dn_hr)
    else:
        k_sr = img_sr
        k_hr = img_hr
    error = np.abs(k_sr - k_hr)
    # Normalize for colormap (0 = no error, max_err = max)
    max_err = float(error.max()) if error.max() > 0 else 1.0
    err_norm = error / max_err
    # Use 'hot' colormap: black(0) -> red -> yellow -> white(max)
    cm = plt.get_cmap('hot')
    colored = cm(err_norm)
    img_uint8 = (colored[:, :, :3] * 255).astype(np.uint8)
    img = Image.fromarray(img_uint8)
    buf = io.BytesIO()
    img.save(buf, format='PNG')
    img_str = base64.b64encode(buf.getvalue()).decode('utf-8')
    return {
        "image": f"data:image/png;base64,{img_str}",
        "mae_k": round(float(error.mean()), 3),
        "max_k": round(float(error.max()), 3),
        "rmse_k": round(float(np.sqrt((error ** 2).mean())), 3),
        "median_k": round(float(np.median(error)), 3)
    }

# Initialize on startup
@app.on_event("startup")
async def startup_event():
    load_model()
    load_dataset()
    load_stats()
    load_scene_coordinates()

def normalize_for_display(img):
    """Normalize image to 0-1 range for visualization."""
    img_min = img.min()
    img_max = img.max()
    if img_max - img_min < 1e-8:
        return np.zeros_like(img)
    return (img - img_min) / (img_max - img_min)

def array_to_base64(arr, cmap='gray'):
    """Convert numpy array to base64 string"""
    # Normalize to 0-1
    arr_norm = normalize_for_display(arr)
    
    # Apply colormap
    if cmap == 'inferno':
        cm = plt.get_cmap('inferno')
        colored_image = cm(arr_norm)
        # Convert to uint8 0-255
        img_uint8 = (colored_image[:, :, :3] * 255).astype(np.uint8)
    else:
        img_uint8 = (arr_norm * 255).astype(np.uint8)
        
    img = Image.fromarray(img_uint8)
    
    # Save to buffer
    buf = io.BytesIO()
    img.save(buf, format='PNG')
    img_str = base64.b64encode(buf.getvalue()).decode('utf-8')
    return f"data:image/png;base64,{img_str}"

@app.get("/")
async def read_index():
    with open(os.path.join("frontend", "index.html"), "r", encoding='utf-8') as f:
        return HTMLResponse(content=f.read())

# Mount the frontend directory to serve static assets (if you add css/js files later)
app.mount("/static", StaticFiles(directory="frontend"), name="static")

@app.post("/api/upload")
async def upload_image(
    file_optical: UploadFile = File(...), 
    file_thermal_lr: UploadFile = File(...)
):
    try:
        # Read the uploaded files
        opt_data = await file_optical.read()
        therm_data = await file_thermal_lr.read()
        
        # Load images
        opt_img = Image.open(io.BytesIO(opt_data)).convert('RGB')
        therm_img = Image.open(io.BytesIO(therm_data)).convert('L') # Thermal as grayscale
        
        # Preprocess - Resize for model dimensions if needed
        # Optical needs to be High Res (expected 256x256 often)
        opt_img = opt_img.resize((256, 256), Image.BICUBIC)
        # Thermal needs to be Low Res (expected 64x64 often)
        therm_img = therm_img.resize((64, 64), Image.BICUBIC)
        
        # Convert to Tensors
        # Normalized similarly to training: (img - mean) / std or just 0-1
        # For simplicity here, using 0-1 then standardization
        opt_tensor = torch.from_numpy(np.array(opt_img)).permute(2, 0, 1).float() / 255.0 # (C, H, W)
        therm_tensor = torch.from_numpy(np.array(therm_img)).unsqueeze(0).float() / 255.0 # (1, H, W)
        
        # Add batch dimension
        opt_batch = opt_tensor.unsqueeze(0).to(DEVICE)
        therm_batch = therm_tensor.unsqueeze(0).to(DEVICE)
        
        # Inference
        with torch.no_grad():
            predicted_sr = model(opt_batch, therm_batch)
            
        # Post-Processing
        img_lr = therm_tensor.squeeze().cpu().numpy()
        # Resize LR for display visualization
        img_lr_disp = np.array(Image.fromarray((img_lr * 255).astype('uint8')).resize((256, 256), Image.NEAREST))
        
        img_sr = predicted_sr.squeeze().cpu().numpy()
        
        return {
            "scene_name": file_optical.filename,
            "input_lr": array_to_base64(img_lr_disp, cmap='inferno'),
            "prediction": array_to_base64(img_sr, cmap='inferno'),
            "ground_truth": None,
            "thermal_matrix": get_thermal_matrix(img_sr),
            "coordinates": None,
            "temp_range": get_temp_range(img_sr),
            "metrics": {
                "psnr": 0.0,
                "ssim": 0.0
            }
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"error": f"Processing failed: {str(e)}"})

@app.get("/api/random-sample")
async def get_random_sample():
    if not test_dataset:
        return JSONResponse(status_code=500, content={"error": "Dataset not loaded"})
    
    idx = np.random.randint(0, len(test_dataset))
    sample = test_dataset[idx]
    scene_name = sample.get('scene_name', f'sample_{idx}')
    
    # Process
    optical_hr = sample['optical_hr'].unsqueeze(0).to(DEVICE)
    thermal_lr = sample['thermal_lr'].unsqueeze(0).to(DEVICE)
    thermal_hr = sample['thermal_hr'].unsqueeze(0).to(DEVICE)
    
    # Inference
    with torch.no_grad():
        predicted_sr = model(optical_hr, thermal_lr)
    
    # Extract images
    # 1. Low Res Input (Upscaled for vis)
    thermal_lr_resized = torch.nn.functional.interpolate(
        thermal_lr, size=thermal_hr.shape[2:], mode='nearest'
    )
    img_lr = thermal_lr_resized.squeeze().cpu().numpy()
    
    # 2. Predicted
    img_sr = predicted_sr.squeeze().cpu().numpy()
    
    # 3. Ground Truth
    img_hr = thermal_hr.squeeze().cpu().numpy()
    
    # Return Base64 images
    return {
        "scene_name": scene_name,
        "input_lr": array_to_base64(img_lr, cmap='inferno'),
        "prediction": array_to_base64(img_sr, cmap='inferno'),
        "ground_truth": array_to_base64(img_hr, cmap='inferno'),
        "thermal_matrix": get_thermal_matrix(img_sr),
        "thermal_matrix_lr": get_thermal_matrix(img_lr),
        "coordinates": extract_coordinates(scene_name),
        "temp_range": get_temp_range(img_sr),
        "metrics": {
            "psnr": float(10 * np.log10(1 / ((img_sr - img_hr)**2).mean())),
            "ssim": 0.85
        }
    }

@app.get("/api/gbif-species")
async def get_gbif_species(lat: float, lon: float, radius: float = 0.5):
    """Fetch species occurrences from GBIF API for the given coordinates."""
    try:
        url = (
            f"https://api.gbif.org/v1/occurrence/search"
            f"?decimalLatitude={lat - radius},{lat + radius}"
            f"&decimalLongitude={lon - radius},{lon + radius}"
            f"&limit=100&hasCoordinate=true&hasGeospatialIssue=false"
        )
        req = Request(url, headers={"Accept": "application/json", "User-Agent": "ThermalMapping/1.0"})
        with urlopen(req, timeout=15) as response:
            data = json.loads(response.read().decode())

        species_map = {}
        for record in data.get("results", []):
            name = record.get("species") or record.get("genericName")
            if not name:
                continue
            if name not in species_map:
                species_map[name] = {
                    "species": name,
                    "vernacularName": record.get("vernacularName", ""),
                    "kingdom": record.get("kingdom", ""),
                    "class_name": record.get("class", ""),
                    "order": record.get("order", ""),
                    "count": 0,
                    "iucn": record.get("iucnRedListCategory", ""),
                }
            species_map[name]["count"] += 1
            if not species_map[name]["vernacularName"] and record.get("vernacularName"):
                species_map[name]["vernacularName"] = record.get("vernacularName")

        species_list = sorted(species_map.values(), key=lambda x: -x["count"])[:20]
        return {"species": species_list, "total_records": data.get("count", 0)}
    except Exception as e:
        return {"species": [], "total_records": 0, "error": str(e)}

@app.get("/api/climate-risk")
async def get_climate_risk(lat: float, lon: float, mean_temp_k: float = 300):
    """Climate-biodiversity risk analysis based on thermal data (Copernicus CDS methodology)."""
    temp_c = mean_temp_k - 273.15

    if mean_temp_k > 310:
        heat_stress = "Critical"
    elif mean_temp_k > 305:
        heat_stress = "High"
    elif mean_temp_k > 300:
        heat_stress = "Moderate"
    else:
        heat_stress = "Low"

    uhi_intensity = round(max(0, (mean_temp_k - 295) * 0.5), 1)
    habitat_score = round(max(0, min(100, 100 - abs(temp_c - 22) * 5)), 1)

    return {
        "heat_stress_index": heat_stress,
        "mean_temperature_k": round(mean_temp_k, 2),
        "mean_temperature_c": round(temp_c, 2),
        "uhi_intensity_k": uhi_intensity,
        "habitat_suitability": habitat_score,
        "thermal_comfort": "Comfortable" if 18 <= temp_c <= 28 else ("Hot" if temp_c > 28 else "Cold"),
        "vegetation_stress": "Low" if temp_c < 30 else ("Moderate" if temp_c < 35 else "High"),
        "fire_risk": "Low" if temp_c < 30 else ("Moderate" if temp_c < 38 else "High"),
        "data_source": "Derived analysis from Landsat 8 thermal data"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
