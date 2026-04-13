"""
Smart Thermal Mapping for Urban Heat & Wildlife Monitoring
Streamlit Cloud entrypoint — do NOT add uvicorn.run() here.

Provides an interactive UI for optical-guided thermal super-resolution
using the Landsat-8 dataset and a pre-trained deep-learning model.
"""

import os
import sys
import io
import base64
import json
import re
from pathlib import Path
from urllib.request import urlopen, Request

import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from PIL import Image

# ---------------------------------------------------------------------------
# Path setup – make Pre processing importable
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
pre_processing_path = BASE_DIR / "Pre processing"
sys.path.insert(0, str(pre_processing_path))

# ---------------------------------------------------------------------------
# Optional heavy imports (torch may not be installed in very lean envs)
# ---------------------------------------------------------------------------
try:
    import torch
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

# ---------------------------------------------------------------------------
# Page config (must be first Streamlit call)
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Smart Thermal Mapping",
    page_icon="🌡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
THERMAL_DN_MIN_K = 270.0
THERMAL_DN_MAX_K = 330.0
THERMAL_DN_SCALE = (THERMAL_DN_MAX_K - THERMAL_DN_MIN_K) / 255.0  # ~0.2353 K/DN

if (BASE_DIR / "dataset").exists():
    DATASET_ROOT = str(BASE_DIR / "dataset")
elif (BASE_DIR / "dataset_demo").exists():
    DATASET_ROOT = str(BASE_DIR / "dataset_demo")
else:
    DATASET_ROOT = str(BASE_DIR / "dataset")

CHECKPOINT_PATH = str(BASE_DIR / "checkpoints" / "best_model.pth")
DEVICE = torch.device("cuda" if TORCH_AVAILABLE and torch.cuda.is_available() else "cpu") if TORCH_AVAILABLE else None


# ---------------------------------------------------------------------------
# Utility helpers (mirror backend_app.py without FastAPI dependency)
# ---------------------------------------------------------------------------

def dn_to_kelvin(dn_value):
    """Convert uint8 DN value (0-255) to brightness temperature in Kelvin."""
    return dn_value * THERMAL_DN_SCALE + THERMAL_DN_MIN_K


def normalize_for_display(img: np.ndarray) -> np.ndarray:
    """Normalise array to [0, 1] for visualisation."""
    lo, hi = img.min(), img.max()
    if hi - lo < 1e-8:
        return np.zeros_like(img)
    return (img - lo) / (hi - lo)


def array_to_pil(arr: np.ndarray, cmap: str = "inferno") -> Image.Image:
    """Convert a 2-D numpy array to a PIL Image using a matplotlib colormap."""
    arr_norm = normalize_for_display(arr)
    cm = plt.get_cmap(cmap)
    colored = cm(arr_norm)
    img_uint8 = (colored[:, :, :3] * 255).astype(np.uint8)
    return Image.fromarray(img_uint8)


def get_temp_range(arr: np.ndarray, dataset_stats: dict | None) -> dict | None:
    if dataset_stats:
        dn_min = float(arr.min()) * dataset_stats.get("thermal_std", 1) + dataset_stats.get("thermal_mean", 0)
        dn_max = float(arr.max()) * dataset_stats.get("thermal_std", 1) + dataset_stats.get("thermal_mean", 0)
        return {"min": round(dn_to_kelvin(dn_min), 2), "max": round(dn_to_kelvin(dn_max), 2)}
    return None


def compute_thermal_histogram(arr: np.ndarray, dataset_stats: dict | None, num_bins: int = 10) -> dict:
    if dataset_stats:
        dn = arr * dataset_stats.get("thermal_std", 1) + dataset_stats.get("thermal_mean", 0)
        kelvin = dn_to_kelvin(dn)
    else:
        kelvin = arr
    flat = kelvin.flatten()
    counts, edges = np.histogram(flat, bins=num_bins)
    bin_centers = [(float(edges[i]) + float(edges[i + 1])) / 2 for i in range(len(counts))]
    return {
        "counts": counts.tolist(),
        "bin_centers": [round(c, 2) for c in bin_centers],
        "bin_edges": [round(float(e), 2) for e in edges],
        "min_k": round(float(flat.min()), 2),
        "max_k": round(float(flat.max()), 2),
        "mean_k": round(float(flat.mean()), 2),
    }


def extract_coordinates(scene_name: str, scene_coords: dict) -> dict | None:
    match = re.search(r"LC08_(\d{3})(\d{3})_", scene_name)
    path = int(match.group(1)) if match else None
    row = int(match.group(2)) if match else None
    if scene_coords and scene_name in scene_coords:
        entry = scene_coords[scene_name]
        result = {"lat": entry["lat"], "lon": entry["lon"], "path": path, "row": row}
        if "bounds" in entry:
            result["bounds"] = entry["bounds"]
        return result
    elif match:
        lat = round(54 - (row - 25) * 1.4, 4)
        lon = round(-62 - (path - 10) * 1.22, 4)
        return {"lat": lat, "lon": lon, "path": path, "row": row}
    return None


# ---------------------------------------------------------------------------
# Cached resource loaders
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner="Loading model…")
def load_model():
    """Load the OpticalGuidedThermalSR model (cached across reruns)."""
    if not TORCH_AVAILABLE:
        return None
    if not os.path.exists(CHECKPOINT_PATH):
        return None
    try:
        from step5_model import OpticalGuidedThermalSR
        model = OpticalGuidedThermalSR(scale_factor=4).to(DEVICE)
        try:
            import numpy
            with torch.serialization.safe_globals(
                [numpy.dtype, numpy._core.multiarray.scalar, numpy.dtypes.Float64DType]
            ):
                checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE, weights_only=False)
        except Exception:
            checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        else:
            model.load_state_dict(checkpoint)
        model.eval()
        return model
    except Exception as exc:
        st.warning(f"Model could not be loaded: {exc}")
        return None


@st.cache_resource(show_spinner="Loading dataset…")
def load_dataset():
    """Load test dataset (cached across reruns)."""
    if not TORCH_AVAILABLE:
        return None
    try:
        from step4_dataloader import ThermalSRDataset
        if os.path.exists(DATASET_ROOT):
            dataset = ThermalSRDataset(DATASET_ROOT, split="test")
            return dataset
    except Exception:
        pass
    return None


@st.cache_resource(show_spinner="Loading stats…")
def load_stats():
    stats_path = os.path.join(DATASET_ROOT, "dataset_stats.npy")
    if os.path.exists(stats_path):
        return np.load(stats_path, allow_pickle=True).item()
    return None


@st.cache_resource(show_spinner="Loading coordinates…")
def load_scene_coordinates():
    coords_path = os.path.join(DATASET_ROOT, "scene_coordinates.json")
    if os.path.exists(coords_path):
        with open(coords_path, "r") as f:
            return json.load(f)
    return {}


# ---------------------------------------------------------------------------
# Inference helpers
# ---------------------------------------------------------------------------

def run_inference_on_tensors(model, optical_tensor, thermal_tensor):
    """Run model inference and return (img_lr_disp, img_sr) numpy arrays."""
    with torch.no_grad():
        opt_batch = optical_tensor.unsqueeze(0).to(DEVICE)
        therm_batch = thermal_tensor.unsqueeze(0).to(DEVICE)
        predicted_sr = model(opt_batch, therm_batch)
    img_lr = thermal_tensor.squeeze().cpu().numpy()
    img_lr_disp = np.array(
        Image.fromarray((normalize_for_display(img_lr) * 255).astype("uint8")).resize((256, 256), Image.NEAREST)
    )
    img_sr = predicted_sr.squeeze().cpu().numpy()
    return img_lr_disp, img_sr


def infer_from_uploaded_files(model, optical_file, thermal_file):
    """Run inference from file-like objects. Returns (img_lr_disp, img_sr)."""
    opt_img = Image.open(optical_file).convert("RGB").resize((256, 256), Image.BICUBIC)
    therm_img = Image.open(thermal_file).convert("L").resize((64, 64), Image.BICUBIC)
    opt_tensor = torch.from_numpy(np.array(opt_img)).permute(2, 0, 1).float() / 255.0
    therm_tensor = torch.from_numpy(np.array(therm_img)).unsqueeze(0).float() / 255.0
    return run_inference_on_tensors(model, opt_tensor, therm_tensor)


# ---------------------------------------------------------------------------
# GBIF & Climate helpers
# ---------------------------------------------------------------------------

def fetch_gbif_species(lat: float, lon: float, radius: float = 0.5) -> dict:
    """Fetch wildlife species occurrences from GBIF near a coordinate.

    Args:
        lat: Latitude in decimal degrees.
        lon: Longitude in decimal degrees.
        radius: Search radius in decimal degrees (approx. 55 km at the equator).

    Returns:
        dict with keys ``species`` (list of dicts), ``total`` (int record count),
        and optionally ``error`` (str) on failure.
    """
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
        species_map: dict = {}
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
                    "count": 0,
                    "iucn": record.get("iucnRedListCategory", ""),
                }
            species_map[name]["count"] += 1
        return {"species": sorted(species_map.values(), key=lambda x: -x["count"])[:20], "total": data.get("count", 0)}
    except Exception as exc:
        return {"species": [], "total": 0, "error": str(exc)}


def compute_climate_risk(lat: float, lon: float, mean_temp_k: float = 300) -> dict:
    """Compute a simple climate-biodiversity risk profile from surface temperature.

    Thresholds follow a Copernicus CDS-inspired methodology:
    - Heat stress:   > 310 K → Critical, > 305 K → High, > 300 K → Moderate, else Low
    - UHI intensity: max(0, (T - 295) × 0.5) K above a 295 K urban baseline
    - Habitat score: 0–100 penalised for deviation from the optimal 22 °C

    Args:
        lat: Latitude in decimal degrees (currently informational).
        lon: Longitude in decimal degrees (currently informational).
        mean_temp_k: Mean surface temperature in Kelvin.

    Returns:
        dict with heat_stress_index, temperatures, UHI intensity, habitat
        suitability, thermal comfort, vegetation stress, and fire risk.
    """
    temp_c = mean_temp_k - 273.15
    if mean_temp_k > 310:
        heat_stress = "Critical"
    elif mean_temp_k > 305:
        heat_stress = "High"
    elif mean_temp_k > 300:
        heat_stress = "Moderate"
    else:
        heat_stress = "Low"
    return {
        "heat_stress_index": heat_stress,
        "mean_temperature_k": round(mean_temp_k, 2),
        "mean_temperature_c": round(temp_c, 2),
        "uhi_intensity_k": round(max(0, (mean_temp_k - 295) * 0.5), 1),
        "habitat_suitability": round(max(0, min(100, 100 - abs(temp_c - 22) * 5)), 1),
        "thermal_comfort": "Comfortable" if 18 <= temp_c <= 28 else ("Hot" if temp_c > 28 else "Cold"),
        "vegetation_stress": "Low" if temp_c < 30 else ("Moderate" if temp_c < 35 else "High"),
        "fire_risk": "Low" if temp_c < 30 else ("Moderate" if temp_c < 38 else "High"),
    }


# ---------------------------------------------------------------------------
# UI helpers
# ---------------------------------------------------------------------------

def display_thermal_images(img_lr, img_sr, img_hr=None):
    """Render LR input, SR prediction and (optional) ground truth side-by-side."""
    cols = st.columns(3 if img_hr is not None else 2)
    with cols[0]:
        st.markdown("**🔻 Low-Res Input (upscaled)**")
        st.image(array_to_pil(img_lr, "inferno"), use_container_width=True)
    with cols[1]:
        st.markdown("**✨ Super-Resolved Output**")
        st.image(array_to_pil(img_sr, "inferno"), use_container_width=True)
    if img_hr is not None and len(cols) > 2:
        with cols[2]:
            st.markdown("**🎯 Ground Truth**")
            st.image(array_to_pil(img_hr, "inferno"), use_container_width=True)


def display_metrics(img_sr, img_hr=None, dataset_stats=None):
    """Show PSNR/SSIM and temperature stats."""
    temp_range = get_temp_range(img_sr, dataset_stats)
    hist = compute_thermal_histogram(img_sr, dataset_stats)

    col1, col2, col3 = st.columns(3)
    with col1:
        if temp_range:
            st.metric("Min Temp (K)", temp_range["min"])
        else:
            st.metric("Min Pixel", round(float(img_sr.min()), 4))
    with col2:
        if temp_range:
            st.metric("Max Temp (K)", temp_range["max"])
        else:
            st.metric("Max Pixel", round(float(img_sr.max()), 4))
    with col3:
        st.metric("Mean Temp (K)", hist["mean_k"])

    if img_hr is not None:
        psnr_val = float(10 * np.log10(1 / max(((img_sr - img_hr) ** 2).mean(), 1e-10)))
        col_a, col_b = st.columns(2)
        with col_a:
            st.metric("PSNR (dB)", round(psnr_val, 2))
        with col_b:
            # Simple approx SSIM placeholder
            rmse = float(np.sqrt(((img_sr - img_hr) ** 2).mean()))
            st.metric("RMSE (pixel)", round(rmse, 4))

    # Histogram
    fig, ax = plt.subplots(figsize=(7, 2.5))
    ax.bar(hist["bin_centers"], hist["counts"], width=(hist["bin_edges"][1] - hist["bin_edges"][0]) * 0.9, color="#FF4B4B")
    ax.set_xlabel("Temperature (K)" if dataset_stats else "Pixel value")
    ax.set_ylabel("Count")
    ax.set_title("SR Thermal Distribution")
    ax.set_facecolor("#0e1117")
    fig.patch.set_facecolor("#0e1117")
    ax.tick_params(colors="white")
    ax.xaxis.label.set_color("white")
    ax.yaxis.label.set_color("white")
    ax.title.set_color("white")
    st.pyplot(fig)
    plt.close(fig)


def display_wildlife_panel(lat: float, lon: float, mean_temp_k: float):
    """Show GBIF species and climate-risk panels."""
    st.subheader("🌿 Wildlife & Climate Analysis")
    with st.spinner("Fetching GBIF species data…"):
        gbif = fetch_gbif_species(lat, lon)
    climate = compute_climate_risk(lat, lon, mean_temp_k)

    col_left, col_right = st.columns(2)
    with col_left:
        st.markdown("**🐾 Nearby Species (GBIF)**")
        if gbif.get("species"):
            st.caption(f"Total GBIF records in area: {gbif['total']}")
            for sp in gbif["species"][:10]:
                iucn = f" — IUCN: {sp['iucn']}" if sp.get("iucn") else ""
                name = sp.get("vernacularName") or sp["species"]
                st.markdown(f"- **{sp['species']}** ({name}){iucn} — {sp['count']} records")
        else:
            err = gbif.get("error", "No species data available.")
            st.info(f"No species found nearby. {err}")

    with col_right:
        st.markdown("**🌡️ Climate Risk Summary**")
        stress_color = {"Low": "🟢", "Moderate": "🟡", "High": "🟠", "Critical": "🔴"}
        emoji = stress_color.get(climate["heat_stress_index"], "⚪")
        st.metric(f"{emoji} Heat Stress Index", climate["heat_stress_index"])
        st.metric("Urban Heat Island (K)", climate["uhi_intensity_k"])
        st.metric("Habitat Suitability (%)", climate["habitat_suitability"])
        st.metric("Vegetation Stress", climate["vegetation_stress"])
        st.metric("Fire Risk", climate["fire_risk"])
        st.metric("Thermal Comfort", climate["thermal_comfort"])


# ---------------------------------------------------------------------------
# Main app
# ---------------------------------------------------------------------------

def main():
    # Sidebar
    st.sidebar.title("🌡️ Smart Thermal Mapping")
    st.sidebar.markdown(
        "Optical-guided thermal super-resolution using Landsat-8 imagery.\n\n"
        "Powered by a gated-fusion deep-learning model."
    )

    mode = st.sidebar.radio(
        "Mode",
        ["📤 Upload Images", "🎲 Random Dataset Sample"],
        index=0,
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown(
        "**Backend API** (optional)\n\n"
        "Run `uvicorn backend_app:app --port 8000` locally to also expose the REST API."
    )

    # Load shared resources
    model = load_model()
    dataset_stats = load_stats()
    scene_coords = load_scene_coordinates()

    if not TORCH_AVAILABLE:
        st.error("❌ PyTorch is not installed. Please add `torch` to requirements.txt.")
        st.stop()

    if model is None:
        st.warning(
            "⚠️ Model checkpoint not found at `checkpoints/best_model.pth`. "
            "Inference is disabled. Please upload a valid checkpoint."
        )

    # Page header
    st.title("🌡️ Smart Thermal Mapping for Urban Heat & Wildlife Monitoring")
    st.markdown(
        "Enhance low-resolution Landsat-8 thermal imagery with optical guidance. "
        "Analyse urban heat islands and assess wildlife habitat suitability."
    )

    # -----------------------------------------------------------------------
    # Mode: Upload Images
    # -----------------------------------------------------------------------
    if mode == "📤 Upload Images":
        st.header("Upload Your Images")
        col_a, col_b = st.columns(2)
        with col_a:
            optical_file = st.file_uploader(
                "Optical image (RGB, any size)", type=["png", "jpg", "jpeg", "tif"]
            )
        with col_b:
            thermal_file = st.file_uploader(
                "Thermal LR image (grayscale, any size)", type=["png", "jpg", "jpeg", "tif"]
            )

        if optical_file and thermal_file:
            if model is None:
                st.error("Model not loaded — cannot run inference.")
                return

            with st.spinner("Running super-resolution…"):
                try:
                    img_lr_disp, img_sr = infer_from_uploaded_files(model, optical_file, thermal_file)
                except Exception as exc:
                    st.error(f"Inference failed: {exc}")
                    return

            st.subheader("Results")
            display_thermal_images(img_lr_disp, img_sr)
            display_metrics(img_sr, dataset_stats=dataset_stats)

            temp_range = get_temp_range(img_sr, dataset_stats)
            mean_k = float(
                img_sr.mean() * dataset_stats.get("thermal_std", 1) + dataset_stats.get("thermal_mean", 0)
                if dataset_stats
                else img_sr.mean()
            )
            mean_k_abs = dn_to_kelvin(mean_k) if dataset_stats else mean_k

            with st.expander("🌍 Wildlife & Climate Analysis (enter coordinates)"):
                lat = st.number_input("Latitude", value=45.0, format="%.4f")
                lon = st.number_input("Longitude", value=-75.0, format="%.4f")
                if st.button("Analyze"):
                    display_wildlife_panel(lat, lon, mean_k_abs)

    # -----------------------------------------------------------------------
    # Mode: Random dataset sample
    # -----------------------------------------------------------------------
    elif mode == "🎲 Random Dataset Sample":
        st.header("Random Dataset Sample")
        dataset = load_dataset()

        if dataset is None:
            st.warning("Dataset not available at `dataset/` or `dataset_demo/`. Please add the dataset.")
            return
        if model is None:
            st.error("Model not loaded — cannot run inference.")
            return

        if st.button("🎲 Load Random Sample"):
            idx = int(np.random.randint(0, len(dataset)))
            sample = dataset[idx]
            scene_name = sample.get("scene_name", f"sample_{idx}")

            with st.spinner(f"Running inference on scene `{scene_name}`…"):
                try:
                    optical_hr = sample["optical_hr"]
                    thermal_lr = sample["thermal_lr"]
                    thermal_hr = sample["thermal_hr"]
                    img_lr_disp, img_sr = run_inference_on_tensors(model, optical_hr, thermal_lr)
                    img_hr = thermal_hr.squeeze().numpy()
                except Exception as exc:
                    st.error(f"Inference failed: {exc}")
                    return

            st.subheader(f"Scene: `{scene_name}`")
            display_thermal_images(img_lr_disp, img_sr, img_hr)
            display_metrics(img_sr, img_hr, dataset_stats)

            coords = extract_coordinates(scene_name, scene_coords)
            if coords:
                st.markdown(f"**📍 Location:** lat={coords['lat']}, lon={coords['lon']}")
                temp_range = get_temp_range(img_sr, dataset_stats)
                mean_k_abs = (
                    dn_to_kelvin(
                        float(img_sr.mean()) * dataset_stats.get("thermal_std", 1)
                        + dataset_stats.get("thermal_mean", 0)
                    )
                    if dataset_stats
                    else float(img_sr.mean())
                )
                with st.expander("🌿 Wildlife & Climate Analysis"):
                    display_wildlife_panel(coords["lat"], coords["lon"], mean_k_abs)
            else:
                st.info("No coordinates available for this scene.")


if __name__ == "__main__":
    main()
