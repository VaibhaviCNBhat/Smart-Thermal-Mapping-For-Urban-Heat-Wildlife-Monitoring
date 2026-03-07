# -Smart-Thermal-Mapping-For-Urban-Heat-Wildlife-Monitoring
📌 Overview
This project focuses on enhancing low-resolution thermal satellite imagery using high-resolution optical imagery as structural guidance. Thermal infrared (TIR) images provide accurate surface temperature measurements but suffer from limited spatial resolution. In contrast, optical images capture sharp spatial details but lack temperature information.
We propose a dual-encoder, gated-fusion deep learning framework that improves thermal spatial resolution while preserving physical temperature fidelity.

🎯 Problem Statement
Thermal satellite imagery (e.g., Landsat 8 Band 10) has lower intrinsic resolution compared to optical bands. Traditional interpolation methods fail to reconstruct structural boundaries, while naïve fusion techniques may introduce artificial textures. The challenge is to enhance spatial detail without distorting thermal measurements.

🏗️ Proposed Architecture
The model follows a structured pipeline:

1. Thermal Encoder
Extracts temperature-specific features from upsampled low-resolution thermal input.

2. Optical Encoder
Extracts high-frequency spatial details from optical imagery.

3. Gated Fusion Module
Regulates optical feature injection using thermal feature guidance to prevent texture hallucination.

4. Reconstruction Decoder
Generates the super-resolved thermal output.

5. Physics-Aware Loss Functions
L1 Reconstruction Loss
SSIM Structural Loss
Thermal Energy Consistency Constraint


📊 Dataset
We use Landsat 8 Collection 2 data:
Optical Bands: B2, B3, B4 (RGB)
Thermal Band: B10
Thermal band values are converted from digital numbers to brightness temperature (Kelvin) using calibration constants from the metadata file.
Cloud-contaminated and invalid patches are filtered prior to training.

📂 Project Structure
Smart-Thermal-Mapping-For-Urban-Heat-Wildlife-Monitoring/
│
├── data/               # Raw and processed satellite data
├── src/                # Model, training, and utility scripts
├── configs/            # Experiment configuration files
├── outputs/            # Checkpoints and generated results
├── demo/               # Live demo script
├── requirements.txt
└── README.md

🚀 Installation
git clone https://github.com/VaibhaviCNBhat/Smart-Thermal-Mapping-For-Urban-Heat-Wildlife-Monitoring.git
cd Smart-Thermal-Mapping-For-Urban-Heat-Wildlife-Monitoring
pip install -r requirements.txt

Recommended:
Python 3.9+
PyTorch
rasterio
numpy
matplotlib


🧪 Training
python src/training/train.py --config configs/default_config.yaml

Checkpoints will be saved in:
outputs/checkpoints/

🔍 Inference & Visualization
Run super-resolution on a sample scene:
python src/inference/infer.py --input data/sample_demo/

The script will generate:
Super-resolved GeoTIFF (scientific output)
Colorized PNG heatmap (presentation output)


📈 Evaluation Metrics
We evaluate performance using:
PSNR (Peak Signal-to-Noise Ratio)
SSIM (Structural Similarity Index)
RMSE (Kelvin) for temperature accuracy


🖼️ Demo Output
During live demonstration, the system displays:
Low-resolution thermal heatmap
Super-resolved thermal heatmap
Quantitative metric comparison
Both images are displayed using identical temperature scales for fair visual comparison.


🌍 Applications
Urban Heat Island Analysis
Wildfire Monitoring
Climate Risk Mapping
Environmental Planning
Precision Agriculture


⚠️ Limitations
Performance depends on optical–thermal structural correlation.
Cloud contamination reduces usable training data.
Enhancement is bounded by physical sensor limitations.
