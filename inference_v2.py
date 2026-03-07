import sys
import os
import random
import torch
import numpy as np
import matplotlib.pyplot as plt

# Add PRE_PROCESSING directory to path so we can import the modules
pre_processing_path = os.path.join(os.getcwd(), 'Pre processing')
sys.path.append(pre_processing_path)

# Try importing the project modules
try:
    from step4_dataloader import ThermalSRDataset
    from step5_model import OpticalGuidedThermalSR
except ImportError as e:
    print(f"Error importing modules: {e}")
    print(f"Make sure 'Pre processing' folder exists and contains step4_dataloader.py and step5_model.py")
    sys.exit(1)

# Configuration
DATASET_ROOT = os.path.join(os.getcwd(), 'dataset')
CHECKPOINT_PATH = os.path.join(os.getcwd(), 'checkpoints', 'best_model.pth')
OUTPUT_DIR = os.path.join(os.getcwd(), 'inference_results_grayscale')
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

def normalize_for_display(img):
    """Normalize image to 0-1 range for visualization."""
    img_min = img.min()
    img_max = img.max()
    if img_max - img_min < 1e-8:
        return np.zeros_like(img)
    return (img - img_min) / (img_max - img_min)

def main():
    print(f"Using device: {DEVICE}")
    print(f"Loading dataset from {DATASET_ROOT}")
    
    # Check if dataset paths exist
    if not os.path.exists(DATASET_ROOT):
        print(f"Dataset root not found: {DATASET_ROOT}")
        sys.exit(1)
        
    try:
        # Load Dataset
        test_dataset = ThermalSRDataset(DATASET_ROOT, split='test')
    except Exception as e:
        print(f"Failed to load dataset: {e}")
        sys.exit(1)

    # Select 5 random files from test set
    total_samples = len(test_dataset)
    num_samples = min(5, total_samples)
    
    if num_samples == 0:
        print("Test dataset is empty.")
        sys.exit(1)
        
    indices = random.sample(range(total_samples), num_samples)
    print(f"Selected {num_samples} random indices: {indices}")

    # Load Model
    print(f"Loading model from {CHECKPOINT_PATH}")
    if not os.path.exists(CHECKPOINT_PATH):
        print(f"Checkpoint not found: {CHECKPOINT_PATH}")
        sys.exit(1)

    model = OpticalGuidedThermalSR(scale_factor=4).to(DEVICE)
    
    try:
        # Handle load with safe_globals for numpy (common issue in newer torch versions)
        import numpy
        with torch.serialization.safe_globals([numpy.dtype, numpy._core.multiarray.scalar, numpy.dtypes.Float64DType]): # Adapt for torch < 2.4/2.6 or compatibility
             checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE, weights_only=False)
    except AttributeError:
        # Fallback for older torch versions
        checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
    except Exception as e:
         print(f"Standard load failed, trying generic load: {e}")
         checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)

    # Load weights
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.eval()

    # Inference Loop
    print("Starting inference...")
    for i, idx in enumerate(indices):
        try:
            sample = test_dataset[idx]
            scene_name = sample.get('scene_name', f'sample_{idx}')
            print(f"Processing {i+1}/{num_samples}: {scene_name}")
            
            # Prepare inputs
            # Add batch dimension (1, C, H, W)
            optical_hr = sample['optical_hr'].unsqueeze(0).to(DEVICE) # (1, 3, 256, 256)
            thermal_lr = sample['thermal_lr'].unsqueeze(0).to(DEVICE) # (1, 1, 64, 64)
            thermal_hr = sample['thermal_hr'].unsqueeze(0).to(DEVICE) # (1, 1, 256, 256)
            
            with torch.no_grad():
                predicted_sr = model(optical_hr, thermal_lr)
            
            # Post-process for visualization
            
            # 1. Resize LR input to HR size for better visual comparison (Nearest Neighbor)
            thermal_lr_resized = torch.nn.functional.interpolate(
                thermal_lr, size=thermal_hr.shape[2:], mode='nearest'
            )
            
            # 2. Extract images as numpy arrays
            img_lr = thermal_lr_resized.squeeze().cpu().numpy()
            img_lr_original = thermal_lr.squeeze().cpu().numpy()  # Native 64x64 resolution
            img_sr = predicted_sr.squeeze().cpu().numpy()
            img_hr = thermal_hr.squeeze().cpu().numpy()
            
            # 3. Normalize for display (0-1)
            img_lr_disp = normalize_for_display(img_lr)
            img_lr_original_disp = normalize_for_display(img_lr_original)
            img_sr_disp = normalize_for_display(img_sr)
            img_hr_disp = normalize_for_display(img_hr)

            # 4. Save images (using 'gray' colormap for thermal)
            sample_idx = i + 1
            # Save the upscaled input (blocky visualization)
            plt.imsave(os.path.join(OUTPUT_DIR, f"ip{sample_idx}.png"), img_lr_disp, cmap='gray')
            # Save the original low-res input (small native size)
            plt.imsave(os.path.join(OUTPUT_DIR, f"ip{sample_idx}_original.png"), img_lr_original_disp, cmap='gray')
            # Save predictions and ground truth
            plt.imsave(os.path.join(OUTPUT_DIR, f"predicted{sample_idx}.png"), img_sr_disp, cmap='gray')
            plt.imsave(os.path.join(OUTPUT_DIR, f"hr{sample_idx}.png"), img_hr_disp, cmap='gray')
            
            # Save predicted image as .tif with raw values
            try:
                import tifffile
                tifffile.imwrite(os.path.join(OUTPUT_DIR, f"predicted{sample_idx}.tif"), img_sr)
            except ImportError:
                try:
                    from PIL import Image
                    im = Image.fromarray(img_sr)
                    im.save(os.path.join(OUTPUT_DIR, f"predicted{sample_idx}.tif"))
                except Exception as e:
                    print(f"Could not save .tif file for sample {sample_idx}: {e}")
            
        except Exception as e:
            print(f"Error processing sample {idx}: {e}")
            import traceback
            traceback.print_exc()

    print(f"Inference completed. Results saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
