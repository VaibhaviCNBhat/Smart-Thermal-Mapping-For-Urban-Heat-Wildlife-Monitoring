import os
import shutil
import random

# Config
SOURCE_DIR = "dataset"
DEST_DIR = "dataset_demo"
NUM_SAMPLES = 1000  # Keep 1000 scenes for GitHub demo

def create_demo_dataset():
    if not os.path.exists(SOURCE_DIR):
        print(f"Error: Source '{SOURCE_DIR}' not found!")
        return

    # 1. Create Destination Folder
    if os.path.exists(DEST_DIR):
        shutil.rmtree(DEST_DIR)
    os.makedirs(DEST_DIR)
    print(f"Created '{DEST_DIR}'")

    # 2. Copy Metadata Files
    meta_files = ["dataset_stats.npy", "scene_coordinates.json"]
    for f in meta_files:
        src = os.path.join(SOURCE_DIR, f)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(DEST_DIR, f))
            print(f"Copied {f}")

    # 3. Pick Random Scenes
    all_scenes = [d for d in os.listdir(SOURCE_DIR) if os.path.isdir(os.path.join(SOURCE_DIR, d))]
    selected_scenes = random.sample(all_scenes, min(len(all_scenes), NUM_SAMPLES))
    
    for i, scene in enumerate(selected_scenes, 1):
        src = os.path.join(SOURCE_DIR, scene)
        dst = os.path.join(DEST_DIR, scene)
        if not os.path.exists(dst): 
            shutil.copytree(src, dst)
        if i % 50 == 0:
            print(f"[{i}/{len(selected_scenes)}] Copied...")

    # 4. Create proper test.txt (Manifest)
    # The app reads 'test.txt' to know which files to load.
    # We must ensure this file ONLY contains the scenes we actually copied.
    with open(os.path.join(DEST_DIR, "test.txt"), "w") as f:
        for scene in selected_scenes:
            f.write(scene + "\n")
    print("\nCreated 'test.txt' with selected scenes.")
    
    # Create dummy train/val mostly to prevent errors if code looks for them
    shutil.copy2(os.path.join(DEST_DIR, "test.txt"), os.path.join(DEST_DIR, "train.txt"))
    shutil.copy2(os.path.join(DEST_DIR, "test.txt"), os.path.join(DEST_DIR, "val.txt"))
    
    print("\n✅ Success! You can now commit 'dataset_demo' to GitHub.")

if __name__ == "__main__":
    create_demo_dataset()
