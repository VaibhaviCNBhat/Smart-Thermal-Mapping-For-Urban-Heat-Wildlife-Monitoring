import os
import random

DATASET_ROOT = r"C:\Users\Admin\Downloads\ssl4eo_l_oli_tirs_toa_benchmark\dataset"
SEED = 42
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1


def main():
    print("=" * 60)
    print("STEP 2: Train/Val/Test Split")
    print("=" * 60)

    scenes_file = os.path.join(DATASET_ROOT, "all_scenes.txt")
    with open(scenes_file, "r") as f:
        scenes = [line.strip() for line in f if line.strip()]

    print(f"Total scenes: {len(scenes)}")

    random.seed(SEED)
    random.shuffle(scenes)

    n = len(scenes)
    train_end = int(n * TRAIN_RATIO)
    val_end = train_end + int(n * VAL_RATIO)

    train_scenes = scenes[:train_end]
    val_scenes = scenes[train_end:val_end]
    test_scenes = scenes[val_end:]

    print(f"  Train: {len(train_scenes)}")
    print(f"  Val:   {len(val_scenes)}")
    print(f"  Test:  {len(test_scenes)}")

    # Save split files
    for split_name, split_scenes in [("train", train_scenes), ("val", val_scenes), ("test", test_scenes)]:
        split_file = os.path.join(DATASET_ROOT, f"{split_name}.txt")
        with open(split_file, "w") as f:
            for s in split_scenes:
                f.write(s + "\n")
        print(f"  Saved: {split_file}")

    print("\nDone!")


if __name__ == "__main__":
    main()
