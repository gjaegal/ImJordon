import os
import cv2
import numpy as np
import yaml
from tqdm import tqdm

def load_mask(path):
    mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"Mask not found: {path}")
    return (mask > 127).astype(np.uint8)

def compute_iou(mask1, mask2):
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    if union == 0:
        return 1.0 if intersection == 0 else 0.0
    return intersection / union

def compute_mean_iou(gen_mask_dir, sim_mask_dir):
    gen_files = sorted(os.listdir(gen_mask_dir))
    sim_files = sorted(os.listdir(sim_mask_dir))

    common_files = sorted(set(gen_files) & set(sim_files))
    if not common_files:
        raise ValueError("No common mask files found between the two directories.")

    iou_list = []

    for fname in tqdm(common_files, desc="Computing IoU"):
        gen_path = os.path.join(gen_mask_dir, fname)
        sim_path = os.path.join(sim_mask_dir, fname)

        gen_mask = load_mask(gen_path)
        sim_mask = load_mask(sim_path)

        if gen_mask.shape != sim_mask.shape:
            raise ValueError(f"Shape mismatch for {fname}: {gen_mask.shape} vs {sim_mask.shape}")

        iou = compute_iou(gen_mask, sim_mask)
        iou_list.append(iou)

    mean_iou = float(np.mean(iou_list))
    return mean_iou

# Optional CLI usage
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    mean_iou = compute_mean_iou(cfg["gen_mask_dir"], cfg["sim_mask_dir"])
    print(f"✅ Mean IoU: {mean_iou:.4f}")
