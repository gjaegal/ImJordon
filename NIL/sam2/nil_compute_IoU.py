import os
import numpy as np
from PIL import Image


def compute_iou(mask1, mask2):
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    return intersection / union if union != 0 else 0.0

def compute_mean_iou(gen_mask_dir: str, gt_mask_dir: str) -> float:
    def load_mask(path):
        return np.array(Image.open(path).convert("L")) > 0  # 이진화

    ious = []
    for fname in sorted(os.listdir(gt_mask_dir)):
        gen_path = os.path.join(gen_mask_dir, fname)
        gt_path = os.path.join(gt_mask_dir, fname)

        if not os.path.exists(gen_path) or not os.path.exists(gt_path):
            continue

        mask1 = load_mask(gen_path)
        mask2 = load_mask(gt_path)
        iou = compute_iou(mask1, mask2)
        ious.append(iou)

    return np.mean(ious) if ious else 0.0

