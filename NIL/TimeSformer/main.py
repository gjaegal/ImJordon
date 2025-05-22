import os
import cv2
import torch
import numpy as np
from Video_Encoder import VideoEncoder  # VideoEncoder 클래스가 정의된 파일

def load_frames_from_folder(folder, sort=True):
    frames = []
    files = sorted(os.listdir(folder)) if sort else os.listdir(folder)
    for fname in files:
        if fname.endswith(('.png', '.jpg')):
            img = cv2.imread(os.path.join(folder, fname))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            frames.append(img)
    return frames

def load_masks_from_folder(folder, sort=True):
    masks = []
    files = sorted(os.listdir(folder)) if sort else os.listdir(folder)
    for fname in files:
        if fname.endswith(('.png', '.jpg')):
            mask = cv2.imread(os.path.join(folder, fname), cv2.IMREAD_GRAYSCALE)
            binary = (mask > 127).astype(np.uint8)
            masks.append(binary)
    return masks

def main():
    # === 경로 설정 ===
    gen_img_path = "/root/TimeSformer/video/"
    gen_mask_path = "/root/TimeSformer/mask/"
    sim_img_path = "/root/TimeSformer/video/"
    sim_mask_path = "/root/TimeSformer/mask/"

    # === 데이터 로딩 ===
    gen_frames = load_frames_from_folder(gen_img_path)
    gen_masks = load_masks_from_folder(gen_mask_path)
    sim_frames = load_frames_from_folder(sim_img_path)
    sim_masks = load_masks_from_folder(sim_mask_path)

    total_timesteps = min(len(gen_frames), len(sim_frames))
    encoder = VideoEncoder()

    print(f"시작: 총 {total_timesteps} 프레임 중 TimeSformer 임베딩 유사도 계산")
    l2_scores = []

    for t in range(7, total_timesteps):
        clip_gen = encoder.create_clip(gen_frames, gen_masks, t)
        clip_sim = encoder.create_clip(sim_frames, sim_masks, t)

        z_gen = encoder.extract_embedding(clip_gen)
        z_sim = encoder.extract_embedding(clip_sim)

        sv = -torch.norm(z_gen - z_sim, p=2).item()  # negative L2 distance
        l2_scores.append(sv)

        # print(f"[t={t:02d}] −L2 distance = {sv:.4f}")
        
    mean_l2 = np.mean(l2_scores)
    print(f"\n 전체 평균 −L2 distance: {mean_l2:.4f}")

if __name__ == "__main__":
    main()
