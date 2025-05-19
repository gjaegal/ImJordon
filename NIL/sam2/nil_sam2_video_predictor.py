import os
import torch
import cv2
import numpy as np
from tqdm import tqdm
import argparse
import yaml

from sam2.build_sam import build_sam2_video_predictor

def convert_video_to_frames(video_path, output_dir):
    """
    MP4 비디오를 이미지(JPG) 프레임으로 변환하여 저장
    """
    os.makedirs(output_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        filename = os.path.join(output_dir, f"{idx:05d}.jpg")
        cv2.imwrite(filename, frame)
        idx += 1
    cap.release()

def ensure_frames_exist(video_path):
    """
    비디오 경로가 .mp4이면 프레임(JPG)으로 변환하고, 이미 JPG 폴더가 있으면 그대로 사용합니다.
    """
    if os.path.isdir(video_path):
        return video_path  # 이미 jpg 폴더인 경우

    elif os.path.isfile(video_path) and video_path.endswith(".mp4"):
        output_dir = os.path.splitext(video_path)[0] + "_frames"
        if not os.path.exists(output_dir) or len(os.listdir(output_dir)) == 0:
            print(f"⏳ {video_path} → JPG 프레임 추출 중...")
            convert_video_to_frames(video_path, output_dir)
        return output_dir

    else:
        raise ValueError("video_path는 폴더 또는 .mp4 파일이어야 합니다.")


def run_sam2_from_config(args, config_path):
    """
    SAM2를 사용하여 클릭 기반 비디오 객체 분할 수행 후 마스크 저장.
    PNG 시퀀스만 저장.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config_name = os.path.splitext(os.path.basename(config_path))[0]

    video_dir = ensure_frames_exist(args["video_dir"])
    predictor = build_sam2_video_predictor(config_file=config_name, checkpoint=args["checkpoint"], device=device)

    inference_state = predictor.init_state(video_path=video_dir)
    predictor.reset_state(inference_state)

    # 클릭 포인트와 라벨 처리
    if "click_points" in args and "click_labels" in args:
        points = np.array(args["click_points"], dtype=np.float32)
        labels = np.array(args["click_labels"], dtype=np.int32)

        if points.ndim == 2:
            points = points[None, ...]  # (1, N, 2)
        if labels.ndim == 1:
            labels = labels[None, ...]  # (1, N)

        predictor.add_new_points_or_box(
            inference_state=inference_state,
            frame_idx=args["click_frame"],
            obj_id=args["obj_id"],
            points=points,
            labels=labels,
            clear_old_points=True,
            normalize_coords=True
        )
    else:
        raise ValueError("click_points와 click_labels가 필요합니다.")

    video_segments = predictor.propagate_in_video(inference_state)

    os.makedirs(args["gen_mask_dir"], exist_ok=True)

    # VideoWriter 설정
    #video_segments = list(predictor.propagate_in_video(inference_state))

    # 첫 번째 프레임 결과에서 마스크만 가져옴
    #first_frame_idx, obj_ids, masks = video_segments[0]
    #sample_seg = masks[0].cpu().numpy()
    #height, width = sample_seg.shape[-2:]
    #output_path = os.path.join(args["gen_mask_dir"], "sample_video.mp4")
    #fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    #fps = 30  # 필요시 원본 fps로 변경 가능
    #video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height), isColor=True)

    for frame_idx, obj_ids, masks in tqdm(video_segments, desc="Saving masks"):
        mask = masks[0].cpu().numpy()

        # 안정성 검사: 2차원, finite, 크기 제한
        if mask.ndim > 2:
            mask = mask.squeeze()
        if mask.ndim != 2 or not np.isfinite(mask).all() or mask.size == 0 or max(mask.shape) > 10000:
            print(f"⚠️ Skip frame {frame_idx}: invalid mask shape={mask.shape}, contains NaN or too large.")
            continue

        mask = (mask > 0.1).astype(np.uint8) * 255
        #mask = 255 - mask # 왜 이렇게 해야 제대로 나오는 건질 모르겠음. 

        filename = os.path.join(args["gen_mask_dir"], f"{frame_idx:05d}.png")
        cv2.imwrite(filename, mask)

        mask_rgb = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        #video_writer.write(mask_rgb)

    print("✅ 마스크 PNG 저장 완료:", args["gen_mask_dir"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML file.")
    parser.add_argument("--video_dir", type=str)
    parser.add_argument("--gen_mask_dir", type=str)
    parser.add_argument("--checkpoint", type=str)
    parser.add_argument("--click_x", type=int)
    parser.add_argument("--click_y", type=int)
    parser.add_argument("--click_frame", type=int)
    parser.add_argument("--obj_id", type=int)

    cli_args = parser.parse_args()

    # Load config file
    with open(cli_args.config, "r") as f:
        cfg = yaml.safe_load(f)

    # Override with CLI arguments if provided
    for key in ["video_dir", "gen_mask_dir", "checkpoint", "click_x", "click_y", "click_frame", "obj_id"]:
        val = getattr(cli_args, key)
        if val is not None:
            cfg[key] = val

    run_sam2_from_config(cfg)
