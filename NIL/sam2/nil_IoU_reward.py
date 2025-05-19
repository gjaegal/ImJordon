import yaml
from nil_sam2_video_predictor import run_sam2_from_config
from nil_compute_IoU import compute_mean_iou

def run_reward_function(config_path: str) -> float:
    """
    config.yaml 경로를 받아 SAM2 마스크 생성 후 GT와 IoU 계산하여 reward 반환
    """
    # 1. config.yaml 로드
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    # 2. SAM2 마스크 생성 (클릭 기반 분할 + 저장)
    run_sam2_from_config(cfg, config_path) 

    # 3. IoU 계산 (보상)
    gen_mask_dir = cfg["gen_mask_dir"]
    sim_mask_dir = cfg["sim_mask_dir"]
    mean_iou = compute_mean_iou(gen_mask_dir, sim_mask_dir)

    return mean_iou


# ✅ CLI 실행도 가능하게
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML")
    args = parser.parse_args()

    reward = run_reward_function(args.config)
    print(f"✅ Final Reward (Mean IoU): {reward:.4f}")
