from nil_video_predictor import run_sam2_and_save_masks
from nil_compute_IoU import compute_mean_iou

"""
input_path = "gen_video_frames/sample_video.mp4"
video_dir = "gen_video_frames/sample_video_frames"  # replace with your output directory
output_dir = "outputs"  # replace with your output directory
"""

def IoU_reward_function(input_path, video_dir, output_dir) -> float:
    """
    SAM2 마스크 생성 후 sim mask와 IoU 계산하여 reward 반환
    """
    run_sam2_and_save_masks(input_path, video_dir, output_dir)
    mean_iou = compute_mean_iou(output_dir, output_dir)  # GT와 생성된 마스크의 IoU 계산

    return mean_iou