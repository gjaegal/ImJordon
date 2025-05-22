import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from timesformer.models.vit import TimeSformer
import os
import cv2
import matplotlib.pyplot as plt

class VideoEncoder:
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        """
        TimeSformer의 비디오 인코더 t 초기화
        """
        self.device = device
        
        self.model = TimeSformer(
            img_size=224,
            num_classes=400,
            num_frames=8,
            attention_type='divided_space_time',
            pretrained_model='/root/TimeSformer/TimeSformer/timesformer/pretrained/TimeSformer_divST_8x32_224_K400.pth'
        )
        
        self.model.eval()
        self.model.to(self.device)
        
        # 입력 이미지 전처리를 위한 변환
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),    # [0, 255] -> [0, 1]
            transforms.Normalize(mean=[0.45, 0.45, 0.45],
                                std=[0.225, 0.225, 0.225])
        ])

    def apply_mask(self, frame, mask):
        """
        프레임에 마스크 적용
        Args:
            frame: numpy 배열 (H, W, 3)
            mask: numpy 배열 (H, W)
        Returns:
            마스킹된 numpy 배열
        """
        masked = frame.copy()
        masked[mask == 0] = 0
        return masked
    
    def preprocess_frames(self, frames, masks=None):
        """
        프레임들을 전처리하고 텐서로 변환
        Args:
            frames: [T, H, W, 3] or PIL Image list
            masks: [T, H, W] numpy array list (선택)
        Returns:
            torch.Tensor [1, 3, T, 224, 224]
        """
        processed = []
        for i, frame in enumerate(frames):
            if isinstance(frame, np.ndarray):
                if masks is not None:
                    frame = self.apply_mask(frame, masks[i])
                frame = Image.fromarray(frame.astype(np.uint8))
            processed.append(self.transform(frame))
        tensor = torch.stack(processed).unsqueeze(0)  # [1, T, C, H, W]
        return tensor.permute(0, 2, 1, 3, 4).to(self.device)  # [1, 3, T, H, W]
    
    def create_clip(self, frames, masks, t):
        """
        시점 t에서 8프레임 클립 생성
        """
        if t < 7:
            f_clip = [frames[0]] * (7 - t) + frames[:t+1]
            m_clip = [masks[0]] * (7 - t) + masks[:t+1]
        else:
            f_clip = frames[t-7:t+1]
            m_clip = masks[t-7:t+1]
        return self.preprocess_frames(f_clip, m_clip)
    
    def extract_embedding(self, clip_tensor):
        """
        TimeSformer로부터 임베딩 추출
        Returns:
            torch.Tensor [D]
        """
        with torch.no_grad():
            z = self.model(clip_tensor)
        return z.squeeze(0)

    def compare_embeddings(self, z1, z2):
        """
        L2 거리 계산
        Returns:
            float
        """
        return torch.norm(z1 - z2, p=2).item()