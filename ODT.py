from ultralytics import YOLO
import os
import matplotlib.pyplot as plt

# 1. 한글 폰트 설정
plt.rcParams['font.family'] = 'AppleGothic'  # MacOS 예시 (AppleGothic 폰트 사용)

# 2. YOLO v8 모델 로드
model = YOLO('/Users/gangsiu/PycharmProjects/detect/coopaAI9/weights/best.pt')

# 3. MacBook M2 GPU 설정 (Metal backend 사용)
import torch

if torch.backends.mps.is_available():
    device = 'mps'  # MacOS에서 GPU 사용을 위한 Metal Performance Shaders(MPS) 설정
else:
    device = 'cpu'
    print("MPS is not available, using CPU instead. Ensure you're on a Mac with a Metal-compatible GPU.")

# 4. 모델 학습 (검증 생략)
model.train(
    data='./data.yaml',  # 데이터셋 설정 파일 경로
    epochs=30,           # 학습 에폭 수 (추천 값으로 조정)
    patience=15,         # 조기 종료 기준 (추천 값으로 조정)
    batch=4,             # 배치 크기 (MacBook M2의 메모리 제약에 맞추어 조정)
    imgsz=640,           # 입력 이미지 크기 (추천 값으로 조정)
    save=True,           # 학습된 모델을 저장할지 여부 (기본값 True)
    name='coopaAI',      # 학습 결과가 저장될 폴더 이름
    device=device,       # GPU 설정 추가
    pretrained=True
)