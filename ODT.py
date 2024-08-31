from ultralytics import YOLO
import os
import json

# 1. YOLO v8 모델 로드
model = YOLO('yolov8n.pt')  # 'yolov8n.pt'는 사전 학습된 YOLOv8 모델 가중치 파일입니다.

# 2. 모델 학습
model.train(
    data='./data.yaml',  # 데이터셋 설정 파일 경로
    epochs=10,           # 학습 에폭 수
    patience=10,         # 조기 종료 기준
    batch=32,            # 배치 크기
    imgsz=416,           # 입력 이미지 크기
    save=True,           # 학습된 모델을 저장할지 여부 (기본값 True)
    name='coopaAI'       # 학습 결과가 저장될 폴더 이름
)

# 3. 저장 경로 확인
weights_dir = os.path.join('runs', 'train', 'coopaAI', 'weights')
best_model_path = os.path.join(weights_dir, 'best.pt')
last_model_path = os.path.join(weights_dir, 'last.pt')

# 학습 후 저장된 모델 파일 확인
if os.path.exists(best_model_path) and os.path.exists(last_model_path):
    print(f"Model saved successfully:\nBest model: {best_model_path}\nLast model: {last_model_path}")
else:
    print("Error: Model files not found. Check training settings or paths.")
