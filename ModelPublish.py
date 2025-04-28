import json
import io
import numpy as np
from PIL import Image
import cv2
from fastapi import FastAPI, HTTPException, File, UploadFile
from ultralytics import YOLO
import os
import datetime

modelVer = {
    "0.3": "3",   # +자두맛
    "0.2": "1",   # 0.91
    "0.1": "2"    # 0.78
}
model_path = f"/Users/gangsiu/PycharmProjects/detect/coopaAI{modelVer['0.2']}/weights/best.pt"
model = YOLO(model_path)

app = FastAPI()

def process_image(image: np.ndarray) -> dict:
    """이미지를 모델에 입력하고 예측 결과를 반환합니다."""
    try:
        # 모델 예측 수행
        results = model(image)

        # 결과가 있는지 확인
        if not results or not results[0].boxes.data.size:
            raise ValueError("No predictions found or tensor is empty")

        predictions = results[0].boxes.data.cpu().numpy()  # 예측된 bounding box 데이터 (예시)

        # 예측 결과를 요청된 JSON 형식으로 변환
        objects = [
            {
                "name": "티니핑_음료수_딸기맛",  # 예시로 고정된 이름 사용, 실제로는 모델 예측 결과에서 가져올 수 있음
                "price": 1200,  # 예시로 고정된 가격 사용, 실제로는 모델 예측 결과에서 가져올 수 있음
                "location": {
                    "x": int(pred[0]),  # x 좌표
                    "y": int(pred[1])   # y 좌표
                },
                "size": {
                    "width": int(pred[2] - pred[0]),  # 너비 계산
                    "height": int(pred[3] - pred[1])  # 높이 계산
                }
            }
            for pred in predictions
        ]

        # 최종 JSON 응답 구조
        response = {
            "time": datetime.datetime.now(),
            "count": len(objects),
            "objects": objects
        }

        return response

    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")

@app.post("/scan")
async def predict_image(file: UploadFile = File(...)):
    # 업로드된 파일을 읽고 이미지를 로드합니다.
    try:
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data)).convert('RGB')
        image = np.array(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error reading image: {str(e)}")

    # 예측 처리
    predictions = process_image(image)

    return predictions

@app.get("/")
def read_root():
    return {"message": "Hello World"}