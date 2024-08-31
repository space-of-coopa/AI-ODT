from ultralytics import YOLO
import cv2

model = YOLO("../detect/train")

image_path = './dataset/test/images/frame_0.jpg'
image = cv2.imread(image_path)

results = model.predict(image)

for result in results:
    boxes = result.boxes
    for box in boxes:
        x1, y1, x2, y2 = box.xyxy[0]

        class_id = int(box.cls)
        confidence = box.conf

        class_name = model.names[class_id]

        print(f"Detected {class_name} with confidence {confidence:.2f} at [{x1}, {y1}, {x2}, {y2}]")

        color = (0, 255, 0)
        cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
        cv2.putText(image, f"{class_name} {confidence:.2f}", (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

output_image_path = 'output_image.jpg'
cv2.imwrite(output_image_path, image)
cv2.imshow('Detected Objects', image)
cv2.waitKey(0)
cv2.destroyAllWindows()