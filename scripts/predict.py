import cv2
from ultralytics import YOLO

model = YOLO("runs/detect/traffic_light_v1/weights/best.pt")
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    results = model.predict(frame, imgsz=640, conf=0.5)
    annotated_frame = results[0].plot()
    cv2.imshow("Traffic Light Detection", annotated_frame)
    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()