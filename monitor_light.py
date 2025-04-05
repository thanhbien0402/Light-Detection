import cv2
import time
import threading
from ultralytics import YOLO

class VideoStream:
    def __init__(self, src):
        self.cap = cv2.VideoCapture(src)
        self.ret, self.frame = self.cap.read()
        self.stopped = False
        # Thiết lập buffer size nếu cần
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    
    def start(self):
        threading.Thread(target=self.update, daemon=True).start()
        return self

    def update(self):
        while not self.stopped:
            ret, frame = self.cap.read()
            if not ret:
                self.stop()
                break
            self.ret, self.frame = ret, frame

    def read(self):
        return self.ret, self.frame

    def stop(self):
        self.stopped = True
        self.cap.release()


# Sử dụng đa luồng để giảm delay
ip = 'http://192.168.1.21:4747/video'
vs = VideoStream(ip).start()
model = YOLO(r"runs/detect/detect_light_v15/weights/best.pt")

while True:
    ret, frame = vs.read()
    if not ret:
        break

    # Có thể giảm kích thước đầu vào cho model nếu cần
    results = model.predict(frame, imgsz=640, conf=0.5)
    annotated_frame = results[0].plot()
    cv2.imshow("Traffic Light Detection", annotated_frame)
    if cv2.waitKey(1) == 27:
        break

vs.stop()
cv2.destroyAllWindows()
