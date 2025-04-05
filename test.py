import cv2
import time
from ultralytics import YOLO

# Tải mô hình (điều chỉnh đường dẫn đến tệp trọng số của bạn)
model = YOLO("runs/detect/detect_light_v15/weights/best.pt")

# Đường dẫn đến video đầu vào
input_video_path = r"C:\Users\bienc\Downloads\Monitor Light\dataset\video\20250225_093029.mp4"

# Mở video
cap = cv2.VideoCapture(input_video_path)

# Lấy thông số video: fps, kích thước frame
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Tạo cửa sổ hiển thị với kích thước thay đổi được
cv2.namedWindow("Detection", cv2.WINDOW_NORMAL)

# Biến để tính toán FPS
prev_time = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Xoay frame nếu cần thiết
    frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)

    # Dự đoán và vẽ annotation
    results = model.predict(frame, imgsz=640, conf=0.5)
    annotated_frame = results[0].plot()

    # Tính toán FPS
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time)
    prev_time = curr_time

    # Hiển thị FPS trên frame
    cv2.putText(annotated_frame, f"FPS: {fps:.2f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Hiển thị frame
    cv2.imshow("Detection", annotated_frame)
    if cv2.waitKey(1) & 0xFF == 27:  # Nhấn ESC để thoát
        break

cap.release()
cv2.destroyAllWindows()
