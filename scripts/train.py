from ultralytics import YOLO

model = YOLO("yolov8n.pt")

if __name__ == '__main__':
    model.train(data="data.yaml",
                epochs=150,              # Tăng epochs để học kỹ hơn
                batch=16,                # Batch nhỏ để phù hợp với dữ liệu ít
                imgsz=640,               # Giữ nguyên kích thước ảnh
                lr0=0.005,                # Learning rate ban đầu
                weight_decay=0.0005,     # Giảm overfitting
                hsv_h=0.1,               # Augmentation màu sắc
                hsv_s=0.7,
                hsv_v=0.4,
                degrees=15,              # Xoay ảnh
                flipud=0.5,              # Lật ngang/dọc
                mosaic=0.5,              # Augmentation mosaic
                device="0",              # Hoặc "0" nếu có GPU
                name="detect_light_v1",
                patience=40,             # Dừng sớm nếu không cải thiện
                seed=42)                 # Seed để tái lặp
    
    # Sau khi training xong, chạy validation
    # results = model.val(data="data.yaml", imgsz=640)
    # print(results)
