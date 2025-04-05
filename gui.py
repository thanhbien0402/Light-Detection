import cv2
import os
import time
import threading
import sys
import numpy as np
import pandas as pd
from ultralytics import YOLO
from PIL import Image, ImageTk
from pygrabber.dshow_graph import FilterGraph
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

# Lớp VideoStream dùng đa luồng để đọc frame từ camera
class VideoStream:
    def __init__(self, src):
        self.cap = cv2.VideoCapture(src)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.ret, self.frame = self.cap.read()
        self.last_frame_time = time.time()
        self.stopped = False

    def start(self):
        threading.Thread(target=self.update, daemon=True).start()
        return self

    def update(self):
        while not self.stopped:
            ret, frame = self.cap.read()
            if ret:
                self.ret, self.frame = ret, frame
                self.last_frame_time = time.time()
            else:
                time.sleep(0.01)

    def read(self):
        return self.ret, self.frame

    def stop(self):
        self.stopped = True
        self.cap.release()

class DetectionApp:
    def list_camera_names(self):
        """Liệt kê tên các camera sử dụng DirectShow trên Windows."""
        graph = FilterGraph()
        return graph.get_input_devices()
    
    def open_camera_with_fallback(self, index):
        """Thử mở camera local bằng backend DirectShow, nếu không thành công thì dùng MSMF."""
        cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
        if not cap.isOpened():
            # print("DirectShow không mở được, thử backend MSMF...")
            cap = cv2.VideoCapture(index)
        return cap

    def open_ip_camera(self, url):
        """Mở camera IP bằng URL."""
        cap = cv2.VideoCapture(url)
        return cap

    def __init__(self, root):
        self.root = root
        self.root.title("Detection Light App")
        # Layout gồm 2 phần: top (setup và status) và bottom (preview video)
        self.top_frame = ttk.Frame(root)
        self.top_frame.grid(row=0, column=0, sticky="ew", padx=10, pady=10)
        self.bottom_frame = ttk.Frame(root)
        self.bottom_frame.grid(row=1, column=0, sticky="nsew", padx=10, pady=10)
        root.grid_rowconfigure(1, weight=1)
        root.grid_columnconfigure(0, weight=1)

        self.running = False
        self.last_frame = None
        self.current_state = None
        self.last_update_time = time.time()
        self.update_job = None  # Để lưu ID callback của after

        # Preview hiển thị GUI sẽ được downscale xuống kích thước này (đầu vào cho model vẫn 640x640)
        self.preview_size = (640, 640)
        self.model_input_size = (640, 640)

        # Excel được lưu mỗi 5 giây; ảnh snapshot dựa vào thời gian do người dùng nhập
        self.excel_save_interval = 5  # giây
        self.last_excel_save_time = time.time()
        self.snapshot_save_interval = 10  # mặc định (sẽ cập nhật từ ô nhập)
        self.last_snapshot_save_time = time.time()

        self.durations = {"red": 0.0, "green": 0.0, "yellow": 0.0, "no_light": 0.0}

        self.label_map = {0: "red", 1: "green", 2: "yellow", 3: "no_light"}
        self.colors = {"red": (0, 0, 255), "green": (0, 255, 0), "yellow": (0, 255, 255), "no_light": (128, 128, 128)}

        # --- Phần trên: chia thành 2 cột: bên trái (setup) và bên phải (status) ---
        self.left_frame = ttk.LabelFrame(self.top_frame, text="Cài đặt")
        self.left_frame.grid(row=0, column=0, padx=5, pady=5, sticky="nw")
        self.right_frame = ttk.LabelFrame(self.top_frame, text="Thời gian sáng của từng trạng thái")
        self.right_frame.grid(row=0, column=1, padx=5, pady=5, sticky="ne")

        # Lựa chọn nguồn camera: Local hoặc IP
        ttk.Label(self.left_frame, text="Nguồn Camera:").grid(row=0, column=0, padx=5, pady=5, sticky="w")
        self.camera_type = tk.StringVar(value="local")
        self.local_radio = ttk.Radiobutton(self.left_frame, text="Local", variable=self.camera_type, value="local", command=self.toggle_camera_source)
        self.local_radio.grid(row=0, column=1, padx=5, pady=5, sticky="w")
        self.ip_radio = ttk.Radiobutton(self.left_frame, text="IP", variable=self.camera_type, value="ip", command=self.toggle_camera_source)
        self.ip_radio.grid(row=0, column=2, padx=5, pady=5, sticky="w")

        # Nếu chọn local: hiển thị combobox danh sách camera và nút Refresh
        self.local_frame = ttk.LabelFrame(self.left_frame, text="Chọn Camera:")
        self.local_frame.grid(row=1, column=0, columnspan=3, padx=5, pady=5, sticky="w")
        self.camera_list = self.list_camera_names()
        self.camera_combo = ttk.Combobox(self.local_frame, values=self.camera_list, state="readonly", width=30)
        self.camera_combo.grid(row=1, column=0, padx=5, pady=5)
        if self.camera_list:
            self.camera_combo.current(0)
        else:
            self.camera_combo['values'] = ["Không có camera nào"]
            self.camera_combo.current(0)
        self.refresh_button = ttk.Button(self.left_frame, text="Refresh Camera", command=self.refresh_camera_list)
        self.refresh_button.grid(row=1, column=1, columnspan=2, padx=5, pady=5)

        # Nếu chọn IP: hiển thị ô nhập URL trên một hàng, hàng bên dưới chứa Username và Password
        self.ip_frame = ttk.Frame(self.left_frame)
        self.ip_frame.grid(row=3, column=0, columnspan=3, padx=5, pady=5, sticky="w")
        # Hàng 0: Nhập URL
        ttk.Label(self.ip_frame, text="IP URL:").grid(row=0, column=0, padx=5, pady=5, sticky="w")
        self.ip_url_var = tk.StringVar()
        self.ip_entry = ttk.Entry(self.ip_frame, textvariable=self.ip_url_var, width=30)
        self.ip_entry.grid(row=0, column=1, columnspan=4, padx=5, pady=5, sticky="w")
        # Hàng 1: Nhập Username và Password trên cùng một dòng
        ttk.Label(self.ip_frame, text="Username:").grid(row=1, column=0, padx=5, pady=5, sticky="w")
        self.ip_username_var = tk.StringVar()
        self.ip_username_entry = ttk.Entry(self.ip_frame, textvariable=self.ip_username_var, width=15)
        self.ip_username_entry.grid(row=1, column=1, padx=5, pady=5, sticky="w")
        ttk.Label(self.ip_frame, text="Password:").grid(row=1, column=2, padx=5, pady=5, sticky="w")
        self.ip_password_var = tk.StringVar()
        self.ip_password_entry = ttk.Entry(self.ip_frame, textvariable=self.ip_password_var, width=15, show="*")
        self.ip_password_entry.grid(row=1, column=3, padx=5, pady=5, sticky="w")
        # Ẩn ô nhập IP khi chọn local
        self.ip_frame.grid_remove()

        ttk.Label(self.left_frame, text="Thư mục lưu Excel:").grid(row=4, column=0, padx=5, pady=5, sticky="w")
        self.excel_dir = tk.StringVar()
        self.excel_entry = ttk.Entry(self.left_frame, textvariable=self.excel_dir, width=30)
        self.excel_entry.grid(row=4, column=1, padx=5, pady=5)
        ttk.Button(self.left_frame, text="Chọn...", command=self.choose_excel_dir).grid(row=4, column=2, padx=5, pady=5)

        ttk.Label(self.left_frame, text="Thư mục lưu ảnh:").grid(row=5, column=0, padx=5, pady=5, sticky="w")
        self.snap_dir = tk.StringVar()
        self.snap_entry = ttk.Entry(self.left_frame, textvariable=self.snap_dir, width=30)
        self.snap_entry.grid(row=5, column=1, padx=5, pady=5)
        ttk.Button(self.left_frame, text="Chọn...", command=self.choose_snap_dir).grid(row=5, column=2, padx=5, pady=5)

        # Ô nhập thời gian lưu ảnh (theo giây)
        ttk.Label(self.left_frame, text="Thời gian lưu ảnh (giây):").grid(row=6, column=0, padx=5, pady=5, sticky="w")
        self.snapshot_interval_var = tk.StringVar(value="10")
        self.snapshot_interval_entry = ttk.Entry(self.left_frame, textvariable=self.snapshot_interval_var, width=10)
        self.snapshot_interval_entry.grid(row=6, column=1, padx=5, pady=5, sticky="w")

        self.start_button = ttk.Button(self.left_frame, text="Start Detection", command=self.toggle_detection)
        self.start_button.grid(row=7, column=0, columnspan=3, padx=5, pady=10)

        # Hiển thị trạng thái model (đang tải,...)
        self.model_status = ttk.Label(self.left_frame, text="Đang tải model...", foreground="blue")
        self.model_status.grid(row=8, column=0, columnspan=3, padx=5, pady=5)

        # Bên phải: hiển thị thời gian cho từng trạng thái
        self.status_labels = {}
        for i, state in enumerate(["red", "green", "yellow", "no_light"]):
            ttk.Label(self.right_frame, text=state.capitalize() + ":").grid(row=i, column=0, padx=5, pady=2, sticky="w")
            self.status_labels[state] = ttk.Label(self.right_frame, text="0.0s")
            self.status_labels[state].grid(row=i, column=1, padx=5, pady=2, sticky="w")

        # --- Phần dưới: hiển thị preview video ---
        self.video_frame = ttk.LabelFrame(self.bottom_frame, text="Video Preview")
        self.video_frame.pack(padx=5, pady=5, fill="both", expand=True)
        self.video_label = ttk.Label(self.video_frame)
        self.video_label.pack()

        if getattr(sys, 'frozen', False):
            base_path = sys._MEIPASS
        else:
            base_path = os.path.abspath(".")
        self.model_path = os.path.join(base_path, "best.pt")
        self.model = None  # Sẽ tải sau khi GUI khởi tạo

        # Tải model nền sau khi GUI khởi tạo
        self.root.after(100, self.load_model_thread)
        # Khởi tạo VideoStream là None, sẽ được tạo khi bắt đầu detection
        self.vs = None

    def refresh_camera_list(self):
        """Cập nhật lại danh sách camera khi cắm thêm thiết bị mới."""
        self.camera_list = self.list_camera_names()
        self.camera_combo['values'] = self.camera_list
        if self.camera_list:
            self.camera_combo.current(0)
        else:
            self.camera_combo['values'] = ["Không có camera nào"]
            self.camera_combo.current(0)

    def toggle_camera_source(self):
        """Hiển thị/ẩn các thành phần dựa trên nguồn camera được chọn."""
        if self.camera_type.get() == "local":
            self.local_frame.grid()
            self.refresh_button.grid()  # Hiển thị nút Refresh cho local
            self.ip_frame.grid_remove()
        else:
            self.local_frame.grid_remove()
            self.refresh_button.grid_remove()  # Ẩn nút Refresh khi IP
            self.ip_frame.grid()

    def load_model_thread(self):
        """Tải model trên một luồng riêng và cập nhật giao diện khi hoàn thành."""
        def load():
            try:
                self.model = YOLO(self.model_path)
                # Nếu load thành công, ẩn thông báo trạng thái
                self.model_status.grid_remove()
            except Exception as e:
                self.model_status.config(text=f"Lỗi khi tải model: {e}", foreground="red")
        threading.Thread(target=load, daemon=True).start()


    def choose_excel_dir(self):
        path = filedialog.askdirectory()
        if path:
            self.excel_dir.set(path)

    def choose_snap_dir(self):
        path = filedialog.askdirectory()
        if path:
            self.snap_dir.set(path)

    def toggle_detection(self):
        if not self.running:
            if not self.excel_dir.get() or not self.snap_dir.get():
                messagebox.showwarning("Chưa chọn thư mục", "Vui lòng chọn thư mục lưu file Excel và Ảnh.")
                return

            # Xác định nguồn camera: local hoặc ip
            if self.camera_type.get() == "local":
                selected_index = self.camera_combo.current()
                src = selected_index
            else:
                ip_url = self.ip_url_var.get().strip()
                if not ip_url:
                    messagebox.showwarning("Chưa nhập URL", "Vui lòng nhập URL cho camera IP.")
                    return

                if "://" in ip_url:
                    protocol, url_body = ip_url.split("://", 1)
                    protocol += "://"
                else:
                    protocol = ""
                    url_body = ip_url

                username = self.ip_username_var.get().strip()
                password = self.ip_password_var.get().strip()

                if username and password:
                    src = f"{protocol}{username}:{password}@{url_body}"
                else:
                    src = f"{protocol}{url_body}"

            # Sử dụng lớp VideoStream để đọc frame từ camera trên luồng riêng
            self.vs = VideoStream(src).start()

            if isinstance(src, int):
                self.vs.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                self.vs.cap.set(cv2.CAP_PROP_FPS, 60)
                self.vs.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.model_input_size[0])
                self.vs.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.model_input_size[1])

            try:
                self.snapshot_save_interval = float(self.snapshot_interval_var.get())
            except Exception as e:
                self.snapshot_save_interval = 10.0

            self.running = True
            self.start_button.config(text="Stop Detection")
            self.last_excel_save_time = time.time()
            self.last_snapshot_save_time = time.time()
            self.last_update_time = time.time()
            self.update_video()
        else:
            self.running = False
            self.start_button.config(text="Start Detection")
            if self.update_job is not None:
                self.root.after_cancel(self.update_job)
                self.update_job = None
            if self.vs:
                self.vs.stop()

    def update_video(self):
        if self.running and self.vs is not None:
            ret, frame = self.vs.read()
            current_time = time.time()
            if current_time - self.vs.last_frame_time > 2:
                self.video_label.configure(text="Không nhận được khung hình từ camera.", image="")
                self.running = False
                self.start_button.config(text="Start Detection")
                self.vs.stop()
                return

            if ret:
                self.last_frame = frame.copy()
                if self.model is None:
                    cv2.putText(frame, "Đang tải model...", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                else:
                    try:
                        results = self.model.predict(frame, imgsz=self.model_input_size[0], conf=0.5)
                        new_state = None
                        if results and results[0].boxes is not None and len(results[0].boxes) > 0:
                            boxes = results[0].boxes
                            confs = boxes.conf.cpu().numpy()
                            idx = np.argmax(confs)
                            cls_idx = int(boxes.cls[idx].item())
                            new_state = self.label_map.get(cls_idx, None)
                            box = boxes.xyxy[idx].cpu().numpy().astype(int)
                            color = self.colors.get(new_state, (255, 255, 255))
                            cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), color, 2)
                            text = f"{new_state} ({confs[idx]:.2f})"
                            cv2.putText(frame, text, (box[0], box[1]-10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                        if new_state:
                            elapsed_time = current_time - self.last_update_time
                            if self.current_state:
                                self.durations[self.current_state] += elapsed_time
                            self.current_state = new_state
                            self.last_update_time = current_time
                    except Exception as e:
                        print("Lỗi khi dự đoán:", e)
                self.update_status_labels()

                frame_display = cv2.resize(frame, self.preview_size)
                frame_rgb = cv2.cvtColor(frame_display, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(frame_rgb)
                imgtk = ImageTk.PhotoImage(image=img)
                self.video_label.imgtk = imgtk
                self.video_label.configure(image=imgtk, text="")

                if current_time - self.last_excel_save_time >= self.excel_save_interval:
                    self.save_excel_data()
                    self.last_excel_save_time = current_time
                if current_time - self.last_snapshot_save_time >= self.snapshot_save_interval:
                    self.save_snapshot()
                    self.last_snapshot_save_time = current_time

                self.update_job = self.root.after(10, self.update_video)
            else:
                self.video_label.configure(text="Không nhận được khung hình từ camera.", image="")
                self.update_job = self.root.after(500, self.update_video)

    def update_status_labels(self):
        for state in self.durations:
            self.status_labels[state].config(text=f"{self.durations[state]:.1f}s")

    def save_excel_data(self):
        data = {
            "State": list(self.durations.keys()),
            "Duration (mm:ss)": [time.strftime("%M:%S", time.gmtime(val)) for val in self.durations.values()]
        }
        df = pd.DataFrame(data)
        excel_path = os.path.join(self.excel_dir.get(), "light_duration.xlsx")
        df.to_excel(excel_path, index=False)
        # print(f"Cập nhật Excel: {excel_path}")

    def save_snapshot(self):
        if self.last_frame is None or not self.current_state:
            # print("Không có dữ liệu ảnh để lưu, bỏ qua.")
            return
        if self.current_state not in ["red", "green", "yellow", "no_light"]:
            # print(f"Trạng thái '{self.current_state}' không phù hợp, bỏ qua lưu ảnh.")
            return
        timestamp = time.strftime("%M_%S")
        snapshot_path = os.path.join(self.snap_dir.get(), self.current_state, f"{timestamp}.jpg")
        os.makedirs(os.path.dirname(snapshot_path), exist_ok=True)
        cv2.imwrite(snapshot_path, self.last_frame)
        # print(f"Đã lưu snapshot: {snapshot_path}")


if __name__ == "__main__":
    root = tk.Tk()
    root.geometry("800x800")
    app = DetectionApp(root)
    root.mainloop()
