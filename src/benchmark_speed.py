import glob
import time
import cv2
import numpy as np
from ultralytics import YOLO
import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image

# ================== CONFIG ==================
# Thư mục chứa ảnh test
BENCHMARK_DIR = "../benchmark_images"   # đổi nếu bạn đặt chỗ khác

# Đường dẫn model YOLO & classifier của bạn
YOLO_WEIGHTS = "../models/detection/yolov8n_vn_best.pt"
CLS_WEIGHTS  = "../models/classification/efficientnet_best.pth"

NUM_CLASSES = 43  # số class GTSRB (chỉnh lại nếu khác)
# ===========================================


# --------- Load YOLO ---------
print("[INFO] Loading YOLO model...")
yolo_model = YOLO(YOLO_WEIGHTS)

# --------- Định nghĩa EfficientNet-B0 thô (giống trong project bạn) ---------
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

print("[INFO] Loading EfficientNet-B0...")
base_model = efficientnet_b0(weights=None)  # không load pretrained từ torchvision
in_features = base_model.classifier[1].in_features
base_model.classifier[1] = nn.Linear(in_features, NUM_CLASSES)

state = torch.load(CLS_WEIGHTS, map_location="cpu")
base_model.load_state_dict(state["model_state_dict"] if "model_state_dict" in state else state)
base_model.eval()

# Transform giống lúc train / infer
cls_transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]),
])


def run_classifier_on_crop(bgr_img):
    """Chạy EfficientNet trên 1 ảnh BGR (crop hoặc full)"""
    rgb = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb)
    inp = cls_transform(pil_img).unsqueeze(0)
    with torch.no_grad():
        logits = base_model(inp)
        prob = torch.softmax(logits, dim=1).max().item()
    return prob


def benchmark(folder):
    img_paths = []
    for ext in ("*.jpg", "*.jpeg", "*.png"):
        img_paths.extend(glob.glob(f"{folder}/{ext}"))

    if len(img_paths) == 0:
        print(f"[WARN] Không tìm thấy ảnh nào trong thư mục: {folder}")
        return

    print(f"[INFO] Tìm thấy {len(img_paths)} ảnh benchmark.")

    times_mode1 = []
    times_mode2 = []
    times_mode3 = []

    for path in img_paths:
        img_bgr = cv2.imread(path)
        if img_bgr is None:
            print(f"[WARN] Không đọc được ảnh: {path}")
            continue

        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        # ========== MODE 1: YOLO ONLY ==========
        t0 = time.time()
        det_res = yolo_model(img_rgb)[0]
        t1 = (time.time() - t0) * 1000.0
        times_mode1.append(t1)

        # ========== MODE 2: CLS ONLY ==========
        # giả sử ta đã có 1 crop biển báo -> ở đây đơn giảnResize full ảnh
        t0 = time.time()
        _ = run_classifier_on_crop(img_bgr)
        t2 = (time.time() - t0) * 1000.0
        times_mode2.append(t2)

        # ========== MODE 3: YOLO + CLS ==========
        t0 = time.time()
        det_res = yolo_model(img_rgb)[0]
        if len(det_res.boxes) > 0:
            for box in det_res.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                # bảo vệ border
                x1 = max(0, x1); y1 = max(0, y1)
                x2 = min(img_bgr.shape[1], x2)
                y2 = min(img_bgr.shape[0], y2)
                if x2 <= x1 or y2 <= y1:
                    continue

                crop = img_bgr[y1:y2, x1:x2]
                if crop.size == 0:
                    continue

                _ = run_classifier_on_crop(crop)
        t3 = (time.time() - t0) * 1000.0
        times_mode3.append(t3)

        print(f"[{path}]  M1={t1:.1f} ms,  M2={t2:.1f} ms,  M3={t3:.1f} ms")

    # ========== TỔNG KẾT ==========
    def avg(arr): return sum(arr) / len(arr) if len(arr) > 0 else 0.0

    m1 = avg(times_mode1)
    m2 = avg(times_mode2)
    m3 = avg(times_mode3)

    print("\n=========== TÓM TẮT BENCHMARK ===========")
    print(f"Số ảnh test: {len(times_mode1)}")
    print(f"Mode 1 - YOLO only          : {m1:.2f} ms/ảnh (~{1000.0/m1:.1f} FPS)")
    print(f"Mode 2 - Classifier only    : {m2:.2f} ms/ảnh (~{1000.0/m2:.1f} FPS)")
    print(f"Mode 3 - YOLO + CLS (+rule) : {m3:.2f} ms/ảnh (~{1000.0/m3:.1f} FPS)")
    print(f"Độ chậm Mode 3 so với Mode 1: x{(m3/m1):.2f}")
    print("=========================================")


if __name__ == "__main__":
    benchmark(BENCHMARK_DIR)
