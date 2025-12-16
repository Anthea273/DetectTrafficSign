"""
Unified pipeline demo (Mục 3):
YOLOv8 (VN) + EfficientNet-B0 (GTSRB) trên ảnh tĩnh.

Chạy từ thư mục root của project:
    python src/detection/inference_unified_image.py \
        --image path/to/input.jpg \
        --out_img results/unified/output.jpg
"""

import os
import sys
import json
import argparse

import cv2
import torch
import torch.nn.functional as F
from ultralytics import YOLO
from PIL import Image

# ====== CHUẨN BỊ ĐƯỜNG DẪN (thêm src vào sys.path) ======
THIS_DIR   = os.path.dirname(os.path.abspath(__file__))  # .../src/detection
SRC_DIR    = os.path.dirname(THIS_DIR)                   # .../src
PROJECT_ROOT = os.path.dirname(SRC_DIR)                  # .../
sys.path.append(SRC_DIR)  # để import classification.inference_classification

# ====== IMPORT TỪ MODULE CLASSIFICATION CŨ ======
# Nếu file inference_classification.py ở chỗ khác thì sửa import này cho đúng
from classification.inference_classification import (
    build_model,
    load_class_names as load_cls_names,
    tfm as cls_transform,
    DEVICE,
    MODEL_PATH as CLS_MODEL_PATH,
    CLASS_NAMES_FILE as CLS_CLASSES_FILE,
)

# ====== ĐƯỜNG DẪN MÔ HÌNH YOLO VÀ LABEL VN ======
YOLO_MODEL_PATH   = os.path.join(PROJECT_ROOT, "models", "detection", "yolov8n_vn_best.pt")
YOLO_CLASSES_JSON = os.path.join(PROJECT_ROOT, "models", "detection", "classes_vn.json")

# ====== HÀM LOAD YOLO & CLASSIFIER ======
def load_yolo(model_path=YOLO_MODEL_PATH):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Không tìm thấy YOLO model: {model_path}")
    return YOLO(model_path)

def load_classifier(model_path=CLS_MODEL_PATH, classes_path=CLS_CLASSES_FILE):
    class_names = load_cls_names(classes_path)
    model = build_model(num_classes=len(class_names)).to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()
    return model, class_names

# ====== LOAD TÊN LỚP VN (NẾU CÓ) ======
def load_yolo_class_names(yolo_model, classes_json=YOLO_CLASSES_JSON):
    if os.path.exists(classes_json):
        with open(classes_json, "r", encoding="utf-8") as f:
            return json.load(f)
    return yolo_model.names  # fallback

# ====== HÀM CLASSIFY CROP TỪ BBOX ======
def classify_crop_bgr(img_bgr, bbox, clf_model, class_names, topk=1):
    """
    bbox: (x1, y1, x2, y2) theo pixel, img_bgr: H x W x 3 (BGR)
    Trả về list [(label, prob), ...]
    """
    x1, y1, x2, y2 = [int(v) for v in bbox]
    h, w = img_bgr.shape[:2]

    # giới hạn bbox trong ảnh
    x1 = max(0, min(x1, w - 1))
    x2 = max(0, min(x2, w - 1))
    y1 = max(0, min(y1, h - 1))
    y2 = max(0, min(y2, h - 1))

    if x2 <= x1 or y2 <= y1:
        return None

    crop_bgr = img_bgr[y1:y2, x1:x2]
    if crop_bgr.size == 0:
        return None

    crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(crop_rgb)

    x = cls_transform(pil).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits = clf_model(x)
        probs = F.softmax(logits, dim=1)[0]

    top_probs, top_idx = torch.topk(probs, k=topk)
    top_probs = top_probs.cpu().numpy()
    top_idx   = top_idx.cpu().numpy()

    results = []
    for idx_i, prob_i in zip(top_idx, top_probs):
        idx_i = int(idx_i)
        name_i = class_names[idx_i] if idx_i < len(class_names) else f"class_{idx_i}"
        results.append((name_i, float(prob_i)))

    return results

# ====== UNIFIED PIPELINE TRÊN ẢNH TĨNH ======
def unified_infer_image(
    image_path: str,
    out_img_path: str,
    yolo,
    clf_model,
    cls_names,
    yolo_names,
    topk: int = 1
):
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        raise FileNotFoundError(image_path)

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # --- Bước 1: YOLO detection ---
    det_results = yolo(img_rgb)[0]

    detections = []  # để in bảng
    out_img = img_bgr.copy()

    if len(det_results.boxes) == 0:
        print("⚠️ Không phát hiện được biển báo nào.")
        return detections, out_img

    for box in det_results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        det_conf = float(box.conf[0])
        cls_id   = int(box.cls[0])

        # Nhãn YOLO (VN)
        yolo_label = yolo_names[cls_id] if isinstance(yolo_names, (list, tuple)) else yolo_names.get(cls_id, str(cls_id))

        # --- Bước 2: crop & classify ---
        preds = classify_crop_bgr(out_img, (x1, y1, x2, y2), clf_model, cls_names, topk=topk)

        if preds is not None and len(preds) > 0:
            cls_label, cls_prob = preds[0]
        else:
            # fallback: nếu classifier lỗi → giữ nhãn YOLO
            cls_label, cls_prob = yolo_label, det_conf

        detections.append({
            "YOLO_label": yolo_label,
            "Classifier_label": cls_label,
            "Det_conf": round(det_conf, 4),
            "Cls_conf": round(cls_prob, 4),
            "bbox": [x1, y1, x2, y2],
        })

        # --- Bước 3: vẽ BBox + nhãn cuối cùng ---
        cv2.rectangle(out_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            out_img,
            f"{cls_label} ({cls_prob:.2f})",
            (x1, max(y1 - 5, 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1,
            cv2.LINE_AA
        )

    # Lưu ảnh kết quả
    os.makedirs(os.path.dirname(out_img_path), exist_ok=True)
    cv2.imwrite(out_img_path, out_img)
    print(f"Saved unified result image to: {out_img_path}")

    # In bảng kết quả
    print("Detections (YOLO + Classifier):")
    for d in detections:
        print(
            f"- YOLO: {d['YOLO_label']:25s} | "
            f"CLS: {d['Classifier_label']:25s} | "
            f"Det_conf={d['Det_conf']:.2f} | Cls_conf={d['Cls_conf']:.2f} | "
            f"bbox={d['bbox']}"
        )

    return detections, out_img

# ====== MAIN ======
def main():
    parser = argparse.ArgumentParser(description="Unified pipeline: YOLOv8 + EfficientNet-B0 on a single image.")
    parser.add_argument("--image", required=True, help="Đường dẫn ảnh đầu vào")
    parser.add_argument("--out_img", default="results/unified/unified_output.jpg", help="Ảnh output có vẽ BBox")
    args = parser.parse_args()

    print("Loading YOLOv8 (VN)...")
    yolo = load_yolo()
    yolo_names = load_yolo_class_names(yolo)

    print("Loading EfficientNet-B0 (GTSRB)...")
    clf_model, cls_names = load_classifier()

    unified_infer_image(
        image_path=args.image,
        out_img_path=args.out_img,
        yolo=yolo,
        clf_model=clf_model,
        cls_names=cls_names,
        yolo_names=yolo_names,
        topk=1,
    )

if __name__ == "__main__":
    main()
