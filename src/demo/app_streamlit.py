import threading
import streamlit as st
import time
import torch
import cv2
import numpy as np
import tempfile
import os
import pandas as pd
from ultralytics import YOLO
from torchvision import models, transforms
import torch.nn as nn
from PIL import Image
import json
import torch.nn.functional as F
import unicodedata
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import av
from collections import deque

MODEL_LOCK = threading.Lock()

DET_MIN_CONF = 0.18          # giữ detection khi webcam mờ
UNCERTAIN_MIN_CONF = 0.10    # dưới mức này bỏ hẳn bbox rác
SPECIAL_MIN_CONF = {
    "NO_STRAIGHT_RIGHT": 0.65,  # ép class này phải chắc mới hiện (chống spam)
}

CLS_MIN_PROB = 0.75          # classifier chỉ override khi rất chắc

def strip_accents(text: str) -> str:
    """Bỏ dấu tiếng Việt để in bằng cv2.putText."""
    text_nfkd = unicodedata.normalize("NFD", text)
    return "".join(ch for ch in text_nfkd if unicodedata.category(ch) != "Mn")

# ==================== PAGE CONFIG ====================
st.set_page_config(
    page_title="Traffic Sign Demo System",
    page_icon="🚗",
    layout="wide"
)

# ==================== CONFIG ====================
YOLO_MODEL_PATH = "models/detection/yolov8n_vn_best.pt"

CLASSIFY_MODEL_PATH = "models/classification/efficientnet_best.pth"
CLASS_NAMES_FILE    = "models/classification/class_names_gtsrb.json"

NUM_CLASSES = 43
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ==================== LOAD MODELS ====================
@st.cache_resource
def load_yolo():
    return YOLO(YOLO_MODEL_PATH)

@st.cache_resource
def load_classifier():
    # EfficientNet-B0 backbone
    clf = models.efficientnet_b0(weights=None)
    in_feats = clf.classifier[1].in_features
    clf.classifier[1] = nn.Linear(in_feats, NUM_CLASSES)
    clf.load_state_dict(torch.load(CLASSIFY_MODEL_PATH, map_location=DEVICE))
    clf.eval().to(DEVICE)
    return clf

yolo_model = load_yolo()
clf_model  = load_classifier()

# ==================== RULE: CHỈ RE-FINE MỘT SỐ BIỂN ====================
# Những từ khoá nếu xuất hiện trong YOLO_label thì mới cho classifier "chen vào"
REFINE_WITH_CLS = [
    "speed",            # ví dụ YOLO_label = "Speed limit", "speed-limit-50"...
    "giới hạn tốc độ",  # nếu sau này bạn dùng nhãn tiếng Việt
]

# ==================== CLASS NAMES (YOLO - VN) ====================
CLASSES_FILE = "models/detection/classes_vn.json"

if os.path.exists(CLASSES_FILE):
    with open(CLASSES_FILE, "r", encoding="utf-8") as f:
        CLASS_NAMES_SHORT = json.load(f)
else:
    CLASS_NAMES_SHORT = {}


# ==================== CLASS NAMES (GTSRB) ====================
if os.path.exists(CLASS_NAMES_FILE):
    with open(CLASS_NAMES_FILE, "r", encoding="utf-8") as f:
        CLASS_NAMES = json.load(f)
else:
    # fallback: đặt tên class_i nếu chưa có file
    CLASS_NAMES = [f"class_{i}" for i in range(NUM_CLASSES)]

# ==================== TRANSFORMS ====================
clf_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])

# ==================== HELPER: CLASSIFY CROP TỪ BBOX ====================
def classify_crop_bgr(img_bgr, x1, y1, x2, y2, topk=1):
    """
    Cắt vùng [x1, y1, x2, y2] từ img_bgr (BGR),
    chạy qua EfficientNet-B0 (clf_model),
    trả về list [(label, prob), ...] theo Top-k.
    """
    # Kích thước ảnh
    h, w = img_bgr.shape[:2]

    # Giới hạn toạ độ trong khung ảnh
    x1 = max(0, min(int(x1), w - 1))
    x2 = max(0, min(int(x2), w - 1))
    y1 = max(0, min(int(y1), h - 1))
    y2 = max(0, min(int(y2), h - 1))

    if x2 <= x1 or y2 <= y1:
        return None  # bbox lỗi

    # Cắt ảnh
    crop_bgr = img_bgr[y1:y2, x1:x2]
    if crop_bgr.size == 0:
        return None

    # BGR -> RGB -> PIL
    crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    pil_img  = Image.fromarray(crop_rgb)

    # Transform giống như classification mode
    x = clf_transform(pil_img).unsqueeze(0).to(DEVICE)

    # Chạy qua EfficientNet
    with torch.no_grad():
        logits = clf_model(x)
        probs  = F.softmax(logits, dim=1)[0]

    # Lấy Top-k
    top_probs, top_indices = torch.topk(probs, k=topk)
    top_probs   = top_probs.cpu().numpy()
    top_indices = top_indices.cpu().numpy()

    preds = []
    for idx_i, prob_i in zip(top_indices, top_probs):
        idx_i = int(idx_i)
        name_i = CLASS_NAMES[idx_i] if idx_i < len(CLASS_NAMES) else f"class_{idx_i}"
        preds.append((name_i, float(prob_i)))

    return preds

# ==================== GLOBAL HEADER ====================
st.markdown(
    "<h1 style='text-align: center; margin-bottom: 0.2rem;'>🚗 Traffic Sign Demo System</h1>",
    unsafe_allow_html=True,
)
st.markdown(
    "<p style='text-align: center; color: gray;'>"
    "Hệ thống minh họa nhận diện & phân loại biển báo giao thông sử dụng YOLOv8 và EfficientNet-B0."
    "</p>",
    unsafe_allow_html=True,
)
st.markdown("---")

# =========== SIDEBAR: MODE & INFO ===========
with st.sidebar:
    st.header("⚙️ Cấu hình demo")
    mode = st.radio(
        "Chọn chế độ:",
        [
            "1️⃣ Vietnam Traffic Sign Detection (YOLOv8)",
            "2️⃣ GTSRB Traffic Sign Classification (EfficientNet-B0)",
            "3️⃣ Unified Pipeline (YOLO + EfficientNet)"
        ]
    )
    st.markdown("---")
    st.markdown("### ℹ️ Hướng dẫn nhanh")
    if mode.startswith("1️⃣"):
        st.write(
            "- Chọn **Image** để upload một ảnh giao thông.\n"
            "- Chọn **Video** để upload một video ngắn.\n"
            "- Kết quả sẽ hiển thị khung bao quanh biển báo và bảng tổng hợp."
        )
    elif mode.startswith("2️⃣"):
        st.write(
            "- Upload **ảnh đã crop sẵn** chỉ chứa biển báo.\n"
            "- Hệ thống trả về lớp dự đoán (Top-5) và bảng Top-k."
        )
    else:  # 3️⃣ Unified Pipeline
        st.write(
            "- Chọn **Image / Video / Webcam** làm nguồn đầu vào.\n"
            "- Hệ thống tự động **phát hiện biển báo (YOLOv8)** và **phân loại chi tiết (EfficientNet-B0)**.\n"
            "- Mỗi biển báo được hiển thị **bounding box, nhãn cuối cùng và confidence**.\n"
            "- Bảng kết quả hiển thị **nhãn YOLO, nhãn classifier, Top-5 prediction**.\n"
            "- Các trường hợp **mismatch** giữa detection và classification được đánh dấu để phân tích.\n"
            "- FPS realtime được hiển thị để đánh giá hiệu năng hệ thống."
        )


class Mode3WebcamProcessor(VideoProcessorBase):
    """
    Realtime webcam processor for Mode 3:
    YOLO -> crop -> classifier -> (optional refine) -> draw -> FPS
    """

    def __init__(self):
        # runtime options (set từ Streamlit UI)
        self.enable_refine = True
        self.show_top5 = False
        self.mismatch_only = False

        self.conf = 0.15
        self.imgsz = 640

        self.short_map = {}      # CLASS_NAMES_SHORT
        self.refine_keys = []    # REFINE_WITH_CLS

        # fps smoothing
        self.fps_hist = deque(maxlen=30)

        # optional: expose latest rows to UI (nếu bạn muốn show bảng)
        self.latest_rows = []

        

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img_bgr = frame.to_ndarray(format="bgr24")
        draw = img_bgr.copy()

        t0 = time.time()

        # --- YOLO detect (RGB) ---
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        # quan trọng: khóa model để thread-safe
        with MODEL_LOCK:
            # dùng yolo_model(...) hoặc yolo_model.predict(...) tùy bạn đang viết
            # cách an toàn nhất với ultralytics:
            res = yolo_model.predict(img_rgb, conf=self.conf, imgsz=self.imgsz, verbose=False)[0]

        rows = []

        if res.boxes is not None and len(res.boxes) > 0:
            for box in res.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                det_conf = float(box.conf[0])
                cls_id = int(box.cls[0])

                yolo_label = res.names[cls_id] if hasattr(res, "names") else str(cls_id)

                # --- classifier top-k (để so YOLO vs CLS + top5 nếu bật) ---
                topk = 5 if self.show_top5 else 1
                preds = classify_crop_bgr(img_bgr, x1, y1, x2, y2, topk=topk)

                cls_top1_name, cls_top1_prob = (None, None)
                top5_str = ""

                if preds is not None and len(preds) > 0:
                    cls_top1_name, cls_top1_prob = preds[0]
                    if self.show_top5:
                        top5_str = ", ".join([f"{n} ({p:.2f})" for n, p in preds])

                # --- map short code trước để dùng cho threshold theo class ---
                yolo_short = self.short_map.get(yolo_label, yolo_label)

                # classifier top1 short (để so sánh / cứu nếu YOLO không chắc)
                cls_short = ""
                if cls_top1_name:
                    cls_short = self.short_map.get(cls_top1_name, cls_top1_name)

                # --- DEMO-SAFE decision (chống spam NO_STRAIGHT_RIGHT + tránh nhãn sai khi conf thấp) ---
                min_conf_this = SPECIAL_MIN_CONF.get(yolo_short, DET_MIN_CONF)

                # bỏ bbox quá yếu (rác)
                too_weak = (det_conf < UNCERTAIN_MIN_CONF)


                allow_refine = any(k in yolo_label.lower() for k in self.refine_keys)
                use_refine = False

                # mặc định: nếu YOLO chưa đủ tin -> hiển thị UNCERTAIN (vàng), tránh nhãn sai
                final_short = "UNCERTAIN"
                final_conf  = det_conf

                if too_weak:
                    final_short = "UNCERTAIN"
                    final_conf  = det_conf
                    use_refine  = False
                    mismatch    = False

                    row = {
                        "YOLO_label": yolo_short,
                        "CLS_top1": cls_short,
                        "Final_label": final_short,
                        "Det_conf": round(det_conf, 3),
                        "Final_conf": round(final_conf, 3),
                        "Refine_used": False,
                        "Mismatch": False,
                        "bbox": [x1, y1, x2, y2],
                    }
                    if self.show_top5:
                        row["Top5"] = top5_str

                    rows.append(row)
                    
                    continue


                # nếu YOLO đủ tin -> dùng YOLO
                if det_conf >= min_conf_this:
                    final_short = yolo_short
                    final_conf  = det_conf

                # nếu YOLO chưa đủ tin nhưng classifier rất chắc + cho phép refine -> CLS cứu
                if (final_short == "UNCERTAIN") and self.enable_refine and allow_refine and (cls_top1_name is not None):
                    if (cls_top1_prob is not None) and (float(cls_top1_prob) >= CLS_MIN_PROB):
                        final_short = cls_short
                        final_conf  = float(cls_top1_prob)
                        use_refine  = True

                # mismatch: chỉ tính khi có cls_top1
                mismatch = (cls_top1_name is not None) and (yolo_short != cls_short) and (final_short != "UNCERTAIN")


                row = {
                    "YOLO_label": yolo_short,
                    "CLS_top1": cls_short,
                    "Final_label": final_short,
                    "Det_conf": round(det_conf, 3),
                    "Final_conf": round(final_conf, 3),
                    "Refine_used": bool(use_refine),
                    "Mismatch": bool(mismatch),
                    "bbox": [x1, y1, x2, y2],
                }

                if self.show_top5:
                    row["Top5"] = top5_str

                rows.append(row)

                # --- draw (bbox + label) ---
                if final_short == "UNCERTAIN":
                    color = (0, 255, 255)      # vàng
                elif mismatch:
                    color = (0, 0, 255)        # đỏ
                else:
                    color = (0, 255, 0)        # xanh

                cv2.rectangle(draw, (x1, y1), (x2, y2), color, 2)

                cv2.putText(
                    draw,
                    f"{final_short} ({final_conf:.2f})",
                    (x1, max(y1 - 6, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    color,
                    2,
                    cv2.LINE_AA
                )

        # --- filter mismatch only (UI option) ---
        if self.mismatch_only:
            rows_view = [r for r in rows if r.get("Mismatch")]
        else:
            rows_view = rows

        self.latest_rows = rows_view

        # --- FPS ---
        dt = max(time.time() - t0, 1e-6)
        fps = 1.0 / dt
        self.fps_hist.append(fps)
        fps_avg = sum(self.fps_hist) / len(self.fps_hist)
        self.fps_avg = fps_avg

        cv2.putText(
            draw,
            f"FPS: {fps_avg:.1f} | conf={self.conf:.2f} imgsz={self.imgsz}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )

        return av.VideoFrame.from_ndarray(draw, format="bgr24")

# ===========================================================
# ================ MODE 1: DETECTION (YOLO) =================
# ===========================================================
if mode.startswith("1️⃣"):
    st.subheader("🇻🇳 Vietnam Traffic Sign Detection (YOLOv8)")
    st.markdown(
        "Chế độ này dùng mô hình **YOLOv8** đã huấn luyện trên bộ dữ liệu biển báo Việt Nam "
        "để **phát hiện vị trí** và **gán nhãn** các biển báo trên ảnh hoặc video."
    )

    st.markdown("#### 1. Chọn loại dữ liệu đầu vào")
    io_choice = st.radio(
        "Kiểu input:",
        ["Image", "Video"],
        horizontal=True
    )

    # Dùng 2 cột: trái = upload, phải = kết quả
    col_left, col_right = st.columns([1.1, 1.3])

    # ---------- IMAGE INPUT ----------
    if io_choice == "Image":
        with col_left:
            st.markdown("##### 📷 Upload ảnh")
            file = st.file_uploader(
                "Chọn một ảnh giao thông (JPG/PNG/JPEG):",
                type=["jpg", "png", "jpeg"],
                label_visibility="collapsed"
            )

        with col_right:
            st.markdown("##### 📊 Kết quả nhận diện")

            if file is not None:
                temp_path = tempfile.NamedTemporaryFile(delete=False).name
                with open(temp_path, "wb") as f:
                    f.write(file.read())

                img_bgr = cv2.imread(temp_path)
                img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

                # --- ĐO THỜI GIAN TOÀN PIPELINE MODE 1 (ẢNH) ---
                start_time = time.time()

                # chạy detect YOLO
                results = yolo_model(img_rgb)[0]

                detections = []
                if len(results.boxes) == 0:
                    st.warning("⚠️ Không phát hiện được biển báo nào trong ảnh.")
                else:
                    for box in results.boxes:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        conf = float(box.conf[0])
                        cls_id = int(box.cls[0])

                        label = results.names[cls_id] if hasattr(results, "names") else str(cls_id)

                        detections.append({
                            "Label": label,
                            "Confidence": f"{conf:.2f}",
                            "Box [x1,y1,x2,y2]": f"[{x1}, {y1}, {x2}, {y2}]"
                        })

                        cv2.rectangle(img_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(
                            img_bgr,
                            f"{label} ({conf:.2f})",
                            (x1, max(y1 - 5, 10)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            (0, 255, 0),
                            1,
                            cv2.LINE_AA
                        )

                    st.success(f"✅ Phát hiện {len(detections)} biển báo:")
                    st.table(detections)

                # --- KẾT THÚC ĐO THỜI GIAN ---
                end_time = time.time()
                elapsed_ms = (end_time - start_time) * 1000.0
                st.caption(f"[Mode 1] Processing time (YOLO only): {elapsed_ms:.1f} ms")

                st.image(
                    cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB),
                    caption="Ảnh sau khi nhận diện biển báo (YOLOv8)",
                    use_container_width=True
                )
            else:
                st.info("⬅️ Hãy upload một ảnh ở cột bên trái để bắt đầu.")


    # ---------- VIDEO INPUT ----------
    else:
        with col_left:
            st.markdown("##### 🎞️ Upload video")
            file = st.file_uploader(
                "Chọn video (MP4/AVI/MOV), nên < 200MB:",
                type=["mp4", "avi", "mov"],
                label_visibility="collapsed"
            )
            st.caption("💡 Sau khi upload, video sẽ chạy một lần với khung live preview và bảng tổng hợp kết quả ở cuối.")

        with col_right:
            st.markdown("##### 📊 Live preview & tổng hợp biển báo")

            if file is not None:
                st.info("⏳ Đang xử lý video, vui lòng chờ trong khi các khung hình được xử lý...")

                tfile = tempfile.NamedTemporaryFile(delete=False)
                tfile.write(file.read())

                vidcap = cv2.VideoCapture(tfile.name)

                stframe = st.empty()
                status_box = st.empty()

                all_detections = {}
                frame_idx = 0

                # 🔹 DANH SÁCH LƯU THỜI GIAN/FPS MỖI FRAME
                frame_times = []

                while vidcap.isOpened():
                    ret, frame_bgr = vidcap.read()
                    if not ret:
                        break
                    frame_idx += 1

                    # --- BẮT ĐẦU ĐO THỜI GIAN CHO FRAME NÀY ---
                    t0 = time.time()

                    results = yolo_model(frame_bgr)[0]

                    draw_frame = frame_bgr.copy()
                    detections_this_frame = []

                    for box in results.boxes:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        conf = float(box.conf[0])
                        cls_id = int(box.cls[0])
                        label_name = results.names[cls_id]

                        cv2.rectangle(draw_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(
                            draw_frame,
                            f"{label_name} ({conf:.2f})",
                            (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            (0, 255, 0),
                            1,
                            cv2.LINE_AA
                        )

                        if label_name not in all_detections:
                            all_detections[label_name] = []
                        all_detections[label_name].append(conf)

                        detections_this_frame.append({
                            "class_name": label_name,
                            "confidence": round(conf, 4)
                        })

                    # --- KẾT THÚC ĐO THỜI GIAN ---
                    t1 = time.time()
                    dt = t1 - t0           # giây / frame
                    if dt > 0:
                        fps = 1.0 / dt
                        frame_times.append(fps)

                    frame_rgb = cv2.cvtColor(draw_frame, cv2.COLOR_BGR2RGB)
                    stframe.image(
                        frame_rgb,
                        channels="RGB",
                        caption=f"Frame {frame_idx}",
                        use_container_width=True
                    )

                    if frame_idx % 5 == 0:
                        status_box.markdown(
                            f"🔄 Đang xử lý frame **{frame_idx}** "
                            f"(phát hiện {len(detections_this_frame)} biển báo trong khung hình gần nhất)"
                        )

                vidcap.release()

                # 🔹 SAU KHI XỬ LÝ XONG VIDEO → TÍNH FPS TRUNG BÌNH
                avg_fps = None
                if len(frame_times) > 0:
                    avg_fps = sum(frame_times) / len(frame_times)
                    st.caption(f"[Mode 1] Average FPS (YOLO only, video): {avg_fps:.2f}")

                st.success("✅ Video đã được xử lý xong. Tổng hợp các biển báo phát hiện được:")

                if len(all_detections) == 0:
                    st.write("_Không có biển báo nào được phát hiện trong toàn bộ video._")
                else:
                    summary_rows = []
                    for label_name, conf_list in all_detections.items():
                        max_conf = max(conf_list)
                        count = len(conf_list)
                        summary_rows.append((label_name, max_conf, count))

                    summary_rows.sort(key=lambda x: x[1], reverse=True)

                    table_lines = [
                        "| Biển báo | Độ tin cậy cao nhất | Số lần xuất hiện |",
                        "|----------|---------------------|------------------|",
                    ]
                    for (label_name, max_conf, count) in summary_rows:
                        table_lines.append(
                            f"| {label_name} | {max_conf:.2f} | {count} |"
                        )

                    st.markdown("\n".join(table_lines))
            else:
                st.info("⬅️ Hãy upload một video ở cột bên trái để bắt đầu.")


# ===========================================================
# =========== MODE 2: CLASSIFICATION (GTSRB) ================
# ===========================================================
elif mode.startswith("2️⃣"):
    st.subheader("🚦 GTSRB Traffic Sign Classification (EfficientNet-B0)")
    st.markdown(
        "Chế độ này dùng mô hình **EfficientNet-B0 fine-tune trên GTSRB** để phân loại "
        "ảnh **biển báo đã được crop sẵn** vào 1 trong 43 lớp."
    )

    col_left, col_right = st.columns([1.1, 1.3])

    with col_left:
        st.markdown("##### 📷 Upload ảnh biển báo (crop)")
        img_file = st.file_uploader(
            "Chọn ảnh (JPG/PNG/JPEG):",
            type=["jpg", "png", "jpeg"],
            label_visibility="collapsed"
        )
        st.caption("💡 Nên dùng ảnh chỉ chứa riêng biển báo, không chứa nhiều background.")

    with col_right:
        st.markdown("##### 📊 Kết quả phân loại")

        if img_file is not None:
            pil_img = Image.open(img_file).convert("RGB")

            st.image(
                pil_img,
                caption="Ảnh input (biển báo đã crop)",
                use_container_width=True
            )

            # --- BẮT ĐẦU ĐO THỜI GIAN MODE 2 (CLASSIFIER ONLY) ---
            start_time = time.time()

            x = clf_transform(pil_img).unsqueeze(0).to(DEVICE)

            with torch.no_grad():
                logits = clf_model(x)
                probs = F.softmax(logits, dim=1)
                top_prob, top_idx = torch.max(probs, dim=1)
                top_prob = float(top_prob.item())
                top_idx  = int(top_idx.item())

            # --- KẾT THÚC ĐO THỜI GIAN ---
            end_time = time.time()
            elapsed_ms = (end_time - start_time) * 1000.0

            class_name = CLASS_NAMES[top_idx] if top_idx < len(CLASS_NAMES) else f"class_{top_idx}"

            st.success("✅ Classification Result")
            st.markdown(f"- **Predicted class ID**: `{top_idx}`")
            st.markdown(f"- **Class name**: **{class_name}**")
            st.markdown(f"- **Confidence**: `{top_prob:.4f}`")

            # 👉 Thêm dòng này để hiển thị thời gian thực thi Mode 2
            st.caption(f"[Mode 2] Processing time (Classifier only): {elapsed_ms:.1f} ms")

            topk = 5
            top_probs, top_indices = torch.topk(probs[0], k=min(topk, probs.shape[1]))
            top_view = []
            for rank in range(len(top_probs)):
                idx_i = int(top_indices[rank].item())
                prob_i = float(top_probs[rank].item())
                name_i = CLASS_NAMES[idx_i] if idx_i < len(CLASS_NAMES) else f"class_{idx_i}"
                top_view.append({
                    "Rank": rank + 1,
                    "ClassID": idx_i,
                    "Name": name_i,
                    "Confidence": f"{prob_i:.4f}"
                })

            st.markdown("##### 🔝 Top-k dự đoán")
            st.table(top_view)
        else:
            st.info("⬅️ Hãy upload một ảnh biển báo ở cột bên trái để xem kết quả phân loại.")

# ===========================================================
# ============ MODE 3: UNIFIED PIPELINE (YOLO + CLS) ==========
else:
    st.header("🔗 Unified Pipeline: YOLOv8 + EfficientNet-B0")
    
    st.subheader("⚙️ Options")
    colA, colB, colC = st.columns([1,1,1])
    with colA:
        ENABLE_REFINE = st.checkbox("Enable refine (YOLO → CLS)", value=True)
    with colB:
        SHOW_TOP5 = st.checkbox("Show Top-5 in table", value=True)
    with colC:
        SHOW_MISMATCH_ONLY = st.checkbox("Show mismatch only", value=False)

    st.caption("Ghi chú: Refine = dùng Classifier Top-1 để thay nhãn YOLO (theo rule REFINE_WITH_CLS).")

    io_choice = st.radio("Select input type", ["Image", "Video", "Webcam"])

    # ========================= IMAGE MODE =========================
    if io_choice == "Image":
        uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

        if uploaded_file is not None:
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

            st.image(img_bgr, channels="BGR", caption="Input Image")

            # --- ĐO THỜI GIAN TOÀN PIPELINE MODE 3 (ẢNH) ---
            start_time = time.time()

            # --- YOLO DETECT ---
            with st.spinner("Running YOLOv8 detection..."):
                results = yolo_model(img_rgb)[0]

            unified_detections = []
            out_img = img_bgr.copy()

            if len(results.boxes) == 0:
                st.warning("No traffic sign detected.")
            else:
                for box in results.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    det_conf = float(box.conf[0])
                    cls_id   = int(box.cls[0])

                    yolo_label = results.names[cls_id] if hasattr(results, "names") else str(cls_id)

                    # luôn chạy classifier để lấy Top-5 (phục vụ hiển thị)
                    preds = classify_crop_bgr(out_img, x1, y1, x2, y2, topk=5)

                    # build chuỗi Top-5
                    top5_str = ""
                    cls_top1_name, cls_top1_prob = None, None
                    if preds is not None and len(preds) > 0:
                        cls_top1_name, cls_top1_prob = preds[0]
                        if SHOW_TOP5:
                            top5_str = ", ".join([f"{name} ({prob:.2f})" for name, prob in preds])

                    # quyết định refine hay không
                    allow_refine_by_rule = any(key in yolo_label.lower() for key in REFINE_WITH_CLS)
                    use_refine = ENABLE_REFINE and (cls_top1_name is not None) and allow_refine_by_rule

                    final_label = cls_top1_name if use_refine else yolo_label
                    final_conf  = float(cls_top1_prob) if use_refine else det_conf

                    # mapping sang short label (classes_short.json)
                    yolo_short  = CLASS_NAMES_SHORT.get(yolo_label, yolo_label)
                    cls_short   = CLASS_NAMES_SHORT.get(cls_top1_name, cls_top1_name) if cls_top1_name else ""
                    final_short = CLASS_NAMES_SHORT.get(final_label, final_label)

                    mismatch = (cls_top1_name is not None) and (CLASS_NAMES_SHORT.get(yolo_label, yolo_label) != CLASS_NAMES_SHORT.get(cls_top1_name, cls_top1_name))

                    row = {
                        "YOLO_label": yolo_short,
                        "CLS_top1": cls_short,
                        "Final_label": final_short,
                        "Det_conf": round(det_conf, 3),
                        "Final_conf": round(final_conf, 3),
                        "Refine_used": bool(use_refine),
                        "Mismatch": bool(mismatch),
                        "bbox": [x1, y1, x2, y2],
                    }
                    if SHOW_TOP5:
                        row["Top5"] = top5_str

                    unified_detections.append(row)

                    # DRAW
                    cv2.rectangle(out_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(
                        out_img,
                        f"{final_short} ({final_conf:.2f})",
                        (x1, max(y1 - 5, 10)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 255, 0),
                        1,
                        cv2.LINE_AA
                    )


                st.subheader("Unified Output")
                st.image(out_img, channels="BGR")

                st.subheader("YOLO vs Classifier Results")
                df = pd.DataFrame(unified_detections)

                if SHOW_MISMATCH_ONLY and "Mismatch" in df.columns:
                    df_view = df[df["Mismatch"] == True].copy()
                else:
                    df_view = df

                st.dataframe(df_view, use_container_width=True)

                # Export CSV
                csv_bytes = df_view.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "⬇️ Download results CSV",
                    data=csv_bytes,
                    file_name="mode3_unified_results.csv",
                    mime="text/csv"
                )

                # --- KẾT THÚC ĐO THỜI GIAN ---
                end_time = time.time()
                elapsed_ms = (end_time - start_time) * 1000.0
                st.caption(f"[Mode 3] Processing time (YOLO + CLS + rule): {elapsed_ms:.1f} ms")

    # ========================= VIDEO MODE =========================
    elif io_choice == "Video":
        uploaded_video = st.file_uploader("Upload a video", type=["mp4","avi","mov"])

        if uploaded_video is not None:
            tfile = tempfile.NamedTemporaryFile(delete=False)
            tfile.write(uploaded_video.read())

            cap = cv2.VideoCapture(tfile.name)
            stframe = st.empty()

            all_stats = {}            # summary theo Final_label
            fps_list = []             # FPS từng frame
            frame_time_list = []      # thời gian xử lý từng frame (seconds)
            video_rows = []           # log csv cho từng detection
            frame_idx = 0

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                frame_idx += 1

                t0 = time.time()

                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = yolo_model(frame_rgb)[0]
                draw = frame.copy()

                # Nếu không có box → vẫn tính time + fps + show frame
                if len(results.boxes) == 0:
                    dt = time.time() - t0
                    frame_time_list.append(dt)
                    fps = (1.0 / dt) if dt > 0 else 0.0
                    fps_list.append(fps)

                    cv2.putText(draw, f"FPS: {fps:.1f}", (10, 25),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
                    stframe.image(draw, channels="BGR")
                    continue

                for box in results.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    det_conf = float(box.conf[0])
                    cls_id   = int(box.cls[0])

                    yolo_label = results.names[cls_id] if hasattr(results, "names") else str(cls_id)

                    # 1) luôn chạy classifier để lấy Top-5 (để so sánh YOLO vs CLS + mismatch)
                    preds = classify_crop_bgr(draw, x1, y1, x2, y2, topk=5)

                    top5_str = ""
                    cls_top1_name, cls_top1_prob = None, None
                    if preds is not None and len(preds) > 0:
                        cls_top1_name, cls_top1_prob = preds[0]
                        if SHOW_TOP5:
                            top5_str = ", ".join([f"{name} ({prob:.2f})" for name, prob in preds])

                    # 2) bật/tắt refine + rule refine
                    allow_refine_by_rule = any(key in yolo_label.lower() for key in REFINE_WITH_CLS)
                    use_refine = ENABLE_REFINE and (cls_top1_name is not None) and allow_refine_by_rule

                    final_label = cls_top1_name if use_refine else yolo_label
                    final_conf  = float(cls_top1_prob) if use_refine else det_conf

                    # 3) map sang short label
                    yolo_short  = CLASS_NAMES_SHORT.get(yolo_label, yolo_label)
                    cls_short   = CLASS_NAMES_SHORT.get(cls_top1_name, cls_top1_name) if cls_top1_name else ""
                    final_short = CLASS_NAMES_SHORT.get(final_label, final_label)

                    mismatch = (cls_top1_name is not None) and (yolo_short != cls_short)

                    # 4) highlight mismatch (bbox đỏ nếu mismatch, xanh nếu khớp)
                    color = (0, 0, 255) if mismatch else (0, 255, 0)

                    # DRAW
                    cv2.rectangle(draw, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(
                        draw,
                        f"{final_short} ({final_conf:.2f})",
                        (x1, max(y1 - 5, 10)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        color,
                        1,
                        cv2.LINE_AA
                    )

                    # summary stats
                    if final_short not in all_stats:
                        all_stats[final_short] = []
                    all_stats[final_short].append(float(final_conf))

                    # log row (CSV)
                    row = {
                        "frame": frame_idx,
                        "YOLO_label": yolo_short,
                        "CLS_top1": cls_short,
                        "Final_label": final_short,
                        "Det_conf": round(det_conf, 3),
                        "Final_conf": round(float(final_conf), 3),
                        "Refine_used": bool(use_refine),
                        "Mismatch": bool(mismatch),
                        "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                    }
                    if SHOW_TOP5:
                        row["Top5"] = top5_str
                    video_rows.append(row)

                # FPS + frame time (tính 1 lần sau khi xử lý xong frame)
                dt = time.time() - t0
                frame_time_list.append(dt)

                fps = (1.0 / dt) if dt > 0 else 0.0
                fps_list.append(fps)

                # Vẽ FPS lên khung hình
                cv2.putText(draw, f"FPS: {fps:.1f}", (10, 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

                stframe.image(draw, channels="BGR")

            cap.release()

            # ====== AVG FPS + AVG FRAME TIME ======
            if len(fps_list) > 0:
                avg_fps = sum(fps_list) / len(fps_list)
                st.caption(f"[Mode 3] Average FPS (YOLO + CLS + rule, video): {avg_fps:.2f}")

            if len(frame_time_list) > 0:
                avg_frame_time_ms = (sum(frame_time_list) / len(frame_time_list)) * 1000.0
                st.caption(f"[Mode 3] Average frame time: {avg_frame_time_ms:.1f} ms/frame")

            # ====== SUMMARY TABLE ======
            if len(all_stats) > 0:
                st.subheader("Summary (Unified Pipeline)")
                table = [{"Label": k, "Avg confidence": sum(v)/len(v), "Count": len(v)} for k, v in all_stats.items()]
                st.table(table)

            # ====== CSV DOWNLOAD (lọc mismatch nếu chọn) ======
            if len(video_rows) > 0:
                dfv = pd.DataFrame(video_rows)
                if SHOW_MISMATCH_ONLY and "Mismatch" in dfv.columns:
                    dfv = dfv[dfv["Mismatch"] == True].copy()

                st.download_button(
                    "⬇️ Download video log CSV",
                    data=dfv.to_csv(index=False).encode("utf-8"),
                    file_name="mode3_video_log.csv",
                    mime="text/csv"
                )

    else:
        st.subheader("📷 Webcam realtime (Mode 3)")

        # --- UI options giống Mode 3 --- (NHỚ key riêng)
        col1, col2, col3 = st.columns(3)
        with col1:
            enable_refine_rt = st.checkbox(
                "Enable refine (YOLO → CLS)",
                value=True,
                key="m3_webcam_enable_refine",
            )
        with col2:
            show_top5_rt = st.checkbox(
                "Show Top-5 (slower)",
                value=False,
                key="m3_webcam_show_top5",
            )
        with col3:
            mismatch_only_rt = st.checkbox(
                "Show mismatch only",
                value=False,
                key="m3_webcam_mismatch_only",
            )

        st.caption("Gợi ý: Webcam thường blur/nhỏ → thử conf=0.10–0.20 và imgsz=640–960.")

        conf_rt = st.slider(
            "YOLO conf",
            0.05, 0.80, 0.15, 0.05,
            key="m3_webcam_conf",
        )
        imgsz_rt = st.selectbox(
            "YOLO imgsz",
            [320, 480, 640, 960, 1280],
            index=2,
            key="m3_webcam_imgsz",
        )

        webrtc_ctx = webrtc_streamer(
            key="m3_webcam_stream",  # key riêng luôn
            video_processor_factory=Mode3WebcamProcessor,
            media_stream_constraints={"video": True, "audio": False},
            async_processing=True,
        )

        # placeholders để update UI
        ph_fps = st.empty()
        ph_tbl = st.empty()

        AUTO_UPDATE = st.checkbox("Auto update results (webcam)", value=True, key="m3_webcam_autoupdate")

        if webrtc_ctx.state.playing and webrtc_ctx.video_processor:
            # (tuỳ bạn) hiện FPS/Result realtime khi webcam đang chạy
            while AUTO_UPDATE and webrtc_ctx.state.playing:
                vp = webrtc_ctx.video_processor

                rows = getattr(vp, "latest_rows", []) or []
                df = pd.DataFrame(rows)

                # nếu bạn có lưu fps_avg trong processor thì show, còn không thì bỏ
                fps_avg = getattr(vp, "fps_avg", None)
                if fps_avg is not None:
                    ph_fps.caption(f"[Mode 3 - Webcam] FPS(avg): {fps_avg:.1f}")
                else:
                    ph_fps.caption("[Mode 3 - Webcam] Running...")

                if df.empty:
                    ph_tbl.info("Chưa có detection nào (hoặc đang bị filter).")
                else:
                    ph_tbl.dataframe(df, use_container_width=True)

                time.sleep(0.2)  # 5 lần/giây là đủ mượt


        # gán option vào processor (sau khi stream khởi tạo)
        if webrtc_ctx.video_processor:
            vp = webrtc_ctx.video_processor
            vp.enable_refine = enable_refine_rt
            vp.show_top5 = show_top5_rt
            vp.mismatch_only = mismatch_only_rt
            vp.conf = conf_rt
            vp.imgsz = imgsz_rt

            # mapping + rule (bắt buộc phải có sẵn ngoài scope)
            vp.short_map = CLASS_NAMES_SHORT
            vp.refine_keys = REFINE_WITH_CLS

            st.subheader("Webcam results (latest frame)")

            if st.button("🔄 Refresh results", key="btn_refresh_webcam"):
                pass  # bấm nút để Streamlit rerun và kéo dữ liệu mới

            rows = []
            if webrtc_ctx.video_processor:
                # copy ra để tránh thread đang update
                rows = list(getattr(webrtc_ctx.video_processor, "latest_rows", []))

            df = pd.DataFrame(rows)
            st.dataframe(df, use_container_width=True)

            if not df.empty:
                st.download_button(
                    "⬇️ Download CSV (latest frame)",
                    data=df.to_csv(index=False).encode("utf-8"),
                    file_name="mode3_webcam_latest.csv",
                    mime="text/csv"
                )




