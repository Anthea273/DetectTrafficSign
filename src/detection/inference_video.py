import cv2
import os
import csv
import json
from ultralytics import YOLO

MODEL_PATH   = "models/detection/yolov8n_vn_best.pt"
INPUT_VIDEO  = "video1.mp4"
OUT_VIDEO    = "results/detection/sample_detections/demo_output.mp4"

# NEW: optional — tên lớp tiếng Việt nếu có
CLASSES_JSON = "models/detection/classes_vn.json"
# NEW: lưu bảng tổng hợp
OUT_CSV      = "results/detection/sample_detections/demo_summary.csv"

def load_class_names(yolo_model, json_path):
    if os.path.exists(json_path):
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                names = json.load(f)
            # names: list[str] hoặc dict[int->str] đều chấp nhận
            if isinstance(names, dict):
                # chuyển về list theo thứ tự chỉ số nếu cần
                max_id = max(int(k) for k in names.keys())
                lst = [None]*(max_id+1)
                for k, v in names.items():
                    lst[int(k)] = v
                return lst
            return names
        except Exception:
            pass
    # fallback
    return yolo_model.names

def main():
    model = YOLO(MODEL_PATH)

    cap = cv2.VideoCapture(INPUT_VIDEO)
    if not cap.isOpened():
        print("Không mở được video:", INPUT_VIDEO)
        return

    # Lấy thông số video gốc để ghi lại (fallback nếu thiếu)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps != fps or fps <= 0:  # NaN/0/None
        fps = 25
    w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)  or 1280)
    h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 720)

    os.makedirs(os.path.dirname(OUT_VIDEO), exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(OUT_VIDEO, fourcc, fps, (w, h))

    # NEW: thống kê toàn video
    # stats = {class_name: {"count": int, "best_conf": float}}
    class_names = load_class_names(model, CLASSES_JSON)
    stats = {}

    frame_idx = 0
    while True:
        ret, frame_bgr = cap.read()
        if not ret:
            break

        # YOLO inference
        results = model.predict(frame_bgr, verbose=False)
        r0 = results[0]

        # Annotated frame (ultralytics đã vẽ sẵn bbox + label)
        annotated = r0.plot()  # BGR
        writer.write(annotated)

        # NEW: gom thống kê từ boxes
        # Nếu bạn muốn lọc conf thấp, thêm điều kiện: if conf >= 0.25:
        for box in r0.boxes:
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])
            # tên lớp: ưu tiên classes_json; fallback names trong model
            if isinstance(class_names, (list, tuple)) and 0 <= cls_id < len(class_names):
                name = class_names[cls_id]
            else:
                # model.names có thể là dict
                try:
                    name = r0.names.get(cls_id, f"class_{cls_id}")
                except Exception:
                    name = f"class_{cls_id}"

            # cập nhật stats
            if name not in stats:
                stats[name] = {"count": 0, "best_conf": 0.0}
            stats[name]["count"] += 1
            if conf > stats[name]["best_conf"]:
                stats[name]["best_conf"] = conf

        frame_idx += 1
        if frame_idx % 30 == 0:
            print(f"Processed {frame_idx} frames...")

    cap.release()
    writer.release()
    print("Video kết quả đã lưu:", OUT_VIDEO)

    # NEW: ghi CSV tổng hợp + in tóm tắt
    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        wr = csv.writer(f)
        wr.writerow(["class_name", "best_confidence", "count"])
        for name, obj in sorted(stats.items(), key=lambda kv: kv[1]["best_conf"], reverse=True):
            wr.writerow([name, f"{obj['best_conf']:.4f}", obj["count"]])

    print("Bảng tổng hợp đã lưu:", OUT_CSV)
    if not stats:
        print("Không phát hiện biển báo nào.")
    else:
        print("Top lớp (theo best_conf):")
        for name, obj in sorted(stats.items(), key=lambda kv: kv[1]["best_conf"], reverse=True)[:10]:
            print(f" - {name}: best_conf={obj['best_conf']:.2f}, count={obj['count']}")

if __name__ == "__main__":
    main()
