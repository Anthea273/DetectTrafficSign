import os, argparse, json
import cv2
from ultralytics import YOLO

DEFAULT_MODEL = "models/detection/yolov8n_vn_best.pt"
DEFAULT_CLASSES_JSON = "models/detection/classes_vn.json"

def load_class_names(yolo_model, classes_json):
    if os.path.exists(classes_json):
        with open(classes_json, "r", encoding="utf-8") as f:
            return json.load(f)
    # fallback: dùng tên lớp trong model
    return yolo_model.names

def draw_and_collect(img_bgr, results, class_names):
    dets = []
    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = float(box.conf[0])
        cid = int(box.cls[0])
        name = class_names[cid] if cid < len(class_names) else f"class_{cid}"
        dets.append({"class_id": cid, "name": name, "conf": conf, "bbox": [x1,y1,x2,y2]})

        cv2.rectangle(img_bgr, (x1,y1), (x2,y2), (0,255,0), 2)
        cv2.putText(img_bgr, f"{name} ({conf:.2f})", (x1, y1-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2, cv2.LINE_AA)
    return dets, img_bgr

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True, help="đường dẫn ảnh input")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--classes", default=DEFAULT_CLASSES_JSON)
    ap.add_argument("--save", default="results/detection/sample_detections/infer_image.jpg")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.save), exist_ok=True)

    yolo = YOLO(args.model)
    class_names = load_class_names(yolo, args.classes)

    img_bgr = cv2.imread(args.image)
    if img_bgr is None:
        raise FileNotFoundError(args.image)

    res = yolo(img_bgr)[0]  # 1 ảnh
    dets, out_img = draw_and_collect(img_bgr.copy(), res, class_names)
    cv2.imwrite(args.save, out_img)

    print(f"Saved: {args.save}")
    print("Detections:")
    for d in dets:
        print(f" - {d['name']} | conf={d['conf']:.2f} | bbox={d['bbox']}")

if __name__ == "__main__":
    main()
