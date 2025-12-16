import os, sys, json, argparse
import cv2
import torch
import numpy as np
from torchvision import models, transforms
import torch.nn as nn
from PIL import Image

# --- đường dẫn mặc định ---
MODEL_PATH = "models/classification/efficientnet_best.pth"
CLASS_NAMES_FILE = "models/classification/class_names_gtsrb.json"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

tfm = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
])

def load_class_names(path):
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return [f"class_{i}" for i in range(43)]

def build_model(num_classes=43):
    m = models.efficientnet_b0(weights=None)
    in_feats = m.classifier[1].in_features
    m.classifier[1] = nn.Linear(in_feats, num_classes)
    return m

def softmax(x): return torch.nn.functional.softmax(x, dim=1)

def run_infer(img_path, model, class_names, topk=5):
    img_bgr = cv2.imread(img_path)
    if img_bgr is None:
        raise FileNotFoundError(img_path)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(img_rgb)

    x = tfm(pil).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        logits = model(x)
        probs = softmax(logits).cpu().numpy()[0]

    top_idx = probs.argsort()[::-1][:topk]
    result = [{
        "rank": i,
        "class_id": int(idx),
        "name": class_names[idx] if idx < len(class_names) else f"class_{idx}",
        "confidence": float(probs[idx])
    } for i, idx in enumerate(top_idx)]

    # In ra tóm tắt
    best = result[0]
    print("Classification Result")
    print(f"Top-1: {best['name']} (id={best['class_id']}) | conf={best['confidence']:.4f}")
    print("Top-5:")
    for r in result:
        print(f"  - {r['name']} (id={r['class_id']}) : {r['confidence']:.4f}")

    return {"topk": result}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True, help="đường dẫn ảnh (đã crop biển báo)")
    ap.add_argument("--model", default=MODEL_PATH)
    ap.add_argument("--classes", default=CLASS_NAMES_FILE)
    ap.add_argument("--topk", type=int, default=5)
    args = ap.parse_args()

    class_names = load_class_names(args.classes)
    model = build_model(num_classes=len(class_names)).to(DEVICE)
    model.load_state_dict(torch.load(args.model, map_location=DEVICE))
    model.eval()

    _ = run_infer(args.image, model, class_names, topk=args.topk)

if __name__ == "__main__":
    main()
