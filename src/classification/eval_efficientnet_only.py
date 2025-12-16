import os, sys, json
import cv2
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix

import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms

# ====== import utils_io ======
CURRENT_DIR = os.path.dirname(__file__)
PROJECT_SRC_DIR = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
if PROJECT_SRC_DIR not in sys.path:
    sys.path.append(PROJECT_SRC_DIR)
from utils_io import ensure_dir

# ====== paths ======
DATA_DIR    = "data/processed/classification_gtsrb"
OUT_DIR     = "results/classification"
MODEL_PATH  = "models/classification/efficientnet_best.pth"
METRICS_OUT = os.path.join(OUT_DIR, "metrics_efficientnet.json")

IMG_SIZE    = 224
NUM_CLASSES = 43
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"

class GTSRBDataset(Dataset):
    def __init__(self, csv_file, img_folder, img_size=224):
        self.df = pd.read_csv(csv_file)
        self.img_folder = img_folder
        self.tf = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485,0.456,0.406],
                                 std=[0.229,0.224,0.225])
        ])
    def __len__(self):
        return len(self.df)
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        p = os.path.join(self.img_folder, row["Filename"])
        img_bgr = cv2.imread(p)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        x = self.tf(img_rgb)
        y = int(row["ClassId"])
        return x, y

def build_efficientnet(num_classes=NUM_CLASSES):
    m = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
    in_feats = m.classifier[1].in_features
    m.classifier[1] = nn.Linear(in_feats, num_classes)
    return m

@torch.no_grad()
def eval_model(model, loader, device):
    model.eval()
    total, correct = 0, 0
    y_true, y_pred = [], []
    for xb, yb in tqdm(loader, desc="eval"):
        xb, yb = xb.to(device), yb.to(device)
        out = model(xb)
        preds = out.argmax(dim=1)
        correct += (preds == yb).sum().item()
        total   += yb.size(0)
        y_true.extend(yb.cpu().tolist())
        y_pred.extend(preds.cpu().tolist())
    acc = correct / total if total else 0.0
    return acc, y_true, y_pred

def main():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Không thấy checkpoint: {MODEL_PATH}")
    ensure_dir(OUT_DIR)

    # dataset test
    test_set = GTSRBDataset(
        csv_file=os.path.join(DATA_DIR, "test_labels.csv"),
        img_folder=os.path.join(DATA_DIR, "images_test"),
        img_size=IMG_SIZE
    )
    test_loader = DataLoader(test_set, batch_size=64, shuffle=False, num_workers=2, pin_memory=True)

    # model
    model = build_efficientnet(NUM_CLASSES).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))

    # evaluate
    test_acc, y_true, y_pred = eval_model(model, test_loader, DEVICE)
    report = classification_report(y_true, y_pred, output_dict=True)
    cm = confusion_matrix(y_true, y_pred).tolist()

    # lịch sử training không có (eval-only), để rỗng
    metrics = {
        "model": "EfficientNet-B0",
        "test_accuracy": float(test_acc),
        "classification_report": report,
        "confusion_matrix": cm,
        "history": {
            "epoch": [],
            "train_loss": [], "train_acc": [],
            "val_loss":   [], "val_acc":   []
        }
    }
    with open(METRICS_OUT, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    print("Wrote:", METRICS_OUT)
    print(f"Final test accuracy: {test_acc:.4f}")

if __name__ == "__main__":
    main()
