import os
import sys
import json
import cv2
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix

import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms

# ====== import utils_io ======
CURRENT_DIR = os.path.dirname(__file__)
PROJECT_SRC_DIR = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
if PROJECT_SRC_DIR not in sys.path:
    sys.path.append(PROJECT_SRC_DIR)

from utils_io import ensure_dir

# ================== MANUAL RESUME CONFIG ==================
# Bạn chỉnh ở đây đúng với tình hình hiện tại:
RESUME_START_EPOCH   = 5        # bạn đã train xong epoch 0-4, tiếp tục từ epoch 5
TARGET_TOTAL_EPOCH   = 10       # tổng cộng muốn đủ 10 epoch
ASSUME_BEST_VAL_ACC  = 0.99     # val_acc tốt nhất đã đạt trước đó (dùng để bảo vệ checkpoint hiện tại)

# ================== FIXED CONFIG ==================
DATA_DIR          = "data/processed/classification_gtsrb"

OUT_DIR           = "results/classification"
MODEL_DIR         = "models/classification"

MODEL_OUT         = os.path.join(MODEL_DIR, "efficientnet_best.pth")
STATE_OUT         = os.path.join(MODEL_DIR, "efficientnet_train_state.json")
METRICS_OUT       = os.path.join(OUT_DIR, "metrics_efficientnet.json")

IMG_SIZE          = 224
NUM_CLASSES       = 43
BATCH_SIZE        = 64
LR                = 1e-3
DEVICE            = "cuda" if torch.cuda.is_available() else "cpu"


# ================== DATASET ==================
class GTSRBDataset(Dataset):
    def __init__(self, csv_file, img_folder, img_size=224, train_mode=True):
        self.df = pd.read_csv(csv_file)
        self.img_folder = img_folder

        if train_mode:
            self.tf = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((img_size, img_size)),
                transforms.ColorJitter(
                    brightness=0.2,
                    contrast=0.2,
                    saturation=0.2
                ),
                transforms.RandomRotation(10),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                ),
            ])
        else:
            self.tf = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                ),
            ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.img_folder, row["Filename"])

        img_bgr = cv2.imread(img_path)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        x = self.tf(img_rgb)
        y = int(row["ClassId"])
        return x, y


# ================== MODEL ==================
def build_efficientnet(num_classes=43):
    model = models.efficientnet_b0(
        weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1
    )
    in_feats = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_feats, num_classes)
    return model


# ================== TRAIN / EVAL HELPERS ==================
def train_one_epoch(model, loader, criterion, optimizer):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for xb, yb in tqdm(loader, desc="train", leave=False):
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)

        optimizer.zero_grad()
        out = model(xb)
        loss = criterion(out, yb)
        loss.backward()
        optimizer.step()

        bs_now = xb.size(0)
        total_loss += loss.item() * bs_now
        preds = out.argmax(dim=1)
        correct += (preds == yb).sum().item()
        total += bs_now

    avg_loss = total_loss / total
    acc = correct / total
    return avg_loss, acc


@torch.no_grad()
def eval_model(model, loader, criterion, collect_preds=False):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    all_true = []
    all_pred = []

    for xb, yb in tqdm(loader, desc="eval", leave=False):
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        out = model(xb)
        loss = criterion(out, yb)

        bs_now = xb.size(0)
        total_loss += loss.item() * bs_now

        preds = out.argmax(dim=1)
        correct += (preds == yb).sum().item()
        total += bs_now

        if collect_preds:
            all_true.extend(yb.cpu().tolist())
            all_pred.extend(preds.cpu().tolist())

    avg_loss = total_loss / total
    acc = correct / total

    result = {
        "loss": avg_loss,
        "acc": acc,
    }
    if collect_preds:
        result["y_true"] = all_true
        result["y_pred"] = all_pred
    return result


def main():
    # 1. Chuẩn bị thư mục
    ensure_dir(OUT_DIR)
    ensure_dir(MODEL_DIR)

    # 2. Tạo dataset / dataloader
    train_set = GTSRBDataset(
        csv_file=os.path.join(DATA_DIR, "train_labels.csv"),
        img_folder=os.path.join(DATA_DIR, "images_train"),
        img_size=IMG_SIZE,
        train_mode=True,
    )
    val_set = GTSRBDataset(
        csv_file=os.path.join(DATA_DIR, "val_labels.csv"),
        img_folder=os.path.join(DATA_DIR, "images_val"),
        img_size=IMG_SIZE,
        train_mode=False,
    )
    test_set = GTSRBDataset(
        csv_file=os.path.join(DATA_DIR, "test_labels.csv"),
        img_folder=os.path.join(DATA_DIR, "images_test"),
        img_size=IMG_SIZE,
        train_mode=False,
    )

    train_loader = DataLoader(
        train_set, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=2, pin_memory=True
    )
    val_loader = DataLoader(
        val_set, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=2, pin_memory=True
    )
    test_loader = DataLoader(
        test_set, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=2, pin_memory=True
    )

    # 3. Khởi tạo model + optimizer
    model = build_efficientnet(NUM_CLASSES).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    # 4. Load checkpoint từ 5 epoch đầu
    if not os.path.exists(MODEL_OUT):
        print(" ERROR: Không tìm thấy models/classification/efficientnet_best.pth.")
        return

    print(f" Loading checkpoint from {MODEL_OUT}")
    model.load_state_dict(torch.load(MODEL_OUT, map_location=DEVICE))

    # best_val_acc ban đầu giả định theo kết quả trước đó
    best_val_acc = ASSUME_BEST_VAL_ACC

    # 5. Train phần còn lại: epoch = 5..9
    history = {
        "epoch": [],
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
    }

    for epoch in range(RESUME_START_EPOCH, TARGET_TOTAL_EPOCH):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer)
        val_stats = eval_model(model, val_loader, criterion, collect_preds=False)

        history["epoch"].append(epoch)
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_stats["loss"])
        history["val_acc"].append(val_stats["acc"])

        improved = val_stats["acc"] > best_val_acc
        if improved:
            best_val_acc = val_stats["acc"]
            torch.save(model.state_dict(), MODEL_OUT)
            print(f"✅ Epoch {epoch}: val_acc improved to {best_val_acc:.4f} -> saved {MODEL_OUT}")
        else:
            print(
                f"Epoch {epoch}: val_acc={val_stats['acc']:.4f} "
                f"(best {best_val_acc:.4f})"
            )

        # tạo state.json để lần sau auto-resume không cần chỉnh tay nữa
        state_obj = {
            "last_epoch": epoch,
            "best_val_acc": best_val_acc
        }
        with open(STATE_OUT, "w", encoding="utf-8") as f:
            json.dump(state_obj, f, indent=2, ensure_ascii=False)

    # 6. Đánh giá checkpoint tốt nhất trên test set và tạo metrics_efficientnet.json
    print(" Evaluating best checkpoint on test set ...")
    best_model = build_efficientnet(NUM_CLASSES).to(DEVICE)
    best_model.load_state_dict(torch.load(MODEL_OUT, map_location=DEVICE))
    test_stats = eval_model(best_model, test_loader, criterion, collect_preds=True)

    y_true = test_stats["y_true"]
    y_pred = test_stats["y_pred"]

    cls_report = classification_report(y_true, y_pred, output_dict=True)
    cm = confusion_matrix(y_true, y_pred).tolist()

    results = {
        "model": "EfficientNet-B0",
        "test_accuracy": float(test_stats["acc"]),
        "classification_report": cls_report,
        "confusion_matrix": cm,
        "history": history,
    }

    with open(METRICS_OUT, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(" Done. EfficientNet fully trained + evaluated.")
    print("   Saved best weights ->", MODEL_OUT)
    print("   Saved train state  ->", STATE_OUT)
    print("   Saved metrics      ->", METRICS_OUT)
    print(f"   Final test accuracy: {test_stats['acc']:.4f}")

if __name__ == "__main__":
    main()
