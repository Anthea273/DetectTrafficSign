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

# --- chuẩn hóa import utils_io ---
CURRENT_DIR = os.path.dirname(__file__)               
PROJECT_SRC_DIR = os.path.abspath(os.path.join(CURRENT_DIR, ".."))  
if PROJECT_SRC_DIR not in sys.path:
    sys.path.append(PROJECT_SRC_DIR)

from utils_io import ensure_dir


# ==== cấu hình ====
DATA_DIR    = "data/processed/classification_gtsrb"
OUT_DIR     = "results/classification"
MODEL_OUT   = "models/classification/resnet_best.pth"
METRICS_OUT = os.path.join(OUT_DIR, "metrics_resnet.json")

IMG_SIZE    = 224            # ResNet dùng input 224x224
NUM_CLASSES = 43             # GTSRB có 43 lớp
EPOCHS      = 10
BATCH_SIZE  = 64
LR          = 1e-3
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"


# ==== dataset ====
class GTSRBDataset(Dataset):
    def __init__(self, csv_file, img_folder, img_size=224, train_mode=True):
        self.df = pd.read_csv(csv_file)
        self.img_folder = img_folder

        if train_mode:
            self.tf = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((img_size, img_size)),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
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


# ==== model ====
def build_resnet(num_classes=NUM_CLASSES):
    # load resnet18 pretrained ImageNet
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    in_feats = model.fc.in_features
    model.fc = nn.Linear(in_feats, num_classes)
    return model


# ==== train / eval helpers ====
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

        batch_size_now = xb.size(0)
        total_loss += loss.item() * batch_size_now
        preds = out.argmax(dim=1)
        correct += (preds == yb).sum().item()
        total += batch_size_now

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

        batch_size_now = xb.size(0)
        total_loss += loss.item() * batch_size_now
        preds = out.argmax(dim=1)
        correct += (preds == yb).sum().item()
        total += batch_size_now

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
    # tạo thư mục output
    ensure_dir(OUT_DIR)
    ensure_dir(os.path.dirname(MODEL_OUT))

    # dataset / dataloader
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

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_set,   batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=2, pin_memory=True)
    test_loader  = DataLoader(test_set,  batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=2, pin_memory=True)

    # model / loss / optim
    model = build_resnet(NUM_CLASSES).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    best_val_acc = 0.0
    best_epoch = -1

    # lưu lịch sử để sau này vẽ learning curve
    history = {
        "epoch": [],
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
    }

    print(" Training ResNet18 ...")
    for epoch in range(EPOCHS):
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
            best_epoch = epoch
            torch.save(model.state_dict(), MODEL_OUT)
            print(f"Epoch {epoch+1}/{EPOCHS}: val_acc improved to {best_val_acc:.4f} ✅ saved {MODEL_OUT}")
        else:
            print(f"Epoch {epoch+1}/{EPOCHS}: val_acc={val_stats['acc']:.4f} (best {best_val_acc:.4f} @ epoch {best_epoch+1})")

    # dùng best model để đánh giá test
    print(" Evaluating best model on test set ...")
    best_model = build_resnet(NUM_CLASSES).to(DEVICE)
    best_model.load_state_dict(torch.load(MODEL_OUT, map_location=DEVICE))
    test_stats = eval_model(best_model, test_loader, criterion, collect_preds=True)

    y_true = test_stats["y_true"]
    y_pred = test_stats["y_pred"]

    cls_report = classification_report(y_true, y_pred, output_dict=True)
    cm = confusion_matrix(y_true, y_pred).tolist()

    results = {
        "model": "ResNet18",
        "test_accuracy": float(test_stats["acc"]),
        "classification_report": cls_report,
        "confusion_matrix": cm,
        "history": history,  # gồm train_loss, val_loss, train_acc, val_acc theo epoch
    }

    ensure_dir(OUT_DIR)
    with open(METRICS_OUT, "w") as f:
        json.dump(results, f, indent=2)

    print(" Done training ResNet18.")
    print("   Saved best weights ->", MODEL_OUT)
    print("   Saved metrics      ->", METRICS_OUT)
    print(f"   Final test accuracy: {test_stats['acc']:.4f}")


if __name__ == "__main__":
    main()
