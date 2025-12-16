# src/visualization/plot_from_metrics.py
import os, json
import numpy as np
import matplotlib.pyplot as plt

# === cấu hình đường dẫn ===
RESULTS_DIR = "results/classification"
OUT_DIR     = "results/classification/charts"
MODELS = [
    ("metrics_cnn.json",     "CNN Baseline"),
    ("metrics_resnet.json",  "ResNet18"),
    ("metrics_efficientnet.json", "EfficientNet-B0"),
]

os.makedirs(OUT_DIR, exist_ok=True)

def load_metrics(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def plot_confusion_matrix(cm, title, out_png):
    cm = np.array(cm)
    plt.figure(figsize=(8,7))
    plt.imshow(cm, interpolation="nearest")
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(out_png, dpi=220)
    plt.close()
    print("Saved:", out_png)

def plot_prf_per_class(report, title, out_png):
    # loại các khóa tổng hợp
    rows = [(int(k), v) for k, v in report.items() if k.isdigit()]
    rows.sort(key=lambda x: x[0])
    labels = [str(k) for k, _ in rows]
    prec = [r.get("precision", 0.0) for _, r in rows]
    rec  = [r.get("recall", 0.0)    for _, r in rows]
    f1   = [r.get("f1-score", 0.0)  for _, r in rows]

    x = np.arange(len(labels))
    w = 0.28
    plt.figure(figsize=(12,5))
    plt.bar(x - w, prec, width=w, label="Precision")
    plt.bar(x     , rec , width=w, label="Recall")
    plt.bar(x + w , f1  , width=w, label="F1")
    plt.xticks(x, labels, rotation=90)
    plt.ylim(0, 1.05)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=220)
    plt.close()
    print("Saved:", out_png)

def plot_learning_curves(history, title, out_png):
    # history: {"epoch":[], "train_loss":[], "train_acc":[], "val_loss":[], "val_acc":[]}
    if not history or not history.get("epoch"):
        print("No history -> skip learning curves for", title)
        return
    ep   = history["epoch"]
    tr_l = history.get("train_loss", [])
    va_l = history.get("val_loss", [])
    tr_a = history.get("train_acc", [])
    va_a = history.get("val_acc", [])

    plt.figure(figsize=(10,6))

    # Loss
    if tr_l and va_l:
        plt.plot(ep, tr_l, label="Train Loss", linewidth=2)
        plt.plot(ep, va_l, label="Val Loss", linewidth=2)

    # Acc
    if tr_a and va_a:
        plt.plot(ep, tr_a, label="Train Acc", linewidth=2)
        plt.plot(ep, va_a, label="Val Acc", linewidth=2)

    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel("Metric")
    plt.legend()
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(out_png, dpi=220)
    plt.close()
    print("Saved:", out_png)

def main():
    for metrics_file, nice_name in MODELS:
        path = os.path.join(RESULTS_DIR, metrics_file)
        if not os.path.exists(path):
            print("Skip (not found):", path)
            continue

        data = load_metrics(path)
        acc  = data.get("test_accuracy", None)

        # Confusion Matrix
        cm_png = os.path.join(
            OUT_DIR, f"confusion_matrix_{os.path.splitext(metrics_file)[0]}.png"
        )
        title_cm = f"{nice_name} – Confusion Matrix" + (f" (test acc={acc:.4f})" if acc is not None else "")
        plot_confusion_matrix(data["confusion_matrix"], title_cm, cm_png)

        # Per-class PRF
        prf_png = os.path.join(
            OUT_DIR, f"prf_per_class_{os.path.splitext(metrics_file)[0]}.png"
        )
        title_prf = f"{nice_name} – Per-class Precision/Recall/F1"
        plot_prf_per_class(data["classification_report"], title_prf, prf_png)

        # Learning curves (nếu có)
        lc_png = os.path.join(
            OUT_DIR, f"learning_curves_{os.path.splitext(metrics_file)[0]}.png"
        )
        plot_learning_curves(data.get("history", {}), f"{nice_name} – Learning Curves", lc_png)

if __name__ == "__main__":
    main()
