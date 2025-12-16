import os
import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from utils_io import ensure_dir

OUT_DIR = "results/classification"

def plot_conf_mat(cm, title, out_path):
    cm = np.array(cm)
    plt.figure(figsize=(6,6))
    plt.imshow(cm, interpolation='nearest', cmap='Blues')
    plt.title(title)
    plt.colorbar()
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    print("Saved:", out_path)

def main():
    ensure_dir(OUT_DIR)

    # CNN
    cnn_json = os.path.join(OUT_DIR, "metrics_cnn.json")
    if os.path.exists(cnn_json):
        with open(cnn_json, "r") as f:
            data = json.load(f)
        cm = data["confusion_matrix"]
        plot_conf_mat(cm, "CNN Confusion Matrix", os.path.join(OUT_DIR, "confusion_matrix_cnn.png"))

    # ResNet
    resnet_json = os.path.join(OUT_DIR, "metrics_resnet.json")
    if os.path.exists(resnet_json):
        with open(resnet_json, "r") as f:
            data = json.load(f)
        cm = data["confusion_matrix"]
        plot_conf_mat(cm, "ResNet18 Confusion Matrix", os.path.join(OUT_DIR, "confusion_matrix_resnet.png"))

    # EfficientNet
    eff_json = os.path.join(OUT_DIR, "metrics_efficientnet.json")
    if os.path.exists(eff_json):
        with open(eff_json, "r") as f:
            data = json.load(f)
        cm = data["confusion_matrix"]
        plot_conf_mat(cm, "EfficientNet-B0 Confusion Matrix", os.path.join(OUT_DIR, "confusion_matrix_efficientnet.png"))

if __name__ == "__main__":
    main()
