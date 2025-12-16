import os
import sys

# đảm bảo import được utils_io từ src/
CURRENT_DIR = os.path.dirname(__file__)
PROJECT_SRC_DIR = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
if PROJECT_SRC_DIR not in sys.path:
    sys.path.append(PROJECT_SRC_DIR)

from utils_io import ensure_dir

import json
import cv2
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import models, layers
from tensorflow.keras.callbacks import ModelCheckpoint


DATA_DIR = "data/processed/classification_gtsrb"
OUT_DIR  = "results/classification"
MODEL_OUT = "models/classification/cnn_best.keras"

IMG_SIZE = 48          # phải khớp với gtsrb_prepare.py
NUM_CLASSES = 43       # GTSRB có 43 lớp
EPOCHS = 10            
BATCH = 64


def load_split(split_name):
    """
    split_name: "train" / "val" / "test"
    - Đọc file <split_name>_labels.csv
    - Đọc từng ảnh trong images_<split_name>/
    - Trả về (X, y_onehot)
    """
    label_csv = os.path.join(DATA_DIR, f"{split_name}_labels.csv")
    df = pd.read_csv(label_csv)

    X_list, y_list = [], []
    for _, row in df.iterrows():
        img_path = os.path.join(DATA_DIR, f"images_{split_name}", row["Filename"])

        img = cv2.imread(img_path)  # BGR
        if img is None:
            # ảnh lỗi -> bỏ qua
            continue

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        X_list.append(img)
        y_list.append(int(row["ClassId"]))

    X = np.array(X_list, dtype="float32") / 255.0  # normalize [0,1]
    y = to_categorical(np.array(y_list), NUM_CLASSES)
    return X, y


def build_cnn(input_shape=(IMG_SIZE, IMG_SIZE, 3), num_classes=NUM_CLASSES):
    """
    CNN baseline nhẹ nhàng (3 conv blocks + FC)
    """
    model = models.Sequential([
        layers.Conv2D(32, (3,3), activation="relu", padding="same", input_shape=input_shape),
        layers.MaxPooling2D(2,2),

        layers.Conv2D(64, (3,3), activation="relu", padding="same"),
        layers.MaxPooling2D(2,2),

        layers.Conv2D(128, (3,3), activation="relu", padding="same"),
        layers.MaxPooling2D(2,2),

        layers.Flatten(),
        layers.Dense(256, activation="relu"),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation="softmax")
    ])
    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model

def main():
    # chuẩn bị thư mục output
    ensure_dir(OUT_DIR)
    ensure_dir(os.path.dirname(MODEL_OUT))

    print("Loading train/val/test splits...")
    X_train, y_train = load_split("train")
    X_val,   y_val   = load_split("val")
    X_test,  y_test  = load_split("test")

    print(f"   Train set: {X_train.shape}, {y_train.shape}")
    print(f"   Val set  : {X_val.shape}, {y_val.shape}")
    print(f"   Test set : {X_test.shape}, {y_test.shape}")

    # build model
    model = build_cnn()
    print("Model summary:")
    model.summary()

    # callback để lưu best model theo val_accuracy (Keras mới bắt buộc .keras)
    checkpoint_cb = ModelCheckpoint(
        MODEL_OUT,
        monitor="val_accuracy",
        save_best_only=True,
        mode="max",
        verbose=1
    )

    print("Training CNN baseline...")
    hist = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH,
        callbacks=[checkpoint_cb],
        verbose=1
    )

    # Sau train, load lại best model (.keras)
    from tensorflow.keras.models import load_model
    best_model = load_model(MODEL_OUT)

    # evaluate trên test
    print("Evaluating on test set...")
    test_loss, test_acc = best_model.evaluate(X_test, y_test, verbose=0)

    # predict test set
    y_true = np.argmax(y_test, axis=1)
    y_pred = np.argmax(best_model.predict(X_test), axis=1)

    from sklearn.metrics import classification_report, confusion_matrix
    report = classification_report(y_true, y_pred, output_dict=True)
    cm = confusion_matrix(y_true, y_pred).tolist()

    # ghi file kết quả
    results = {
        "model": "CNN_baseline_tf",
        "test_accuracy": float(test_acc),
        "classification_report": report,
        "confusion_matrix": cm,
        "history": {
            "loss":         hist.history.get("loss", []),
            "val_loss":     hist.history.get("val_loss", []),
            "accuracy":     hist.history.get("accuracy", []),
            "val_accuracy": hist.history.get("val_accuracy", []),
        }
    }

    out_metrics_path = os.path.join(OUT_DIR, "metrics_cnn.json")
    with open(out_metrics_path, "w") as f:
        json.dump(results, f, indent=2)

    print("Done training CNN baseline.")
    print("   Saved best model ->", MODEL_OUT)
    print("   Saved metrics    ->", out_metrics_path)
    print(f"   Final test accuracy: {test_acc:.4f}")


if __name__ == "__main__":
    main()
