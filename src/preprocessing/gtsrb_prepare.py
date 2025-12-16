import os
import cv2
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from utils_io import ensure_dir

# đường dẫn trong repo của bạn
RAW_DIR = "data/raw/gtsrb"
OUT_DIR = "data/processed/classification_gtsrb"
IMG_SIZE = 48


def crop_and_resize_row(row):
    """
    Đọc 1 dòng từ Train.csv:
    - row["Path"] = 'Train/20/00020_00000_00000.png'
    - row["Roi.X1"]..["Roi.Y2"]: toạ độ bounding box của biển báo
    Trả về ảnh RGB đã crop và resize về IMG_SIZE x IMG_SIZE.
    """
    # Tạo đường dẫn tuyệt đối tới file ảnh gốc
    img_path = os.path.join(RAW_DIR, os.path.normpath(row["Path"]))

    img_bgr = cv2.imread(img_path)
    if img_bgr is None:
        # ảnh bị hỏng / không đọc được
        return None

    x1, y1, x2, y2 = int(row["Roi.X1"]), int(row["Roi.Y1"]), int(row["Roi.X2"]), int(row["Roi.Y2"])

    # Chặn trường hợp ROI ngoài biên (phòng hờ dữ liệu bẩn)
    h, w = img_bgr.shape[:2]
    x1 = max(0, min(x1, w-1))
    x2 = max(0, min(x2, w))
    y1 = max(0, min(y1, h-1))
    y2 = max(0, min(y2, h))

    if x2 <= x1 or y2 <= y1:
        return None  # ROI rỗng, bỏ

    crop = img_bgr[y1:y2, x1:x2]

    # BGR -> RGB
    crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)

    # resize về kích thước cố định
    crop = cv2.resize(crop, (IMG_SIZE, IMG_SIZE))

    return crop


def save_split(images, labels, split_name):
    """
    Lưu ảnh và ghi file CSV nhãn cho từng split (train / val / test)
    """
    img_dir = os.path.join(OUT_DIR, f"images_{split_name}")
    ensure_dir(img_dir)

    rows = []
    for i, (img_rgb, cls_id) in enumerate(zip(images, labels)):
        fname = f"{split_name}_{i:06d}.png"
        out_path = os.path.join(img_dir, fname)

        # Lưu ra BGR để tiện xem lại ngoài code (Windows Photo Viewer, v.v.)
        cv2.imwrite(out_path, cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))

        rows.append({"Filename": fname, "ClassId": int(cls_id)})

    # Ghi file nhãn
    labels_csv_path = os.path.join(OUT_DIR, f"{split_name}_labels.csv")
    pd.DataFrame(rows).to_csv(labels_csv_path, index=False)

    print(f"Saved {split_name} split:")
    print(f"   {len(images)} images -> {img_dir}")
    print(f"   labels -> {labels_csv_path}")


def main():
    ensure_dir(OUT_DIR)

    csv_path = os.path.join(RAW_DIR, "Train.csv")
    print("Reading:", csv_path)
    df = pd.read_csv(csv_path)

    imgs = []
    labs = []

    print("Cropping + resizing...")
    for idx, r in df.iterrows():
        img = crop_and_resize_row(r)
        if img is None:
            continue
        imgs.append(img)
        labs.append(int(r["ClassId"]))

    imgs = np.array(imgs)
    labs = np.array(labs)

    print(f"Total usable samples after crop: {len(imgs)}")

    # Chia tập: 15% test, rồi tiếp tục 15% của phần còn lại làm val
    X_tmp, X_test, y_tmp, y_test = train_test_split(
        imgs, labs, test_size=0.15, stratify=labs, random_state=42
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_tmp, y_tmp, test_size=0.15, stratify=y_tmp, random_state=42
    )

    # Lưu kết quả ra processed/
    save_split(X_train, y_train, "train")
    save_split(X_val,   y_val,   "val")
    save_split(X_test,  y_test,  "test")

    print("Done. GTSRB processed ->", OUT_DIR)


if __name__ == "__main__":
    main()
