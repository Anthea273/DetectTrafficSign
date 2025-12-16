import os
import glob
import cv2
import pandas as pd
import numpy as np
from collections import Counter
from utils_io import ensure_dir

GTSRB_DIR = "data/processed/classification_gtsrb"
VN_DIR    = "data/processed/detection_vn_yolo"
STATS_OUT = "data/processed/stats"

def analyze_gtsrb_distribution():
    df = pd.read_csv(os.path.join(GTSRB_DIR, "train_labels.csv"))
    counts = df["ClassId"].value_counts().sort_index()
    out_path = os.path.join(STATS_OUT, "class_distribution_gtsrb.csv")
    counts.to_csv(out_path, header=["count"])
    print("GTSRB class dist saved:", out_path)

def analyze_vn_distribution_and_bbox():
    class_counter = Counter()
    bbox_rows = []

    train_img_dir = os.path.join(VN_DIR, "images/train")
    train_lbl_dir = os.path.join(VN_DIR, "labels/train")

    img_list = glob.glob(os.path.join(train_img_dir, "*.jpg")) \
             + glob.glob(os.path.join(train_img_dir, "*.png"))

    for img_path in img_list:
        base = os.path.splitext(os.path.basename(img_path))[0]
        lbl_path = os.path.join(train_lbl_dir, base + ".txt")
        if not os.path.exists(lbl_path):
            continue

        img_bgr = cv2.imread(img_path)
        if img_bgr is None:
            continue
        h, w = img_bgr.shape[:2]

        with open(lbl_path, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) != 5:
                    continue
                cls_id, cx, cy, bw, bh = parts
                cls_id = int(cls_id)
                class_counter[cls_id] += 1

                bw_px = float(bw) * w
                bh_px = float(bh) * h
                bbox_rows.append({
                    "class_id": cls_id,
                    "img_w": w,
                    "img_h": h,
                    "bbox_w_px": bw_px,
                    "bbox_h_px": bh_px
                })

    # lưu phân bố class VN
    class_dist_path = os.path.join(STATS_OUT, "class_distribution_vn.csv")
    pd.DataFrame.from_dict(class_counter, orient="index", columns=["count"]) \
        .sort_index() \
        .to_csv(class_dist_path)

    # lưu kích thước bbox
    bbox_df = pd.DataFrame(bbox_rows)
    bbox_stats = bbox_df[["bbox_w_px","bbox_h_px"]].describe()
    bbox_stats_path = os.path.join(STATS_OUT, "bbox_size_distribution_vn.csv")
    bbox_stats.to_csv(bbox_stats_path)

    # lưu thống kê kích thước ảnh
    img_size_rows = bbox_df[["img_w","img_h"]].drop_duplicates()
    img_size_rows.to_csv(
        os.path.join(STATS_OUT, "image_resolution_stats.csv"),
        index=False
    )

    print("VN stats saved to:", STATS_OUT)

def main():
    ensure_dir(STATS_OUT)
    analyze_gtsrb_distribution()
    analyze_vn_distribution_and_bbox()
    print("Dataset stats generated at", STATS_OUT)

if __name__ == "__main__":
    main()
