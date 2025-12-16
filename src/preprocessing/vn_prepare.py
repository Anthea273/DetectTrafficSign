import os
import glob
import random
import shutil
from utils_io import ensure_dir

RAW_IMG_DIR = "data/raw/vn_signs/images"
RAW_LBL_DIR = "data/raw/vn_signs/labels"
RAW_CLASSES = "data/raw/vn_signs/classes.txt"

OUT_DIR = "data/processed/detection_vn_yolo"
TRAIN_IMG_OUT = os.path.join(OUT_DIR, "images/train")
VAL_IMG_OUT   = os.path.join(OUT_DIR, "images/val")
TRAIN_LBL_OUT = os.path.join(OUT_DIR, "labels/train")
VAL_LBL_OUT   = os.path.join(OUT_DIR, "labels/val")
DATA_YAML_PATH = os.path.join(OUT_DIR, "data.yaml")

VAL_RATIO = 0.15  # 15% val


def list_images_recursive(base_dir):
    """
    Lấy toàn bộ ảnh .jpg/.png kể cả trong subfolder
    """
    patterns = ["**/*.jpg", "**/*.jpeg", "**/*.png"]
    out = []
    for p in patterns:
        out.extend(glob.glob(os.path.join(base_dir, p), recursive=True))
    return out


def copy_pairs(pairs, img_out, lbl_out):
    for img_src, lbl_src in pairs:
        img_name = os.path.basename(img_src)
        lbl_name = os.path.basename(lbl_src)
        shutil.copy2(img_src, os.path.join(img_out, img_name))
        shutil.copy2(lbl_src, os.path.join(lbl_out, lbl_name))


def main():
    # tạo thư mục output trước
    ensure_dir(TRAIN_IMG_OUT)
    ensure_dir(VAL_IMG_OUT)
    ensure_dir(TRAIN_LBL_OUT)
    ensure_dir(VAL_LBL_OUT)

    print("Scanning images from:", RAW_IMG_DIR)
    all_imgs = list_images_recursive(RAW_IMG_DIR)
    print(f"   Found {len(all_imgs)} total candidate images")

    valid_pairs = []
    missing_label = 0

    for img_path in all_imgs:
        base = os.path.splitext(os.path.basename(img_path))[0]
        lbl_path = os.path.join(RAW_LBL_DIR, base + ".txt")
        if os.path.exists(lbl_path):
            valid_pairs.append((img_path, lbl_path))
        else:
            missing_label += 1

    print(f"   Matched {len(valid_pairs)} image/label pairs")
    print(f"   Missing labels for {missing_label} images (ignored)")

    if len(valid_pairs) == 0:
        print("Không tìm thấy cặp ảnh-nhãn nào. Kiểm tra lại RAW_IMG_DIR/RAW_LBL_DIR.")
        return

    random.seed(42)
    random.shuffle(valid_pairs)

    val_size = int(len(valid_pairs) * VAL_RATIO)
    val_pairs = valid_pairs[:val_size]
    train_pairs = valid_pairs[val_size:]

    print(f"   -> train pairs: {len(train_pairs)}")
    print(f"   -> val pairs  : {len(val_pairs)}")

    copy_pairs(train_pairs, TRAIN_IMG_OUT, TRAIN_LBL_OUT)
    copy_pairs(val_pairs,   VAL_IMG_OUT,   VAL_LBL_OUT)

    # đọc danh sách class
    if not os.path.exists(RAW_CLASSES):
        print("Không tìm thấy classes.txt, vẫn tiếp tục nhưng data.yaml sẽ lỗi nếu thiếu.")
        class_names = []
    else:
        with open(RAW_CLASSES, "r", encoding="utf-8") as f:
            class_names = [line.strip() for line in f if line.strip()]

    # ghi YAML cho YOLOv8 / YOLOv5
    with open(DATA_YAML_PATH, "w", encoding="utf-8") as f:
        f.write(f"train: {os.path.abspath(TRAIN_IMG_OUT)}\n")
        f.write(f"val: {os.path.abspath(VAL_IMG_OUT)}\n")
        f.write(f"nc: {len(class_names)}\n")
        f.write(f"names: {class_names}\n")

    print("VN dataset prepared ->", OUT_DIR)
    print("   train imgs :", TRAIN_IMG_OUT)
    print("   val imgs   :", VAL_IMG_OUT)
    print("   data.yaml  :", DATA_YAML_PATH)


if __name__ == "__main__":
    main()
