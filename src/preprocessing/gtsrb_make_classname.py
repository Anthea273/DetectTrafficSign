import os
import pandas as pd
import json

DATA_DIR = "data/processed/classification_gtsrb"
OUT_PATH = "models/classification/class_names_gtsrb.json"

def main():
    # Đọc tất cả các file label
    dfs = []
    for split in ["train_labels.csv", "val_labels.csv", "test_labels.csv"]:
        path = os.path.join(DATA_DIR, split)
        if os.path.exists(path):
            dfs.append(pd.read_csv(path))
    df = pd.concat(dfs, ignore_index=True)

    # Lấy danh sách class ID thực tế
    class_ids = sorted(df["ClassId"].unique())
    print(f"Found {len(class_ids)} unique classes:", class_ids)

    # Nếu đủ 43 lớp thì dùng tên chuẩn
    default_names = [
        "Speed limit (20km/h)",
        "Speed limit (30km/h)",
        "Speed limit (50km/h)",
        "Speed limit (60km/h)",
        "Speed limit (70km/h)",
        "Speed limit (80km/h)",
        "End of speed limit (80km/h)",
        "Speed limit (100km/h)",
        "Speed limit (120km/h)",
        "No passing",
        "No passing for vehicles over 3.5 metric tons",
        "Right-of-way at the next intersection",
        "Priority road",
        "Yield",
        "Stop",
        "No vehicles",
        "Vehicles over 3.5 metric tons prohibited",
        "No entry",
        "General caution",
        "Dangerous curve to the left",
        "Dangerous curve to the right",
        "Double curve",
        "Bumpy road",
        "Slippery road",
        "Road narrows on the right",
        "Road work",
        "Traffic signals",
        "Pedestrians",
        "Children crossing",
        "Bicycles crossing",
        "Beware of ice/snow",
        "Wild animals crossing",
        "End of all speed and passing limits",
        "Turn right ahead",
        "Turn left ahead",
        "Ahead only",
        "Go straight or right",
        "Go straight or left",
        "Keep right",
        "Keep left",
        "Roundabout mandatory",
        "End of no passing",
        "End of no passing by vehicles over 3.5 metric tons",
        "Other danger"
    ]

    # Chỉ giữ lại tên tương ứng với ID có trong dataset
    class_names = {cid: default_names[cid] for cid in class_ids if cid < len(default_names)}

    # Xuất ra JSON
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump([class_names[cid] for cid in class_ids], f, indent=2, ensure_ascii=False)

    print(f"Saved to {OUT_PATH}")
    print(f"Example mapping: {list(class_names.items())[:5]}")

if __name__ == "__main__":
    main()
