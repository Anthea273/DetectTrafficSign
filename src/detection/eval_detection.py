import os
import json
import pandas as pd

LOG_CSV = "results/detection/training_log_yolov8/results.csv"
OUT_JSON = "results/detection/metrics_yolov8.json"

def main():
    if not os.path.exists(LOG_CSV):
        print("Không tìm thấy log:", LOG_CSV)
        return

    df = pd.read_csv(LOG_CSV)

    # thường hàng cuối cùng là epoch cuối, bạn có thể chọn hàng có mAP50 cao nhất nếu muốn
    last_row = df.iloc[-1]

    metrics = {
        "precision": float(last_row.get("metrics/precision(B)", last_row.get("precision", 0))),
        "recall": float(last_row.get("metrics/recall(B)", last_row.get("recall", 0))),
        "mAP@0.5": float(last_row.get("metrics/mAP50(B)", last_row.get("map50", 0))),
        "mAP@0.5:0.95": float(last_row.get("metrics/mAP50-95(B)", last_row.get("map50-95", 0))),
        "train/box_loss": float(last_row.get("train/box_loss", 0)),
        "train/cls_loss": float(last_row.get("train/cls_loss", 0)),
        "val/box_loss": float(last_row.get("val/box_loss", 0)),
        "val/cls_loss": float(last_row.get("val/cls_loss", 0)),
        "epoch": int(last_row.get("epoch", -1))
    }

    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    with open(OUT_JSON, "w") as f:
        json.dump(metrics, f, indent=2)

    print("metrics_yolov8.json generated at", OUT_JSON)
    print(metrics)

if __name__ == "__main__":
    main()
