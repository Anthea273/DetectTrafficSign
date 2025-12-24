# Dữ liệu gốc

## 1. GTSRB (German Traffic Sign Recognition Benchmark)

- Nguồn: Kaggle / benchmark 2011
- Mục đích: PHÂN LOẠI biển báo (classification)
- Cấu trúc gốc:
  - Train/: ảnh train
  - Test/: ảnh test
  - Train.csv, Test.csv: chứa bounding box (Roi.X1..Y2) và ClassId
  - label_names.csv: map classId -> tên biển báo
- Không chỉnh sửa thủ công trong thư mục này.

## 2. Vietnamese Traffic Signs

- Nguồn: Kaggle Vietnamese traffic signs dataset
- Mục đích: PHÁT HIỆN biển báo (detection)
- Cấu trúc gốc:
  - images/: ảnh đường phố VN
  - labels/: annotation YOLO cho từng ảnh (.txt với (class cx cy w h))
  - classes.txt: danh sách class theo index
- Không chỉnh sửa thủ công trong thư mục này.

## Ghi chú

Mọi thay đổi / lọc nhiễu / chuẩn hoá sẽ đưa qua `data/interim/` và kết quả dùng train sẽ ở `data/processed/`.
