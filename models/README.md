# models/

Chứa trọng số đã huấn luyện.

## detection/

- `yolov8n_vn_best.pt`
  - Copy từ `runs/detect/.../best.pt` sau khi train YOLOv8n trên dataset biển báo Việt Nam.
- `classes_vn.json`
  - Danh sách class tương ứng với index YOLO (parse từ data/raw/vn_signs/classes.txt).

## classification/

- `cnn_best.pth` (hoặc .h5)
- `resnet_best.pth`
- `efficientnet_best.pth`
  - Trọng số sau khi train trên GTSRB.
- `class_names_gtsrb.json`
  - Map {class_id: "tên biển báo"} để hiển thị ra chữ thay vì chỉ số.

Các file này KHÔNG push public nếu có policy riêng, nhưng nên giữ nội bộ nhóm để demo.
