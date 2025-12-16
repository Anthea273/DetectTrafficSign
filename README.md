# Traffic Sign Recognition

## Mục tiêu

Xây dựng hệ thống có thể:

1. Phát hiện vị trí biển báo giao thông trong ảnh/video (detection).
2. Phân loại loại biển báo (classification).
3. Chạy demo dạng ứng dụng (upload ảnh → hiển thị kết quả).

## Kiến trúc

- `data/`
  - raw/: dữ liệu gốc (GTSRB, Vietnamese Traffic Signs)
  - interim/: dữ liệu đã làm sạch tạm
  - processed/: dữ liệu chuẩn hoá, đã split train/val/test, dùng trực tiếp để train
- `src/`
  - preprocessing/: script chuẩn hoá data, thống kê dữ liệu
  - classification/: train/eval/inference CNN, ResNet, EfficientNet (GTSRB)
  - detection/: train/eval/inference YOLOv5/YOLOv8 (biển báo Việt Nam)
  - visualization/: vẽ biểu đồ phân bố class, confusion matrix, v.v.
  - demo/: app Streamlit chạy inference local
- `models/`: lưu trọng số đã huấn luyện (.pt / .pth / .h5)
- `results/`: lưu metric (accuracy, mAP, FPS...), confusion matrix, PR curve
- `docs/`: slide, báo cáo và hình minh hoạ

## Quy trình chuẩn

1. Chuẩn hoá dữ liệu
   - `python src/preprocessing/gtsrb_prepare.py`
   - `python src/preprocessing/vn_prepare.py`
   - `python src/preprocessing/analyze_dataset.py`
2. Train mô hình classification (GTSRB)
   - `python src/classification/train_cnn.py`
   - (tương tự cho ResNet / EfficientNet)
3. Train mô hình detection (Vietnam YOLO)
   - Thực hiện (có thể trên Colab), copy best.pt về `models/detection/yolov8n_vn_best.pt`
4. Đánh giá & xuất kết quả
   - Lưu metric vào `results/`
5. Demo local (không cần internet)
   - `python src/detection/inference_image.py`
   - `streamlit run src/demo/app_streamlit.py`

## Demo offline

- Không cần train lại.
- Không cần GPU.
- Chỉ load trọng số đã có và chạy suy luận (inference).

## Installation

```bash
pip install -r requirements.txt

```

## Run Demo

streamlit run src/demo/app_streamlit.py
