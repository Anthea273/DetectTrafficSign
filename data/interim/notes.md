Ghi chú xử lý dữ liệu (data curation log)

- GTSRB:

  - Đã crop theo ROI, resize về IMG_SIZE=48.
  - Loại bỏ ảnh nào bị hỏng / không đọc được.
  - Ghi kết quả crop tạm thời ở data/interim/gtsrb_cropped/.

- VN Traffic Signs:
  - Đã kiểm tra file label .txt tương ứng mỗi ảnh.
  - Ảnh bị thiếu nhãn hoặc nhãn lệch hoàn toàn → tạm loại và lưu phần sạch vào data/interim/vn_signs_cleaned/.
  - Những vấn đề label đặc biệt / nghi vấn được note lại ở đây.

Lý do tồn tại file này: dùng nội dung này vào báo cáo phần "Tiền xử lý dữ liệu".
