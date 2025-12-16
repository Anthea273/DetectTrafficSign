import os
import cv2

def ensure_dir(path: str):
    if not os.path.exists(path):
        os.makedirs(path)

def load_image_cv2_rgb(path: str):
    """Đọc ảnh bằng cv2, trả về ảnh RGB (hoặc None nếu hỏng)."""
    img_bgr = cv2.imread(path)
    if img_bgr is None:
        return None
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    return img_rgb
