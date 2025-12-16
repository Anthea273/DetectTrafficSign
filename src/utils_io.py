import os

def ensure_dir(path):
    """Tạo thư mục nếu chưa tồn tại."""
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
