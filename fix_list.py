import os

# Cấu hình đường dẫn
IMAGE_DIR = "./data/raw/samples/Images"
OUTPUT_LIST = "./data/raw/samples/list"

with open(OUTPUT_LIST, "w") as f:
    for root, dirs, files in os.walk(IMAGE_DIR):
        for filename in files:
            if filename.endswith((".png", ".jpg", ".jpeg")):
                # Lấy đường dẫn tương đối từ sau chữ Images/
                full_path = os.path.join(root, filename)
                relative_path = os.path.relpath(full_path, IMAGE_DIR)
                
                # Loại bỏ đuôi file 
                name_without_ext = os.path.splitext(relative_path)[0]
                
                # Chuyển dấu gạch chéo ngược \ của Windows thành /
                clean_path = name_without_ext.replace("\\", "/")
                f.write(clean_path + "\n")

print(f"✅ Đã tạo lại file {OUTPUT_LIST} thành công với {len(files)} ảnh!")