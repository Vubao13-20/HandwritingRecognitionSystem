import cv2
import numpy as np
import tensorflow as tf
import os

class HandwritingRecognizer:
    def __init__(self):
        # Tạm thời để None, khi nào có file .h5 sẽ điền đường dẫn vào đây
        self.model_path = "model_final.h5" 
        self.model = None
        
        if os.path.exists(self.model_path):
            self.model = tf.keras.models.load_model(self.model_path, compile=False)
            print("✅ Model AI đã được nạp thành công!")
        else:
            print("⚠️ Chưa tìm thấy file model_final.h5, hệ thống sẽ chạy ở chế độ Demo.")

    def preprocess(self, img):
        # Repo này yêu cầu ảnh Grayscale và kích thước 128x32
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (128, 32))
        normalized = resized / 255.0
        # Thêm chiều để khớp với input của CNN (Batch, Width, Height, Channel)
        final_img = np.expand_dims(normalized, axis=(0, -1))
        return final_img

    def predict(self, img_array):
        if self.model is None:
            # Nếu chưa có model thật, trả về kết quả giả lập để test Web
            return "Dòng chữ được nhận diện (Demo Mode)"
        
        processed_img = self.preprocess(img_array)
        prediction = self.model.predict(processed_img)
        
        # Ở đây sẽ cần hàm CTC Decode từ file util.py của repo gốc
        # Tạm thời trả về text thô từ prediction
        return "Kết quả từ AI"