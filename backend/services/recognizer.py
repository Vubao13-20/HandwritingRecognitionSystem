import os
import cv2
import numpy as np
import math
import tensorflow.compat.v1 as tf
tf.compat.v1.disable_eager_execution()

from backend.utils.config import cfg
from backend.utils.util import LoadClasses, LoadModel, ReadData
from backend.core.models.cnn import CNN, WND_HEIGHT, WND_WIDTH, MPoolLayers_H
from backend.core.models.rnn import RNN

class HandwritingRecognizer:
    def __init__(self):
        print(" Đang khởi tạo bộ não AI (CRNN + CTC) thực thụ...")
        
        tf.keras.backend.clear_session()
        tf.compat.v1.reset_default_graph()

        self.Classes = LoadClasses(cfg.CHAR_LIST)
        self.NClasses = len(self.Classes)

        self.WND_SHIFT = WND_WIDTH - 2
        self.VEC_PER_WND = WND_WIDTH / math.pow(2, MPoolLayers_H)

        # Xây dựng mô hình trên Default Graph (giống hệt file test.py gốc)
        self.phase_train = tf.Variable(False, name='phase_train')
        self.x = tf.placeholder(tf.float32, shape=[None, WND_HEIGHT, WND_WIDTH])
        self.SeqLens = tf.placeholder(shape=[None], dtype=tf.int32)
        
        x_expanded = tf.expand_dims(self.x, 3)
        self.Inputs = CNN(x_expanded, self.phase_train, 'CNN_1')
        self.logits = RNN(self.Inputs, self.SeqLens, 'RNN_1')
        self.decoded, self.log_prob = tf.nn.ctc_beam_search_decoder(self.logits, self.SeqLens)

        self.session = tf.Session()
        self.session.run(tf.global_variables_initializer())

        print("Đang nạp trí nhớ từ model.ckpt...")
        LoadModel(self.session, cfg.SaveDir+'/')
        self.session.run(tf.assign(self.phase_train, False))
        print("Lắp não thành công! Hệ thống đã sẵn sàng.")


    def predict(self, img_array):
        if img_array is None:
            return "Lỗi: Không nhận được ảnh từ Web."

        try:
            gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
            h, w = gray.shape
            
            new_h = WND_HEIGHT
            new_w = int((w / float(h)) * new_h)
            
            if new_w < WND_WIDTH:
                new_w = WND_WIDTH
                
            resized = cv2.resize(gray, (new_w, new_h))
            normalized = resized / 255.0
            
            # --- 2. CẮT ẢNH THÀNH TỪNG Ô VUÔNG 64x64 ---
            windows = []
            x = 0
            while x <= (new_w - WND_WIDTH):
                window = normalized[:, x:x + WND_WIDTH]
                windows.append(window)
                x += int(self.WND_SHIFT)
            
            if x < new_w and (new_w - WND_WIDTH) > 0:
                windows.append(normalized[:, new_w - WND_WIDTH:])
                
            if len(windows) == 0:
                windows.append(cv2.resize(normalized, (WND_WIDTH, WND_HEIGHT)))

            # --- 3. ĐÁNH LỪA MÔ HÌNH (FAKE BATCH SIZE) ---
            # Mạng RNN bắt buộc phải nhận đủ số lượng = cfg.BatchSize
            windows_batch = []
            seq_lens_batch = []
            
            # Vị trí số 1: Nhét ảnh thật vào
            windows_batch.extend(windows)
            seq_lens_batch.append(len(windows))
            
            # Các vị trí còn lại: Nhét ảnh giả (trống) cho đủ sĩ số
            dummy_window = np.zeros((WND_HEIGHT, WND_WIDTH), dtype=np.float32)
            for _ in range(cfg.BatchSize - 1):
                windows_batch.append(dummy_window)
                seq_lens_batch.append(1) # Ảnh giả có độ dài = 1

            # --- 4. BƠM VÀO NÃO AI ---
            feed = {self.x: windows_batch, self.SeqLens: seq_lens_batch}
            Decoded = self.session.run([self.decoded], feed_dict=feed)[0]
            trans = self.session.run(tf.sparse_tensor_to_dense(Decoded[0]))

            # --- 5. DỊCH MÃ SỐ THÀNH CHỮ ---
            decodedStr = ""
            # Chúng ta CHỈ quan tâm đến kết quả đầu tiên (trans[0] là ảnh thật)
            if len(trans) == 0 or len(trans[0]) == 0:
                 return "AI không nhìn thấy chữ gì rõ ràng cả."

            for j in range(0, len(trans[0])):
                if trans[0][j] == 0:
                    if (j != (len(trans[0]) - 1)):
                        if trans[0][j+1] == 0: break
                        else: decodedStr += self.Classes[trans[0][j]]
                    else: break
                else:
                    if trans[0][j] == (self.NClasses - 2):
                        if (j != 0): decodedStr += " "
                        else: continue
                    else:
                        decodedStr += self.Classes[trans[0][j]]

            decodedStr = decodedStr.replace("<SPACE>", " ").strip()
            
            if not decodedStr:
                return "AI quét được nét nhưng không dịch ra chữ."
            return decodedStr

        except Exception as e:
            print(f"Lỗi giải mã ảnh: {e}")
            return "Lỗi nội bộ khi AI đọc ảnh."