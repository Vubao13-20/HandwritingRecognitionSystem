from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np
from backend.services.recognizer import HandwritingRecognizer

app = FastAPI()

# Cấu hình CORS để Next.js (port 3000) gọi được vào đây
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Khai báo biến chứa AI nhưng CHƯA khởi tạo ngay
ai_engine = None

# Hàm này đảm bảo AI CHỈ ĐƯỢC LOAD ĐÚNG 1 LẦN khi server đã bật xong
@app.on_event("startup")
def startup_event():
    global ai_engine
    print("Web đã khởi động, bắt đầu load model AI...")
    ai_engine = HandwritingRecognizer()

@app.post("/predict")
async def predict_api(file: UploadFile = File(...)):
    # 1. Đọc file ảnh người dùng gửi lên
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # 2. Gọi AI xử lý
    text_result = ai_engine.predict(img)

    # 3. Trả về đúng format Frontend đang chờ
    return {
        "data": [
            {"id": 1, "text": text_result}
        ]
    }

if __name__ == "__main__":
    import uvicorn
    # TẮT reload=True vì nó là khắc tinh của TensorFlow
    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=False)