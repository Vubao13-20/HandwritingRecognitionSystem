from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np
from services.recognizer import HandwritingRecognizer

app = FastAPI()

# Cấu hình CORS để Next.js (port 3000) gọi được vào đây
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Khởi tạo recognizer
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
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)