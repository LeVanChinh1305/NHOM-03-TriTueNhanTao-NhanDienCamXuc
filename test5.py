import cv2 # thư viện xử lý ảnh, đọc camera và hiển thị cửa sổ
import FaceDetectionModule as fdm # module phát hiện khuôn mặt đã tạo
import keras.models # thư viện keras để tải mô hình học sâu
import numpy as np # thư viện xử lý mảng số học
from PIL import ImageFont, ImageDraw, Image # thư viện PIL để vẽ text đẹp hơn

# Font chữ
font = ImageFont.truetype("./arial.ttf", 28) # tải font chữ Arial với kích thước 28

# Mở camera
cap = cv2.VideoCapture(0) # mở camera

if not cap.isOpened(): # kiểm tra camera có mở được không
    print("❌ Không mở được camera")
    exit()

# Tải model
model = keras.models.load_model("Model6.h5") # tải mô hình học sâu đã huấn luyện trước trong tệp Model6.h5 

# Bộ phát hiện khuôn mặt
detector = fdm.FaceDetector() # khởi tạo bộ phát hiện khuôn mặt từ module FaceDetectionModule

# Nhãn cảm xúc
label = ["Bất ngờ", "Bình thường", "Buồn", "Tức giận", "Vui vẻ"]

while True: # đọc từng frame(khung hình) từ camera
    success, img = cap.read() # đọc frame
    
    if not success:# kiểm tra đọc frame có thành công không
        print("❌ Không thể lấy ảnh từ camera")
        break

    # Lật ảnh cho giống gương
    img = cv2.flip(img, 1) # lật ngang khung hình để giống gương

    # Chuyển sang RGB 
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Lấy kết quả từ mediapipe
    results = detector.faceDetection.process(imgRGB) # phát hiện khuôn mặt

    # Sử dụng PIL để vẽ text đẹp hơn (tránh nhòe chữ so với OpenCV)
    img_pil = Image.fromarray(img) # chuyển ảnh từ OpenCV sang PIL
    draw = ImageDraw.Draw(img_pil) # tạo đối tượng vẽ trên ảnh PIL

    # Nếu phát hiện khuôn mặt
    if results.detections: 
        for detection in results.detections: # duyệt qua từng khuôn mặt được phát hiện

            # Lấy tọa độ khung mặt
            bboxC = detection.location_data.relative_bounding_box # lấy hộp giới hạn khuôn mặt
            h, w, _ = img.shape # lấy kích thước khung hình

            x = int(bboxC.xmin * w) # tính tọa độ x1
            y = int(bboxC.ymin * h) # tính tọa độ y1
            bw = int(bboxC.width * w)   # tính chiều rộng khung mặt
            bh = int(bboxC.height * h)  # tính chiều cao khung mặt

            # Cắt mặt
            face = imgRGB[y:y+bh, x:x+bw] # cắt khuôn mặt từ ảnh gốc

            # Kiểm tra mặt rỗng
            if face.size == 0:
                continue

            # Resize về 48x48 grayscale
            face_gray = cv2.resize(face, (48,48)) # thay đổi kích thước về 48x48 vì phổ biến/ thường dùng cho nhận diện cảm xúc
            face_gray = cv2.cvtColor(face_gray, cv2.COLOR_RGB2GRAY) # chuyển sang ảnh xám
            face_gray = face_gray.reshape(1,48,48,1) / 255.0 # chuẩn hóa và thay đổi kích thước để phù hợp với đầu vào của mô hình

            # Dự đoán cảm xúc
            y_pred = model.predict(face_gray, verbose=0) # dự đoán cảm xúc từ khuôn mặt đã xử lý
                # đưa khuôn mặt đã chuẩn hoá vào model6.h5 để dự đoán
            result = np.argmax(y_pred) # lấy chỉ số cảm xúc có xác suất cao nhất

            # Vẽ khung vuông trên mặt (OpenCV)
            cv2.rectangle(img, (x, y), (x + bw, y + bh), (255, 0, 255), 2)

            # Vẽ nhãn cảm xúc trên mặt (PIL)
            draw.text((x, y - 35), label[result], font=font, fill=(255, 0, 0))

    # Chuyển PIL về OpenCV
    img_show = np.array(img_pil) # chuyển ảnh từ PIL sang OpenCV

    cv2.imshow("Emotion Recognition", img_show) # hiển thị khung hình

    # Thoát bằng phím Q
    if cv2.waitKey(10) & 0xFF == ord("q"):
        break

cap.release() # giải phóng camera
cv2.destroyAllWindows() # đóng tất cả cửa sổ hiển thị
