import cv2 # thư viện xử lý ảnh, đọc camera và hiển thị cửa sổ
import mediapipe as mp # thư viện xử lý nhận diện khuôn mặt
import os # thư viện thao tác với hệ thống file
import time # thư viện xử lý thời gian (trễ)

# =========================
# DANH SÁCH CẢM XÚC THEO THỨ TỰ
# =========================
EMOTIONS = ["batngo", "binhThuong", "buon", "tucgian", "vuive"]

# =========================
# THƯ MỤC CHỨA DỮ LIỆU
# =========================
DATA_DIR = "Data"# tên thư mục gốc chứa dữ liệu cảm xúc

# =========================
# CAMERA INDEX
# =========================
CAMERA_INDEX = 0     # Camera của bạn

# =========================
# MEDIAPIPE SETUP khởi tạo bộ phát hiện khuôn mặt
# =========================
mp_face = mp.solutions.face_detection # sử dụng mô-đun phát hiện khuôn mặt
detector = mp_face.FaceDetection(0.6) # khởi tạo bộ phát hiện với ngưỡng tin cậy 0.6

# =========================
# TẠO THƯ MỤC CẢM XÚC NẾU CHƯA TỒN TẠI
# =========================
for emo in EMOTIONS: # duyệt qua từng cảm xúc
    folder = os.path.join(DATA_DIR, emo) # đường dẫn thư mục cảm xúc
    if not os.path.exists(folder): # nếu thư mục chưa tồn tại
        os.makedirs(folder) # tạo thư mục


# =========================
# THU THẬP THEO PHÍM T
# =========================
def collect(emotion): # hàm thu thập dữ liệu cho cảm xúc được chọn

    save_folder = os.path.join(DATA_DIR, emotion) # thư mục lưu ảnh
    print(f"\n📸 CHẾ ĐỘ CHỤP THỦ CÔNG – cảm xúc: {emotion}") 
    print("➡ Nhấn phím T để chụp ảnh")
    print("➡ Nhấn phím Q để thoát\n")

    cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW) # mở camera
    if not cap.isOpened(): # kiểm tra camera có mở được không
        print("❌ Không mở được camera")
        return
    
    count = len(os.listdir(save_folder))  # tiếp tục từ số ảnh hiện có tránh ghi đè

    while True: # đọc từng frame(khung hình) từ camera
        ret, frame = cap.read()# đọc frame
         # kiểm tra đọc frame có thành công không
        if not ret:
            print("⚠ Không đọc được frame")
            continue

        frame = cv2.flip(frame, 1) # lật ngang khung hình để giống gương
        h, w, _ = frame.shape # lấy kích thước khung hình

        # Detect face
        results = detector.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))# phát hiện khuôn mặt

        if results.detections: # nếu phát hiện được khuôn mặt
            det = results.detections[0].location_data.relative_bounding_box # lấy hộp giới hạn khuôn mặt đầu tiên
            x1 = int(det.xmin * w) # tính tọa độ x1
            y1 = int(det.ymin * h) # tính tọa độ y1
            x2 = int((det.xmin + det.width) * w) # tính tọa độ x2
            y2 = int((det.ymin + det.height) * h) # tính tọa độ y2

            # Vẽ khung mặt
            cv2.rectangle(frame, (x1,y1), (x2,y2), (255,0,255), 2) # vẽ hình chữ nhật quanh mặt
            cv2.putText(frame, "Nhan phim T de chup", (10,40), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

        cv2.imshow("Manual Capture", frame) # hiển thị khung hình

        key = cv2.waitKey(1) & 0xFF # đợi phím nhấn

        # Thoát
        if key == ord('q'): 
            break

        # Chụp ảnh nếu nhấn T
        if key == ord('t'):
            if results.detections: # nếu phát hiện được khuôn mặt
                face = frame[y1:y2, x1:x2] # cắt khuôn mặt từ khung hình

                if face.size > 0: # kiểm tra khuôn mặt có kích thước hợp lệ
                    face_gray = cv2.cvtColor(
                        cv2.resize(face, (48,48)), # thay đổi kích thước về 48x48 vì phổ biến/ thường dùng cho nhận diện cảm xúc
                        cv2.COLOR_BGR2GRAY  # chuyển sang ảnh xám
                    )

                    filepath = os.path.join(save_folder, f"{count}.jpg")# đường dẫn lưu ảnh
                    cv2.imwrite(filepath, face_gray) # lưu ảnh
                    print(f"✔ ĐÃ CHỤP: {filepath}") # in thông báo đã chụp

                    count += 1 # tăng bộ đếm ảnh
                    time.sleep(0.3)  # tránh chụp trùng khi giữ T

            else:
                print("⚠ Không thấy mặt – không thể chụp")

    cap.release() # giải phóng camera
    cv2.destroyAllWindows() # đóng tất cả cửa sổ hiển thị
    print("\n🎉 ĐÃ THOÁT CHẾ ĐỘ CHỤP\n") 


# =========================
# MENU CHỌN CẢM XÚC
# =========================
print("Chọn cảm xúc muốn chụp:") # in ra menu chọn cảm xúc
for i, emo in enumerate(EMOTIONS): # duyệt qua từng cảm xúc với chỉ số
    print(f"{i}. {emo}") # in chỉ số và tên cảm xúc

choice = int(input("\nNhập số: ")) # nhập lựa chọn từ người dùng
collect(EMOTIONS[choice]) # gọi hàm thu thập dữ liệu cho cảm xúc được chọn
