# Nhận Diện Cảm Xúc Khuôn Mặt (Facial Emotion Recognition)

## 1. Giới thiệu
Dự án này sử dụng **machine learning** và **computer vision** để nhận diện cảm xúc trên khuôn mặt từ hình ảnh hoặc video. Mục tiêu là phân loại cảm xúc thành các loại như:
- Vui vẻ (Happy)
- Buồn (Sad)
- Giận dữ (Angry)
- Ngạc nhiên (Surprised)
- Bình thường (Neutral)

## 2. Công nghệ sử dụng
- **Python**  
- **OpenCV**: Xử lý ảnh và nhận diện khuôn mặt  
- **TensorFlow / PyTorch**: Xây dựng và huấn luyện mô hình deep learning  
- **Keras**: Giao diện mạng nơ-ron  
- **NumPy, Pandas**: Xử lý dữ liệu  
- **Matplotlib / Seaborn**: Trực quan hóa kết quả  

## 3. Cấu trúc dự án
```bash
NHOM-03-TriTueNhanTao-NhanDienCamXuc/
├── Data/                   # Thư mục chứa dữ liệu hình ảnh khuôn mặt đã được phân loại
│   ├── vuiVe/
│   ├── buon/
│   └── ...
├── Model6.h5               # Mô hình học sâu (CNN) đã được huấn luyện
├── FaceDetectionModule.py  # Module phát hiện khuôn mặt (sử dụng OpenCV)
├── TrainData.ipynb         # Notebook dùng để tiền xử lý, huấn luyện và đánh giá mô hình
├── collect.py              # Script thu thập dữ liệu (nếu có)
├── test5.py                # Script chạy ứng dụng nhận diện real-time
└── README.md
```
## 4. Cài đặt


## 4. Cài đặt

### 4.1. Clone dự án
```bash
git clone https://github.com/LeVanChinh1305/NHOM-03-TriTueNhanTao-NhanDienCamXuc.git
cd NHOM-03-TriTueNhanTao-NhanDienCamXuc 
```
### 4.2. Tạo môi trường ảo  (tùy chọn nhưng khuyến nghị)
```bash
    # 1. Tạo môi trường ảo
    python -m venv venv

    # 2. Kích hoạt môi trường (Windows)
    venv\Scripts\activate

    # 2. Kích hoạt môi trường (Linux / macOS)
    source venv/bin/activate
```
### 4.3. Cài đặt các thư viện
Cài đặt tất cả các thư viện cần thiết:
```bash
    pip install --upgrade pip
    pip install opencv-python tensorflow keras numpy pandas matplotlib seaborn
```
### 4.4. Chuẩn bị dữ liệu
Đảm bảo dữ liệu khuôn mặt đã được phân loại nằm trong thư mục Data/ theo cấu trúc yêu cầu của mô hình.
### 4.5. Kiểm tra môi trường
Chạy lệnh kiểm tra sau để xác nhận các thư viện chính đã được cài đặt thành công:
```bash
python -c "import cv2, tensorflow, keras, numpy; print('Môi trường đã sẵn sàng!')"
```
## 5. Sử dụng
### 5.1. Huấn luyện mô hình
Mở Notebook để tiền xử lý và huấn luyện:

bash
Copy code
jupyter notebook TrainData.ipynb
### 5.2. Nhận diện cảm xúc từ ảnh
bash
Copy code
python test5.py --image path_to_image.jpg
### 5.3. Nhận diện cảm xúc từ webcam
bash
Copy code
python test5.py --webcam
## 6. Kết quả
Mô hình có thể dự đoán cảm xúc với độ chính xác cao trên dữ liệu test.

Ví dụ trực quan:

Ảnh	Dự đoán
Happy
Sad

## 7. Ghi chú
Dataset và mô hình cần được tải trước khi chạy script.

Chạy trên GPU sẽ nhanh hơn đáng kể so với CPU.