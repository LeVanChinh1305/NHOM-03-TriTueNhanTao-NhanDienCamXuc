# FaceDetectionModule.py Module này cung cấp dịch vụ phát hiện và chuẩn bị dữ liệu khuôn mặt theo thời gian thực.
# Bằng cách trả về ảnh khuôn mặt 48x48 ảnh xám, nó đáp ứng chính xác yêu cầu định dạng đầu vào của một 
# mô hình học sâu (CNN) phổ biến được huấn luyện để nhận dạng cảm xúc.

import cv2 # thư viện xử lý ảnh, đọc camera và hiển thị cửa sổ
import mediapipe as mp # thư viện xử lý nhận diện khuôn mặt
import time # thư viện xử lý thời gian (trễ)


class FaceDetector(): # Lớp phát hiện khuôn mặt sử dụng Mediapipe
    def __init__(self, minDetection = 0.75): # khởi tạo với ngưỡng tin cậy minDetection 
        self.minDetection = minDetection # ngưỡng tin cậy để phát hiện khuôn mặt
    
        self.mpFaceDetection = mp.solutions.face_detection # sử dụng mô-đun phát hiện khuôn mặt

        self.mpDraw = mp.solutions.drawing_utils # công cụ vẽ của mediapipe
        self.faceDetection = self.mpFaceDetection.FaceDetection(self.minDetection) # khởi tạo bộ phát hiện khuôn mặt với ngưỡng tin cậy

    # Tìm mặt trong ảnh và vẽ khung mặt
    def findFaces(self, img, draw = True): # Tìm mặt trong ảnh, nếu draw=True thì vẽ khung mặt lên ảnh
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # chuyển ảnh từ BGR sang RGB vì mediapipe sử dụng ảnh RGB
        
        self.results = self.faceDetection.process(imgRGB) # xử lý ảnh để phát hiện khuôn mặt
        self.bboxs = [] # danh sách chứa thông tin khung mặt
        # Hiển thị khung nhận dạng
        if self.results.detections: # nếu phát hiện được khuôn mặt
            for id, detection in enumerate(self.results.detections): # duyệt qua từng khuôn mặt được phát hiện
                
                bboxC = detection.location_data.relative_bounding_box # lấy hộp giới hạn khuôn mặt
                ih, iw, ic = img.shape # lấy kích thước ảnh
                bbox = (int(bboxC.xmin * iw), int(bboxC.ymin * ih),int(bboxC.width * iw), int(bboxC.height * ih)) # tính tọa độ hộp giới hạn khuôn mặt
                
                self.bboxs.append([id,bbox, detection.score]) # lưu thông tin khung mặt vào danh sách
                if draw: # nếu draw=True thì vẽ khung mặt lên ảnh
                    cv2.rectangle(img, bbox, (255,0,255), 2) # vẽ hình chữ nhật quanh mặt
                
                    cv2.putText(img, str(int((detection.score[0]*100))),(bbox[0], bbox[1]-10), cv2.FONT_HERSHEY_PLAIN, 2, (0,255,0), 1)# hiển thị độ tin cậy
        
        return img, self.bboxs # trả về ảnh đã vẽ khung mặt và danh sách khung mặt


    def facegrayDetection(self, img): # cắt và chuẩn hóa khuôn mặt 
        ok = True # trạng thái phát hiện mặt
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # chuyển ảnh từ BGR sang RGB vì mediapipe sử dụng ảnh RGB
        
        self.results = self.faceDetection.process(imgRGB) # xử lý ảnh để phát hiện khuôn mặt
        if self.results.detections: # nếu phát hiện được khuôn mặt
            
            detection = self.results.detections[0] # lấy khuôn mặt đầu tiên
            bboxC = detection.location_data.relative_bounding_box # lấy hộp giới hạn khuôn mặt
            ih, iw, ic = img.shape # lấy kích thước ảnh
            bbox = (int(bboxC.xmin * iw), int(bboxC.ymin * ih),int(bboxC.width * iw), int(bboxC.height * ih)) # tính tọa độ hộp giới hạn khuôn mặt
            # Cắt khung chứa mặt 
            cropped_face = imgRGB[bbox[1]:bbox[1]+bbox[3], bbox[0]:bbox[0]+bbox[2]] # cắt khuôn mặt từ ảnh gốc
            if len(cropped_face[0])!=0: # kiểm tra khuôn mặt có kích thước hợp lệ
                # Thanh đổi kích thước 
                cropped_face = cv2.resize(cropped_face,(48,48)) # thay đổi kích thước về 48x48 vì phổ biến/ thường dùng cho nhận diện cảm xúc
                # Chuyển sang ảnh xám
                cropped_face = cv2.cvtColor(cropped_face, cv2.COLOR_RGB2GRAY) # chuyển sang ảnh xám
                return cropped_face, ok, bbox # trả về khuôn mặt đã xử lý, trạng thái ok và tọa độ khung mặt
            else : 
                ok = False # trạng thái không phát hiện được mặt
                return None, ok, None # trả về None nếu không có khuôn mặt
        else:
            ok = False
            return None, ok, None
    
   

def main(): # hàm chính để chạy module phát hiện khuôn mặt
    cap = cv2.VideoCapture(0) # mở camera

    pTime = 0 # thời gian trước đó để tính FPS
    detector = FaceDetector(0.4) # khởi tạo bộ phát hiện khuôn mặt với ngưỡng tin cậy 0.4
    bboxs = [] # danh sách chứa thông tin khung mặt
    while True: # đọc từng frame(khung hình) từ camera
        success, img = cap.read() # đọc frame
        cTime = time.time() # thời gian hiện tại

        fps = 1/(cTime-pTime) # tính FPS (Frames Per Second)
        
        pTime = cTime # cập nhật thời gian trước đó
        img, bboxs = detector.findFaces(img) # phát hiện khuôn mặt và vẽ khung mặt
        imgDetection, ok = detector.facegrayDetection(img) # cắt và chuẩn hóa khuôn mặt
        
        cv2.putText(img, f"FPS: {int(fps)}",(20,50), cv2.FONT_HERSHEY_PLAIN, 2, (0,255,0), 2) # hiển thị FPS
        
        
        
        
        if (ok): # nếu phát hiện được khuôn mặt
            cv2.imshow("Image2", imgDetection) # hiển thị khuôn mặt đã xử lý
        cv2.imshow("Image1", img) # hiển thị khung hình gốc với khung mặt
        
    
    
        if cv2.waitKey(2) == ord("q"): 
            break
  
    
if __name__ == "__main__": # chạy hàm chính khi file được thực thi trực tiếp
    main()
    
    