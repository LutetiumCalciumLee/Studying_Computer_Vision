import cv2
from ultralytics import YOLO
from pose_display import mirrored_pose_display
from webcam_zoom import digital_zoom
import os

# VS Code에서 직접 수정: 0.5 = 축소, 1.0 = 원본, 1.5 = 확대
ZOOM_FACTOR = 1.0

#=================================================================
#초기값 설정
#=================================================================
#실행 경로 설정 
# 경로 설정
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)
Models_dir = os.path.abspath(os.path.join(script_dir, "../00_Models"))
Videos_dir = os.path.abspath(os.path.join(script_dir, "../00_Sample_Video"))

model_file = os.path.join(Models_dir, "yolov8n-pose.pt" )

# 1. YOLOv8 Pose 모델 불러오기 (미리 학습된 모델 사용)
model = YOLO(model_file)  # 'n'은 모델 크기 옵션 (n, s, m, l, x)

# 2. 실시간 카메라 비디오 스트림 열기
cap = cv2.VideoCapture(0)  # 0번 카메라 열기

# 3. 비디오 스트림에서 프레임을 받아와서 포즈 추정
while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # YOLO Pose 모델로 포즈 추정 수행
    results = model(frame)

    # 4. 영상과 관절만 좌우 반전, 사람 라벨과 신뢰도는 정방향으로 표시
    annotated_display = mirrored_pose_display(results[0])
    annotated_display = digital_zoom(annotated_display, ZOOM_FACTOR)
    cv2.imshow("YOLOv8 Pose Estimation", annotated_display)

    # q / Esc: 종료
    if cv2.waitKey(1) & 0xFF in (ord('q'), ord('Q'), 27):
        break

# 5. 카메라와 창 닫기
cap.release()
cv2.destroyAllWindows()
