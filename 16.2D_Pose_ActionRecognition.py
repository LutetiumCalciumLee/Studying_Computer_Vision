from ultralytics import YOLO
from webcam_zoom import digital_zoom
from pose_data import FEATURE_MODE, pose_features
from pose_config import CLASS_NAMES, DATASET_RELATIVE_PATH
import cv2
import xgboost as xgb
import pandas as pd
import os
import json
import numpy as np

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
model_file = os.path.join(Models_dir, "yolov8n-pose.pt" )

save_dir = os.path.join(script_dir, DATASET_RELATIVE_PATH)
weight_file = os.path.join(save_dir, 'model_weights.xgb')
labels_file = os.path.join(save_dir, 'labels.json')
features_file = os.path.join(save_dir, 'features.json')

if not os.path.isfile(weight_file):
    raise FileNotFoundError('앉음·서 있음 모델이 없습니다. 13번으로 두 자세를 촬영한 뒤 14 → 15를 실행하세요.')

# YOLOv8 모델 로드
model_yolo = YOLO(model_file)

# XGBoost 모델 로드 (Booster 대신 XGBClassifier 사용)
model = xgb.XGBClassifier()
# Windows 한글 경로를 XGBoost의 파일 API에 직접 전달하지 않음
with open(weight_file, 'rb') as weight_stream:
    model.load_model(bytearray(weight_stream.read()))

if model.n_classes_ != len(CLASS_NAMES):
    raise ValueError('앉음·서 있음 모델이 아닙니다. 14 → 15를 다시 실행하세요.')

# 라벨 매핑 로드 (없으면 숫자 라벨 그대로 사용)
id_to_name = {str(i): name for i, name in enumerate(CLASS_NAMES)}
if os.path.exists(labels_file):
    try:
        with open(labels_file, 'r', encoding='utf-8') as f:
            labels = json.load(f)
            id_to_name = labels.get('id_to_name', {})
    except Exception:
        id_to_name = {}

# 피처 컬럼 순서 로드 (없으면 추론 시 생성한 순서를 사용)
feature_columns = None
preprocessing = 'frame_normalized_v1'
if os.path.exists(features_file):
    try:
        with open(features_file, 'r', encoding='utf-8') as f:
            meta = json.load(f)
            feature_columns = meta.get('feature_columns', None)
            preprocessing = meta.get('preprocessing', 'frame_normalized_v1')
    except Exception:
        feature_columns = None

if preprocessing not in ('frame_normalized_v1', FEATURE_MODE):
    raise ValueError(f'Unknown preprocessing: {preprocessing}')

def get_label_name(class_id):
    try:
        key = str(int(class_id))
    except Exception:
        key = str(class_id)
    return id_to_name.get(key, key)

# 카메라에서 비디오 캡처 시작
cap = cv2.VideoCapture(0)

# 총 프레임 수 출력 (일부 카메라에서는 0일 수 있음)
print('Total Frame', cap.get(cv2.CAP_PROP_FRAME_COUNT))

# 비디오 속성 가져오기
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

frame_tot = 0  # 처리된 프레임 수

# 비디오 프레임 반복 처리
while cap.isOpened():
    # 비디오 프레임 읽기
    success, frame = cap.read()

    if success:
        # YOLOv8로 프레임에서 객체 감지 수행
        results = model_yolo(frame, verbose=False)

        # 추론은 원본으로 수행하고, 표시 화면만 원본 크기로 좌우 반전
        annotated_frame = cv2.flip(results[0].plot(boxes=False), 1)
        display_width = annotated_frame.shape[1]

        for r in results:
            bound_box = r.boxes.xyxy  # 경계 상자 정보
            conf = r.boxes.conf.tolist()  # 신뢰도 값
            keypoints = r.keypoints.xyn.tolist()  # 키포인트 좌표

            for index, box in enumerate(bound_box):
                if conf[index] > 0.75:  # 신뢰도 0.75 이상일 때만 처리
                    x1, y1, x2, y2 = box.tolist()
                    # 학습 모델의 좌표 기준과 동일하게 변환 (구 모델도 지원)
                    if preprocessing == FEATURE_MODE:
                        data = pose_features(r, index)
                    else:
                        data = {}
                        for j in range(len(keypoints[index])):
                            data[f'x{j}'] = keypoints[index][j][0]
                            data[f'y{j}'] = keypoints[index][j][1]

                    # 데이터프레임 생성 및 학습 시 컬럼 순서와 정렬
                    df = pd.DataFrame([data])
                    if feature_columns is not None:
                        # 누락 컬럼은 0으로 채우고, 여분 컬럼은 제거
                        for col in feature_columns:
                            if col not in df.columns:
                                df[col] = 0.0
                        df = df.reindex(columns=feature_columns)

                    # 확률 기반 예측
                    proba = model.predict_proba(df)[0]
                    pred_class = int(np.argmax(proba))
                    label_text = str(get_label_name(pred_class)).title()

                    # 사각형 및 라벨 그리기
                    # 클래스별 색상
                    color_map = {
                        0: (255, 0, 0), 1: (0, 255, 0),
                    }
                    color = color_map.get(pred_class, (255, 0, 0))
                    # 반전된 화면에 맞춰 경계 상자 위치 변환 (라벨 글자는 정방향)
                    display_x1 = display_width - 1 - int(x2)
                    display_x2 = display_width - 1 - int(x1)
                    cv2.rectangle(annotated_frame, (display_x1, int(y1)), (display_x2, int(y2)), color, 2)
                    # 텍스트를 사각형 내부에 보이도록 살짝 안쪽으로
                    cv2.putText(annotated_frame, label_text, (display_x1 + 5, int(y1) + 25), cv2.FONT_HERSHEY_DUPLEX, 1.0, color, 2)

        # 원본 화면 크기를 유지하고, 표시 화면에 디지털 줌 적용
        annotated_display = digital_zoom(annotated_frame, ZOOM_FACTOR)
        cv2.imshow("Action Recognition", annotated_display)

        frame_tot += 1  # 처리된 프레임 수 증가
        # print('Processed Frame : ', frame_tot)

        # q / Esc: 종료
        if cv2.waitKey(1) & 0xFF in (ord('q'), ord('Q'), 27):
            break
    else:
        # 비디오 끝에 도달하면 종료
        break

# 비디오 캡처 객체 및 창 닫기
cap.release()
cv2.destroyAllWindows()
