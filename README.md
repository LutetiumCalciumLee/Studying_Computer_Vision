<details>
<summary>ENG (English Version)</summary>

# Human Pose Estimation

### 1. Pose Estimation Overview

- **Definition:** Detects human body keypoints (joints) for skeleton reconstruction in images/videos.
- **Applications:** Action recognition, motion analysis, sports analytics, surveillance.
- **Keypoint Standards:** COCO (17 keypoints), OpenPose (25 body + 68 face + 42 hands + 6 feet).

### 2. Detection Approaches

- **Top-Down:** Detect person → crop → estimate pose per person (AlphaPose, Mask R-CNN); scales poorly with crowds.
- **Bottom-Up:** Detect all keypoints → group into persons (OpenPose, HRNet, DeepCut); efficient for multi-person.
- **2D vs 3D:** 2D (x,y coordinates) vs 3D (x,y,z); 3D requires depth estimation.

### 3. YOLOv8 Pose Implementation

- **Model Variants:** yolov8n-pose.pt (nano), s/m/l/x sizes for speed/accuracy trade-off.
- **Real-time Pipeline:** Webcam input → YOLO pose inference → keypoint visualization.
- **Output Format:** Bounding box + confidence + 17 keypoints (x,y,norm-confidence).

### 4. Action Recognition Pipeline

- **Data Collection:** 13.2DPoseSave.py extracts keypoints from sitting/standing videos → keypoints.csv.
- **Dataset Prep:** 14.2DPoseDatasetSave.py creates labeled dataset.csv (sitting=0, standing=1).
- **XGBoost Training:** 15.2DPoseTraining.py trains binary classifier on keypoint features.

### 5. Pose-based Classification

- **Real-time Inference:** 16.2DPoseActionRecognition.py → YOLO pose → XGBoost predict → label overlay.
- **Performance:** Train/test split (80/20), accuracy scoring, model weights saved as modelweights.xgb.
- **Libraries:** ultralytics, xgboost, scikit-learn, pandas for end-to-end pipeline.

### 6. Key Advantages

- **Efficiency:** Bottom-up single-stage processing handles crowds better than top-down.
- **Real-time:** YOLOv8 achieves webcam inference with keypoint extraction + classification.
- **Extensibility:** 133+ keypoints (COCO wholebody) support complex action analysis.

</details>

<details>
<summary>KOR (한국어 버전)</summary>

# 인간 자세 추정

### 1. 자세 추정 개요

- **정의:** 이미지/비디오에서 인간 신체 키포인트(관절) 검출로 골격 재구성.
- **응용:** 행동 인식, 동작 분석, 스포츠 분석, 감시.
- **키포인트 표준:** COCO(17개), OpenPose(25몸체+68얼굴+42손+6발).

### 2. 검출 접근법

- **Top-Down:** 사람 검출 → 크롭 → 개인별 자세 추정(AlphaPose, Mask R-CNN); 군중에서 확장성 저하.
- **Bottom-Up:** 모든 키포인트 검출 → 사람별 그룹화(OpenPose, HRNet, DeepCut); 다중인원 효율적.
- **2D vs 3D:** 2D(x,y 좌표) vs 3D(x,y,z); 3D는 깊이 추정 필요.

### 3. YOLOv8 Pose 구현

- **모델 변형:** yolov8n-pose.pt(나노), s/m/l/x 속도/정확도 트레이드오프.
- **실시간 파이프라인:** 웹캠 입력 → YOLO pose 추론 → 키포인트 시각화.
- **출력 형식:** 바운딩 박스 + 신뢰도 + 17 키포인트(x,y,정규화신뢰도).

### 4. 행동 인식 파이프라인

- **데이터 수집:** 13.2DPoseSave.py로 앉기/서기 비디오에서 키포인트 추출 → keypoints.csv.
- **데이터셋 준비:** 14.2DPoseDatasetSave.py로 레이블링된 dataset.csv(sitting=0, standing=1) 생성.
- **XGBoost 훈련:** 15.2DPoseTraining.py로 키포인트 특징 기반 이진 분류기 학습.

### 5. 자세 기반 분류

- **실시간 추론:** 16.2DPoseActionRecognition.py → YOLO pose → XGBoost 예측 → 레이블 오버레이.
- **성능:** 훈련/테스트 분할(80/20), 정확도 평가, modelweights.xgb로 모델 저장.
- **라이브러리:** ultralytics, xgboost, scikit-learn, pandas로 엔드투엔드 파이프라인.

### 6. 주요 장점

- **효율성:** Bottom-up 단일단계 처리로 Top-down 대비 군중 처리 우수.
- **실시간:** YOLOv8로 웹캠 추론 + 키포인트 추출 + 분류 가능.
- **확장성:** COCO wholebody 133+ 키포인트로 복잡 행동 분석 지원.

</details>
