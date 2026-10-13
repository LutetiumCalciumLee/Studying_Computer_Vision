<details>
<summary>ENG (English Version)</summary>

# Human Pose Estimation: Sitting & Standing Classification

### 1. Project Overview

Developed a webcam application that distinguishes a person sitting on a chair from a person standing. The application combines a pretrained YOLOv8 Pose model with an XGBoost binary classifier and displays the predicted posture alongside the detected skeleton.

The initial exercise explored ten artillery hand signals. Small differences between similar arm poses were difficult to distinguish consistently, so the project scope was revised to two postures with clearer structural differences.

### 2. Technology Stack & Processing Pipeline

| Component | Technology / Role |
| --- | --- |
| Language | Python |
| Pose estimation | Ultralytics YOLOv8 Pose (`yolov8n-pose.pt`) |
| Posture classification | XGBoost (`binary:logistic`) |
| Webcam input and visualization | OpenCV |
| Dataset processing | NumPy, pandas |
| Training and evaluation | scikit-learn |

```text
Webcam → YOLOv8 Pose → 17 body keypoints → Coordinate normalization
       → XGBoost classifier → Sitting / Standing label
```

Used the pretrained pose model to extract body keypoints and trained a separate classifier for the two target postures.

### 3. Data Collection & Labeling

- Collected webcam images of sitting and standing postures and organized them by class.
- Assigned `sitting = 0` and `standing = 1`.
- Built a dataset containing **116 images: 60 sitting and 56 standing**.
- Used the 17 keypoints' x and y coordinates as **34 numerical features** per image.
- Stored data and model files under `pose_img/sitting_standing`.
- Implemented unique image filenames and incremental CSV saving for repeated capture sessions.
- Re-extracted keypoints from class-folder images when rebuilding the dataset, using folder-relative paths to distinguish identical filenames.

### 4. Feature Processing & Model Training

- Normalized keypoint coordinates relative to each person's bounding box, using the same preprocessing during training and inference.
- Encoded undetected or low-confidence keypoints as zero.
- Checked duplicate image paths, missing values, feature validity, and the presence of both classes before training.
- Split the dataset into **92 training images and 24 test images**, maintaining the class distribution with stratified sampling.
- Trained an XGBoost binary classifier with `binary:logistic` and `logloss`.
- Saved `model_weights.xgb`, the feature order and preprocessing metadata, and evaluation results for consistent inference.

### 5. Webcam Inference & Display

- Extracted body keypoints from incoming webcam frames and predicted `Sitting` or `Standing`.
- Displayed the skeleton, bounding box, and readable posture label.
- Mirrored the webcam image while rendering text in its original reading direction.
- Added a hardcoded `ZOOM_FACTOR` setting for display enlargement or reduction while preserving the frame dimensions.
- Applied mirroring and zoom to the display only, retaining the original image coordinates for classification.

### 6. Problems Addressed

| Problem | Implementation |
| --- | --- |
| Repeated capture overwrote images and CSV data | Used session-specific filenames and appended new CSV rows |
| Identical filenames across classes caused incorrect labels | Identified images by folder-relative path and assigned labels from their class folders |
| Cropped training images and webcam frames used different coordinate scales | Shared bounding-box-relative normalization between dataset preparation and inference |
| Mirroring reversed labels and confidence values | Mirrored the image and skeleton first, then rendered readable annotations |
| XGBoost model loading failed with a Korean Windows path | Read model bytes through Python and loaded them from a memory buffer |

### 7. Evaluation & Lessons Learned

| Metric | Recorded Result |
| --- | --- |
| Dataset size | 116 images |
| Training / test images | 92 / 24 |
| Correct test predictions | 24 / 24 |
| Test accuracy | 100% |

**Evaluation scope:** These results come from a random image split of webcam capture data. Adjacent frames from the same recording may appear in both sets, so this score does not establish accuracy for new people, camera positions, or recording conditions.

The project connected data capture, labeling, feature processing, classifier training, and webcam inference into one workflow. It also highlighted the importance of correct labels, consistent preprocessing, and a task scope appropriate to the available pose information.

Future improvements include evaluating separate recording sessions, collecting more varied camera views, and combining predictions across multiple frames to reduce unstable labels. Hips and knees should remain visible to provide useful evidence for sitting-versus-standing classification.

### 8. Demo Video

https://github.com/user-attachments/assets/3f8ff787-0993-4567-ae1a-4804891ebe41

</details>

<details>
<summary>KOR (한국어 버전)</summary>

# 인간 자세 추정 기반 앉은 자세·서 있는 자세 분류

### 1. 프로젝트 개요

웹캠 영상에서 **의자에 앉은 자세와 일어서 있는 자세를 구분하는 프로그램**을 구현했습니다. 사전 학습된 YOLOv8 Pose 모델로 신체 관절을 검출하고, XGBoost 이진 분류기로 자세를 판단해 골격과 함께 화면에 표시했습니다.

초기에는 포병 수신호 10가지를 분류하는 실습을 진행했습니다. 비슷한 팔 자세의 미세한 차이를 안정적으로 구분하는 데 한계가 있어, 신체 구조의 차이가 비교적 명확한 **앉음과 서 있음 두 자세로 목표를 조정**했습니다.

### 2. 사용 기술 및 처리 흐름

| 구분  | 사용 기술 및 역할 |
| --- | --- |
| 개발 언어 | Python |
| 자세 추정 | Ultralytics YOLOv8 Pose (`yolov8n-pose.pt`) |
| 자세 분류 | XGBoost (`binary:logistic`) |
| 웹캠 입력 및 시각화 | OpenCV |
| 데이터 처리 | NumPy, pandas |
| 학습 데이터 분할 및 평가 | scikit-learn |

```text
웹캠 입력 → YOLOv8 Pose → 신체 관절 17개 추출 → 좌표 정규화
          → XGBoost 분류 → Sitting / Standing 표시
```

사전 학습된 자세 추정 모델을 이용해 관절 정보를 얻고, 두 자세를 구분하는 분류기를 별도로 학습했습니다.

### 3. 데이터 수집 및 라벨링

- 웹캠으로 앉은 자세와 서 있는 자세를 촬영하고, 실제 자세에 따라 클래스 폴더로 분류했습니다.
- **앉음(`sitting`)은 0, 서 있음(`standing`)은 1**로 라벨을 지정했습니다.
- **앉음 60장, 서 있음 56장으로 총 116장의 데이터셋**을 구성했습니다.
- 신체 관절 17개의 x·y 좌표를 추출해 이미지당 **34개 수치 특징**을 사용했습니다.
- 데이터와 모델은 `pose_img/sitting_standing`에 저장했습니다.
- 반복 촬영 시 기존 자료가 덮어써지지 않도록 고유 파일명과 CSV 누적 저장을 적용했습니다.
- 데이터셋 생성 시 클래스 폴더의 이미지에서 관절을 다시 추출하고, 폴더를 포함한 상대 경로로 이미지를 식별했습니다.

### 4. 특징 처리 및 모델 학습

- 사람의 경계 상자를 기준으로 관절 위치를 정규화하고, 학습과 실시간 인식에서 같은 좌표 기준을 사용했습니다.
- 검출되지 않거나 신뢰도가 낮은 관절은 0으로 처리했습니다.
- 학습 전 중복 이미지 경로, 누락값, 좌표의 유효성, 두 클래스의 존재 여부를 검사했습니다.
- 클래스 비율을 유지하는 층화 분할을 적용해 **학습 92장, 테스트 24장**으로 나눴습니다.
- XGBoost의 **`binary:logistic` 이진 분류 모델**을 학습하고 `logloss`를 평가 지표로 설정했습니다.
- 학습된 `model_weights.xgb`와 특징 순서·전처리 정보를 저장해 인식 단계에서도 동일하게 사용했습니다.

### 5. 실시간 자세 인식 및 화면 표시

- 웹캠 프레임에서 관절을 추출하고, **`Sitting` 또는 `Standing`**을 예측하도록 구현했습니다.
- 신체 골격, 사람의 경계 상자, 자세 라벨을 화면에 표시했습니다.
- 웹캠 영상은 좌우 반전하되, 사람 이름·신뢰도 숫자·자세 라벨은 읽을 수 있는 방향으로 표시했습니다.
- 코드의 **`ZOOM_FACTOR` 값**으로 화면을 확대·축소하면서 원본 프레임 크기를 유지하도록 했습니다.
- 좌우 반전과 줌은 표시 화면에만 적용하고, 분류에는 원본 방향의 관절 좌표를 사용했습니다.

### 6. 문제 해결 과정

| 발생한 문제 | 적용한 해결 방법 |
| --- | --- |
| 재촬영 시 이미지와 CSV가 덮어써짐 | 촬영 세션별 고유 파일명과 CSV 누적 저장 적용 |
| 클래스 폴더 간 같은 파일명으로 라벨이 잘못 지정됨 | 폴더를 포함한 상대 경로로 식별하고 클래스 폴더 기준으로 라벨 지정 |
| 잘라 저장한 학습 이미지와 웹캠 영상의 좌표 크기가 다름 | 데이터셋 생성과 인식에 동일한 사람 경계 상자 기준 정규화 적용 |
| 좌우 반전 시 글자와 숫자까지 뒤집힘 | 영상·관절을 먼저 반전한 뒤 글자를 정방향으로 표시 |
| 한글 Windows 경로에서 XGBoost 모델 로딩 오류 발생 | Python으로 모델 파일을 읽고 메모리 버퍼로 전달해 로드 |

### 7. 평가 결과 및 배운 점

| 평가 항목 | 기록된 결과 |
| --- | --- |
| 전체 데이터 | 116장 |
| 학습 / 테스트 데이터 | 92장 / 24장 |
| 테스트 정답 수 | 24장 / 24장 |
| 테스트 정확도 | 100% |

**평가 범위:** 위 결과는 웹캠 촬영 이미지의 무작위 분할 결과입니다. 같은 촬영의 연속 프레임이 학습과 테스트에 함께 포함될 수 있으므로, 새로운 사람·카메라 위치·촬영 환경에서의 정확도를 보장하는 수치는 아닙니다.

이번 실습을 통해 **데이터 수집 → 라벨링 → 특징 처리 → 분류 모델 학습 → 웹캠 인식**의 전체 흐름을 구현했습니다. 특히 올바른 라벨 관리와 학습·인식 간 전처리 일치가 중요하다는 점을 확인했고, 관절 정보로 구분할 수 있는 범위를 고려해 프로젝트 목표를 조정했습니다.

향후에는 촬영 세션을 분리해 평가하고, 다양한 거리와 각도의 데이터를 추가하며, 여러 프레임의 예측을 모아 라벨이 흔들리는 현상을 줄일 계획입니다. 앉음과 서 있음을 구분할 수 있도록 골반과 무릎이 보이는 촬영 구도를 확보하는 것도 중요합니다.

### 8. 시연 영상

https://github.com/user-attachments/assets/82184883-860c-41c6-9f57-498b1e7938c9

</details>
