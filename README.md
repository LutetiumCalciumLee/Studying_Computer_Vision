<details>
<summary>ENG (English Version)</summary>

# Monocular 3D Object Detection & BEV Visualization

> A computer vision practice project combining YOLOv8 detection and tracking with MobileNetV2-based 3D regression.

### 1. Project Overview

This project explores how objects detected in a single RGB video can be represented using 3D bounding boxes and a Bird’s Eye View (BEV).

Based on educational reference code, I organized the project structure, configured the execution environment, improved compatibility, and verified the main processing components.

- **Input:** A user-provided driving video.
- **Default detection target:** Cars.
- **Main outputs:** 3D bounding boxes, BEV visualization, tracking IDs, and an output video.

### 2. Processing Pipeline

```text
User-provided video
        ↓
YOLOv8 detection and tracking
        ↓
Object crop → Resize to 224 × 224
        ↓
MobileNetV2-based 3D regression
        ↓
Dimensions / Orientation / Orientation-bin confidence
        ↓
Camera projection and geometric calculations
        ↓
3D bounding boxes + BEV visualization
        ↓
Display and output video
```

### 3. Main Features

| Feature | Description |
|---|---|
| Vehicle detection | YOLOv8-based detection |
| Object tracking | Tracking IDs and recent trajectories |
| 3D regression | Dimensions, orientation, and bin confidence |
| Orientation representation | Six orientation bins |
| Geometry | KITTI-based camera projection matrix |
| Visualization | 3D bounding boxes and BEV |
| Video output | Processing FPS display and result recording |
| Training utilities | Early stopping, checkpointing, and TensorBoard |

The YOLO segmentation model supplies detection boxes and tracking information. Segmentation masks are not directly used in the 3D calculations.

### 4. My Contributions

- Organized source code, utility modules, models, and dataset paths.
- Configured the `vision_ai` conda environment.
- Resolved NumPy and Keras compatibility issues.
- Corrected backbone selection and added model-name validation.
- Improved TensorBoard logging and image loading for Korean file paths.
- Added validation for missing images, decoding failures, and invalid crop coordinates.
- Verified model loading, test inference, dataset batches, and logging.

### 5. Development Environment

| Component | Technology |
|---|---|
| Language | Python 3.12.12 |
| Environment | Conda `vision_ai` |
| Detection and tracking | Ultralytics YOLOv8 |
| 3D regression | TensorFlow 2.21.0 / Keras 3.15.1 |
| Default backbone | MobileNetV2 |
| Video processing | OpenCV 4.10.0 |
| Numerical processing | NumPy 1.26.4 |
| Visualization | Matplotlib |
| Training data | KITTI images and labels |
| Current inference configuration | CPU |

### 6. Project Structure

```text
Visoin_AI_Github_Project_1/
├── 12_3D object Detection/
│   ├── 12_3D_Object_Detection.py
│   ├── Train.py
│   └── libs/
│       ├── bbox3d_utils.py
│       └── Plotting.py
├── 00_Models/
│   ├── yolov8n-seg.pt
│   └── mobilenetv2/
│       ├── mobilenetv2_weights.keras
│       ├── mobilenetv2_training_history.csv
│       └── mobilenetv2_results_plot.png
├── 00_Dataset/
│   └── kitti/
│       ├── image_2/training/
│       └── label_2/
├── 00_Sample_Video/
│   ├── input/
│   └── output/
├── requirements.txt
└── README.md
```

### 7. How to Run

Activate the environment:

```powershell
conda activate vision_ai
```

For a new environment:

```powershell
conda create -n vision_ai python=3.12.12
conda activate vision_ai
python -m pip install -r requirements.txt
```

**Input videos are not included in this repository.**

Prepare your own driving video and place it at:

```text
00_Sample_Video/input/3d_drive.mp4
```

Prepare the required model weights and KITTI files before execution. The current inference script imports `Train.py`, which reads KITTI labels and initializes an ImageNet backbone.

Run inference from the project root:

```powershell
python "./12_3D object Detection/12_3D_Object_Detection.py"
```

Run training:

```powershell
python "./12_3D object Detection/Train.py"
```

The output video is saved to:

```text
00_Sample_Video/output/mobilenetv2_output_video.mp4
```

Press `q` to exit.

### 8. Verification Results

| Check | Result |
|---|---|
| Required library and utility imports | Passed |
| 3D model loading and dummy-input prediction | Passed |
| YOLO inference on a test image | Passed |
| Video first-frame reading | Passed |
| Supported backbone selection | Passed |
| Image-label filename matching | 7,481 pairs checked |
| Sampled object preprocessing | 64 samples passed |
| TensorFlow dataset batch generation | Passed |
| TensorBoard logging with a small model | Passed |

These checks confirm component behavior. Full video processing, complete 3D model training, benchmark accuracy, and real-time performance have not been verified.

### 9. Future Improvements

- Separate inference utilities from training initialization.
- Apply camera calibration corresponding to each input video.
- Review empirical angle and position corrections.
- Align training and inference preprocessing.
- Review validation splits at the image level.
- Evaluate 3D detection accuracy and processing speed.


### 10. Demo Video

https://github.com/user-attachments/assets/6ee32b18-d955-4867-b918-548e15e5c51a


</details>

<details>
<summary>KOR (한국어 버전)</summary>

# 단안 영상 기반 3D 객체 탐지 및 BEV 시각화

> YOLOv8의 객체 탐지·추적 결과와 MobileNetV2 기반 회귀 모델을 결합해 차량을 3D 바운딩 박스와 BEV로 표현하는 컴퓨터 비전 실습 프로젝트입니다.

### 1. 프로젝트 소개

2D 객체 탐지는 영상에서 객체가 차지하는 영역을 찾아냅니다. 이 프로젝트에서는 탐지된 차량 영역을 회귀 모델에 입력해 크기와 방향을 예측하고, 카메라 투영 관계를 이용해 3D 공간에 표현하는 과정을 다뤘습니다.

교육용 원본 코드를 바탕으로 프로젝트 구조와 실행 환경을 정리하고, 코드 호환성과 파일 경로 처리를 개선했습니다.

- **입력:** 사용자가 직접 준비한 주행 영상
- **기본 탐지 대상:** 차량(Car)
- **주요 출력:** 3D 바운딩 박스, BEV 화면, 객체 ID, 결과 영상
- **프로젝트 목표:** 2D 탐지 결과를 3D 표현으로 연결하는 처리 과정 이해

### 2. 처리 파이프라인

```text
사용자 입력 영상
      ↓
YOLOv8 객체 탐지 및 추적
      ↓
차량 영역 추출 및 224 × 224 크기 변환
      ↓
MobileNetV2 기반 3D 회귀 모델
      ↓
크기 / 방향 / 방향 구간 신뢰도 예측
      ↓
카메라 투영 행렬과 기하학적 계산
      ↓
3D 바운딩 박스 및 BEV 시각화
      ↓
화면 출력 및 결과 영상 저장
```

### 3. 주요 기능

#### 3.1 차량 탐지 및 추적

- `yolov8n-seg.pt`를 이용한 객체 탐지
- 차량 클래스 중심의 탐지 결과 필터링
- 객체 ID와 최근 이동 궤적 표시
- 탐지 신뢰도와 바운딩 박스 표시

현재 파이프라인에서는 Segmentation 모델의 탐지 박스와 추적 정보를 사용하며, 분할 마스크를 직접 이용해 3D 정보를 계산하지는 않습니다.

#### 3.2 다중 출력 회귀 모델

탐지된 차량 이미지를 입력으로 받아 세 종류의 값을 예측합니다.

| 출력 | 역할 |
|---|---|
| Dimension | 클래스 평균 크기에 더할 크기 보정값 예측 |
| Orientation | 방향 벡터 예측 |
| Confidence | 6개 방향 구간에 대한 신뢰도 예측 |

방향 정보는 6개 구간으로 표현하며, 방향 벡터와 구간 신뢰도를 조합해 객체의 회전 각도를 복원합니다.

#### 3.3 3D 및 BEV 시각화

- KITTI 기반 카메라 투영 행렬 `P2` 사용
- 객체 크기, 방향, 2D 박스를 이용한 기하학적 계산
- 영상 위에 3D 바운딩 박스 표시
- 위에서 내려다본 BEV 형태로 객체 배치 표현
- 처리 FPS 표시 및 결과 영상 저장

### 4. 수행한 작업

| 구분 | 수행 내용 |
|---|---|
| 프로젝트 구성 | 실행 코드, 유틸리티, 모델, 데이터셋 경로 정리 |
| 환경 구성 | Conda `vision_ai`, Python 3.12.12 환경 설정 |
| 의존성 관리 | 필요한 라이브러리 설치 및 버전 기록 |
| 호환성 개선 | NumPy 및 Keras import 경로 정리 |
| 모델 선택 개선 | 지원 백본 명시, 잘못된 모델 연결 수정 |
| 로그 처리 개선 | TensorBoard 로그를 임시 폴더에 저장하도록 변경 |
| 이미지 읽기 개선 | 한글 경로를 지원하는 파일 읽기·디코딩 적용 |
| 입력 검증 | 파일 누락, 디코딩 실패, 잘못된 크롭 좌표 검사 |
| 동작 확인 | 모델 로드, 테스트 추론, 데이터 배치 및 로그 생성 검증 |

### 5. 개발 환경

| 구분 | 사용 기술 및 설정 |
|---|---|
| Python | 3.12.12 |
| 가상환경 | Conda `vision_ai` |
| TensorFlow | 2.21.0 |
| Keras | 3.15.1 |
| 객체 탐지 | Ultralytics YOLOv8 |
| 기본 회귀 백본 | MobileNetV2 |
| 영상 처리 | OpenCV 4.10.0 |
| 수치 연산 | NumPy 1.26.4 |
| 시각화 | Matplotlib |
| 데이터 처리 | pandas, tqdm |
| 기타 의존성 | imgaug, lap, TensorBoard |
| 학습 데이터 | KITTI 이미지 및 라벨 |
| 현재 추론 설정 | CPU |

### 6. 프로젝트 구조

```text
Visoin_AI_Github_Project_1/
├── 12_3D object Detection/
│   ├── 12_3D_Object_Detection.py
│   ├── Train.py
│   └── libs/
│       ├── bbox3d_utils.py
│       └── Plotting.py
├── 00_Models/
│   ├── yolov8n-seg.pt
│   └── mobilenetv2/
│       ├── mobilenetv2_weights.keras
│       ├── mobilenetv2_training_history.csv
│       └── mobilenetv2_results_plot.png
├── 00_Dataset/
│   └── kitti/
│       ├── image_2/
│       │   └── training/
│       └── label_2/
├── 00_Sample_Video/
│   ├── input/
│   │   └── 3d_drive.mp4  # 사용자 직접 준비
│   └── output/
├── requirements.txt
└── README.md
```

데이터셋과 입력·출력 영상은 로컬에서 준비하는 실행 자원입니다.

### 7. 실행 방법

#### 7.1 환경 준비

기존 환경을 사용하는 경우:

```powershell
conda activate vision_ai
```

새 환경을 구성하는 경우:

```powershell
conda create -n vision_ai python=3.12.12
conda activate vision_ai
python -m pip install -r requirements.txt
```

#### 7.2 모델 및 데이터 준비

추론 전 아래 모델 파일을 준비합니다.

```text
00_Models/yolov8n-seg.pt
00_Models/mobilenetv2/mobilenetv2_weights.keras
```

KITTI 데이터는 다음 구조로 배치합니다.

```text
00_Dataset/kitti/image_2/training/
00_Dataset/kitti/label_2/
```

현재 추론 코드는 `Train.py`를 import하면서 KITTI 라벨을 읽고 ImageNet 백본을 초기화합니다. 따라서 추론 시에도 라벨 경로가 필요하며, 백본 가중치가 캐시에 없으면 다운로드가 발생할 수 있습니다.

#### 7.3 입력 영상 준비

**입력 영상은 저장소에 포함하지 않습니다. 직접 준비한 주행 영상으로 테스트해 주세요.**

영상을 아래 경로에 배치합니다.

```text
00_Sample_Video/input/3d_drive.mp4
```

다른 파일명을 사용하려면 `12_3D_Object_Detection.py`의 영상 로드 부분을 변경합니다.

```python
video = cv2.VideoCapture(
    os.path.join(video_in_dir, "사용할_영상.mp4")
)
```

폴더가 없다면 프로젝트 루트에서 생성합니다.

```powershell
New-Item -ItemType Directory -Force "00_Sample_Video/input"
New-Item -ItemType Directory -Force "00_Sample_Video/output"
```

#### 7.4 추론 실행

프로젝트 루트에서 실행합니다.

```powershell
python "./12_3D object Detection/12_3D_Object_Detection.py"
```

결과 영상은 다음 경로에 저장됩니다.

```text
00_Sample_Video/output/mobilenetv2_output_video.mp4
```

실행 중 `q` 키로 종료할 수 있습니다.

#### 7.5 학습 실행

학습 데이터와 라벨을 준비한 뒤 실행합니다.

```powershell
python "./12_3D object Detection/Train.py"
```

| 학습 항목 | 코드 설정 |
|---|---|
| 입력 크기 | 224 × 224 × 3 |
| Batch size | 8 |
| Optimizer | Adam |
| Learning rate | 0.00001 |
| 학습·검증 분할 | 유효 객체 기준 85% / 15% |
| 방향 구간 수 | 6 |
| 실행당 학습 범위 | 기존 기록 이후 최대 10 epoch 추가 |
| 학습 관리 | EarlyStopping, ModelCheckpoint, TensorBoard |

기존 가중치와 학습 기록이 있으면 이어서 학습하도록 구성되어 있습니다.

TensorBoard 로그의 실제 위치는 실행 시 출력되는 다음 메시지에서 확인할 수 있습니다.

```text
TensorBoard log directory: ...
```

### 8. 검증 결과

| 확인 항목 | 결과 |
|---|---|
| 필수 라이브러리 및 유틸리티 import | 통과 |
| 기존 3D 모델 로드 | 통과 |
| 더미 입력을 이용한 3D 예측 | 통과 |
| 테스트 이미지를 이용한 YOLO 추론 | 통과 |
| 입력 영상 첫 프레임 읽기 | 통과 |
| 지원 백본 선택 및 MobileNetV2 생성 | 통과 |
| 이미지·라벨 파일명 대응 검사 | 7,481쌍 확인 |
| 실제 객체 이미지 전처리 | 표본 64개 통과 |
| TensorFlow 학습 배치 생성 | 통과 |
| TensorBoard 학습·검증 로그 생성 | 소형 모델 테스트 통과 |

확인한 학습 배치의 형태는 다음과 같습니다.

```text
Input       : (8, 224, 224, 3)
Dimension   : (8, 3)
Orientation : (8, 6, 2)
Confidence  : (8, 6)
```

위 결과는 주요 구성 요소와 데이터 처리 경로에 대한 검증입니다. 전체 3D 모델의 학습 완료, 전체 영상 처리, KITTI 기준 정량 평가까지 확인한 결과는 아닙니다.

출력 영상에 설정된 15 FPS는 저장 영상의 재생 속도 설정이며, 측정된 추론 처리 속도를 의미하지 않습니다.

### 9. 한계 및 개선 방향

#### 카메라별 보정값 적용

현재는 고정된 KITTI 기반 투영 행렬을 사용합니다. 입력 영상의 카메라 조건에 따라 위치와 크기 추정 결과가 달라질 수 있어, 영상별 카메라 보정값을 적용하는 구조가 필요합니다.

#### 추론과 학습 초기화 분리

모델 복원에 필요한 사용자 정의 함수를 별도 모듈로 분리해, 추론 과정에서 학습 데이터와 백본 초기화에 대한 의존성을 줄일 계획입니다.

#### 전처리와 위치 보정 검토

학습과 추론의 전처리 방식을 일치시키고, 코드에 포함된 경험적 각도·위치 보정값의 영향을 검증할 필요가 있습니다.

#### 평가 방식 개선

- 이미지 단위 학습·검증 분리 검토
- 검증 데이터에 대한 3D 검출 정확도 측정
- 입력 영상별 평균 처리 FPS 측정
- 거리와 가림 정도에 따른 오차 비교
- 차량 외 클래스에 대한 적용 범위 검토

### 10. 시연 영상

https://github.com/user-attachments/assets/6ee32b18-d955-4867-b918-548e15e5c51a

</details>
