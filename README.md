<details> <summary>ENG (English Version)</summary>

# Facial Landmarks Outline Project

## Project Overview

This demo detects faces from a webcam feed in real time and draws outlines by connecting the 68 dlib facial landmark points. Each facial region is rendered with polylines so the shape is easy to inspect.

### Key Features
- Face landmark detection based on dlib 68-point model
- Region-wise polyline visualization (jaw, eyebrows, nose, eyes, mouth)
- Model path copy workaround for Korean paths on Windows
- Real-time webcam stream processing

## Technology Stack

| Technology | Version/Description |
|------|----------|
| **Python** | 3.12 |
| **OpenCV (cv2)** | Video capture and rendering |
| **dlib** | Face detection and landmark prediction |
| **imutils** | Convenience utilities (resize, transforms) |
| **NumPy** | Array operations |
| **OS/Tempfile/Shutil** | Path handling and temp copy |

## Project Goals

1. Detect faces and extract landmarks in real time
2. Draw region-wise outlines for intuitive visualization
3. Load the model reliably in Windows environments with Korean paths
4. Provide a demo that terminates with a simple key press

## Algorithm Explanation

### Overall Process Flow
```
Webcam frame
    ↓
Grayscale conversion
    ↓
dlib face detection
    ↓
Landmark prediction (68 points)
    ↓
Split points by facial region
    ↓
Draw outlines with cv2.polylines
```

### Step 1: Model prep and path handling
- Locate `shape_predictor_68_face_landmarks.dat` using the script path
- Copy the model to a temp directory to avoid Korean-path issues, then load

### Step 2: Video input
- Open webcam with `cv2.VideoCapture(0)`, exit with an error message on failure
- Wait 2 seconds, then start frame processing

### Step 3: Face detection and landmarks
- Resize frame to width 720 and flip horizontally for mirror mode
- Convert to grayscale and detect faces via `dlib.get_frontal_face_detector()`
- For each detected face, predict 68 landmark points with `shape_predictor`

### Step 4: Region-wise outline rendering
- Split jaw, eyebrows, nose, eyes, mouth indices via `face_utils.FACIAL_LANDMARKS_IDXS`
- Render each region with `cv2.polylines`, using distinct colors and closure flags

### Step 5: Interaction and exit
- Exit loop on `q` key
- Release capture and close all windows

## Code Structure

### Main functions/objects
- `dlib.get_frontal_face_detector()`: face detector
- `dlib.shape_predictor(temp_file)`: landmark predictor
- `face_utils.shape_to_np(shape)`: convert dlib shape to NumPy array
- `cv2.polylines(frame, pts, isClosed, color, thickness)`: draw region outlines

### Code sections

1. **Initial setup**  
   - Script/dataset paths, model existence check, temp copy
2. **Model load**  
   - Initialize face detector and landmark predictor
3. **Webcam prep**  
   - Open capture and warm up
4. **Realtime loop**  
   - Read frame, resize/flip, grayscale, detect faces, predict landmarks
5. **Outline rendering**  
   - Draw region-wise polylines (jaw, brows, nose, eyes, mouth)
6. **Exit handling**  
   - Break on `q`, release resources, close windows

## How to Run

### 1. Environment setup
```bash
conda activate vision_ai
pip install opencv-python dlib imutils numpy
```

### 2. File structure
```
Vision AI/
├── 00_Dataset/
│   └── shape_predictor_68_face_landmarks.dat
└── 04_Setting/
    └── 05_Face_Points_Lines.py
```

### 3. Execution
```bash
python 04_Setting/05_Face_Points_Lines.py
```

### 4. View results
- Check the `Facial Landmarks Outline` window for live outlines
- Press `q` to exit

## Performance and Results
<img width="640" height="480" alt="image" src="https://github.com/user-attachments/assets/cd7e8017-022f-4ecf-b310-7c052afe70b0" />

### Demo traits
- Lightweight processing with 720px width resizing
- Mirror mode for intuitive feedback
- Region-specific colors for clarity

### Example message
```
Copied model file to temp path: C:\...\shape_predictor_68_face_landmarks.dat
```

## Technical Notes and Issues Solved

### 1. Korean path support
**Problem**: Loading the dlib model directly from a Korean path can fail  
**Solution**: Copy to an English temp path (`tempfile.gettempdir()`) and load

### 2. Realtime robustness
**Problem**: Blank screen if capture fails or frame read fails  
**Solution**: Check `cap.isOpened()` and `ret`; exit with message on failure

### 3. Region index management
**Problem**: Hardcoding 68-point indices is error-prone  
**Solution**: Use `face_utils.FACIAL_LANDMARKS_IDXS` for automatic mapping

## Possible Improvements

- Show FPS and performance logs
- Different colors/thickness per face when multiple faces appear
- Preprocessing for landmark stability (blur, histogram equalization, etc.)
- ROI cropping for speed optimization

## Learning Points

1. dlib face detection and 68-point landmark pipeline
2. Region-wise visualization with OpenCV polylines
3. Korean-path handling via temp directory copy on Windows
4. Realtime streaming loop structure and exit handling

## Developer Info

**Project**: Vision AI Practice  
**Filename**: `05_Face_Points_Lines.py`  
**Environment**: Windows, Python, OpenCV, dlib, NumPy, imutils

</details>

<details open> <summary>KOR (한국어 버전)</summary>

# 얼굴 랜드마크 윤곽선 프로젝트 (Facial Landmarks Outline)

## 프로젝트 개요

웹캠으로 얼굴을 실시간 감지하고 dlib 68개의 점 랜드마크를 연결해 윤곽선을 그리는 예제입니다. 각 부위별 폴리라인을 그려 얼굴 형태를 시각적으로 확인할 수 있습니다.

### 주요 특징
- dlib 68 포인트 기반 얼굴 랜드마크 감지
- 부위별 폴리라인 시각화(턱, 눈썹, 코, 눈, 입)
- 한글 경로 모델 파일 복사 처리(Windows)
- 실시간 웹캠 스트림 처리

## 기술 스택

| 기술 | 버전/설명 |
|------|----------|
| **Python** | 3.12 |
| **OpenCV (cv2)** | 영상 캡처 및 렌더링 |
| **dlib** | 얼굴 감지 및 랜드마크 예측 |
| **imutils** | 편의 유틸(리사이즈, 변환 등) |
| **NumPy** | 배열 연산 |
| **OS/Tempfile/Shutil** | 파일 경로 및 임시 복사 처리 |

## 프로젝트 목표

1. 실시간 얼굴 감지 및 랜드마크 추출
2. 부위별 윤곽선을 그려 직관적 시각화
3. Windows 한글 경로 환경에서도 안정적 모델 로딩
4. 간단한 키 입력으로 종료 가능한 데모 제공

## 알고리즘 설명

### 전체 프로세스 흐름
```
웹캠 프레임
    ↓
그레이스케일 변환
    ↓
dlib 얼굴 감지
    ↓
랜드마크 예측 (68점)
    ↓
부위별 포인트 분리
    ↓
cv2.polylines로 윤곽선 시각화
```

### 1단계: 모델 준비 및 경로 처리
- 스크립트 경로 기반으로 `shape_predictor_68_face_landmarks.dat` 위치 확인
- 한글 경로 문제를 피하기 위해 임시 디렉터리로 모델 복사 후 로드

### 2단계: 영상 입력
- `cv2.VideoCapture(0)`으로 웹캠 오픈, 실패 시 오류 출력 후 종료
- 2초 대기 후 프레임 스트림 처리 시작

### 3단계: 얼굴 감지와 랜드마크
- 프레임 리사이즈(폭 720) 후 좌우 반전으로 거울 모드
- 그레이스케일 변환 후 `dlib.get_frontal_face_detector()`로 얼굴 검출
- 검출된 얼굴마다 `shape_predictor`로 68점 좌표 추출

### 4단계: 부위별 윤곽선 시각화
- 턱, 눈썹, 코, 눈, 입 구간 인덱스를 `face_utils.FACIAL_LANDMARKS_IDXS`로 분리
- `cv2.polylines`로 각 부위를 서로 다른 색상과 닫힘 옵션으로 렌더링

### 5단계: 인터랙션 및 종료
- `q` 키 입력 시 루프 종료
- 캡처 자원 해제 후 모든 창 종료

## 코드 구조

### 주요 함수/객체
- `dlib.get_frontal_face_detector()`: 얼굴 감지기
- `dlib.shape_predictor(temp_file)`: 랜드마크 예측기
- `face_utils.shape_to_np(shape)`: dlib shape를 NumPy 배열로 변환
- `cv2.polylines(frame, pts, isClosed, color, thickness)`: 부위별 윤곽선 그리기

### 코드 섹션

1. **초기값 설정**  
   - 스크립트/데이터셋 경로, 모델 파일 존재 여부 확인 및 임시 복사
2. **모델 로드**  
   - 얼굴 감지기와 랜드마크 예측기 초기화
3. **웹캠 준비**  
   - 캡처 열기 및 예열 대기
4. **실시간 처리 루프**  
   - 프레임 읽기, 리사이즈/반전, 그레이 변환, 얼굴 감지, 랜드마크 예측
5. **윤곽선 시각화**  
   - 부위별 폴리라인 그리기(턱, 눈썹, 코, 눈, 입)
6. **종료 처리**  
   - `q` 키로 루프 탈출, 자원 해제 및 창 종료

## 실행 방법

### 1. 환경 설정
```bash
conda activate vision_ai
pip install opencv-python dlib imutils numpy
```

### 2. 파일 구조
```
Vision AI/
├── 00_Dataset/
│   └── shape_predictor_68_face_landmarks.dat
└── 04_Setting/
    └── 05_Face_Points_Lines.py
```

### 3. 실행
```bash
python 04_Setting/05_Face_Points_Lines.py
```

### 4. 결과 확인
- `Facial Landmarks Outline` 창에서 얼굴 윤곽선을 실시간 확인
- `q` 키 입력 시 종료

## 성능 및 결과
<img width="640" height="480" alt="image" src="https://github.com/user-attachments/assets/535a7280-6c57-4e4c-b606-ee68869324c3" />

### 시연 특징
- 웹캠 해상도 720px 리사이즈로 경량 처리
- 거울 모드로 직관적 피드백
- 부위별 색상 차별화로 식별 용이

### 예시 메시지
```
모델 파일을 임시 경로로 복사했습니다: C:\...\shape_predictor_68_face_landmarks.dat
```

## 기술적 특징 및 해결한 문제

### 1. 한글 경로 지원
**문제**: dlib 모델을 한글 경로에서 직접 로드 시 실패 가능  
**해결**: 임시 영문 경로(`tempfile.gettempdir()`)로 복사 후 로드

### 2. 실시간 안정성
**문제**: 캡처 실패나 프레임 미수신 시 빈 화면 발생  
**해결**: `cap.isOpened()` 체크 및 `ret` 검증, 실패 시 메시지 후 종료

### 3. 부위별 구간 분리
**문제**: 68점 인덱스를 일일이 하드코딩 시 유지보수 어려움  
**해결**: `face_utils.FACIAL_LANDMARKS_IDXS`로 인덱스 자동 매핑

## 개선 가능한 사항

- FPS 표시 및 성능 로깅
- 다중 얼굴 렌더링 시 색상/두께 차등 적용
- 랜드마크 추적 안정화를 위한 전처리(블러, 히스토그램 평활화 등)
- ROI 크롭 후 처리로 속도 최적화

## 학습 포인트

1. dlib 얼굴 감지와 68점 랜드마크 파이프라인
2. OpenCV 폴리라인을 활용한 부위별 시각화
3. Windows 한글 경로 대응 기법(temp 디렉터리 복사)
4. 실시간 스트림 처리 루프 구조와 종료 핸들링

## 개발자 정보

**프로젝트**: Vision AI 실습  
**파일명**: `05_Face_Points_Lines.py`  
**개발 환경**: Windows, Python, OpenCV, dlib, NumPy, imutils

</details>

