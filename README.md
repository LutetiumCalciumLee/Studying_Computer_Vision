<details> <summary>ENG (English Version) </summary>

# Apple Detection Project

## Project Overview

This project is a system that automatically detects red apples in images and numbers them using computer vision technology. It combines HSV color space analysis with the Hough Circle Transform algorithm to achieve accurate apple detection.

### Key Features
- HSV color space-based color filtering
- Separation of red and green regions
- Circular object detection using Hough Circle Transform
- Korean path support (Windows environment)
- Real-time visualization and result display


## Technology Stack

| Technology | Version/Description |
|------|----------|
| **Python** | 3.12 |
| **OpenCV (cv2)** | Image processing and computer vision |
| **NumPy** | Array operations and numerical calculations |
| **OS** | File path management |


## Project Goals

1. Automatically detect red apples in images
2. Number and visualize detected apples
3. Accurate object separation through color-based filtering
4. Improve detection accuracy through noise removal and post-processing


## Algorithm Explanation

### Overall Process Flow

```
Input Image
    ↓
HSV Color Space Conversion
    ↓
    (Parallel Processing)
    - Red mask generation
    - Green mask generation
    ↓
Remove overlapping regions (extract pure red only)
    ↓
Gaussian Blur (noise removal)
    ↓
Hough Circle Transform (circular detection)
    ↓
Result Visualization (numbering)
```

### Step 1: HSV Color Space Conversion

```python
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
```

- **Purpose**: Convert to HSV color space, which is more suitable for color separation than RGB
- **Advantages**: Robust to lighting changes and suitable for color-based filtering

### Step 2: Red Region Mask Generation

Since red is located near 0° and 180° in the HSV color space, it is processed in two ranges:

```python
# Red range 1: 0° ~ 10°
lower_red1 = np.array([0, 100, 80])
upper_red1 = np.array([10, 255, 255])

# Red range 2: 165° ~ 180°
lower_red2 = np.array([165, 100, 80])
upper_red2 = np.array([180, 255, 255])
```

### Step 3: Green Region Mask Generation

Mask to remove green from apple leaves or background:

```python
lower_green = np.array([35, 40, 40])
upper_green = np.array([85, 255, 255])
```

### Step 4: Pure Red Extraction

Remove overlapping regions between red and green (e.g., where apples meet leaves) to keep only pure red:

```python
overlapping_mask = cv2.bitwise_and(red_mask, green_mask)
pure_red_mask = cv2.subtract(red_mask, overlapping_mask)
```

### Step 5: Noise Removal

Apply Gaussian blur to remove noise from the mask and smooth boundaries:

```python
pure_red_mask = cv2.GaussianBlur(pure_red_mask, (9, 9), 2, 2)
```

### Step 6: Circular Detection (Hough Circle Transform)

Detect circular objects from the pure red mask:

```python
circles = cv2.HoughCircles(
    pure_red_mask,
    cv2.HOUGH_GRADIENT,
    dp=1.1,          # Resolution ratio
    minDist=40,      # Minimum distance between circle centers
    param1=50,       # Upper threshold for Canny edge detection
    param2=25,       # Accumulator threshold
    minRadius=15,    # Minimum radius
    maxRadius=80     # Maximum radius
)
```

**Parameter Description:**
- `dp`: Resolution ratio of the accumulator array (smaller = more accurate but slower)
- `minDist`: Minimum distance between detected circle centers (prevents duplicate detection)
- `param1`: Upper threshold for Canny edge detector
- `param2`: Accumulator threshold for circle detection (lower = detects more circles)
- `minRadius/maxRadius`: Size range of circles to detect


## Code Structure

### Main Functions

#### `imread_korean(path)`
Function to read image files including Korean paths

```python
def imread_korean(path):
    """Function to read image files including Korean paths"""
    with open(path, "rb") as f:
        image_bytes = f.read()
    numpy_array = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(numpy_array, cv2.IMREAD_COLOR)
    return image
```

**Features:**
- Solves Korean path issues in Windows environment
- Reads files in binary mode and decodes with `cv2.imdecode()`

### Code Sections

1. **Initial Value Settings** (Lines 5-17)
   - Script directory path setup
   - Image file path configuration

2. **Image Reading** (Lines 19-35)
   - Korean path-supported image loading
   - Error handling

3. **HSV Conversion** (Lines 39-42)
   - BGR → HSV color space conversion

4. **Color Mask Generation** (Lines 44-66)
   - Red/green mask generation

5. **Mask Refinement** (Lines 68-78)
   - Remove overlapping regions
   - Noise removal

6. **Circular Detection** (Lines 80-93)
   - Apply Hough Circle Transform

7. **Result Visualization** (Lines 95-122)
   - Number detected apples
   - Draw blue borders and numbers

8. **Result Output** (Lines 124-134)
   - Display intermediate processes and final results in multiple windows


## Execution Method

### 1. Environment Setup

```bash
# Activate conda environment
conda activate vision_ai

# Install required packages (skip if already installed)
pip install opencv-python numpy
```

### 2. File Structure

```
Vision AI/
├── 00_Sample_Video/
│   └── input/
│       └── apple.png  # Input image
└── 04_Setting/
    └── 09_Apple_detection_test.py
```

### 3. Execution

```bash
python 04_Setting/09_Apple_detection_test.py
```

### 4. View Results

When the program runs, 5 windows will open:

1. **Original Image**: Original image
2. **Red Mask (Before)**: Initial red mask
3. **Green Mask**: Green mask
4. **Pure Red Mask (After)**: Pure red mask after removing green
5. **Detected Apples with Numbers**: Final detection result (with numbering)

Press any key to close all windows.


## Performance and Results

### Detection Accuracy
- Successfully detects red apples
- Successfully distinguishes from green leaves
- Accurately recognizes circular objects

### Output Example
```
Total 4 apples detected.
```

<img width="200" height="200" alt="image" src="https://github.com/user-attachments/assets/f1763e40-1407-41fa-ae6a-22be20a3101f" />
<img width="200" height="200" alt="image" src="https://github.com/user-attachments/assets/1e39c669-4552-46fc-8721-58c50ee2a3fe" />
<img width="200" height="200" alt="image" src="https://github.com/user-attachments/assets/7987c42e-6f45-4b15-a79c-32688727def4" />
<img width="200" height="200" alt="image" src="https://github.com/user-attachments/assets/f29b13cf-5b9d-4970-a7fa-28a58d485951" />
<img width="200" height="200" alt="image" src="https://github.com/user-attachments/assets/22330c25-b3b3-47f7-85fa-fc585b7e9a62" />

### Visualization Results
- Blue circular borders on each detected apple
- Numbers displayed at the center of apples (1, 2, 3, ...)


## Technical Features and Problems Solved

### 1. Korean Path Support
**Problem**: OpenCV's `cv2.imread()` does not support Korean paths on Windows

**Solution**: 
- Read files in binary mode and use `cv2.imdecode()`
- Wrapped in `imread_korean()` function for reusability

### 2. Color Separation Accuracy
**Problem**: False detection at areas where apples meet leaves

**Solution**:
- Find and remove the intersection of red and green masks
- Extract only pure red regions to improve detection accuracy

### 3. Noise Removal
**Problem**: Noise exists in mask after color filtering

**Solution**:
- Apply Gaussian blur (kernel size: 9x9)
- Improve circular detection accuracy with smooth boundaries



## Potential Improvements

### 1. Dynamic Parameter Adjustment
- Automatically adjust `minRadius`, `maxRadius` based on image size
- Automatically adjust HSV ranges based on lighting conditions

### 2. Multiple Color Support
- Current: Only detects red apples
- Improvement: Support various colors like yellow apples, green apples, etc.

### 3. Batch Processing
- Process multiple images instead of single image
- Automatic result image saving functionality

### 4. Performance Optimization
- Utilize GPU acceleration
- Parallel processing through multithreading

### 5. Deep Learning-based Detection
- Apply deep learning models like YOLO, Faster R-CNN
- Higher accuracy and detection of various objects


## Learning Points

What was learned through this project:

1. **Computer Vision Basics**
   - Color space conversion (BGR → HSV)
   - Mask operations and bitwise operations
   - Image filtering

2. **OpenCV Usage**
   - `cv2.inRange()`: Color range-based filtering
   - `cv2.HoughCircles()`: Circular object detection
   - `cv2.GaussianBlur()`: Noise removal

3. **Image Preprocessing**
   - Color-based segmentation
   - Noise removal techniques
   - Mask refinement methods

4. **Problem-solving Skills**
   - Korean path handling in Windows environment
   - Solving color overlap issues
   - Optimization through parameter tuning


## Developer Information

**Project**: Vision AI Practice  
**Filename**: `09_Apple_detection_test.py`  
**Development Environment**: Windows, Python, OpenCV, NumPy

</details>
<details open> <summary>KOR (한국어 버전) </summary>

# 사과 검출 프로젝트 (Apple Detection Project)

## 프로젝트 개요

이 프로젝트는 컴퓨터 비전 기술을 활용하여 이미지에서 빨간 사과를 자동으로 검출하고 번호를 매기는 시스템입니다. HSV 색상 공간 분석과 Hough Circle Transform 알고리즘을 결합하여 정확한 사과 검출을 구현했습니다.

### 주요 특징
- HSV 색상 공간 기반 색상 필터링
- 빨간색과 초록색 영역 분리 처리
- Hough Circle Transform을 이용한 원형 객체 검출
- 한글 경로 지원 (Windows 환경)
- 실시간 시각화 및 결과 표시


## 기술 스택

| 기술 | 버전/설명 |
|------|----------|
| **Python** | 3.12 |
| **OpenCV (cv2)** | 이미지 처리 및 컴퓨터 비전 |
| **NumPy** | 배열 연산 및 수치 계산 |
| **OS** | 파일 경로 관리 |


## 프로젝트 목표

1. 이미지에서 빨간 사과를 자동으로 검출
2. 검출된 사과에 번호를 매겨 시각화
3. 색상 기반 필터링을 통한 정확한 객체 분리
4. 노이즈 제거 및 후처리를 통한 검출 정확도 향상


## 알고리즘 설명

### 전체 프로세스 흐름

```
입력 이미지
    ↓
HSV 색상 공간 변환
    ↓
    (병렬 처리)
    - 빨간색 마스크 생성
    - 초록색 마스크 생성
    ↓
겹치는 영역 제거 (순수 빨간색만 추출)
    ↓
가우시안 블러 (노이즈 제거)
    ↓
Hough Circle Transform (원형 검출)
    ↓
결과 시각화 (번호 표시)
```

### 1단계: HSV 색상 공간 변환

```python
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
```

- **목적**: RGB보다 색상 분리가 용이한 HSV 색상 공간으로 변환
- **장점**: 조명 변화에 강건하고, 색상 기반 필터링에 적합

### 2단계: 빨간색 영역 마스크 생성

빨간색은 HSV 색상 공간에서 0°와 180° 근처에 위치하므로 두 개의 범위로 나누어 처리합니다:

```python
# 빨간색 범위 1: 0° ~ 10°
lower_red1 = np.array([0, 100, 80])
upper_red1 = np.array([10, 255, 255])

# 빨간색 범위 2: 165° ~ 180°
lower_red2 = np.array([165, 100, 80])
upper_red2 = np.array([180, 255, 255])
```

### 3단계: 초록색 영역 마스크 생성

사과의 잎이나 배경의 초록색을 제거하기 위한 마스크:

```python
lower_green = np.array([35, 40, 40])
upper_green = np.array([85, 255, 255])
```

### 4단계: 순수 빨간색 추출

빨간색과 초록색이 겹치는 영역(예: 사과와 잎이 만나는 부분)을 제거하여 순수한 빨간색만 남깁니다:

```python
overlapping_mask = cv2.bitwise_and(red_mask, green_mask)
pure_red_mask = cv2.subtract(red_mask, overlapping_mask)
```

### 5단계: 노이즈 제거

가우시안 블러를 적용하여 마스크의 노이즈를 제거하고 경계를 부드럽게 만듭니다:

```python
pure_red_mask = cv2.GaussianBlur(pure_red_mask, (9, 9), 2, 2)
```

### 6단계: 원형 검출 (Hough Circle Transform)

순수한 빨간색 마스크에서 원형 객체를 검출합니다:

```python
circles = cv2.HoughCircles(
    pure_red_mask,
    cv2.HOUGH_GRADIENT,
    dp=1.1,          # 해상도 비율
    minDist=40,      # 원 중심 간 최소 거리
    param1=50,       # Canny 엣지 상위 임계값
    param2=25,       # accumulator 임계값
    minRadius=15,    # 최소 반지름
    maxRadius=80     # 최대 반지름
)
```

**파라미터 설명:**
- `dp`: 누적 배열의 해상도 비율 (작을수록 정확하지만 느림)
- `minDist`: 검출된 원 중심 간 최소 거리 (중복 검출 방지)
- `param1`: Canny 엣지 검출기의 상위 임계값
- `param2`: 원 검출을 위한 누적 임계값 (낮을수록 더 많은 원 검출)
- `minRadius/maxRadius`: 검출할 원의 크기 범위


## 코드 구조

### 주요 함수

#### `imread_korean(path)`
한글 경로를 포함한 이미지 파일을 읽는 함수

```python
def imread_korean(path):
    """한글 경로를 포함한 이미지 파일을 읽는 함수"""
    with open(path, "rb") as f:
        image_bytes = f.read()
    numpy_array = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(numpy_array, cv2.IMREAD_COLOR)
    return image
```

**특징:**
- Windows 환경에서 한글 경로 문제 해결
- 바이너리 모드로 파일을 읽어 `cv2.imdecode()`로 디코딩

### 코드 섹션

1. **초기값 설정** (라인 5-17)
   - 스크립트 디렉토리 경로 설정
   - 이미지 파일 경로 구성

2. **이미지 읽기** (라인 19-35)
   - 한글 경로 지원 이미지 로딩
   - 에러 처리

3. **HSV 변환** (라인 39-42)
   - BGR → HSV 색상 공간 변환

4. **색상 마스크 생성** (라인 44-66)
   - 빨간색/초록색 마스크 생성

5. **마스크 정제** (라인 68-78)
   - 겹치는 영역 제거
   - 노이즈 제거

6. **원형 검출** (라인 80-93)
   - Hough Circle Transform 적용

7. **결과 시각화** (라인 95-122)
   - 검출된 사과에 번호 표시
   - 파란색 테두리 및 번호 그리기

8. **결과 출력** (라인 124-134)
   - 다중 윈도우로 중간 과정 및 최종 결과 표시


## 실행 방법

### 1. 환경 설정

```bash
# conda 환경 활성화
conda activate vision_ai

# 필요한 패키지 설치 (이미 설치되어 있다면 생략)
pip install opencv-python numpy
```

### 2. 파일 구조

```
Vision AI/
├── 00_Sample_Video/
│   └── input/
│       └── apple.png  # 입력 이미지
└── 04_Setting/
    └── 09_Apple_detection_test.py
```

### 3. 실행

```bash
python 04_Setting/09_Apple_detection_test.py
```

### 4. 결과 확인

프로그램 실행 시 다음 5개의 윈도우가 열립니다:

1. **Original Image**: 원본 이미지
2. **Red Mask (Before)**: 초기 빨간색 마스크
3. **Green Mask**: 초록색 마스크
4. **Pure Red Mask (After)**: 초록색 제거 후 순수 빨간색 마스크
5. **Detected Apples with Numbers**: 최종 검출 결과 (번호 표시)

아무 키나 누르면 모든 윈도우가 닫힙니다.


## 성능 및 결과

### 검출 정확도
- 빨간 사과 검출 성공
- 초록색 잎과의 구분 성공
- 원형 객체 정확히 인식

### 출력 예시
```
총 4개의 사과를 검출했습니다.
```

<img width="200" height="200" alt="image" src="https://github.com/user-attachments/assets/f1763e40-1407-41fa-ae6a-22be20a3101f" />
<img width="200" height="200" alt="image" src="https://github.com/user-attachments/assets/1e39c669-4552-46fc-8721-58c50ee2a3fe" />
<img width="200" height="200" alt="image" src="https://github.com/user-attachments/assets/7987c42e-6f45-4b15-a79c-32688727def4" />
<img width="200" height="200" alt="image" src="https://github.com/user-attachments/assets/f29b13cf-5b9d-4970-a7fa-28a58d485951" />
<img width="200" height="200" alt="image" src="https://github.com/user-attachments/assets/22330c25-b3b3-47f7-85fa-fc585b7e9a62" />


### 시각화 결과
- 검출된 각 사과에 파란색 원형 테두리 표시
- 사과 중심에 번호 표시 (1, 2, 3, ...)


## 기술적 특징 및 해결한 문제

### 1. 한글 경로 지원
**문제**: OpenCV의 `cv2.imread()`는 Windows에서 한글 경로를 지원하지 않음

**해결**: 
- 바이너리 모드로 파일을 읽은 후 `cv2.imdecode()` 사용
- `imread_korean()` 함수로 래핑하여 재사용성 향상

### 2. 색상 분리 정확도
**문제**: 사과와 잎이 만나는 부분에서 오검출 발생

**해결**:
- 빨간색과 초록색 마스크의 교집합을 찾아 제거
- 순수한 빨간색 영역만 추출하여 검출 정확도 향상

### 3. 노이즈 제거
**문제**: 색상 필터링 후 마스크에 노이즈 존재

**해결**:
- 가우시안 블러 적용 (커널 크기: 9x9)
- 부드러운 경계로 원형 검출 정확도 향상



## 개선 가능한 사항

### 1. 동적 파라미터 조정
- 이미지 크기에 따라 `minRadius`, `maxRadius` 자동 조정
- 조명 조건에 따른 HSV 범위 자동 조정

### 2. 다양한 색상 지원
- 현재: 빨간 사과만 검출
- 개선: 노란 사과, 초록 사과 등 다양한 색상 지원

### 3. 배치 처리
- 단일 이미지가 아닌 여러 이미지 일괄 처리
- 결과 이미지 자동 저장 기능

### 4. 성능 최적화
- GPU 가속 활용
- 멀티스레딩을 통한 병렬 처리

### 5. 딥러닝 기반 검출
- YOLO, Faster R-CNN 등 딥러닝 모델 적용
- 더 높은 정확도와 다양한 객체 검출 가능


## 학습 포인트

이 프로젝트를 통해 학습한 내용:

1. **컴퓨터 비전 기초**
   - 색상 공간 변환 (BGR → HSV)
   - 마스크 연산 및 비트 연산
   - 이미지 필터링

2. **OpenCV 활용**
   - `cv2.inRange()`: 색상 범위 기반 필터링
   - `cv2.HoughCircles()`: 원형 객체 검출
   - `cv2.GaussianBlur()`: 노이즈 제거

3. **이미지 전처리**
   - 색상 기반 세그멘테이션
   - 노이즈 제거 기법
   - 마스크 정제 방법

4. **문제 해결 능력**
   - Windows 환경에서의 한글 경로 처리
   - 색상 겹침 문제 해결
   - 파라미터 튜닝을 통한 최적화


## 개발자 정보

**프로젝트**: Vision AI 실습  
**파일명**: `09_Apple_detection_test.py`  
**개발 환경**: Windows, Python, OpenCV, NumPy
