<details>
<summary>ENG (English Version)</summary>

# Age & Gender Recognition

### 1. System Overview

- **Purpose:** Estimates age (0-100) and gender (Male/Female) from face images in real-time.
- **Face Detection:** OpenCV CascadeClassifier or Dlib get_frontal_face_detector.
- **Architecture:** WideResNet CNN (depth=16, width=8) pre-trained on IMDB-WIKI dataset.

### 2. Dataset and Models

- **IMDB-WIKI Dataset:** 500K+ face images with age/gender labels (IMDb: 460K, Wiki: 523K).
- **DEX Model:** Deep EXpectation (2016) predicts apparent age without facial landmarks.
- **Weights:** Pre-trained WideResNet-28-3.73.hdf5; HDF5 format for Keras/TensorFlow.

### 3. Wide Residual Network (WRN)

- **Improvements over ResNet:** Wider channels reduce gradient vanishing; excels on CIFAR/SVHN/COCO.
- **Architecture:** Conv1 → 3 residual layers → Global Avg Pool → Age(101)/Gender(2) classifiers.
- **Key Features:** Dropout (0.0-0.5), L2 regularization, He normal initialization.

### 4. FaceCV Implementation

- **Preprocessing:** Crop faces with 40px margin → resize to 64x64 → normalize.
- **Detection Pipeline:** Grayscale input → Dlib detector → multi-face batch prediction.
- **Visualization:** Bounding boxes + age/gender labels + real-time FPS overlay.

### 5. Real-time Processing

- **Video Input:** Webcam or sample video; ESC key to exit.
- **Batch Inference:** Multiple detected faces processed simultaneously for efficiency.
- **Performance:** FPS calculation; handles varying frame rates dynamically.

### 6. Code Structure

- **WideResNet Class:** Builds residual blocks with configurable depth/width.
- **FaceCV Class:** Orchestrates detection, cropping, prediction, visualization.
- **Dependencies:** Keras, TensorFlow, OpenCV, Dlib, argparse for CLI args.

### 7. Example Video

The following video demonstrates real-time face detection and age/gender prediction.

https://github.com/user-attachments/assets/e2d8fa68-9415-451c-af99-a8627e2ab1a2

The subject in the demo is a 34-year-old male. The model estimated the age close to the actual age and correctly predicted the gender.

</details>

<details>
<summary>KOR (한국어 버전)</summary>

# 연령 및 성별 인식

### 1. 시스템 개요

- **목적:** 얼굴 이미지에서 실시간 연령(0-100) 및 성별(Male/Female) 추정.
- **얼굴 검출:** OpenCV CascadeClassifier 또는 Dlib get_frontal_face_detector.
- **구조:** IMDB-WIKI 데이터셋 사전학습 WideResNet CNN(depth=16, width=8).

### 2. 데이터셋 및 모델

- **IMDB-WIKI 데이터셋:** 연령/성별 레이블 50만+ 얼굴 이미지(IMDb:46만, Wiki:52만).
- **DEX 모델:** Deep EXpectation(2016); 랜드마크 없이 겉보기 연령 예측.
- **가중치:** 사전학습 WideResNet-28-3.73.hdf5; Keras/TensorFlow용 HDF5 형식.

### 3. Wide Residual Network (WRN)

- **ResNet 개선:** 더 넓은 채널로 기울기 소실 감소; CIFAR/SVHN/COCO 우수 성능.
- **구조:** Conv1 → 3 잔차 층 → Global Avg Pool → 연령(101)/성별(2) 분류기.
- **특징:** Dropout(0.0-0.5), L2 정규화, He normal 초기화.

### 4. FaceCV 구현

- **전처리:** 40px 마진 크롭 → 64x64 리사이즈 → 정규화.
- **검출 파이프라인:** 그레이스케일 입력 → Dlib 검출기 → 다중 얼굴 배치 예측.
- **시각화:** 바운딩 박스 + 연령/성별 레이블 + 실시간 FPS 오버레이.

### 5. 실시간 처리

- **비디오 입력:** 웹캠 또는 샘플 비디오; ESC 키 종료.
- **배치 추론:** 검출된 다중 얼굴 동시 처리로 효율성 향상.
- **성능:** FPS 계산; 다양한 프레임 속도 동적 처리.

### 6. 코드 구조

- **WideResNet 클래스:** 구성가능 깊이/너비의 잔차 블록 구축.
- **FaceCV 클래스:** 검출, 크롭, 예측, 시각화 조율.
- **종속성:** Keras, TensorFlow, OpenCV, Dlib, CLI 인자용 argparse.

### 7. 예시 영상

아래 영상은 실시간 얼굴 검출 및 연령/성별 예측 결과를 보여줍니다.

https://github.com/user-attachments/assets/585eafd5-3036-4488-82a5-0197c58737eb

영상 속 인물의 실제 나이는 34세이며 남성입니다. 모델은 실제 나이와 비슷한 연령을 예측했으며, 성별은 정확하게 분류했습니다.

</details>
