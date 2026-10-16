<details>
<summary>ENG (English Version)</summary>

# Road Segmentation & Object Detection

**Pedestrian Danger Warning Using Road Segmentation and Crosswalk Estimation**

### 1. OpenVINO Toolkit Overview

- **Project Goal:** Integrate road segmentation and object detection to identify pedestrians on the roadway outside estimated crosswalk regions.
- **OpenVINO Application:** Ran the pretrained `road-segmentation-adas-0001` model and generate pixel-level road labels.
- **Development Environment:** Python 3.12.12, Conda `vision_ai`, OpenVINO 2024.5.0, PyTorch, Ultralytics, OpenCV, and NumPy.
- **Environment Troubleshooting:** Resolved missing OpenVINO dependencies in the active Conda environment and prepared both XML and BIN model files at the expected paths.

### 2. ADAS Applications

- **Application:** Developed a camera-based road perception and pedestrian warning prototype for an ADAS-related scenario.
- **Scene Understanding:** Combined object locations with road segmentation to distinguish pedestrians on the roadway, inside estimated crosswalks, or outside the roadway.
- **Danger Warning:** Displayed a red `DANGER` caption when at least one pedestrian met the roadway-outside-crosswalk condition.
- **Scope:** The prototype uses visual location rules; traffic signals, vehicle distance, speed, and collision prediction are outside the current implementation.

### 3. Model Integration

- **YOLOv8 Detection:** Used `yolov8n.pt` to detect objects such as people and vehicles, with CUDA FP16 inference on an NVIDIA RTX 3060 Laptop GPU.
- **Road Segmentation:** Used the supplied FP16-INT8 OpenVINO model on Intel Iris Xe Graphics. Its four classes are background, road, curb, and road marking.
- **Model Input:** Used 640×640 frames for YOLO and output visualization; internally resized segmentation inputs to 512×896.
- **Crosswalk Estimation:** Since the integrated models do not provide a dedicated crosswalk class, combined white stripe geometry with road marking labels.
- **Implementation Scope:** Integrated pretrained models and implemented preprocessing, postprocessing, and warning logic without training a new model.

### 4. Processing Pipeline

- **Input Handling:** Resized video frames to 640×640 and ran road segmentation and object detection on every input frame.
- **Parallel Inference:** Started OpenVINO segmentation asynchronously on the Intel GPU while YOLO processed the frame on the NVIDIA GPU.
- **Crosswalk Regions:** Grouped at least three compatible white stripes using direction, length, and spacing conditions; included the gaps between stripes and tracked validated regions for up to 20 frames using optical flow.
- **Pedestrian Ground Assessment:** Evaluated `person` detections with confidence ≥0.5. Treated the ground below the bounding box as roadway when road or road marking pixels occupied at least 60% of that area; excluded estimated crosswalk regions from the warning condition.
- **Output Visualization:** Detected objects before applying the road overlay, then drew boxes, processing FPS, and conditional `DANGER` captions. Crosswalk estimates are used internally without separate annotations; clipped or insufficiently sized boxes are treated as unknown.

### 5. Real-time Performance

- **GPU Comparison:** Changing road segmentation from CPU to Intel GPU increased average processing speed from 10.47 FPS to 13.41 FPS, approximately **28.2%**, while YOLO remained on NVIDIA CUDA FP16.
- **Measurement Conditions:** Measured 80 frames after 10 warmup frames, with display and video saving disabled. This comparison was performed before adding the `DANGER` feature.
- **Full Video Validation:** Processed all 1,364 input frames and decoded all 1,364 saved frames successfully, confirming 640×640 output and 30 FPS playback.
- **Behavior Tests:** Passed 13 automated tests using synthetic inputs for pedestrian ground assessment, crosswalk geometry, and tracking behavior.
- **Metric Interpretation:** Playback FPS differs from processing FPS. These functional tests do not establish detection accuracy against real-world ground truth.

### 6. Optimization Techniques

- **Asynchronous Processing:** Overlapped Intel GPU segmentation with NVIDIA GPU object detection to reduce sequential waiting.
- **Graph-based Postprocessing:** Selected the highest-scoring class and performed input type/layout conversions within the OpenVINO graph to reduce CPU processing.
- **Vectorized Overlay:** Replaced repeated class-wise color operations with NumPy palette indexing.
- **Precision and Thread Settings:** Used the supplied FP16-INT8 segmentation model, enabled YOLO CUDA FP16, and adjusted OpenCV/PyTorch CPU thread usage.
- **Future Improvements:** Evaluate a dedicated crosswalk model, improve foot localization and warning stability, and measure false alarms and missed warnings using labeled videos.

### 7. Demo Video

https://github.com/user-attachments/assets/07299e0a-910c-4d1f-a88d-2e0d1204eace

</details>

<details>
<summary>KOR (한국어 버전)</summary>

# 도로 분할 및 객체 탐지

**도로 분할과 횡단보도 추정을 활용한 보행자 위험 경고 시스템**

### 1. OpenVINO 툴킷 개요

- **프로젝트 목표:** 도로 분할과 객체 탐지를 통합하여, 횡단보도로 추정되는 영역 밖의 차도에 있는 보행자를 판단합니다.
- **OpenVINO 활용:** 사전학습된 `road-segmentation-adas-0001` 모델을 실행하고 픽셀 단위 도로 분류 결과를 생성합니다.
- **개발 환경:** Python 3.12.12, Conda `vision_ai`, OpenVINO 2024.5.0, PyTorch, Ultralytics, OpenCV, NumPy를 사용했습니다.
- **환경 문제 해결:** 실제 실행 중인 Conda 환경에 OpenVINO를 설치하고, 모델 XML·BIN 파일을 기본 경로에 준비하여 라이브러리 및 모델 경로 오류를 해결했습니다.

### 2. ADAS 응용

- **적용 분야:** ADAS 관련 상황을 대상으로 카메라 기반 도로 인지 및 보행자 경고 프로토타입을 구현했습니다.
- **주변 상황 분석:** 객체 위치와 도로 분할 결과를 결합하여 보행자를 차도 위, 횡단보도 추정 영역 안, 차도 밖으로 구분합니다.
- **위험 경고:** 횡단보도 추정 영역 밖 차도의 보행자가 한 명 이상 있으면 빨간색 `DANGER` 자막을 표시합니다.
- **구현 범위:** 영상 속 위치 관계를 이용해 판단하며, 신호등 상태·차량 거리·속도·충돌 예측은 현재 구현 범위에 포함하지 않습니다.

### 3. 모델 통합

- **YOLOv8 탐지:** `yolov8n.pt`로 사람과 차량 등의 객체를 탐지하고, NVIDIA RTX 3060 Laptop GPU에서 CUDA FP16 추론을 수행합니다.
- **도로 분할:** 사전 제공된 FP16-INT8 OpenVINO 모델을 Intel Iris Xe GPU에서 실행합니다. 분류 결과는 배경·도로·연석·도로 표식의 4개 클래스입니다.
- **모델 입력:** YOLO 입력과 결과 영상은 640×640을 유지하고, 도로 분할 모델에는 내부적으로 512×896 크기의 입력을 전달합니다.
- **횡단보도 추정:** 사용 모델에 별도 횡단보도 클래스가 없어 흰 줄무늬의 기하 조건과 도로 표식 분류 결과를 결합했습니다.
- **개발 범위:** 사전학습 모델을 통합하고 전처리·후처리·경고 판단 로직을 구현했으며, 별도의 모델 학습은 수행하지 않았습니다.

### 4. 처리 파이프라인

- **입력 처리:** 영상을 640×640으로 조정하고 모든 입력 프레임에서 도로 분할과 객체 탐지를 수행합니다.
- **병렬 추론:** Intel GPU에서 OpenVINO 분할을 비동기로 시작한 뒤, NVIDIA GPU에서 같은 프레임의 YOLO 추론을 수행합니다.
- **횡단보도 영역:** 방향·길이·간격 조건을 만족하는 흰 줄무늬를 최소 3개 묶고, 줄 사이 공간까지 포함합니다. 광학 흐름과 영역 검증을 통과한 경우에만 최대 20프레임 동안 추적합니다.
- **보행자 지면 판단:** 신뢰도 0.5 이상인 `person`을 대상으로 박스 하단의 지면을 분석합니다. 도로·도로 표식 픽셀이 60% 이상이면 차도로 판단하고, 횡단보도 추정 영역 안의 보행자는 경고 조건에서 제외합니다.
- **결과 시각화:** 도로 오버레이 적용 전에 객체를 탐지하고, 결과 영상에 객체 박스·처리 FPS·조건부 `DANGER` 자막을 표시합니다. 횡단보도 영역은 별도 표시하지 않으며, 화면 경계에서 잘리거나 크기가 부족한 박스는 판단 불가로 처리합니다.

### 5. 실시간 성능

- **GPU 전환 비교:** YOLO의 NVIDIA CUDA FP16 설정을 유지한 상태에서 도로 분할을 CPU에서 Intel GPU로 변경하여, 평균 처리 속도를 10.47 FPS에서 13.41 FPS로 약 **28.2% 개선**했습니다.
- **측정 조건:** 워밍업 10프레임을 제외한 80프레임을 측정했으며, 화면 표시와 영상 저장을 비활성화했습니다. 이 비교는 `DANGER` 기능 추가 전에 수행했습니다.
- **전체 영상 검증:** 입력 영상 1,364프레임을 모두 처리하고 저장 영상 1,364프레임을 다시 읽어, 640×640 해상도와 30 FPS 재생을 확인했습니다.
- **기능 테스트:** 합성 입력을 이용한 보행자 지면 판단, 횡단보도 줄무늬 및 추적 관련 자동 테스트 13개를 통과했습니다.
- **지표 해석:** 영상 재생 FPS와 실제 처리 FPS는 구분했습니다. 기능 테스트 결과는 실제 정답 데이터에 대한 탐지 정확도를 의미하지 않습니다.

### 6. 최적화 기법

- **비동기 처리:** Intel GPU의 도로 분할과 NVIDIA GPU의 객체 탐지를 겹쳐 실행하여 순차 처리의 대기 시간을 줄였습니다.
- **그래프 내부 후처리:** OpenVINO 그래프에서 최대 점수 클래스 선택과 입력 자료형·레이아웃 변환을 수행하여 CPU 처리 부담을 줄였습니다.
- **오버레이 벡터화:** 클래스별 색상 처리 반복을 NumPy 팔레트 인덱싱으로 변경했습니다.
- **정밀도 및 스레드 설정:** 사전 제공된 FP16-INT8 분할 모델과 YOLO CUDA FP16을 활용하고, OpenCV·PyTorch의 CPU 스레드 사용을 조정했습니다.
- **향후 개선:** 횡단보도 전용 모델 도입, 발 위치 추정과 경고 안정성 개선, 정답 영상 기반 오경고·미경고 평가를 진행하고자 합니다.

### 7. 시연 영상

https://github.com/user-attachments/assets/07299e0a-910c-4d1f-a88d-2e0d1204eace

</details>
