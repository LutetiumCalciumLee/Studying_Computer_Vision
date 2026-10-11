<details>
<summary>ENG (English Version)</summary>

# Real-Time Open-Vocabulary Object Detection

### 1. Environment Setup

- **GPU Requirements:** NVIDIA GeForce RTX 3060 Laptop GPU; verifies GPU availability with `torch.cuda.is_available()`.
- **Python Environment:** Windows, Conda `vision_ai`, Python 3.12.12.
- **PyTorch + CUDA:** Uses `torch 2.10.0+cu126`, a CUDA 12.6 build, for GPU inference; supports automatic device selection, `--device cuda`, and `--device cpu`.

### 2. YOLOv8 Foundation

- **Ultralytics Installation:** `pip install ultralytics`; basic detection workflow using `yolov8n.pt`.
- **Basic Detection:** Webcam input → GPU inference → frames annotated with bounding boxes and confidence scores.
- **Custom Dataset Pipeline:** Learning workflow: `labelme` → `labelme2yolo` → `dataset.yaml` → training with `yolov8n.pt`.

### 3. Open-Vocabulary Detection (OVOD)

- **YOLO-World Architecture:** CLIP text encoder → text–image feature fusion → object detection based on user prompts.
- **Zero-Shot Capability:** Detects objects specified by English names, such as `person`, `cup`, and `dog`, without retraining for each vocabulary change.
- **Prompt-then-Detect:** Applies target labels with `set_classes()`; supports changing detection targets through the web interface.

### 4. YOLO-World Implementation

- **Model Loading:** `YOLOWorld('00_Models/yolov8x-worldv2.pt')` → `set_classes(['person', 'cup'])` → webcam inference.
- **DETR Comparison:** Uses `facebook/detr-resnet-50` as a fixed-COCO-class comparison; optimizes both webcam and web scripts with smaller inputs, CUDA FP16, and GPU preprocessing where supported.
- **Performance:** DETR frame processing improved from approximately 5.0 FPS to 9.9 FPS at the default 480px setting, and 15.6 FPS at 320px.
- **Execution Structure:** Four standalone Python scripts for YOLO-World/DETR webcam and web applications, with `00_Models` and Git LFS configuration prepared for GitHub upload.

*Performance was measured on the RTX 3060 Laptop GPU using 640×480 synthetic frames, four warm-up iterations, and the median of 12 CUDA-synchronized processing times. Camera capture, JPEG encoding, and HTTP transmission are excluded; smaller inputs may reduce small-object accuracy.*

### 5. Web Application

- **Flask Backend:** `/video_feed` streams MJPEG; `/update_classes` changes target labels; `/upload_image` returns annotated images.
- **Frontend:** Embedded HTML, CSS, and JavaScript provide class input, webcam start/stop, and image upload.
- **Local Execution:** YOLO-World at `http://127.0.0.1:5000`; DETR at `http://127.0.0.1:5001`. DETR accepts only supported COCO labels.

### 6. Key Advantages

- **Flexibility:** Changes YOLO-World detection targets through user-defined text without model retraining.
- **Efficiency:** Uses GPU inference and DETR input-size/mixed-precision optimization to improve frame-processing speed.
- **Versatility:** Supports webcam and uploaded-image detection; standalone scripts and script-relative model paths simplify execution in other environments.

### 7. Demo Video

- Open Vocabulary Detection ResNet  


https://github.com/user-attachments/assets/79523ff9-c2e6-4769-b90c-dd43c5b4f988


- Open Vocabulary Detection YOLO World  


https://github.com/user-attachments/assets/53078da6-49da-4fd5-a7a9-583728952220


</details>

<details>
<summary>KOR (한국어 버전)</summary>

# 실시간 Open Vocabulary 객체 탐지

### 1. 환경 구축

- **GPU 요구사항:** NVIDIA GeForce RTX 3060 Laptop GPU 사용; `torch.cuda.is_available()`로 GPU 사용 가능 여부 확인.
- **Python 환경:** Windows, Conda `vision_ai`, Python 3.12.12.
- **PyTorch + CUDA:** CUDA 12.6 지원 빌드인 `torch 2.10.0+cu126`으로 GPU 추론; 장치 자동 선택, `--device cuda`, `--device cpu` 지원.

### 2. YOLOv8 기반

- **Ultralytics 설치:** `pip install ultralytics`; `yolov8n.pt`를 활용한 기본 객체 탐지 흐름.
- **기본 탐지:** 웹캠 입력 → GPU 추론 → 탐지 박스와 신뢰도를 표시한 결과 프레임.
- **커스텀 데이터셋:** 학습 흐름 정리: `labelme` → `labelme2yolo` → `dataset.yaml` → `yolov8n.pt` 훈련.

### 3. 개방형 어휘 탐지 (OVOD)

- **YOLO-World 구조:** CLIP 텍스트 인코더 → 텍스트·이미지 특징 융합 → 사용자 프롬프트에 따른 객체 탐지.
- **제로샷 기능:** `person`, `cup`, `dog` 등의 영어 객체 이름을 지정하고, 어휘 변경마다 재학습 없이 탐지.
- **프롬프트-탐지:** `set_classes()`로 탐지 대상을 적용하고 웹 인터페이스에서 변경.

### 4. YOLO-World 구현

- **모델 로딩:** `YOLOWorld('00_Models/yolov8x-worldv2.pt')` → `set_classes(['person', 'cup'])` → 웹캠 추론.
- **DETR 비교:** COCO 고정 클래스 모델인 `facebook/detr-resnet-50`을 비교 대상으로 구성; 웹캠·웹용 파일 모두 입력 크기 축소, CUDA FP16, 지원되는 GPU 전처리 적용.
- **성능:** DETR 프레임 처리 성능을 약 5.0 FPS에서 기본 480px 설정의 9.9 FPS, 속도 우선 320px 설정의 15.6 FPS로 개선.
- **실행 구성:** YOLO-World·DETR 각각 웹캠용·웹용으로 총 네 Python 파일 독립 실행; `00_Models`와 Git LFS 설정을 포함해 GitHub 업로드 준비.

*성능은 RTX 3060 Laptop GPU에서 640×480 합성 프레임으로 워밍업 4회 후 12회 처리 시간의 중앙값을 사용하고, CUDA 연산 완료까지 동기화해 측정했습니다. 카메라 캡처·JPEG 인코딩·HTTP 전송은 제외하며, 입력 축소 시 작은 객체의 탐지 정확도가 낮아질 수 있습니다.*

### 5. 웹 애플리케이션

- **Flask 백엔드:** `/video_feed` MJPEG 스트리밍; `/update_classes` 탐지 클래스 변경; `/upload_image` 탐지 결과 이미지 반환.
- **프론트엔드:** HTML·CSS·JavaScript 내장; 클래스 입력, 웹캠 시작·중지, 이미지 업로드 기능 제공.
- **로컬 실행:** YOLO-World는 `http://127.0.0.1:5000`, DETR는 `http://127.0.0.1:5001`로 접속. DETR는 지원하는 COCO 클래스만 입력 가능.

### 6. 주요 장점

- **유연성:** 모델 재학습 없이 사용자 입력 텍스트로 YOLO-World 탐지 대상 변경.
- **효율성:** GPU 추론과 DETR 입력 크기·혼합 정밀도 최적화로 프레임 처리 속도 개선.
- **다재다능:** 웹캠·업로드 이미지 탐지 지원; 독립 실행 파일과 파일 위치 기준 모델 경로로 다른 환경에서의 실행 편의성 확보.

### 7. 실행 영상

- Open Vocabulary Detection ResNet  


https://github.com/user-attachments/assets/79523ff9-c2e6-4769-b90c-dd43c5b4f988


- Open Vocabulary Detection YOLO World  


https://github.com/user-attachments/assets/53078da6-49da-4fd5-a7a9-583728952220

</details>
