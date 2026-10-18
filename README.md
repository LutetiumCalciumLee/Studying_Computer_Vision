<details>
<summary>ENG (English Version)</summary>

# Korean License Plate Recognition

## 1. Project Overview

Built a Korean license plate recognition pipeline that combines vehicle detection, license plate localization, and OCR.

The project integrates pretrained models and separates AI inference from video rendering to reduce playback interruptions.

Users can prepare their own recorded vehicle videos and specify the input through command-line arguments.

| Item | Description |
|---|---|
| Main script | `13_LPRecognition.py` |
| Input | User-recorded vehicle videos |
| Output | Annotated preview and a combined MP4 video |
| Implementation | Model integration, OCR post-processing, asynchronous inference, and visualization |
| Training | Existing pretrained models are used |
| Development environment | Windows · Anaconda · Python 3.12 |
| Development GPU | NVIDIA GeForce RTX 3060 Laptop GPU · 6GB VRAM |

Input videos and model weights are prepared locally rather than bundled with the repository.

## 2. Recognition Pipeline

```text
Input Frame
    ↓
Vehicle Detection — YOLOv8s
    ↓
Vehicle ROI
    ↓
License Plate Detection — Custom YOLOv5
    ↓
License Plate ROI → Grayscale Conversion
    ↓
Text Recognition — EasyOCR Custom Model
    ↓
Text Validation → Bounding Box and Korean Text Overlay
```

| Component | Model / Resource | Preparation |
|---|---|---|
| Vehicle detection | `yolov8s.pt` | Download through Ultralytics |
| License plate detection | `lp_det.pt` | Download from the original model repository |
| OCR recognition | `best_acc.pth` | Download from the original model repository |
| OCR text detection | `craft_mlt_25k.pth` | Automatically downloaded by EasyOCR when missing |
| Custom OCR network | `best_acc.py`, `best_acc.yaml`, `transformation.py` | Download from the original model repository |

Vehicle detection targets COCO classes **2, 3, 5, and 7**: car, motorcycle, bus, and truck.

Although EasyOCR is initialized with `['en']`, the custom `best_acc` network and character list support Korean license plate recognition.

Recognized text is cleaned and validated against:

```regex
^\d{2,3}[가-힣]\d{4}$
```

Examples: `12가3456`, `123가4567`

Regional prefixes and special plate formats are outside the current validation pattern.

## 3. Key Features

### Flexible Video Input

- Accepts one video, multiple videos, or a folder through `--input`.
- Processes explicitly listed files in the supplied order.
- Sorts videos in a folder using natural filename order.
- Uses `00_Sample_video/input/my_cars.mp4` when `--input` is omitted.
- Validates input paths, resolution, and FPS before loading the models.
- Prevents the output path from overwriting an input video.
- Supports a custom output path through `--output`.

Multiple videos must have matching resolution and FPS to be combined into one output.

### Asynchronous Inference

| Setting | Behavior |
|---|---|
| `DETECTION_INTERVAL = 5` | Submits inference at intervals of at least five frames |
| `max_workers = 1` | Uses one dedicated AI worker |
| `pending.done()` | Retrieves completed results without waiting for unfinished inference |
| One outstanding task | Prevents inference tasks from accumulating |
| `RESULT_MAX_AGE_SECONDS = 0.5` | Limits displayed results by source-frame age |

Playback continues while inference is running. Inference requests may be skipped when the worker is busy, while video frames continue to be displayed and saved.

### Visualization and Output

- Red bounding boxes around recognized plates
- Green Korean text at **40px with a 2px stroke**
- Cached Korean font using `lru_cache`
- On-screen FPS display
- No intro countdown or black-screen padding
- `q` to stop the entire session
- Default output: `00_Sample_video/output/lp_output.mp4`

## 4. Validation and Limitations

### Development Validation

- Verified sequential processing of multiple video clips.
- Checked bounded inference scheduling under slow-inference conditions.
- Verified quit handling and resource cleanup.
- A short test with actual GPU inference and video encoding measured approximately **28fps**.
- The performance measurement excluded the native preview window.

These measurements describe development tests. They do not guarantee the same performance on newly recorded videos or different hardware.

Recognition accuracy has not been measured against a labeled evaluation dataset.

### Limitations and Next Steps

- The regex supports only selected Korean plate formats.
- Recognition results may appear slightly behind the displayed video.
- Recent detections are reused without persistent vehicle tracking IDs.
- Small, blurred, dark, or tilted plates can be difficult to recognize.
- Slow inference reduces the frequency of recognition updates.

Planned improvements include object tracking, multi-frame OCR voting, plate correction, and accuracy evaluation using labeled data and personal recordings.

## 5. Installation and Run

The following instructions use **Windows PowerShell**.

Download the repository and open the project root containing `13_Korean_LP_Recognition` and `00_Models`.

### 5-1. Create a Python Environment

Anaconda or Miniconda is required for these environment commands.

```powershell
conda create -n lp_recognition python=3.12 -y
conda activate lp_recognition

python -m pip install --upgrade pip
```

### 5-2. Install PyTorch

Choose **one** installation option.

#### NVIDIA GPU — CUDA 12.6

An NVIDIA driver compatible with the CUDA 12.6 PyTorch build is required.

```powershell
python -m pip install torch==2.10.0 torchvision==0.25.0 --index-url https://download.pytorch.org/whl/cu126
```

#### CPU Only

```powershell
python -m pip install torch==2.10.0 torchvision==0.25.0 --index-url https://download.pytorch.org/whl/cpu
```

Check CUDA availability:

```powershell
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA:', torch.cuda.is_available())"
```

CPU execution is supported, but processing speed will differ.

### 5-3. Install Libraries

```powershell
python -m pip install ultralytics==8.4.178 easyocr==1.7.2 pillow==12.1.0 "numpy<3" pandas matplotlib seaborn pyyaml requests tqdm scipy gitpython "setuptools>=70" "ultralytics-thop>=2.2.1"
```

EasyOCR may install the headless OpenCV package. Replace it with the GUI-supported package to use the preview window.

```powershell
python -m pip uninstall -y opencv-python opencv-python-headless
python -m pip install opencv-python==4.10.0.84
```

Check GUI support:

```powershell
python -c "import cv2; print(cv2.__version__); print([s.strip() for s in cv2.getBuildInformation().splitlines() if 'GUI:' in s])"
```

On Windows, check that the output contains `GUI: WIN32UI`.

### 5-4. Download Models and Font

The custom license plate and OCR resources come from `gyupro/EasyKoreanLpDetector`.

Run the following commands from the project root. Existing files are skipped.

```powershell
$ErrorActionPreference = "Stop"
$baseUrl = "https://raw.githubusercontent.com/gyupro/EasyKoreanLpDetector/main"

$files = @{
    "lp_det.pt" = "00_Models/lp_det.pt"
    "lp_models/models/best_acc.pth" = "00_Models/lp_models/models/best_acc.pth"
    "lp_models/user_network/best_acc.py" = "00_Models/lp_models/user_network/best_acc.py"
    "lp_models/user_network/best_acc.yaml" = "00_Models/lp_models/user_network/best_acc.yaml"
    "lp_models/user_network/transformation.py" = "00_Models/lp_models/user_network/transformation.py"
    "SpoqaHanSansNeo-Light.ttf" = "00_font/SpoqaHanSansNeo-Light.ttf"
}

foreach ($relativePath in $files.Keys) {
    $destination = $files[$relativePath]

    New-Item -ItemType Directory -Force -Path (Split-Path -Parent $destination) | Out-Null

    if (-not (Test-Path -LiteralPath $destination)) {
        Invoke-WebRequest -Uri "$baseUrl/$relativePath" -OutFile $destination
    }
}
```

Prepare YOLOv8s:

```powershell
python -c "from pathlib import Path; from ultralytics import YOLO; p=Path('00_Models/yolov8s.pt'); p.parent.mkdir(parents=True, exist_ok=True); YOLO(str(p))"
```

The large `craft_mlt_25k.pth` file is downloaded automatically by EasyOCR when missing.

The first run may also download YOLOv5 code through `torch.hub`. An internet connection is required during initial setup.

### 5-5. Prepare a Personal Video

Create the input folder:

```powershell
New-Item -ItemType Directory -Force -Path "00_Sample_video/input" | Out-Null
```

Place a recorded vehicle video at:

```text
00_Sample_video/input/my_cars.mp4
```

Expected project structure:

```text
Project Root/
├── 13_Korean_LP_Recognition/
│   └── 13_LPRecognition.py
├── 00_Models/
│   ├── yolov8s.pt
│   ├── lp_det.pt
│   └── lp_models/
│       ├── models/
│       │   ├── best_acc.pth
│       │   └── craft_mlt_25k.pth
│       └── user_network/
│           ├── best_acc.py
│           ├── best_acc.yaml
│           └── transformation.py
├── 00_font/
│   └── SpoqaHanSansNeo-Light.ttf
└── 00_Sample_video/
    ├── input/
    │   └── my_cars.mp4
    └── output/
        └── lp_output.mp4
```

The input video is supplied by the user. The output file is created when the script runs.

### 5-6. Run

Use the default `my_cars.mp4`:

```powershell
python .\13_Korean_LP_Recognition\13_LPRecognition.py
```

Specify a video:

```powershell
python .\13_Korean_LP_Recognition\13_LPRecognition.py --input ".\00_Sample_video\input\my_cars.mp4"
```

Process multiple videos in order:

```powershell
python .\13_Korean_LP_Recognition\13_LPRecognition.py --input ".\00_Sample_video\input\my_cars_01.mp4" ".\00_Sample_video\input\my_cars_02.mp4"
```

Process all supported videos in a folder:

```powershell
python .\13_Korean_LP_Recognition\13_LPRecognition.py --input ".\00_Sample_video\input"
```

Specify an output path:

```powershell
python .\13_Korean_LP_Recognition\13_LPRecognition.py --input ".\00_Sample_video\input\my_cars.mp4" --output ".\00_Sample_video\output\my_result.mp4"
```

Show available arguments:

```powershell
python .\13_Korean_LP_Recognition\13_LPRecognition.py --help
```

Press `q` in the preview window to stop processing.

These instructions reflect the development environment. A complete fresh installation has not been independently revalidated.

## 6. Demo Video

Existing implementation demonstration:

https://github.com/user-attachments/assets/000004e4-61c5-45e4-be8d-9258df46cb66

</details>

<details>
<summary>KOR (한국어 버전)</summary>

# 한국 차량 번호판 인식

## 1. 프로젝트 개요

차량 영상에서 차량과 번호판 영역을 단계적으로 검출하고, 번호판의 한글과 숫자를 인식하는 파이프라인을 구현했습니다.

사전학습 모델을 연결한 뒤, OCR 처리 중 발생하는 영상 끊김을 분석해 **영상 표시와 AI 추론을 분리하는 구조**로 개선했습니다.

사용자가 직접 촬영한 차량 영상을 준비하고, 명령어로 입력 경로를 지정해 실행할 수 있습니다.

| 항목 | 내용 |
|---|---|
| 실행 파일 | `13_LPRecognition.py` |
| 입력 | 사용자가 직접 촬영한 차량 영상 |
| 출력 | 번호판 인식 화면 및 통합 MP4 영상 |
| 구현 범위 | 모델 통합, OCR 후처리, 비동기 처리, 시각화, 영상 저장 |
| 모델 학습 | 기존 사전학습 모델 활용 |
| 개발 환경 | Windows · Anaconda · Python 3.12 |
| 개발 시 검증 GPU | NVIDIA GeForce RTX 3060 Laptop GPU · VRAM 6GB |

입력 영상과 모델 가중치는 저장소에 첨부하는 대신 로컬에서 준비하도록 구성했습니다.

## 2. 인식 파이프라인

```text
영상 프레임 입력
    ↓
YOLOv8s 차량 검출
    ↓
차량 영역 추출
    ↓
커스텀 YOLOv5 번호판 검출
    ↓
번호판 영역 추출 및 그레이스케일 변환
    ↓
EasyOCR 문자 인식
    ↓
문자열 후처리 및 번호판 형식 검증
    ↓
원본 영상에 번호판 위치와 인식 결과 표시
```

| 구성 요소 | 모델·리소스 | 준비 방법 |
|---|---|---|
| 차량 검출 | `yolov8s.pt` | Ultralytics로 다운로드 |
| 번호판 검출 | `lp_det.pt` | 원본 모델 저장소에서 다운로드 |
| 번호판 문자 인식 | `best_acc.pth` | 원본 모델 저장소에서 다운로드 |
| OCR 문자 영역 검출 | `craft_mlt_25k.pth` | 파일이 없으면 EasyOCR가 자동 다운로드 |
| 커스텀 OCR 네트워크 | `best_acc.py`, `best_acc.yaml`, `transformation.py` | 원본 모델 저장소에서 다운로드 |

차량 검출에는 COCO 클래스 **2·3·5·7**, 즉 자동차·오토바이·버스·트럭을 사용합니다.

EasyOCR의 언어 설정은 `['en']`이지만, 한글이 포함된 커스텀 문자 목록과 `best_acc` 네트워크를 불러와 번호판을 인식합니다.

OCR 결과의 공백과 반복 문자열을 정리한 뒤 다음 정규식에 맞는 결과만 표시합니다.

```regex
^\d{2,3}[가-힣]\d{4}$
```

- 앞자리 숫자: 2~3자리
- 한글: 1글자
- 뒷자리 숫자: 4자리
- 예시: `12가3456`, `123가4567`

지역명이 포함된 번호판이나 특수 형식은 현재 검증 범위에 포함하지 않습니다.

## 3. 주요 구현 기능

### 유연한 영상 입력

- `--input`으로 영상 하나, 여러 영상 또는 폴더 지정
- 여러 파일을 직접 지정하면 입력한 순서로 처리
- 폴더를 지정하면 파일명의 숫자를 고려해 정렬
- `--input`을 생략하면 `00_Sample_video/input/my_cars.mp4` 사용
- 모델 로딩 전에 입력 경로·해상도·FPS 검사
- 입력 영상을 결과 파일로 덮어쓰는 경우 차단
- `--output`으로 결과 저장 경로 지정

여러 영상을 하나로 연결하려면 입력 영상들의 해상도와 FPS가 같아야 합니다.

### 영상 표시와 AI 추론 분리

| 설정 | 동작 |
|---|---|
| `DETECTION_INTERVAL = 5` | 마지막 작업 제출 이후 최소 5프레임 간격으로 추론 요청 |
| `max_workers = 1` | 하나의 전용 AI 작업 스레드 사용 |
| `pending.done()` | 완료된 결과만 가져와 화면 대기 방지 |
| 미완료 작업 1개 유지 | AI 작업이 누적되지 않도록 제한 |
| `RESULT_MAX_AGE_SECONDS = 0.5` | 원본 프레임 기준 약 0.5초 이내 결과만 표시 |

추론 중에도 영상 재생은 계속됩니다. AI가 바쁘면 일부 추론 요청을 생략하지만, 영상 프레임 자체를 생략하도록 구현하지는 않았습니다.

### 인식 결과 시각화 및 저장

- 번호판 위치: 빨간색 테두리
- 번호판 문자열: 초록색 한글 표시
- 글자 크기: **40px**
- 글자 굵기: **2px 획 추가**
- 한글 폰트: `lru_cache`로 캐싱
- 화면 상단: 처리 FPS 표시
- 앞뒤 검은 화면이나 카운트다운 없이 본 영상 처리
- 종료 키: **`q`**
- 기본 결과 경로: `00_Sample_video/output/lp_output.mp4`

## 4. 검증 및 한계

### 개발 과정의 검증

- 여러 영상의 순차 처리 확인
- 느린 추론 상황에서 작업이 누적되지 않는지 확인
- `q` 종료 및 오류 발생 시 자원 해제 확인
- 실제 GPU 인식과 영상 저장을 포함한 짧은 테스트에서 **약 28fps** 측정
- 해당 성능 측정은 실제 화면 창을 제외한 환경에서 수행

위 수치는 기존 개발 테스트에서 관찰한 결과입니다. 새로 촬영한 영상이나 다른 장비에서 동일한 성능을 보장하는 값은 아닙니다.

정답 라벨을 갖춘 평가 데이터셋으로 인식 정확도를 측정하지 않았으므로, 정확도 수치는 제시하지 않았습니다.

### 현재 한계 및 향후 개선

- 정규식으로 허용한 일부 번호판 형식만 처리
- 비동기 처리로 인해 인식 결과가 현재 영상보다 늦게 표시될 수 있음
- 최근 인식 결과를 재사용하며 차량별 추적 ID는 관리하지 않음
- 작거나 흐릿한 번호판, 역광, 기울어진 번호판에서 인식 품질 저하 가능
- AI 처리 시간이 길면 인식 갱신 빈도 감소

향후 객체 추적, 여러 프레임의 OCR 결과 결합, 번호판 보정, 개인 촬영 영상과 정답 라벨을 활용한 정확도 평가를 적용할 계획입니다.

## 5. 설치 및 실행

아래 명령은 **Windows PowerShell** 기준입니다.

저장소를 내려받고, `13_Korean_LP_Recognition`과 `00_Models`가 들어 있는 프로젝트 루트 폴더에서 실행합니다.

### 5-1. Python 환경 생성

Anaconda 또는 Miniconda가 설치된 환경에서 실행합니다.

```powershell
conda create -n lp_recognition python=3.12 -y
conda activate lp_recognition

python -m pip install --upgrade pip
```

### 5-2. PyTorch 설치

사용하는 장비에 맞는 방법 **하나만** 실행합니다.

#### NVIDIA GPU — CUDA 12.6

CUDA 12.6용 PyTorch를 지원하는 NVIDIA 드라이버가 필요합니다.

```powershell
python -m pip install torch==2.10.0 torchvision==0.25.0 --index-url https://download.pytorch.org/whl/cu126
```

#### CPU 전용

```powershell
python -m pip install torch==2.10.0 torchvision==0.25.0 --index-url https://download.pytorch.org/whl/cpu
```

GPU 인식 여부 확인:

```powershell
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA:', torch.cuda.is_available())"
```

GPU용 환경에서 `CUDA: True`가 표시되면 CUDA를 사용할 수 있습니다.

CPU 실행도 지원하지만 처리 속도는 달라집니다.

### 5-3. 라이브러리 설치

```powershell
python -m pip install ultralytics==8.4.178 easyocr==1.7.2 pillow==12.1.0 "numpy<3" pandas matplotlib seaborn pyyaml requests tqdm scipy gitpython "setuptools>=70" "ultralytics-thop>=2.2.1"
```

EasyOCR 설치 과정에서 GUI 기능이 없는 OpenCV가 함께 설치될 수 있습니다. 재생 창을 사용하도록 GUI 지원 버전으로 교체합니다.

```powershell
python -m pip uninstall -y opencv-python opencv-python-headless
python -m pip install opencv-python==4.10.0.84
```

GUI 지원 확인:

```powershell
python -c "import cv2; print(cv2.__version__); print([s.strip() for s in cv2.getBuildInformation().splitlines() if 'GUI:' in s])"
```

Windows 환경에서는 `GUI: WIN32UI`가 표시되는지 확인합니다.

### 5-4. 모델과 폰트 다운로드

커스텀 번호판·OCR 리소스의 출처는 `gyupro/EasyKoreanLpDetector`입니다.

프로젝트 루트에서 다음 명령을 실행합니다. 이미 존재하는 파일은 건너뜁니다.

```powershell
$ErrorActionPreference = "Stop"
$baseUrl = "https://raw.githubusercontent.com/gyupro/EasyKoreanLpDetector/main"

$files = @{
    "lp_det.pt" = "00_Models/lp_det.pt"
    "lp_models/models/best_acc.pth" = "00_Models/lp_models/models/best_acc.pth"
    "lp_models/user_network/best_acc.py" = "00_Models/lp_models/user_network/best_acc.py"
    "lp_models/user_network/best_acc.yaml" = "00_Models/lp_models/user_network/best_acc.yaml"
    "lp_models/user_network/transformation.py" = "00_Models/lp_models/user_network/transformation.py"
    "SpoqaHanSansNeo-Light.ttf" = "00_font/SpoqaHanSansNeo-Light.ttf"
}

foreach ($relativePath in $files.Keys) {
    $destination = $files[$relativePath]

    New-Item -ItemType Directory -Force -Path (Split-Path -Parent $destination) | Out-Null

    if (-not (Test-Path -LiteralPath $destination)) {
        Invoke-WebRequest -Uri "$baseUrl/$relativePath" -OutFile $destination
    }
}
```

YOLOv8s 모델 준비:

```powershell
python -c "from pathlib import Path; from ultralytics import YOLO; p=Path('00_Models/yolov8s.pt'); p.parent.mkdir(parents=True, exist_ok=True); YOLO(str(p))"
```

대용량 파일인 `craft_mlt_25k.pth`는 별도로 첨부하지 않습니다. 파일이 없으면 첫 실행 시 EasyOCR가 다운로드합니다.

첫 실행 시 `torch.hub`가 YOLOv5 코드를 내려받을 수도 있으므로 초기 준비 과정에서는 인터넷 연결이 필요합니다.

### 5-5. 개인 촬영 영상 준비

입력 폴더 생성:

```powershell
New-Item -ItemType Directory -Force -Path "00_Sample_video/input" | Out-Null
```

직접 촬영한 차량 영상을 다음 경로에 저장합니다.

```text
00_Sample_video/input/my_cars.mp4
```

프로젝트 구성:

```text
Project Root/
├── 13_Korean_LP_Recognition/
│   └── 13_LPRecognition.py
├── 00_Models/
│   ├── yolov8s.pt
│   ├── lp_det.pt
│   └── lp_models/
│       ├── models/
│       │   ├── best_acc.pth
│       │   └── craft_mlt_25k.pth
│       └── user_network/
│           ├── best_acc.py
│           ├── best_acc.yaml
│           └── transformation.py
├── 00_font/
│   └── SpoqaHanSansNeo-Light.ttf
└── 00_Sample_video/
    ├── input/
    │   └── my_cars.mp4
    └── output/
        └── lp_output.mp4
```

입력 영상은 사용자가 준비하며, 결과 파일은 실행 후 생성됩니다.

### 5-6. 실행

기본 입력 영상인 `my_cars.mp4` 사용:

```powershell
python .\13_Korean_LP_Recognition\13_LPRecognition.py
```

영상 하나 지정:

```powershell
python .\13_Korean_LP_Recognition\13_LPRecognition.py --input ".\00_Sample_video\input\my_cars.mp4"
```

여러 영상을 지정한 순서대로 처리:

```powershell
python .\13_Korean_LP_Recognition\13_LPRecognition.py --input ".\00_Sample_video\input\my_cars_01.mp4" ".\00_Sample_video\input\my_cars_02.mp4"
```

폴더 안의 영상 전체 처리:

```powershell
python .\13_Korean_LP_Recognition\13_LPRecognition.py --input ".\00_Sample_video\input"
```

결과 저장 경로 지정:

```powershell
python .\13_Korean_LP_Recognition\13_LPRecognition.py --input ".\00_Sample_video\input\my_cars.mp4" --output ".\00_Sample_video\output\my_result.mp4"
```

사용 가능한 옵션 확인:

```powershell
python .\13_Korean_LP_Recognition\13_LPRecognition.py --help
```

재생 창에서 `q`를 누르면 전체 처리를 종료합니다.

위 명령은 개발 환경을 참고한 설치 안내이며, 새로운 환경에서 전체 설치 과정을 별도로 재검증한 결과는 아닙니다.

## 6. 시연 영상

기존 구현 시연 영상입니다.

https://github.com/user-attachments/assets/000004e4-61c5-45e4-be8d-9258df46cb66

</details>
