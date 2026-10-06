<details>
<summary>ENG (English Version)</summary>

# Object Tracking Systems

### 1. Object Tracking Fundamentals

- **Definition:** Extends object detection by assigning consistent IDs across frames; answers "Where, What, Who".
- **Detection vs Tracking:** Detection finds objects per frame; tracking maintains identity over time.
- **Core Challenge:** Maintain object identity despite appearance and motion changes.

### 2. Real-world Applications

- **Surveillance:** CCTV monitoring with person and vehicle ID tracking.
- **Autonomous Systems:** Traffic analysis and pedestrian following.
- **Augmented Reality:** Object overlays that follow real-world movement.

### 3. Tracking Challenges

- **Occlusion:** Objects temporarily hidden by others.
- **Illumination Changes:** Lighting variations alter appearance.
- **Deformation:** Non-rigid objects change shape.
- **Fast Motion:** Rapid movement can cause detection gaps.

### 4. Single Object Tracking (SOT)

- **Correlation Filters:** KCF (Kernelized Correlation Filters), CSRT (Discriminative Correlation Filter with Channel and Spatial Reliability).
- **Siamese Networks:** Template matching through learned similarity metrics.
- **Use Case:** Tracking a single target from an initial bounding box.

### 5. Multi-Object Tracking (MOT)

- **Tracking-by-Detection:** Detect → Associate detections across frames.
- **SORT:** Simple Online and Realtime Tracking using a Kalman filter and Hungarian matching based on IoU.
- **DeepSORT:** Extends SORT with appearance features for improved identity preservation during occlusion.

### 6. Modern Implementations

- **YOLO + BoT-SORT:** Ultralytics YOLOv8 with BoT-SORT for multi-object tracking.
- **Tracker Selection:** Configurable through Ultralytics tracker settings.

### 7. Project Implementation

An object tracking system was implemented using YOLOv8n and BoT-SORT to detect objects in a webcam stream and display the ID and movement path of a user-selected object.

- **Object Detection and Tracking:** YOLO detections from each frame are passed to BoT-SORT to track objects across consecutive frames.
- **Target Selection:** Users can click a detected object's bounding box or press `S` and drag a region to select a target.
- **Trajectory Visualization:** The selected object's center positions are connected to display its movement path over the last five seconds.
- **Display:** Frames are displayed on a 960×540 canvas while preserving the original aspect ratio and adding padding where necessary.
- **Horizontal Mirroring:** The webcam view is mirrored, with bounding boxes and selection coordinates aligned with the mirrored image.
- **Controls:** Press `R` to clear the selection and `Q` or `Esc` to exit.

The initial implementation contained an incorrect detection confidence calculation and reused previous tracking coordinates as new detections. These issues were addressed by switching to actual YOLO detections from every frame.

### 8. Live Demonstration

The demonstration covers the following workflow:

1. Selecting an object on the screen
2. Maintaining its ID and displaying its movement path
3. Tracking behavior when the object is temporarily occluded or leaves the frame
4. Reselecting the target when tracking is lost

https://github.com/user-attachments/assets/7c25a7d8-bb6d-43f8-9e5b-224c3f103bc9

### 9. Execution Environment

| Item | Configuration |
|---|---|
| Detection Model | YOLOv8n |
| Tracking Algorithm | BoT-SORT |
| Main Libraries | Ultralytics, OpenCV, PyTorch |
| Display Canvas Size | 960×540 |


### 10. Operating Conditions and Limitations

Tracking stability can vary depending on camera specifications, recording conditions, object characteristics, detection results, and the computer's processing performance.

- **Image Quality:** Low resolution, poor focus, low lighting, or motion blur can make object detection more difficult.
- **Frame Rate and Processing Speed:** A low input frame rate or substantial processing latency can make fast-moving objects more difficult to track.
- **Object Size and Occlusion:** Detections may be missed when the target appears small or is obscured by another object.
- **Detection Model Limitations:** The current implementation only allows selection of objects detected by YOLO. Selecting an arbitrary region does not guarantee that every object can be tracked.
- **ID Continuity:** An object may receive a new ID after occlusion or leaving the frame, requiring the user to select it again.

Comparative testing is needed to establish tracking performance across cameras and determine minimum hardware requirements. Tracking failures should be investigated by considering image quality, detection results, and processing speed together.

### 11. How to Run

#### Prerequisites

- Use a Python environment with Ultralytics, OpenCV, PyTorch, NumPy, and a `lap`-compatible package such as `lapx` installed.
- Prepare the YOLOv8n detection model file, `yolov8n.pt`, locally.
- Open a terminal in the directory containing `ObjectTracking.py`.

#### Specify the Model Path

The script's default model path points to the developer's local environment. When running it on another computer, specify your own model location using `--model`.

If the model is stored in a `models` folder inside the current directory:

```bash
python ObjectTracking.py --model "./models/yolov8n.pt"
```

An absolute path can also be used. Windows example:

```bash
python ObjectTracking.py --model "C:\models\yolov8n.pt"
```

Replace the example path with the actual location of your model file. Enclose paths containing spaces in quotation marks.

The model file must already exist at the specified location. This script does not automatically download a missing model file.

#### Select the Video Source

The default source is webcam `0`.

To use another camera:

```bash
python ObjectTracking.py --model "./models/yolov8n.pt" --source 1
```

To use a video file:

```bash
python ObjectTracking.py --model "./models/yolov8n.pt" --source "./videos/demo.mp4"
```

Replace the video path with an existing file on your computer.

#### Controls

| Input | Action |
|---|---|
| Click a detected bounding box | Select the tracking target |
| `S` → Drag → Enter | Select a target using a region |
| `R` | Clear the selection |
| `Q` / `Esc` | Exit |

</details>

<details>
<summary>KOR (한국어 버전)</summary>

# 객체 추적 시스템

### 1. 객체 추적 기초

- **정의:** 객체 검출에 프레임 간 ID 유지를 더하여 객체의 위치와 정체성을 추적한다.
- **검출 vs 추적:** 검출은 각 프레임에서 객체를 찾고, 추적은 시간에 따른 객체의 동일성을 유지한다.
- **핵심 과제:** 외관과 움직임이 변해도 동일한 객체로 인식하는 것이다.

### 2. 실제 응용 분야

- **감시:** CCTV에서 사람과 차량의 ID 및 이동 경로 추적.
- **자율 시스템:** 교통 분석과 보행자 추적.
- **증강현실:** 현실 객체의 움직임에 맞춘 가상 콘텐츠 표시.

### 3. 추적 도전 과제

- **가림(Occlusion):** 다른 객체에 일시적으로 가려지는 경우.
- **조명 변화:** 조명에 따라 객체의 외관이 달라지는 경우.
- **변형(Deformation):** 사람처럼 자세나 형태가 변하는 경우.
- **빠른 움직임:** 프레임 사이의 급격한 이동으로 검출이 누락되는 경우.

### 4. 단일 객체 추적(SOT)

- **상관 필터:** KCF, CSRT 등 객체와 주변 영상의 상관관계를 활용하는 방식.
- **시암 네트워크:** 학습된 유사도 정보를 이용하여 템플릿과 현재 영상을 비교하는 방식.
- **사용 사례:** 초기 바운딩 박스로 지정한 단일 대상을 추적한다.

### 5. 다중 객체 추적(MOT)

- **검출 기반 추적:** 검출 → 프레임 간 검출 결과 연결.
- **SORT:** 칼만 필터와 IoU 기반 헝가리안 매칭을 사용하는 추적 방식.
- **DeepSORT:** SORT에 외형 특징을 추가하여 가림 상황에서 ID 유지를 개선한 방식.

### 6. 현대 구현

- **YOLO + BoT-SORT:** Ultralytics YOLOv8과 BoT-SORT를 활용한 다중 객체 추적.
- **트래커 선택:** Ultralytics의 트래커 설정을 통해 구성할 수 있다.

### 7. 프로젝트 구현

YOLOv8n과 BoT-SORT를 활용하여 웹캠 영상에서 객체를 탐지하고, 사용자가 선택한 객체의 ID와 이동 경로를 표시하는 시스템을 구현하였다.

- **객체 탐지 및 추적:** 매 프레임의 YOLO 탐지 결과를 BoT-SORT에 전달하여 객체를 추적한다.
- **추적 대상 선택:** 탐지된 객체의 박스를 클릭하거나, `S` 키를 누른 후 드래그하여 대상을 선택한다.
- **이동 경로 시각화:** 선택한 객체의 중심 좌표를 연결하여 최근 5초 동안의 이동 경로를 표시한다.
- **출력 화면:** 960×540 크기로 표시하며, 영상 비율을 유지하고 남는 영역은 여백으로 처리한다.
- **좌우 반전:** 웹캠 영상을 거울처럼 표시하고, 추적 박스와 선택 좌표도 반전된 화면에 맞춘다.
- **조작:** `R` 키로 선택을 해제하고, `Q` 또는 `Esc` 키로 종료한다.

초기 코드의 검출 신뢰도 계산 오류와 이전 추적 좌표를 검출 결과로 재사용하는 문제를 개선하여, 매 프레임의 실제 탐지 결과를 사용하는 구조로 변경하였다.

### 8. 실제 구동 영상

영상에는 다음 과정을 담는다.

1. 화면에서 추적할 객체 선택
2. 객체 이동에 따른 ID 유지 및 이동 경로 표시
3. 객체가 잠시 가려지거나 화면을 벗어났을 때의 동작
4. 추적이 끊긴 경우 대상 재선택

https://github.com/user-attachments/assets/221d1400-1216-41f0-970a-60a7faa0dd37

### 9. 실행 환경

| 항목 | 사용 환경 |
|---|---|
| 탐지 모델 | YOLOv8n |
| 추적 알고리즘 | BoT-SORT |
| 주요 라이브러리 | Ultralytics, OpenCV, PyTorch |
| 출력 화면 크기 | 960×540 |


### 10. 동작 조건과 한계

추적 안정성은 카메라 사양뿐 아니라 촬영 환경, 객체의 특성, 모델의 탐지 결과, 컴퓨터의 처리 속도에 따라 달라질 수 있다.

- **영상 품질:** 낮은 해상도, 초점 불량, 저조도 또는 움직임으로 인한 흐림은 객체 탐지를 어렵게 할 수 있다.
- **프레임률과 처리 속도:** 입력 프레임률이 낮거나 처리 지연이 커지면 빠르게 움직이는 객체를 추적하기 어려울 수 있다.
- **객체 크기와 가림:** 대상이 화면에서 작게 보이거나 다른 물체에 가려지면 검출이 누락될 수 있다.
- **탐지 모델의 한계:** 현재 구현은 YOLO가 탐지한 객체만 선택할 수 있으며, 임의의 영역을 선택한다고 모든 물체를 추적할 수 있는 것은 아니다.
- **ID 유지의 한계:** 가림이나 화면 이탈 후 객체에 새로운 ID가 부여되면 다시 선택해야 할 수 있다.

카메라별 추적 성능과 최소 요구 사양은 별도의 비교 실험이 필요하다. 추적 실패가 발생하면 영상 품질과 검출 결과, 처리 속도를 함께 확인해야 한다.

### 11. 실행 방법

#### 준비 사항

- Ultralytics, OpenCV, PyTorch, NumPy 및 `lapx`와 같은 `lap` 호환 패키지가 설치된 Python 환경을 사용한다.
- YOLOv8n 탐지 모델 파일인 `yolov8n.pt`를 로컬에 준비한다.
- `ObjectTracking.py`가 있는 폴더에서 터미널을 연다.

#### 모델 경로 지정

코드의 기본 모델 경로는 개발자의 로컬 환경을 기준으로 설정되어 있다. 다른 컴퓨터에서 실행할 때는 `--model` 옵션으로 자신의 모델 파일 위치를 지정한다.

현재 폴더의 `models` 폴더에 모델 파일을 저장한 경우:

```bash
python ObjectTracking.py --model "./models/yolov8n.pt"
```

절대경로로 지정할 수도 있다. Windows 환경의 예시는 다음과 같다.

```bash
python ObjectTracking.py --model "C:\models\yolov8n.pt"
```

예시 경로를 실제 모델 파일 위치로 변경한다. 경로에 공백이 있으면 큰따옴표로 감싼다.

지정한 위치에 모델 파일이 실제로 존재해야 한다. 이 코드는 누락된 모델 파일을 자동으로 다운로드하지 않는다.

#### 영상 입력 선택

기본 입력은 웹캠 `0`이다.

다른 카메라를 사용하는 경우:

```bash
python ObjectTracking.py --model "./models/yolov8n.pt" --source 1
```

영상 파일을 사용하는 경우:

```bash
python ObjectTracking.py --model "./models/yolov8n.pt" --source "./videos/demo.mp4"
```

영상 경로는 자신의 컴퓨터에 존재하는 파일 위치로 변경한다.

#### 조작 방법

| 입력 | 동작 |
|---|---|
| 탐지 박스 클릭 | 추적 대상 선택 |
| `S` → 드래그 → Enter | 영역을 지정하여 대상 선택 |
| `R` | 선택 해제 |
| `Q` / `Esc` | 프로그램 종료 |

</details>
