<details>
<summary>ENG (English Version)</summary>

# Object Tracking Systems

### 1. Object Tracking Fundamentals

- **Definition:** Extends object detection by assigning consistent IDs across frames; answers "Where, What, Who".
- **Detection vs Tracking:** Detection finds objects per frame; tracking maintains identity over time.
- **Core Challenge:** Maintain object identity despite appearance/motion changes.

### 2. Real-world Applications

- **Surveillance:** CCTV monitoring with person/vehicle ID tracking.
- **Autonomous Systems:** Traffic analysis, pedestrian following.
- **Augmented Reality:** AR object overlay and interaction.

### 3. Tracking Challenges

- **Occlusion:** Objects temporarily hidden by others.
- **Illumination Changes:** Lighting variations alter appearance.
- **Deformation:** Non-rigid object shape changes.
- **Fast Motion:** Rapid movement causes detection gaps.

### 4. Single Object Tracking (SOT)

- **Correlation Filters:** KCF (Kernelized Correlation Filters), CSRT (Channel/Spatial Reliability).
- **Siamese Networks:** Template matching via learned similarity metrics.
- **Use Case:** Track single target from initial bounding box.

### 5. Multi-Object Tracking (MOT)

- **Tracking-by-Detection:** Detect → Associate detections across frames.
- **SORT:** Simple Online Realtime Tracking using Kalman Filter + Hungarian IoU matching.
- **DeepSORT:** SORT + deep appearance features (ReID) for robust ID maintenance.

### 6. Modern Implementations

- **YOLO + BoT-SORT:** Ultralytics YOLOv8 with BoT-SORT tracker for real-time multi-object tracking.
- **Tracker Selection:** Configurable via Ultralytics docs for different scenarios.

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

The demonstration will show:

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
| Camera Model | To be confirmed |
| Actual Camera Input Resolution | To be confirmed |
| Actual Input Frame Rate | To be confirmed |
| CPU / GPU | To be confirmed |
| Measured Processing FPS | To be measured |
| Display Canvas Size | 960×540 |

*The display canvas size, camera input resolution, and model inference size are separate settings.*

### 10. Operating Conditions and Limitations

Tracking stability can vary depending on camera specifications, recording conditions, object characteristics, detection results, and the computer's processing performance.

- **Image Quality:** Low resolution, poor focus, low lighting, or motion blur can make object detection more difficult.
- **Frame Rate and Processing Speed:** A low input frame rate or substantial processing latency can make fast-moving objects more difficult to track.
- **Object Size and Occlusion:** Detections may be missed when the target appears small or is obscured by another object.
- **Detection Model Limitations:** The current implementation only allows selection of objects detected by YOLO. Selecting an arbitrary region does not guarantee that every object can be tracked.
- **ID Continuity:** An object may receive a new ID after occlusion or leaving the frame, requiring the user to select it again.

Comparative testing is needed to establish tracking performance across cameras and determine minimum hardware requirements. Tracking failures should therefore be investigated by considering image quality, detection results, and processing speed together, rather than attributing them solely to camera specifications.

</details>

<details>
<summary>KOR (한국어 버전)</summary>

# 객체 추적 시스템

### 1. 객체 추적 기초

- **정의:** 프레임 간 일관된 ID 부여로 객체 검출 확장; "어디, 무엇, 누구" 답변.
- **검출 vs 추적:** 검출은 프레임별 객체 찾기; 추적은 시간적 ID 유지.
- **핵심 과제:** 외관/운동 변화에도 객체 신원 유지.

### 2. 실제 응용 분야

- **감시:** CCTV 사람/차량 ID 추적 모니터링.
- **자율 시스템:** 교통 분석, 보행자 추종.
- **증강현실:** AR 객체 오버레이 및 상호작용.

### 3. 추적 도전 과제

- **가림(Occlusion):** 다른 객체에 일시 가려짐.
- **조명 변화:** 조명 변동으로 외관 변화.
- **변형(Deformation):** 비강체 객체 형상 변화.
- **빠른 움직임:** 급격한 이동으로 검출 갭 발생.

### 4. 단일 객체 추적(SOT)

- **상관 필터:** KCF(Kernelized Correlation Filters), CSRT(Channel/Spatial Reliability).
- **시암 네트워크:** 학습된 유사도 메트릭으로 템플릿 매칭.
- **사용 사례:** 초기 바운딩 박스로부터 단일 타겟 추적.

### 5. 다중 객체 추적(MOT)

- **검출 기반 추적:** 검출 → 프레임 간 연관성 부여.
- **SORT:** 칼만 필터 + 헝가리안 IoU 매칭의 간단 실시간 추적.
- **DeepSORT:** ReID 심층 외관 특징 + SORT로 견고한 ID 유지.

### 6. 현대 구현

- **YOLO + BoT-SORT:** Ultralytics YOLOv8 + BoT-SORT 실시간 다중 객체 추적.
- **트래커 선택:** 다양한 시나리오별 Ultralytics 문서 설정 가능.

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
| 카메라 모델 | 확인 후 입력 |
| 실제 카메라 입력 해상도 | 확인 후 입력 |
| 실제 입력 프레임률 | 확인 후 입력 |
| CPU / GPU | 확인 후 입력 |
| 측정된 처리 FPS | 측정 후 입력 |
| 출력 화면 크기 | 960×540 |

※ 출력 화면 크기, 카메라 입력 해상도, 모델의 추론 크기는 서로 다른 설정이다.

### 10. 동작 조건과 한계

추적 안정성은 카메라 사양뿐 아니라 촬영 환경, 객체의 특성, 모델의 탐지 결과, 컴퓨터의 처리 속도에 따라 달라질 수 있다.

- **영상 품질:** 낮은 해상도, 초점 불량, 저조도 또는 움직임으로 인한 흐림은 객체 탐지를 어렵게 할 수 있다.
- **프레임률과 처리 속도:** 입력 프레임률이 낮거나 처리 지연이 커지면 빠르게 움직이는 객체를 추적하기 어려울 수 있다.
- **객체 크기와 가림:** 대상이 화면에서 작게 보이거나 다른 물체에 가려지면 검출이 누락될 수 있다.
- **탐지 모델의 한계:** 현재 구현은 YOLO가 탐지한 객체만 선택할 수 있으며, 임의의 영역을 선택한다고 모든 물체를 추적할 수 있는 것은 아니다.
- **ID 유지의 한계:** 가림이나 화면 이탈 후 객체에 새로운 ID가 부여되면 다시 선택해야 할 수 있다.

카메라별 추적 성능과 최소 요구 사양은 별도의 비교 실험이 필요하다. 따라서 특정 카메라에서 발생한 추적 실패를 카메라 사양만의 문제로 단정하지 않고, 영상 품질과 검출 결과, 처리 속도를 함께 확인해야 한다.

</details>
