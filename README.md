<details>
<summary>ENG (English Version)</summary>

# Korean License Plate Recognition

## 1. Project Overview

Built a Korean license plate recognition pipeline that combines vehicle detection, license plate localization, and OCR.

The project integrates pretrained models and improves video playback by separating AI inference from frame rendering.

- **Input:** 15 sequential video clips
- **Output:** Annotated preview and a combined MP4 video
- **Main Script:** `13_LPRecognition.py`
- **Implementation:** Model integration, OCR post-processing, asynchronous inference, and visualization

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

| Component | Model / Resource |
|---|---|
| Vehicle detection | `yolov8s.pt` |
| License plate detection | `lp_det.pt` |
| OCR recognition | `best_acc.pth` |
| OCR text detection | `craft_mlt_25k.pth` |
| Custom network | `best_acc.py`, `best_acc.yaml`, `transformation.py` |

Vehicle detection targets COCO classes **2, 3, 5, and 7**: car, motorcycle, bus, and truck.

Recognized text is validated against:

```regex
^\d{2,3}[가-힣]\d{4}$
```

Example: `123가4567`

## 3. Key Features

- Plays `sample_lp_part_01.mp4` through `sample_lp_part_15.mp4` in order.
- Loads the models once and reuses them across all clips.
- Runs AI inference in a dedicated worker thread.
- Submits inference at intervals of at least five video frames.
- Continues playback when the AI worker is busy instead of accumulating tasks.
- Displays recent results only within a source-frame age of approximately 0.5 seconds.
- Uses red plate boxes and green Korean text at **40px with a 2px stroke**.
- Saves the processed sequence as `output/lp_output.mp4`.
- Supports `q` to stop the entire playlist.
- Starts directly with the video, without an intro countdown or black-screen padding.


## 4. Validation

- Prepared **15 silent clips** from a 79.7-second source while preserving **1280×720 at 30fps**.
- Kept every input clip below **10MB**.
- Verified that all **2,391 frames** were preserved during splitting.
- During optimization, a short test with actual GPU inference and video encoding measured approximately **28fps**.
- The performance test excluded the native preview window and does not represent a guaranteed playback rate.
- Separately validated playlist order, frame pacing, bounded inference scheduling, and cleanup using simulated inference and GUI behavior.

Recognition accuracy was not evaluated against a labeled test set.

## 5. Limitations and Next Steps

- The current regex does not cover every Korean plate format.
- Recognition results may appear slightly behind the displayed video.
- The implementation reuses recent detections; it does not assign persistent tracking IDs.
- Motion blur, distant plates, lighting, and camera angles can reduce recognition quality.

Planned improvements include object tracking, multi-frame OCR voting, plate image correction, and accuracy evaluation with labeled data.


## 6. Demo Video

https://github.com/user-attachments/assets/000004e4-61c5-45e4-be8d-9258df46cb66

</details>

<details>
<summary>KOR (한국어 버전)</summary>

# 한국 차량 번호판 인식

## 1. 프로젝트 개요

주행 영상에서 차량을 찾고, 차량 내부의 번호판 영역을 검출한 뒤 한글과 숫자를 인식하는 파이프라인을 구현했습니다.

사전학습된 모델을 연결하는 데서 시작해, 번호판 인식 중 발생하는 재생 끊김을 분석하고 **영상 표시와 AI 추론을 분리하는 구조**로 개선했습니다.

| 항목 | 내용 |
|---|---|
| 실행 파일 | `13_LPRecognition.py` |
| 입력 | 분할 영상 15개 |
| 출력 | 번호판 인식 화면 및 통합 MP4 영상 |
| 구현 범위 | 모델 통합, OCR 후처리, 비동기 처리, 시각화, 영상 저장 |
| 실행 환경 | Windows · Anaconda · Python 3.12.12 |
| 주요 라이브러리 | PyTorch 2.10.0+cu126 · OpenCV 4.10.0.84 · EasyOCR 1.7.2 · Ultralytics |
| 검증 GPU | NVIDIA GeForce RTX 3060 Laptop GPU · VRAM 6GB |

별도의 모델 학습보다는 **사전학습 모델을 활용한 추론 파이프라인 구성과 실행 성능 개선**에 집중했습니다.

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

### 차량 검출

`yolov8s.pt`를 사용해 다음 객체를 검출합니다.

| 클래스 ID | 객체 |
|---|---|
| 2 | 자동차 |
| 3 | 오토바이 |
| 5 | 버스 |
| 7 | 트럭 |

### 번호판 검출 및 OCR

차량 전체 영상을 OCR에 전달하지 않고, 차량 영역과 번호판 영역을 단계적으로 추출해 인식합니다.

| 파일 | 역할 |
|---|---|
| `lp_det.pt` | 차량 내부의 번호판 검출 |
| `best_acc.pth` | 번호판 문자 인식 가중치 |
| `craft_mlt_25k.pth` | OCR 문자 영역 검출 가중치 |
| `best_acc.py` | 커스텀 OCR 네트워크 |
| `best_acc.yaml` | 문자 목록 및 네트워크 설정 |
| `transformation.py` | 커스텀 네트워크에서 사용하는 변환 모듈 |

EasyOCR의 언어 설정은 `['en']`이지만, 커스텀 문자 목록과 `best_acc` 네트워크를 불러와 한글 번호판 인식에 사용합니다.

### 문자열 후처리

OCR 결과의 공백을 제거하고 반복 문자열을 정리한 뒤, 다음 정규식에 맞는 결과만 표시합니다.

```regex
^\d{2,3}[가-힣]\d{4}$
```

- 앞자리 숫자: 2~3자리
- 한글: 1글자
- 뒷자리 숫자: 4자리
- 예시: `12가3456`, `123가4567`

지역명이 포함된 번호판이나 특수 형식은 현재 검증 범위에 포함하지 않습니다.

## 3. 주요 구현 기능

### 분할 영상 연속 처리

`sample_lp_part_01.mp4`부터 `sample_lp_part_15.mp4`까지 순서대로 재생합니다.

- 실행 전에 입력 파일 15개의 존재 여부 확인
- 모델은 한 번만 로드하고 전체 영상에서 재사용
- 영상 전환 시 프레임 번호와 인식 상태 유지
- 분할 영상 간 FPS와 해상도 일치 여부 확인
- 전체 처리 결과를 하나의 `lp_output.mp4`로 저장

현재 버전은 **앞뒤 검은 화면이나 카운트다운 없이 본 영상을 바로 처리**합니다.

### 영상 표시와 AI 추론 분리

메인 스레드는 영상 읽기·화면 표시·저장을 담당하고, 하나의 작업 스레드는 차량 검출과 번호판 인식을 담당합니다.

| 설정 | 동작 |
|---|---|
| `DETECTION_INTERVAL = 5` | 마지막 작업 제출 이후 최소 5프레임 간격으로 추론 요청 |
| `max_workers = 1` | 하나의 전용 작업 스레드 사용 |
| `pending.done()` | 완료된 결과만 가져와 화면 대기 방지 |
| 미완료 작업 1개 유지 | AI가 바쁘면 새 작업을 쌓지 않고 영상 재생 지속 |
| `RESULT_MAX_AGE_SECONDS = 0.5` | 원본 프레임 기준 약 0.5초 이내의 결과만 표시 |

추론 처리 시간이 길면 일부 추론 요청을 생략하지만, **영상 프레임 자체를 생략하도록 구현하지는 않았습니다.**

### 인식 결과 시각화

- 번호판 위치: 빨간색 테두리
- 번호판 문자열: 초록색 한글 표시
- 글자 크기: **40px**
- 글자 굵기: **2px 획 추가**
- 화면 상단: 처리 FPS 표시
- 종료 키: **`q`**

한글 폰트는 `lru_cache`로 캐싱해 매 프레임 다시 읽지 않도록 했습니다.


## 4. 검증 및 결과

### 입력 영상 준비

원본 영상을 재인코딩 없이 키프레임 기준으로 분할했습니다.

| 항목 | 결과 |
|---|---|
| 전체 영상 길이 | 79.7초 |
| 해상도 | 1280×720 |
| FPS | 30 |
| 분할 개수 | 15개 |
| 개별 파일 용량 | 모두 10MB 미만 |
| 최대 파일 용량 | 약 9.79MB |
| 전체 프레임 | 2,391개 |
| 음성 | 제거 |

분할 전후 영상의 전체 프레임을 비교해 누락이나 화질 변경 없이 이어지는지 확인했습니다.

### 실행 성능 및 동작 검증

- 최적화 과정에서 실제 GPU 인식과 영상 저장을 포함한 짧은 테스트에서 **약 28fps** 측정
- 해당 측정은 실제 화면 창을 제외한 환경에서 수행
- 실제 번호판 인식 결과 생성 및 다음 분할 영상 전환 확인
- 별도 시뮬레이션으로 15개 영상의 처리 순서와 2,391프레임 흐름 검증
- 느린 추론 상황에서 작업이 누적되지 않는지 확인
- `q` 종료와 오류 발생 시 자원 해제 확인

약 28fps는 제한된 테스트에서 관찰한 값이며, 전체 영상이나 모든 실행 환경에서 보장되는 성능은 아닙니다.

정답 라벨을 갖춘 평가 데이터셋으로 번호판 인식 정확도를 측정하지 않았으므로, 정확도 수치는 제시하지 않았습니다.

## 5. 한계 및 향후 개선

### 현재 한계

- 정규식으로 허용한 일부 번호판 형식만 처리
- 비동기 처리로 인해 번호판 표시가 현재 영상보다 늦을 수 있음
- 최근 인식 결과를 재사용하며, 차량별 추적 ID는 관리하지 않음
- 작은 번호판, 흔들림, 역광, 기울어진 영상에서 인식 품질 저하 가능
- AI 처리 시간이 길면 영상은 계속 재생되지만 인식 갱신 빈도는 감소

### 향후 개선 방향

- 객체 추적을 적용해 번호판 위치의 시간적 일관성 개선
- 여러 프레임의 OCR 결과를 모아 번호판 문자열 안정화
- 번호판 기울기 보정과 영상 전처리 비교
- 지역명·영업용·특수 번호판 형식 지원 확대
- 정답 라벨 기반 인식 정확도 및 지연 시간 평가

이번 프로젝트를 통해 모델 추론 속도뿐 아니라 **작업 대기, 스레드 구성, 화면 표시, 영상 저장까지 함께 고려해야 재생 품질을 개선할 수 있음**을 확인했습니다.


## 6. 시연 영상

https://github.com/user-attachments/assets/000004e4-61c5-45e4-be8d-9258df46cb66

</details>
