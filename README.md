<details>
<summary>ENG (English Version)</summary>

# Real-Time Drowsiness Detection

## 1. Project Overview

A Windows webcam project that detects sustained eye closure using facial landmarks, an Eye Aspect Ratio (EAR), and elapsed time.

The implementation improves an existing drowsiness detection program with eye-contour visualization, face-boundary alerts, asynchronous beep playback, and rapid alarm cancellation when the eyes reopen.

The warning threshold is based on elapsed seconds rather than frame counts.

## 2. Key Features

| Feature | Implementation |
|---|---|
| Mirrored webcam preview | Resize frames to 720px wide and flip horizontally |
| Face detection | Use the dlib frontal face detector |
| Facial landmarks | Extract 68 landmarks with a pretrained model |
| Eye visualization | Draw eye contours with polylines only |
| Eye-closure warning | Display WARNING and start a beep after 1 second of detected eye closure |
| Drowsiness warning | Display an additional message after 2 seconds |
| Rapid alarm cancellation | Reset the timer and stop the eye-closure alarm when the eyes reopen |
| Face-boundary alert | Play a beep when no face is detected or face coordinates reach the frame boundary |
| Asynchronous audio | Continue processing video while the alarm plays |
| Resource cleanup | Stop audio, remove the temporary alarm WAV file, and release the camera on exit |

> Timing starts when eye closure is first detected. A warning is activated on the first processed frame at or after the threshold. Actual response latency still depends on frame acquisition and processing.

## 3. Technology Stack

| Technology | Role |
|---|---|
| Python | Application implementation |
| OpenCV | Webcam capture, preprocessing, and visualization |
| dlib | Face detection and 68-point landmark prediction |
| imutils | Frame resizing and landmark conversion |
| NumPy | Distance calculations and audio waveform generation |
| Pillow | Korean warning text rendering |
| time | Elapsed-time measurement with `monotonic()` |
| winsound | Asynchronous Windows audio playback |
| wave | WAV file generation |
| tempfile / shutil / os | Temporary files and model path handling |

The current source also imports `pygame`, so it must be installed. Beep playback is handled by `winsound`.

## 4. Algorithm

### Processing Flow

```text
Webcam frame
    ↓
Resize → Horizontal flip → Grayscale conversion
    ↓
Face detection and boundary check
    ├─ Missing or incomplete face
    │      → Reset eye-closure timer → Play beep
    │
    └─ Valid face
           ↓
       Predict landmarks and check their boundaries
           ↓
       Extract eye coordinates and calculate EAR
           ├─ EAR ≥ 0.30
           │      → Reset eye-closure timer
           │      → Stop beep if no other warning remains
           │
           └─ EAR < 0.30
                  → Measure elapsed eye-closure time
                  → At least 1 second: WARNING + beep
                  → At least 2 seconds: Additional drowsiness message
```

### Eye Aspect Ratio

The implementation uses the following formula:

```python
def eye_aspect_ratio(eye):
    a = euclidean_dist(eye[1], eye[5])
    b = euclidean_dist(eye[2], eye[4])
    c = euclidean_dist(eye[0], eye[3])

    return (a + b) / (1.5 * c)
```

- `a`, `b`: Vertical distances between eye landmarks
- `c`: Horizontal distance between the eye corners
- Final EAR: Average of the left and right eye values
- Eye-closure candidate threshold: `EAR < 0.30`

### Elapsed-Time Detection

```python
EAR_THRESHOLD = 0.30
EYE_CLOSED_WARNING_SECONDS = 1.0
DROWSINESS_WARNING_SECONDS = 2.0
g_eye_closed_since = None

def get_eye_closed_duration(ear, now):
    global g_eye_closed_since

    if ear >= EAR_THRESHOLD:
        g_eye_closed_since = None
        return 0.0

    if g_eye_closed_since is None:
        g_eye_closed_since = now

    return now - g_eye_closed_since
```

The processing loop calls:

```python
closed_duration = get_eye_closed_duration(
    ear,
    time.monotonic()
)
```

This replaces the previous frame counter and moving-average logic. The timer measures elapsed time without relying on a target FPS or changes to the system clock.

### Detection Parameters

| Parameter | Value |
|---|---|
| Frame width | 720px |
| EAR threshold | 0.30 |
| Eye-closure warning threshold | 1.0 second |
| Additional drowsiness warning threshold | 2.0 seconds |
| Face-boundary margin | 10px |
| Beep frequency | 1000Hz |
| Tone duration | 200ms |
| Silence between tones | 100ms |

### Asynchronous Beep Playback

The `BeepAlarm` class generates a temporary WAV file and starts asynchronous looping playback:

```python
winsound.PlaySound(
    self.filename,
    winsound.SND_FILENAME
    | winsound.SND_ASYNC
    | winsound.SND_LOOP
    | winsound.SND_NODEFAULT
)
```

When the warning ends, playback is stopped:

```python
winsound.PlaySound(None, 0)
```

Playback commands are issued only when the alarm state changes.

If the eyes reopen while the face remains missing or near the frame boundary, the face alert continues.

## 5. Improvements and Problem Solving

| Problem | Improvement |
|---|---|
| Eye-point markers overlapped the contours | Removed point markers and retained polylines |
| Incomplete faces produced unreliable eye measurements | Added face-box and landmark boundary checks |
| Thirty frames took longer than one second at low FPS | Replaced frame counts with elapsed-time thresholds |
| Historical measurements delayed alarm cancellation | Used the current EAR and reset the timer when the eyes reopen |
| Synchronous beeps delayed video processing | Replaced blocking `Beep()` calls with asynchronous `PlaySound()` |
| Audio and camera resources needed cleanup | Added cleanup in `finally` |

## 6. How to Run

### Clone the Repository

```powershell
git clone --branch Project-Test.03 --single-branch https://github.com/LutetiumCalciumLee/Studying_Computer_Vision.git
cd Studying_Computer_Vision
```

### Install Dependencies

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
python -m pip install opencv-python dlib imutils numpy pillow pygame
```

Some environments may require CMake and C++ build tools to install dlib.

### Prepare the Landmark Model

Download and extract:

https://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2

The current source expects the model in `../00_Dataset`, relative to the script:

```text
workspace/
├── 00_Dataset/
│   └── shape_predictor_68_face_landmarks.dat
└── Studying_Computer_Vision/
    ├── README.md
    └── 06.detect_drowsiness.py
```

Modify `Dataset_dir` or `dataset_file` if using a different location.

### Run

```powershell
python 06.detect_drowsiness.py
```

- Default camera index: `0`
- Korean font: `C:\Windows\Fonts\gulim.ttc`
- Alarm WAV file: Generated automatically
- Exit: Press `q` in the video window

## 7. Validation

Mock timestamps, frames, landmark coordinates, and audio playback functions were used to verify the control logic.

| Check | Result |
|---|---|
| Python syntax | Passed |
| One-second and two-second thresholds at simulated 5–60 FPS | Passed |
| Warning thresholds with irregular frame intervals | Passed |
| Alarm cancellation on the first open-eye frame | Passed |
| Timer reset after a brief eye closure | Passed |
| Timer reset when face detection becomes invalid | Passed |
| No repeated playback restart during an active alarm | Passed |
| Missing-face and boundary alerts | Passed |
| Alarm waveform generation and shutdown cleanup | Passed |

Real-camera FPS, detection accuracy, and end-to-end response latency have not been measured.

## 8. Limitations and Future Improvements

### Current Limitations

- A fixed EAR threshold may be affected by individual differences, glasses, lighting, and face angle.
- Time thresholds do not depend on frame counts, but state changes can only be detected when a frame is processed.
- Boundary checks depend on detected face boxes and predicted landmarks.
- Multiple detected faces share the same timer, so the current implementation is intended for a single monitored person.
- Audio playback and the font path depend on Windows.

### Planned Improvements

- Calibrate EAR thresholds for each user
- Use separate opening and closing thresholds to reduce state flicker
- Separate tracking and warning state for multiple faces
- Measure FPS, false alarms, and activation/cancellation latency
- Move configuration values into a settings file or command-line options
- Support platform-specific audio and font handling

## 9. Learning Points

- Applying pretrained facial landmark models
- Calculating geometric eye features
- Measuring eye-closure duration with a monotonic clock
- Designing timer reset and warning-state transitions
- Integrating asynchronous audio with video processing
- Handling model paths and Korean text rendering
- Releasing resources with `try/finally`

## 10. Demo Video

https://github.com/user-attachments/assets/9c694820-18da-49e7-9ca0-6a4fa41f6cbc

Demo checklist:

- Mirrored webcam preview and eye-contour visualization
- WARNING and beep playback after 1 second of detected eye closure
- An additional drowsiness message after 2 seconds
- Rapid beep cancellation when the eyes reopen
- Beep alerts when the face leaves the frame or is not detected

</details>

<details>
<summary>KOR (한국어 버전)</summary>

# 실시간 졸음 감지 시스템

## 1. 프로젝트 개요

웹캠 영상에서 얼굴과 눈의 랜드마크를 추출하고, 눈 종횡비(EAR)와 실제 눈 감음 지속 시간을 이용해 경고를 제공하는 Windows 기반 컴퓨터 비전 프로젝트입니다.

기존 졸음 감지 프로그램을 개선하여 눈 윤곽선 시각화, 얼굴 이탈 알림, 비동기 경고음 재생, 눈을 다시 떴을 때 빠르게 경고를 해제하는 기능을 구현했습니다.

경고 발생 기준을 프레임 수에서 실제 경과 시간으로 변경했습니다.

## 2. 주요 기능

| 기능 | 구현 내용 |
|---|---|
| 좌우반전 웹캠 화면 | 프레임을 폭 720px로 조정하고 거울처럼 표시 |
| 얼굴 검출 | dlib 정면 얼굴 검출기 사용 |
| 얼굴 랜드마크 | 사전 학습된 모델로 68개 좌표 추출 |
| 눈 윤곽선 표시 | 눈 좌표를 선으로 연결하고 점 표시는 제거 |
| 눈 감음 경고 | 눈 감음 감지 후 1초가 경과하면 WARNING 문구와 경고음 제공 |
| 졸음 경고 | 2초가 경과하면 추가 졸음 경고 문구 표시 |
| 빠른 경고 해제 | 눈 뜸을 감지하면 타이머 초기화 및 눈 감음 경고음 중지 |
| 얼굴 이탈 알림 | 얼굴 미검출 또는 얼굴 좌표가 화면 경계에 걸리면 경고음 재생 |
| 비동기 소리 재생 | 경고음 재생 중에도 영상 처리 계속 |
| 자원 정리 | 종료 시 경고음 중지, 임시 경고음 파일 삭제, 카메라 해제 |

> 눈 감음을 처음 감지한 시점부터 시간을 측정합니다. 설정 시간이 지난 후 처리되는 첫 프레임에서 경고를 시작하므로, 실제 반응 지연은 영상 수신과 처리 속도의 영향을 받습니다.

## 3. 기술 스택

| 기술 | 역할 |
|---|---|
| Python | 전체 프로그램 구현 |
| OpenCV | 웹캠 입력, 전처리 및 화면 표시 |
| dlib | 얼굴 검출 및 68점 랜드마크 예측 |
| imutils | 영상 크기 조정 및 랜드마크 배열 변환 |
| NumPy | 좌표 거리 계산과 경고음 파형 생성 |
| Pillow | 한글 경고 문구 렌더링 |
| time | `monotonic()`을 이용한 경과 시간 측정 |
| winsound | Windows 비동기 경고음 재생 및 중지 |
| wave | WAV 파일 생성 |
| tempfile / shutil / os | 임시 파일과 모델 경로 관리 |

현재 소스에는 `pygame` import도 남아 있어 실행 환경에 설치가 필요합니다. 경고음 재생은 `winsound`가 담당합니다.

## 4. 알고리즘

### 전체 처리 흐름

```text
웹캠 프레임 수신
    ↓
크기 조정 → 좌우반전 → 그레이스케일 변환
    ↓
얼굴 검출 및 화면 경계 확인
    ├─ 얼굴 미검출 / 경계 이탈
    │      → 눈 감음 타이머 초기화 → 경고음 재생
    │
    └─ 얼굴 영역 정상
           ↓
       랜드마크 예측 및 화면 경계 확인
           ↓
       양쪽 눈 좌표 추출 및 EAR 계산
           ├─ EAR ≥ 0.30
           │      → 눈 감음 타이머 초기화
           │      → 다른 경고 원인이 없으면 경고음 중지
           │
           └─ EAR < 0.30
                  → 눈 감음 경과 시간 측정
                  → 1초 이상: WARNING + 경고음
                  → 2초 이상: 졸음 경고 문구 추가
```

### 눈 종횡비 계산

각 눈의 6개 랜드마크에서 세로 거리 두 개와 가로 거리 한 개를 계산합니다.

```python
def eye_aspect_ratio(eye):
    a = euclidean_dist(eye[1], eye[5])
    b = euclidean_dist(eye[2], eye[4])
    c = euclidean_dist(eye[0], eye[3])

    return (a + b) / (1.5 * c)
```

- `a`, `b`: 눈 랜드마크 사이의 세로 거리
- `c`: 눈 양 끝점 사이의 가로 거리
- 최종 EAR: 왼쪽 눈과 오른쪽 눈 EAR의 평균
- 눈 감음 후보 기준: `EAR < 0.30`

### 실제 경과 시간 기반 판단

```python
EAR_THRESHOLD = 0.30
EYE_CLOSED_WARNING_SECONDS = 1.0
DROWSINESS_WARNING_SECONDS = 2.0
g_eye_closed_since = None

def get_eye_closed_duration(ear, now):
    global g_eye_closed_since

    if ear >= EAR_THRESHOLD:
        g_eye_closed_since = None
        return 0.0

    if g_eye_closed_since is None:
        g_eye_closed_since = now

    return now - g_eye_closed_since
```

영상 처리 루프에서는 다음과 같이 호출합니다.

```python
closed_duration = get_eye_closed_duration(
    ear,
    time.monotonic()
)
```

기존 프레임 카운터와 이동 평균 판단을 제거하고, 눈 감음 시작 시점부터 실제 경과 시간을 계산하도록 변경했습니다.

`time.monotonic()`을 사용하므로 목표 FPS나 시스템 시각 변경에 의존하지 않고 지속 시간을 측정합니다.

### 주요 설정값

| 항목 | 설정값 |
|---|---|
| 영상 폭 | 720px |
| EAR 임계값 | 0.30 |
| 눈 감음 경고 기준 | 1.0초 |
| 추가 졸음 경고 기준 | 2.0초 |
| 얼굴 경계 여유 공간 | 10px |
| 경고음 주파수 | 1000Hz |
| 경고음 길이 | 200ms |
| 경고음 사이 무음 | 100ms |

### 비동기 경고음 재생

`BeepAlarm` 클래스에서 임시 WAV 파일을 생성하고 비동기 반복 재생을 시작합니다.

```python
winsound.PlaySound(
    self.filename,
    winsound.SND_FILENAME
    | winsound.SND_ASYNC
    | winsound.SND_LOOP
    | winsound.SND_NODEFAULT
)
```

경고 상태가 해제되면 재생 중인 경고음을 중지합니다.

```python
winsound.PlaySound(None, 0)
```

알람 상태가 바뀔 때만 재생 또는 중지 요청을 보내므로, 매 프레임마다 경고음이 처음부터 다시 재생되지 않습니다.

눈을 떠도 얼굴이 검출되지 않거나 화면 경계에 걸려 있다면 얼굴 이탈 경고음은 유지됩니다.

## 5. 핵심 개선과 문제 해결

| 문제 | 개선 방법 |
|---|---|
| 눈 주변의 점이 윤곽선과 겹침 | 점 표시를 제거하고 눈 윤곽선만 유지 |
| 얼굴이 잘리면 눈 측정값이 불안정해짐 | 얼굴 검출 박스와 랜드마크 경계 검사 추가 |
| 처리 FPS가 낮으면 30프레임 도달 시간이 길어짐 | 프레임 수 대신 실제 1초·2초 경과 시간으로 판단 |
| 이전 측정값 때문에 눈 뜸 이후 경고 해제가 늦어짐 | 현재 EAR로 눈 뜸을 판단하고 타이머 즉시 초기화 |
| 경고음 재생이 영상 처리를 지연시킴 | 동기식 `Beep()`을 비동기 `PlaySound()`으로 교체 |
| 종료 시 소리와 영상 자원 정리가 필요함 | `finally`에서 경고음 중지와 자원 해제 수행 |

## 6. 실행 방법

### 저장소 다운로드

```powershell
git clone --branch Project-Test.03 --single-branch https://github.com/LutetiumCalciumLee/Studying_Computer_Vision.git
cd Studying_Computer_Vision
```

### 실행 환경 준비

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
python -m pip install opencv-python dlib imutils numpy pillow pygame
```

환경에 따라 dlib 설치 시 CMake와 C++ 빌드 도구가 필요할 수 있습니다.

### 랜드마크 모델 준비

다음 모델을 다운로드하고 압축을 해제합니다.

https://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2

현재 코드는 스크립트 기준 `../00_Dataset`에서 모델을 찾습니다.

```text
workspace/
├── 00_Dataset/
│   └── shape_predictor_68_face_landmarks.dat
└── Studying_Computer_Vision/
    ├── README.md
    └── 06.detect_drowsiness.py
```

다른 위치를 사용하려면 `Dataset_dir` 또는 `dataset_file`을 수정해야 합니다.

### 프로그램 실행

```powershell
python 06.detect_drowsiness.py
```

- 기본 카메라 인덱스: `0`
- 한글 폰트: `C:\Windows\Fonts\gulim.ttc`
- 경고음 WAV 파일: 실행 시 자동 생성
- 종료 방법: 영상 창에서 `q` 키 입력

## 7. 검증 내용

모의 시간값, 프레임, 랜드마크 좌표, 소리 재생 함수를 사용해 프로그램의 상태 전환을 확인했습니다.

| 검증 항목 | 결과 |
|---|---|
| Python 문법 검사 | 통과 |
| 모의 5~60 FPS 환경에서 1초·2초 경고 기준 확인 | 통과 |
| 불규칙한 프레임 간격에서 경과 시간 판단 | 통과 |
| 첫 눈 뜸 프레임에서 경고음 중지 | 통과 |
| 짧은 눈 감음 이후 타이머 초기화 | 통과 |
| 얼굴 검출이 유효하지 않을 때 타이머 초기화 | 통과 |
| 경고 중 매 프레임 경고음 재시작 방지 | 통과 |
| 얼굴 미검출 및 경계 이탈 경고 | 통과 |
| 경고음 파형 생성과 종료 시 자원 정리 | 통과 |

실제 웹캠 환경의 FPS, 검출 정확도, 경고음 시작·중지 지연 시간은 아직 정량 측정하지 않았습니다.

## 8. 현재 한계와 개선 계획

### 현재 한계

- 고정 EAR 임계값은 눈 모양, 안경, 조명, 얼굴 각도의 영향을 받을 수 있습니다.
- 경고 기준 시간은 프레임 수에 의존하지 않지만, 상태 변화는 프레임이 처리될 때 확인됩니다.
- 얼굴 경계 판단은 검출 박스와 예측된 랜드마크를 기준으로 합니다.
- 여러 얼굴이 검출되어도 타이머를 공유하므로, 현재 구조는 한 사람을 대상으로 사용하는 것이 적합합니다.
- 경고음 재생과 폰트 경로가 Windows 환경에 의존합니다.

### 개선 계획

- 사용자별 EAR 임계값 보정
- 눈 뜸·눈 감음 임계값 분리로 상태 전환의 흔들림 완화
- 다중 얼굴 환경에서 대상 선택과 경고 상태 분리
- FPS, 오탐률, 경고음 시작·중지 지연 시간 측정
- 설정값을 별도 파일 또는 실행 옵션으로 관리
- 운영체제별 경고음 재생 및 폰트 처리 지원

## 9. 학습 포인트

- 사전 학습된 얼굴 랜드마크 모델 활용
- 좌표 거리와 EAR를 이용한 기하학적 특징 계산
- 단조 시계를 이용한 눈 감음 지속 시간 측정
- 타이머 초기화와 경고 상태 전환 설계
- 비동기 경고음 재생과 영상 처리의 분리
- 모델 경로 처리와 한글 텍스트 렌더링
- `try/finally`를 이용한 자원 해제

## 10. 실행 영상

https://github.com/user-attachments/assets/b65d9b3f-7128-4c64-b170-097145f23c21

영상 시연 항목:

- 좌우반전된 웹캠 화면과 눈 윤곽선 표시
- 눈 감음 감지 후 1초가 경과했을 때 WARNING 문구와 경고음 발생
- 2초가 경과했을 때 추가 졸음 경고 문구 표시
- 눈을 다시 떴을 때 빠른 경고음 중지
- 얼굴이 화면을 벗어나거나 검출되지 않을 때 경고음 발생

</details>
