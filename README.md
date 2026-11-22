<details>
<summary>ENG (English Version)</summary>

# Line Detection Techniques

### 1. Line Detection Motivation
- **Parametric Space Search:** Line fitting treats detection as search problem in parameter space (ρ, θ).
- **Practical Need:** Essential for shape analysis, object recognition, geometric feature extraction.
- **Challenges:** Noise, missing segments, outliers complicate direct fitting.

### 2. Hough Transform Fundamentals
- **Core Principle:** Maps edge points from image space to parameter space (sinusoids intersect at line parameters).
- **Vertical Line Example:** Discrete parameter space; accumulator array votes for line presence.
- **Preprocessing:** Edge detection (Canny/Sobel) + thresholding creates binary edge map.

### 3. Hough Transform Process
- **Parameter Space Discretization:** Non-linear quantization for angle θ; constrained ρ range.
- **Voting Mechanism:** Each edge point votes for possible lines; peaks indicate strong lines.
- **Accumulator Analysis:** Local maxima detection finds best line fits despite gaps/noise.

### 4. Robustness Advantages
- **Noise Insensitivity:** Accumulator smooths local data imprecision and image noise.
- **Gap Tolerance:** Handles missing line parts, non-linear structures effectively.
- **Outlier Handling:** Distinguishes inliers from outliers through voting consensus.

### 5. Practical Implementation
- **Edge Map Quality:** Canny with varying thresholds (0.1, 0.15) affects detection sensitivity.
- **Peak Detection:** Non-trivial local maxima finding in accumulator array.
- **Applications:** MR brain images, geometric pattern recognition.

### 6. Line Fitting Examples
- **Input Processing:** Original → Edge detection → Hough voting → Line extraction.
- **Threshold Impact:** Lower thresholds detect more edges but increase noise/false positives.

</details>

<details open>
<summary>KOR (한국어 버전)</summary>

# 직선 검출 기법

### 1. 직선 검출 동기
- **매개변수 공간 탐색:** 직선 피팅을 (ρ, θ) 매개변수 공간에서의 탐색 문제로 처리.
- **실용적 필요:** 형상 분석, 객체 인식, 기하학적 특징 추출에 필수.
- **도전 과제:** 노이즈, 누락된 세그먼트, 이상치가 직접 피팅 복잡화.

### 2. Hough 변환 기초
- **핵심 원리:** 이미지 공간 엣지점을 매개변수 공간으로 매핑(사인파 교차로 직선 매개변수 결정).
- **수직선 예시:** 이산 매개변수 공간; 어큐뮬레이터 배열로 직선 존재 투표.
- **전처리:** 엣지 검출(Canny/Sobel) + 임계값으로 이진 엣지 맵 생성.

### 3. Hough 변환 과정
- **매개변수 공간 이산화:** 각도 θ 비선형 양자화; ρ 범위 제한.
- **투표 메커니즘:** 각 엣지점이 가능한 직선에 투표; 피크가 강한 직선 나타냄.
- **어큐뮬레이터 분석:** 갭/노이즈에도 불구하고 국부 최대값으로 최적 직선 탐지.

### 4. 견고성 장점
- **노이즈 비감도:** 어큐뮬레이터가 국소 데이터 부정확성/이미지 노이즈 완화.
- **갭 허용:** 누락된 직선 부분, 비선형 구조 효과적 처리.
- **이상치 처리:** 투표 합의로 인라이어와 아웃라이어 구분.

### 5. 실무 구현
- **엣지 맵 품질:** Canny 다양한 임계값(0.1, 0.15)으로 감도 조절.
- **피크 탐지:** 어큐뮬레이터 배열에서 비자명 국부 최대값 찾기.
- **응용:** MR 뇌 영상, 기하학적 패턴 인식.

### 6. 직선 피팅 예시
- **입력 처리:** 원본 → 엣지 검출 → Hough 투표 → 직선 추출.
- **임계값 영향:** 낮은 임계값은 더 많은 엣지 탐지만 노이즈/오탐 증가.

</details>
