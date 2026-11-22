<details>
<summary>ENG (English Version)</summary>

# Edge Detection Techniques

### 1. Edge Detection Fundamentals
- **Historical Context:** Edges central to human drawing from cave paintings (30,000 BC) to modern line art.
- **Biological Basis:** Hubel & Wiesel (1960s) discovered edge-sensitive neurons (R-G double opponent cells).
- **Importance:** Edges encode geometry (vanishing points), surface properties, depth discontinuities.

### 2. Edge Origins and Descriptors
- **Sources:** Surface normal changes, depth discontinuities, color changes, shadows, illumination variations.
- **Descriptors:** Edge strength (gradient magnitude), direction (normal), precise position.
- **Mathematical Basis:** Edges as intensity function extrema; first derivative peaks, second derivative zero-crossings.

### 3. Gradient-Based Detection
- **Image Gradient:** ∇f = [Gx, Gy]; direction perpendicular to edge; Sobel/Prewitt operators approximate.
- **Noise Effects:** Finite differences amplify noise; Gaussian smoothing + differentiation solves this.
- **Derivative of Gaussian (DoG):** Combines smoothing (σ) and differentiation for robust edge detection.

### 4. Edge Detection Pipeline
- **Steps Overview:** Gradient computation → Non-maximum suppression → Hysteresis thresholding → Edge linking.
- **Noise-Localization Tradeoff:** Larger Gaussian σ smooths more but localizes edges poorly.
- **Scale Effects:** Small σ detects fine details; large σ captures major structural edges.

### 5. Canny Edge Detector
- **Algorithm Steps:** 1) Gaussian smoothing, 2) Gradient computation, 3) Non-max suppression, 4) Double-threshold hysteresis.
- **Advantages:** Optimal for step edges + Gaussian noise; thin, continuous edges.
- **Example Results:** Lena image shows clean edges after full pipeline processing.

### 6. Advanced Considerations
- **Multi-scale Detection:** Varying σ balances fine/coarse features.
- **Beyond Intensity:** Color, texture gradients for complex edge detection.
- **Applications:** Object recognition, 3D reconstruction, image segmentation.

</details>

<details open>
<summary>KOR (한국어 버전)</summary>

# 엣지 검출 기법

### 1. 엣지 검출 기초
- **역사적 맥락:** 동굴 벽화(기원전 30,000년)부터 현대 선화까지 엣지가 핵심.
- **생물학적 기반:** Hubel & Wiesel(1960년대)이 엣지 감지 뉴런(R-G 이중 반대 세포) 발견.
- **중요성:** 기하학(소실점), 표면 특성, 깊이 불연속성 인코딩.

### 2. 엣지 기원과 기술자
- **발생 원인:** 표면 법선 변화, 깊이 불연속, 색상 변화, 그림자, 조명 변화.
- **기술자:** 엣지 강도(기울기 크기), 방향(법선), 정확 위치.
- **수학적 기반:** 강도 함수 극대값; 1차 미분 피크, 2차 미분 영교차.

### 3. 기울기 기반 검출
- **이미지 기울기:** ∇f = [Gx, Gy]; 엣지와 수직 방향; Sobel/Prewitt 연산자 근사.
- **노이즈 영향:** 유한 차분이 노이즈 증폭; 가우시안 스무딩 + 미분 해결.
- **가우시안 미분(DoG):** 스무딩(σ) + 미분 결합으로 견고한 엣지 검출.

### 4. 엣지 검출 파이프라인
- **단계 개요:** 기울기 계산 → 비최대 억제 → 히스테리시스 임계 → 엣지 연결.
- **노이즈-지역화 트레이드오프:** 큰 가우시안 σ는 스무딩 우수하나 엣지 위치 부정확.
- **스케일 효과:** 작은 σ는 세밀 특징, 큰 σ는 주요 구조 엣지 탐지.

### 5. Canny 엣지 검출기
- **알고리즘 단계:** 1) 가우시안 스무딩, 2) 기울기 계산, 3) 비최대 억제, 4) 이중 임계 히스테리시스.
- **장점:** 계단 엣지 + 가우시안 노이즈에 최적; 얇고 연속 엣지.
- **예시 결과:** Lena 이미지 전체 파이프라인 후 깨끗한 엣지 생성.

### 6. 고급 고려사항
- **다중 스케일 검출:** σ 변화로 세밀/거친 특징 균형.
- **강도 초월:** 색상, 텍스처 기울기로 복잡 엣지 검출.
- **응용:** 객체 인식, 3D 재구성, 이미지 분할.

</details>
