<details>
<summary>ENG (English Version)</summary>

# Face Detection Techniques

### 1. Face Detection Applications
- **Security Uses:** Entrance monitoring, criminal identification, driver license verification.
- **Biometrics Comparison:** Faces offer non-intrusive, natural identification advantages.
- **Historical Development:** From 1959 research to OpenCV Viola-Jones implementation.

### 2. Face Detection Importance
- **Special Nature:** Faces have unique structural properties making detection feasible.
- **Video Coding:** ATR applications compress facial regions efficiently.
- **Real-world Impact:** Powers surveillance, access control, human-computer interaction.

### 3. Detection Approaches Overview
- **Knowledge-based:** Top-down rules for facial features (eyes, nose, mouth).
- **Color-based:** HSV/YCrCb skin segmentation creates binary face candidate masks.
- **Template/feature Methods:** Matching patterns or Haar-like features.

### 4. Learning-based Methods
- **Neural Networks:** Rowley et al. (PAMI 98) trained nets for face classification.
- **SVMs:** Heisele & Poggio (CVPR 01) support vector machines for detection.
- **Boosting:** Viola-Jones (ICCV 01) uses Haar features with AdaBoost cascade.

### 5. Viola-Jones Algorithm
- **Haar Features:** 160,000 possible 24x24 features; detect edges, lines efficiently.
- **Cascade Classifier:** Multi-scale detection; rejects non-faces quickly.
- **Key Innovation:** Real-time performance through integral images and boosting.

### 6. Advanced Features
- **Edge Orientation Histogram (EOH):** Levy & Weiss (2004); small training sets, good generalization.
- **EOH Process:** Edge detection → orientation histograms → classification.
- **Advantages:** Faster classifiers, better results from limited positive/negative samples.

</details>

<details open>
<summary>KOR (한국어 버전)</summary>

# 얼굴 검출 기법

### 1. 얼굴 검출 응용
- **보안 활용:** 출입문 모니터링, 범죄자 식별, 운전면허 인증.
- **생체인식 비교:** 비침습적, 자연스러운 식별 장점.
- **발전 역사:** 1959 연구부터 OpenCV Viola-Jones 구현까지.

### 2. 얼굴 검출 중요성
- **특수성:** 얼굴 고유 구조적 특성으로 검출 가능.
- **비디오 코딩:** ATR에서 얼굴 영역 효율적 압축.
- **실제 영향:** 감시, 출입 제어, 인간-컴퓨터 상호작용 동력.

### 3. 검출 접근법 개요
- **지식 기반:** 상향식 규칙으로 눈, 코, 입 등 얼굴 특징.
- **색상 기반:** HSV/YCrCb 피부 분할로 이진 얼굴 후보 마스크 생성.
- **템플릿/특징 방법:** 패턴 매칭 또는 Haar-like 특징.

### 4. 학습 기반 방법
- **신경망:** Rowley et al.(PAMI 98)이 얼굴 분류용 네트워크 학습.
- **SVM:** Heisele & Poggio(CVPR 01)이 검출용 서포트 벡터 머신.
- **부스팅:** Viola-Jones(ICCV 01)가 Haar 특징 + AdaBoost 캐스케이드.

### 5. Viola-Jones 알고리즘
- **Haar 특징:** 24x24에서 160,000 가능 특징; 엣지, 선 효율적 탐지.
- **캐스케이드 분류기:** 다중 스케일 검출; 비얼굴 빠른 거부.
- **핵심 혁신:** 적분 이미지와 부스팅으로 실시간 성능.

### 6. 고급 특징
- **엣지 방향 히스토그램(EOH):** Levy & Weiss(2004); 소규모 학습세트, 우수 일반화.
- **EOH 과정:** 엣지 검출 → 방향 히스토그램 → 분류.
- **장점:** 빠른 분류기, 제한적 양/음 샘플로 우수 결과.

</details>
