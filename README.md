<details>
<summary>ENG (English Version)</summary>

# Image Processing Fundamentals

### 1. Image Processing vs Computer Vision
- **Image Processing:** Transforms input images to output images using filtering, FFT; no scene understanding required.
- **Computer Vision:** Extracts knowledge from images (objects, activities, distances) using image processing + machine learning.
- **Relationship:** Image processing serves as foundation for higher-level vision tasks.

### 2. Biological Inspiration
- **Human Vision Speed:** Thorpe et al. (1996) show animal/non-animal recognition in 150ms; highlights efficiency gap.
- **Nobel Recognition:** Hubel & Wiesel (1981) Nobel Prize for visual cortex neuron discoveries influencing computational models.
- **Vision Goals:** Bridge human perception (scenes) vs computer pixel grids.

### 3. Vision Applications
- **Measurement Device:** Feature matching for 3D structure, motion recovery, dense depth maps, 3D modeling.
- **Real-World Uses:** Medical imaging, surveillance, face/smile detection, biometrics (iris, fingerprints), OCR, license plates.
- **Consumer Tech:** Google Goggles visual search, automotive safety (pedestrian detection), Kinect, robotics.

### 4. Vision in Everyday Products
- **Digital Cameras:** Face detection/recognition, smile shutter capture.
- **Software:** iPhoto face tagging, Microsoft Photosynth 3D modeling, Google Street View.
- **Industry:** Supermarket checkout monitoring, NASA Mars rover navigation (panorama, obstacle detection).

### 5. Image Representation Basics
- **Pixel Values:** Grayscale (0-255 intensity), Color (RGB, Lab, HSV); images as discrete 2D functions.
- **Matrix Format:** OpenCV stores grayscale as m×n matrix, color as m×n×3; upper-left (0,0) origin.
- **From Continuous to Discrete:** Sampling 2D space on regular grid creates pixel arrays.

### 6. Why Study Computer Vision
- **Ubiquity:** Images/videos everywhere; enables AI understanding of visual world.
- **Practical Impact:** Powers medical diagnosis, security, entertainment, autonomous systems.

</details>

<details open>
<summary>KOR (한국어 버전)</summary>

# 이미지 처리 기초

### 1. 이미지 처리 vs 컴퓨터 비전
- **이미지 처리:** 필터링, FFT 등으로 입력 이미지를 출력 이미지로 변환; 장면 이해 불필요.
- **컴퓨터 비전:** 이미지에서 객체, 활동, 거리 등 지식 추출; 이미지 처리 + 머신러닝 활용.
- **관계:** 이미지 처리가 고수준 비전 작업의 기반 역할.

### 2. 생물학적 영감
- **인간 시각 속도:** Thorpe et al.(1996)이 동물/비동물 인식 150ms; 효율성 격차 강조.
- **노벨상:** Hubel & Wiesel(1981)이 시각 피질 뉴런 발견으로 계산 모델 영향.
- **비전 목표:** 인간 인식(장면)과 컴퓨터 픽셀 격자 연결.

### 3. 비전 응용 분야
- **측정 장치:** 특징 매칭으로 3D 구조, 움직임 복원, 깊이 맵, 3D 모델링.
- **실제 활용:** 의료 영상, 감시, 얼굴/미소 탐지, 생체인식(홍채, 지문), OCR, 번호판.
- **소비자 기술:** Google Goggles 시각 검색, 자동차 안전(보행자 탐지), Kinect, 로보틱스.

### 4. 일상 제품에서의 비전
- **디지털 카메라:** 얼굴 탐지/인식, 미소 셔터 캡처.
- **소프트웨어:** iPhoto 얼굴 태깅, Microsoft Photosynth 3D 모델링, Google 스트리트뷰.
- **산업:** 슈퍼마켓 체크아웃 모니터링, NASA 화성 로버 항법(파노라마, 장애물 탐지).

### 5. 이미지 표현 기초
- **픽셀 값:** 그레이스케일(0-255 밝기), 컬러(RGB, Lab, HSV); 2D 이산 함수.
- **행렬 형식:** OpenCV에서 그레이스케일 m×n 행렬, 컬러 m×n×3; 좌상단 (0,0) 원점.
- **연속→이산:** 2D 공간을 규칙 격자로 샘플링해 픽셀 배열 생성.

### 6. 컴퓨터 비전 학습 이유
- **보편성:** 이미지/비디오 어디에나 존재; AI 시각 세계 이해 가능.
- **실제 영향:** 의료 진단, 보안, 엔터테인먼트, 자율 시스템 동력.

</details>
