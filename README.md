<details>
<summary>ENG (English Version)</summary>

# Face Recognition Techniques

### 1. Face Recognition Challenges
- **Variability Factors:** Aging, occlusion, pose variations, lighting, facial expressions complicate matching.
- **Identity Consistency:** Same person appears drastically different (e.g., Madonna's many faces).
- **Real-world Complexity:** Dynamic conditions require robust feature extraction.

### 2. Principal Component Analysis (PCA)
- **Eigenfaces Method:** Turk & Pentland; represents faces as linear combinations of principal components.
- **Dimensionality Reduction:** Projects high-dimensional face images onto low-dimensional eigenspace.
- **Recognition Process:** Minimize reconstruction error to nearest training face.

### 3. Commercial Systems
- **Cognitec FaceVACS:** Local image transforms at fixed locations capture distinguishing features.
- **Feature Extraction:** Spatial frequency amplitudes form feature vectors; cluster centers as references.
- **Optimization:** Maximizes between-class scatter (Sb) over within-class scatter (Sw).

### 4. Performance Benchmarks
- **FRVT Evaluations:** State-of-the-art results from 2006 to 2013 (1.6M subjects).
- **Progress Tracking:** NIST Face Recognition Vendor Tests measure accuracy improvements.
- **Large-scale Testing:** Validates systems on massive identity databases.

### 5. 3D Morphable Models
- **3D Face Representation:** Linear combinations of shape/texture from 3D scan database.
- **Model Fitting:** Estimates optimal coefficients to match input 2D/3D face images.
- **Active Appearance Model (AAM):** 3D extension for pose/appearance normalization.

### 6. 3D Acquisition and Processing
- **Scanning Technology:** Minolta Vivid 910 (1sec, 320x240, 0.1mm depth); produces intensity + range images.
- **Model Construction:** Stitch 5 scans, denoise, decimate to 50K polygons using Geomagic/RapidForm.
- **Applications:** Pose correction, aging modeling in 3D domain for frontal normalization.

</details>

<details open>
<summary>KOR (한국어 버전)</summary>

# 얼굴 인식 기법

### 1. 얼굴 인식 도전 과제
- **변동성 요인:** 노화, 가림, 자세 변화, 조명, 표정으로 매칭 복잡.
- **신원 일관성:** 동일 인물 극단적 차이(예: Madonna 다중 얼굴).
- **실세계 복잡성:** 동적 조건에서 견고한 특징 추출 필요.

### 2. 주성분 분석 (PCA)
- **Eigenfaces 방법:** Turk & Pentland; 주성분 선형 결합으로 얼굴 표현.
- **차원 축소:** 고차원 얼굴 이미지를 저차원 고유공간으로 투영.
- **인식 과정:** 훈련 얼굴 중 재구성 오차 최소화.

### 3. 상용 시스템
- **Cognitec FaceVACS:** 고정 위치 국소 이미지 변환으로 구별 특징 포착.
- **특징 추출:** 공간 주파수 진폭으로 특징 벡터 생성; 클러스터 중심 참조.
- **최적화:** 클래스 내 산포(Sw) 대비 클래스 간 산포(Sb) 최대화.

### 4. 성능 벤치마크
- **FRVT 평가:** 2006년부터 2013년(160만 주체) 최신 결과.
- **진행 추적:** NIST 얼굴 인식 벤더 테스트로 정확도 향상 측정.
- **대규모 테스트:** 대형 신원 데이터베이스에서 시스템 검증.

### 5. 3D 모르퍼블 모델
- **3D 얼굴 표현:** 3D 스캔 데이터베이스 형상/텍스처 선형 결합.
- **모델 피팅:** 입력 2D/3D 얼굴 이미지 최적 계수 추정.
- **활성 외관 모델(AAM):** 자세/외관 정규화를 위한 3D 확장.

### 6. 3D 획득 및 처리
- **스캐닝 기술:** Minolta Vivid 910(1초, 320x240, 0.1mm 깊이); 강도+범위 이미지.
- **모델 구성:** 5개 스캔 스티칭, 노이즈 제거, 50K 폴리곤으로 디시메이션(Geomagic/RapidForm).
- **응용:** 3D 영역 정면 정규화 위한 자세 보정, 노화 모델링.

</details>
