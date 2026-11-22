<details>
<summary>ENG (English Version)</summary>

# Color Image Processing

### 1. Color Image Processing Pipeline
- **Image Processing Hierarchy:** Color processing fits within acquisition, enhancement, segmentation, recognition workflow.
- **Image Representations:** Raster (pixel-based, f(x,y)), Vector (geometric primitives like circles, lines).
- **Acquisition Devices:** Video, infrared, hyperspectral, omnidirectional cameras capture diverse color data.

### 2. Color Fundamentals
- **Physics Basis:** Newton (1666) prism experiments; visible spectrum 400-700nm within electromagnetic range.
- **Perception Model:** Eye + illumination + scene reflection produces perceived color (radiance → luminance).
- **CIE Standards:** RGB color-matching functions define human color response across wavelengths.

### 3. Color Models Overview
- **RGB Model:** Additive primaries (Red, Green, Blue); basis for displays, 24-bit full color (8 bits/channel).
- **CMY(K) Model:** Subtractive primaries for printing (Cyan, Magenta, Yellow, Black); complements RGB.
- **HSI Model:** Hue (color type), Saturation (purity), Intensity (brightness); intuitive for human perception.

### 4. HSI Color Space Details
- **Conversion from RGB:** Separates chromaticity (H,S) from achromatic intensity; useful for segmentation.
- **Visualization:** Intensity line from black-white; constant hue triangles; saturation from white to pure color.
- **Applications:** Face detection via skin color clustering in HSI space.

### 5. Color Image Processing Techniques
- **Intensity Slicing:** Threshold-based color highlighting in 3D intensity-color space (e.g., radiation patterns, X-rays).
- **Multi-spectral Imaging:** Combine RGB + Near-Infrared for vegetation analysis, material discrimination.
- **Pixel Addressing:** C-language loops over rows, columns, channels; 8-neighbor connectivity.

### 6. Practical Color Processing
- **Spatial Masks:** Color pixels as vectors in RGB space; kernels applied channel-wise.
- **Pixel Depth:** 24-bit RGB enables 16.7M colors; grayscale single channel.
- **Applications:** Color-based object detection, image enhancement, segmentation.

</details>

<details open>
<summary>KOR (한국어 버전)</summary>

# 컬러 이미지 처리

### 1. 컬러 이미지 처리 파이프라인
- **이미지 처리 계층:** 획득, 향상, 분할, 인식 워크플로우 내 컬러 처리 위치.
- **이미지 표현:** 래스터(픽셀 기반, f(x,y)), 벡터(원, 선 등 기하 도형).
- **획득 장치:** 비디오, 적외선, 초분광, 전방위 카메라로 다양한 컬러 데이터 수집.

### 2. 컬러 기초
- **물리학 기반:** Newton(1666) 프리즘 실험; 가시광 스펙트럼 400-700nm.
- **인식 모델:** 눈 + 조명 + 장면 반사로 인지 색상(복사광 → 휘도).
- **CIE 표준:** RGB 색상 매칭 함수로 파장별 인간 색상 응답 정의.

### 3. 컬러 모델 개요
- **RGB 모델:** 가산 혼합 기본색(Red, Green, Blue); 디스플레이 기반, 24비트 풀컬러(채널당 8비트).
- **CMY(K) 모델:** 인쇄용 감산 혼합(Cyan, Magenta, Yellow, Black); RGB 보완.
- **HSI 모델:** Hue(색상), Saturation(순도), Intensity(밝기); 인간 직관적.

### 4. HSI 색 공간 상세
- **RGB 변환:** 색도(H,S)와 무색 강도 분리; 분할에 유용.
- **시각화:** 흑백 강도선; 등색상 삼각형; 백색→순수색 채도.
- **응용:** HSI 공간에서 피부색 클러스터링으로 얼굴 탐지.

### 5. 컬러 이미지 처리 기법
- **강도 슬라이싱:** 3D 강도-컬러 공간 임계값 강조(방사선 패턴, X선 등).
- **초분광 영상:** RGB + 근적외선 결합으로 식생 분석, 재질 구분.
- **픽셀 주소 지정:** C언어 행/열/채널 루프; 8-이웃 연결성.

### 6. 실무 컬러 처리
- **공간 마스크:** RGB 공간 벡터 픽셀; 채널별 커널 적용.
- **픽셀 깊이:** 24비트 RGB로 1,670만 색상; 그레이스케일 단일 채널.
- **응용:** 컬러 기반 객체 탐지, 이미지 향상, 분할.

</details>
