<details>
<summary>ENG (English Version)</summary>

# Convolutional Neural Networks (CNN)

### 1. CNN Introduction and History
- **Origin:** Developed by Yann LeCun in 1989 with LeNet-5 for document recognition; excels in image processing by preserving spatial topology.
- **Key Advantages:** Parameter sharing reduces overfitting; captures local patterns and spatial hierarchies effectively.
- **LeNet-5 Architecture:** 32x32 input → CONV1 (28x28x6) → POOL → CONV2 (10x10x16) → POOL → FC → Softmax (10 classes).

### 2. Convolution Layer Fundamentals
- **Operation:** Filter (kernel) slides over input creating feature maps (activation/response maps) that detect edges, textures.
- **RGB Processing:** Separate filters for Red, Green, Blue channels produce color-specific feature maps; ReLU activation follows.
- **Filter vs Kernel:** Filters optimized during training; Sobel filters detect vertical/horizontal edges explicitly.

### 3. Pooling Layer Operations
- **Types:** Max Pooling (largest value), Average Pooling (mean), Min Pooling; downsamples feature maps reducing computation.
- **Purpose:** Spatial invariance, overfitting prevention, dimensionality reduction (e.g., 6x6 → 3x3 with 2x2 stride=2).
- **Stride Effect:** Controls output size; larger stride reduces spatial dimensions faster.

### 4. Advanced CNN Components
- **Padding:** 'Valid' (no padding, shrinks size), 'Same' (zero-padding maintains input size); preserves edge features.
- **Dropout:** Randomly drops nodes in fully connected layers to prevent co-adaptation and overfitting.
- **Fully Connected Layers:** Flattens feature maps for final classification; follows convolutional/pooling stack.

### 5. CNN Architecture Flow
- **Typical Pipeline:** Input → Conv → Pool → Conv → Pool → Flatten → Dense (ReLU) → Softmax classification.
- **Feature Extraction:** Early layers detect edges; deeper layers learn complex patterns (hierarchical representation).
- **Backpropagation:** End-to-end training optimizes all filters/weights via gradient descent.

### 6. Implementation Example
- **Keras Structure:** Conv2D → MaxPooling2D → Flatten → Dense → Output; uses ReLU activation, softmax for multiclass.

</details>

<details open>
<summary>KOR (한국어 버전)</summary>

# 합성곱 신경망 (CNN)

### 1. CNN 소개와 역사
- **기원:** 1989 Yann LeCun의 LeNet-5로 문서 인식 개발; 공간적 토폴로지 보존하며 이미지 처리에 탁월.
- **주요 장점:** 매개변수 공유로 과적합 감소; 국부 패턴과 공간 계층 효과적 포착.
- **LeNet-5 구조:** 32x32 입력 → CONV1(28x28x6) → POOL → CONV2(10x10x16) → POOL → FC → Softmax(10 클래스).

### 2. 합성곱 계층 기초
- **연산:** 필터(커널)가 입력 위 슬라이딩하며 특징 맵(활성화/응답 맵) 생성; 모서리, 질감 탐지.
- **RGB 처리:** Red, Green, Blue 채널별 필터로 색상별 특징 맵 생성; ReLU 활성화 적용.
- **필터 vs 커널:** 학습 중 최적화되는 필터; Sobel 필터는 수직/수평 모서리 명시적 탐지.

### 3. 풀링 계층 연산
- **종류:** Max Pooling(최대값), Average Pooling(평균), Min Pooling; 특징 맵 다운샘플링으로 연산 감소.
- **목적:** 공간 불변성, 과적합 방지, 차원 축소(예: 6x6 → 3x3, 2x2 stride=2).
- **스트라이드 효과:** 출력 크기 제어; 큰 스트라이드일수록 공간 차원 빠르게 축소.

### 4. 고급 CNN 구성요소
- **패딩:** 'Valid'(패딩 없음, 크기 축소), 'Same'(제로 패딩으로 입력 크기 유지); 가장자리 특징 보존.
- **드롭아웃:** 완전 연결층에서 노드 랜덤 제거로 공동 적응 및 과적합 방지.
- **완전 연결층:** 특징 맵 평탄화 후 최종 분류; 합성곱/풀링 스택 뒤따름.

### 5. CNN 구조 흐름
- **전형적 파이프라인:** 입력 → Conv → Pool → Conv → Pool → Flatten → Dense(ReLU) → Softmax 분류.
- **특징 추출:** 초기 층은 모서리 탐지; 깊은 층은 복잡 패턴 학습(계층적 표현).
- **역전파:** 모든 필터/가중치 그라디언트 디센트로 엔드투엔드 학습.

### 6. 구현 예시
- **Keras 구조:** Conv2D → MaxPooling2D → Flatten → Dense → 출력; ReLU 활성화, 다중 클래스 소프트맥스.

</details>
