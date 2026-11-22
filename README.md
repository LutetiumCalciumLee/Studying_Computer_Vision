<details>
<summary>ENG (English Version)</summary>

# Deep Learning Fundamentals

### 1. Artificial Neural Networks Basics
- **Neuron Structure:** Composed of inputs (X), weights (W), bias (b), linear combination (z = wx + b), activation function, producing output (Y).
- **Perceptron:** Single neuron model capable of logic gates like OR, AND, NAND; fails on XOR (non-linearly separable).
- **XOR Solution:** Multi-layer approach using NAND → OR → XOR gate combinations.

### 2. Multi-Layer Perceptron (MLP)
- **Architecture:** Input layer, multiple hidden layers, output layer; 1986 development by Rumelhart, Hinton et al.
- **Deep Neural Network (DNN):** Multiple hidden layers enable complex pattern learning; overcomes single-layer limitations.
- **Overfitting Risk:** Too many hidden layers or units can lead to memorization.

### 3. Activation Functions
- **Types Overview:** Step, Sigmoid (0-1 output, vanishing gradient issue), ReLU (f(x)=max(0,x)), Leaky ReLU, ELU.
- **Sigmoid Function:** σ(x) = 1/(1+e^(-x)); used in logistic regression for binary classification.
- **Gradient Issues:** Sigmoid causes vanishing gradients in deep networks; ReLU preferred for hidden layers.

### 4. Training Process
- **Forward Propagation:** Data flows from input through layers to output, computing predictions.
- **Loss Calculation:** Measures error using functions like Mean Squared Error (MSE).
- **Backpropagation:** Computes gradients via chain rule, updates weights using Gradient Descent.

### 5. Gradient Descent and Optimization
- **Weight Update:** w_new = w_old - η * ∂E/∂w (learning rate η, error E).
- **Backpropagation Algorithm:** 1974 Werbos, popularized 1986 Hinton; propagates errors backward through layers.
- **Vanishing Gradient Problem:** Sigmoid gradients approach 0 in deep layers; solutions include ReLU, Xavier/He initialization.

### 6. Advanced Techniques
- **Chain Rule Application:** Essential for computing partial derivatives across multiple layers.
- **Numerical Example:** Demonstrates forward pass, activation, error computation in simple 2-layer network.
- **Mitigation Strategies:** Pretraining, Dropout, proper weight initialization for stable training.

</details>

<details open>
<summary>KOR (한국어 버전)</summary>

# 딥러닝 기초

### 1. 인공 신경망 기본
- **뉴런 구조:** 입력(X), 가중치(W), 편향(b), 선형 결합(z=wx+b), 활성화 함수를 거쳐 출력(Y) 생성.
- **퍼셉트론:** 단일 뉴런으로 OR, AND, NAND 게이트 구현 가능; XOR(비선형 분리)에서는 실패.
- **XOR 해결:** NAND → OR → XOR 다층 게이트 조합으로 구현.

### 2. 다층 퍼셉트론 (MLP)
- **구조:** 입력층, 다중 은닉층, 출력층; 1986 Rumelhart, Hinton 등이 개발.
- **심층 신경망 (DNN):** 다중 은닉층으로 복잡한 패턴 학습; 단일층 한계 극복.
- **과적합 위험:** 은닉층/유닛 과다 시 암기 발생.

### 3. 활성화 함수
- **종류 개요:** Step, Sigmoid(0-1 출력, 기울기 소실 문제), ReLU(f(x)=max(0,x)), Leaky ReLU, ELU.
- **시그모이드 함수:** σ(x)=1/(1+e^(-x)); 이진 분류 로지스틱 회귀에 사용.
- **기울기 문제:** 시그모이드 깊은 층에서 기울기 0에 수렴; 은닉층에는 ReLU 선호.

### 4. 학습 과정
- **순전파(Forward Propagation):** 입력부터 출력까지 데이터 흐름, 예측값 계산.
- **손실 계산:** 평균 제곱 오차(MSE) 등 손실 함수로 오차 측정.
- **역전파(Backpropagation):** 체인 룰로 기울기 계산, 그라디언트 디센트로 가중치 업데이트.

### 5. 그라디언트 디센트와 최적화
- **가중치 업데이트:** w_new = w_old - η * ∂E/∂w (학습률 η, 오차 E).
- **역전파 알고리즘:** 1974 Werbos 개발, 1986 Hinton 대중화; 오차를 층별 후방 전파.
- **기울기 소실 문제:** 시그모이드 깊은 층에서 기울기 소실; ReLU, Xavier/He 초기화로 해결.

### 6. 고급 기법
- **체인 룰 적용:** 다층 간 편미분 계산 핵심.
- **수치 예시:** 2층 간단 네트워크에서 순전파, 활성화, 오차 계산 데모.
- **완화 전략:** 사전학습, 드롭아웃, 적절한 가중치 초기화로 안정적 학습.

</details>
