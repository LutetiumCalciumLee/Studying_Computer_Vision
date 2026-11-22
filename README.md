<details>
<summary>ENG (English Version)</summary>

# Machine Learning Learning Types

### 1. Overview of Machine Learning Categories
- **Four Main Types:** The document covers Supervised Learning, Unsupervised Learning, Semi-Supervised Learning, and Reinforcement Learning as core ML paradigms.
- **Feature Engineering:** Emphasizes the role of features (input variables) in all learning types.
- **Labeled vs Unlabeled Data:** Distinguishes between data with hints/answers (supervised) and without (unsupervised).

### 2. Supervised Learning
- **Definition and Data:** Learns from labeled data containing both inputs (features) and outputs (targets). Handles continuous (regression) and categorical (classification) targets.
- **Regression Methods:** Includes Linear Regression (y = w0 + w1x1 + ...), Nonlinear Regression, Polynomial Regression, and optimization via Gradient Descent.
- **Classification Algorithms:** Covers Logistic Regression (Sigmoid/Softmax), Decision Trees (Entropy, Information Gain, Gini), SVM (hyperplane, kernel trick), KNN (K-Nearest Neighbors), Naive Bayes.

### 3. Unsupervised Learning
- **Pattern Discovery:** Finds structure in unlabeled data through clustering, dimensionality reduction (PCA), and anomaly detection.
- **Clustering Techniques:** Details K-Means (centroid-based), Hierarchical Clustering, DBSCAN (density-based), GMM (Gaussian Mixture Models).
- **Applications:** Group similar data points, identify outliers, and generative models like GANs.

### 4. Semi-Supervised and Reinforcement Learning
- **Semi-Supervised:** Combines small labeled datasets with large unlabeled ones to improve efficiency (e.g., labeling 100 out of 5000 data points).
- **Reinforcement Learning Basics:** Agent interacts with environment via states, actions, rewards; includes Q-Learning, Model-Free/Model-Based, Value/Policy-Based approaches.
- **Key Components:** State transitions, reward signals, discrete time stochastic control.

### 5. Key Algorithms in Depth
- **Decision Trees:** Root node, impurity measures (Entropy, Gini), splitting criteria, pruning to avoid overfitting.
- **KNN:** Distance-based classification/regression; weighted variants; KNeighborsClassifier/Regressor.
- **SVM:** Finds optimal hyperplane maximizing margin; support vectors define boundaries.

</details>

<details open>
<summary>KOR (한국어 버전)</summary>

# 인공지능 학습 유형

### 1. 머신러닝 분류 개요
- **4대 주요 유형:** 지도 학습, 비지도 학습, 반지도 학습, 강화 학습을 핵심 패러다임으로 다룸.
- **특징 공학:** 모든 학습 유형에서 입력 변수(특징)의 중요성 강조.
- **레이블 데이터 구분:** 힌트/정답 포함 데이터(지도) vs 미포함 데이터(비지도).

### 2. 지도 학습
- **정의 및 데이터:** 입력(특징)과 출력(타겟)이 레이블링된 데이터로 학습. 연속형(회귀), 범주형(분류) 타겟 처리.
- **회귀 방법:** 선형 회귀(y = w0 + w1x1 + ...), 비선형 회귀, 다항 회귀, 그라디언트 디센트 최적화.
- **분류 알고리즘:** 로지스틱 회귀(시그모이드/소프트맥스), 결정 트리(엔트로피, 정보 이득, 지니), SVM(초평면, 커널 트릭), KNN(K최근접 이웃), 나이브 베이즈.

### 3. 비지도 학습
- **패턴 발견:** 레이블 없는 데이터에서 클러스터링, 차원 축소(PCA), 이상 탐지로 구조 발견.
- **클러스터링 기법:** K-평균(중심 기반), 계층 클러스터링, DBSCAN(밀도 기반), GMM(가우시안 혼합).
- **응용:** 유사 데이터 그룹화, 이상치 식별, GAN 같은 생성 모델.

### 4. 반지도 및 강화 학습
- **반지도 학습:** 소량 레이블 데이터와 대량 비레이블 데이터를 결합해 효율성 향상(예: 5000개 중 100개 레이블링).
- **강화 학습 기초:** 에이전트가 환경과 상태, 행동, 보상으로 상호작용; Q-러닝, Model-Free/Model-Based, Value/Policy-Based 포함.
- **핵심 요소:** 상태 전이, 보상 신호, 이산 시간 확률 제어.

### 5. 주요 알고리즘 상세
- **결정 트리:** 루트 노드, 불순도 측정(엔트로피, 지니), 분할 기준, 과적합 방지 프루닝.
- **KNN:** 거리 기반 분류/회귀; 가중치 변형; KNeighborsClassifier/Regressor.
- **SVM:** 마진 최대화 최적 초평면 찾기; 서포트 벡터로 경계 정의.

</details>
