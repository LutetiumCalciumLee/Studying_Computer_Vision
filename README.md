<details>
<summary>ENG (English Version)</summary>

# Machine Learning Fundamentals

### 1. Machine Learning Overview
- **Definition and History:** ML defined by Arthur Samuel (1959) as programs improving via experience; Tom Mitchell (1998) formalized as performance improvement on tasks. Approaches include rule-based and learning-based.
- **ML Workflow:** Involves EDA (Exploratory Data Analysis) as initial step.
- **Task Categories:** Predictive (regression, classification), Descriptive (clustering, summarization), Time-series (forecasting, sequencing), Association rules.

### 2. Model Fitting Issues
- **Overfitting:** Model memorizes training data, fails on test data; detected by comparing training vs validation loss. Solutions: more data, augmentation, regularization, early stopping, dropout, batch normalization.
- **Underfitting:** High bias, poor performance on both training and test data.
- **Bias-Variance Tradeoff:** Balances model complexity; high variance (overfitting), high bias (underfitting). Optimal via validation error monitoring.

### 3. Good Predictive Models
- **Evaluation Criteria:** Minimizes Expected MSE = Bias² + Variance + Irreducible Error. Generalization gap between training and test error.
- **Regularization Techniques:** Adds penalty to loss function (L1 Lasso, L2 Ridge) to control complexity and improve generalization.
- **Ridge vs Lasso:** Ridge (L2 norm) shrinks weights; Lasso (L1 norm) induces sparsity by zeroing features.

### 4. Data Handling Practices
- **Data Splitting:** Typical 80% training, 10% validation, 10% test. Validation for tuning, test for final evaluation.
- **Cross-Validation:** K-fold CV and LOOCV (Leave-One-Out) for robust validation.
- **Feature Scaling:** Normalization and zero-centering essential for gradient-based optimization; apply consistently across splits.

### 5. Practical Considerations
- **Model Capacity Control:** Reduce features, early stopping to prevent overfitting.
- **Hyperparameter Tuning:** Regularization strength (alpha, lambda) affects bias-variance balance.
- **Preprocessing Pipeline:** Original → Zero-centered → Normalized data sequence recommended.

</details>

<details open>
<summary>KOR (한국어 버전)</summary>

# 머신러닝 기초

### 1. 머신러닝 개요
- **정의와 역사:** Arthur Samuel(1959)이 경험을 통한 성능 향상으로 정의; Tom Mitchell(1998)이 형식화. 규칙 기반과 학습 기반 접근법 존재.
- **ML 워크플로우:** EDA(탐색적 데이터 분석)가 초기 단계.
- **작업 분류:** 예측(회귀, 분류), 기술(클러스터링, 요약), 시계열(예측, 순차), 연관 규칙.

### 2. 모델 적합 문제
- **과적합(Overfitting):** 훈련 데이터 암기, 테스트 데이터 실패; 훈련 vs 검증 손실 비교로 진단. 해결: 데이터 증강, 정규화, 조기 종료, 드롭아웃, 배치 정규화.
- **과소적합(Underfitting):** 높은 편향으로 훈련/테스트 모두 저성능.
- **편향-분산 트레이드오프:** 모델 복잡도 균형; 높은 분산(과적합), 높은 편향(과소적합). 검증 오차로 최적화.

### 3. 우수한 예측 모델
- **평가 기준:** 기대 MSE = 편향² + 분산 + 비환원 오차 최소화. 훈련-테스트 오차 격차(일반화 갭) 확인.
- **정규화 기법:** 손실 함수에 페널티 추가(L1 Lasso, L2 Ridge)로 복잡도 제어 및 일반화 향상.
- **Ridge vs Lasso:** Ridge(L2)는 가중치 축소; Lasso(L1)는 희소성 유발로 특징 선택.

### 4. 데이터 처리 기법
- **데이터 분할:** 일반적 80% 훈련, 10% 검증, 10% 테스트. 검증은 튜닝, 테스트는 최종 평가.
- **교차 검증:** K-폴드 CV와 LOOCV(Leave-One-Out)로 견고한 검증.
- **특징 스케일링:** 정규화와 제로-센터링 필수; 모든 분할에 일관 적용.

### 5. 실무 고려사항
- **모델 용량 제어:** 특징 감소, 조기 종료로 과적합 방지.
- **하이퍼파라미터 튜닝:** 정규화 강도(alpha, lambda)가 편향-분산 균형 결정.
- **전처리 파이프라인:** 원본 → 제로-센터링 → 정규화 순서 권장.

</details>
