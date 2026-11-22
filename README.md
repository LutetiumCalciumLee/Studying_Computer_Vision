<details>
<summary>ENG (English Version)</summary>

# Transformer Model

### 1. Transformer Introduction
- **Problem with RNN/LSTM:** Long-term dependency issues in sequence processing; Transformer solves with Self-Attention mechanism.
- **Origin:** Google 2017 paper "Attention is All You Need"; parallelizable, no recurrence or convolution needed.
- **Key Innovation:** Self-Attention captures relationships between all sequence positions simultaneously.

### 2. Evolution and Applications
- **Model Family:** BERT (Encoder-only, bidirectional), GPT (Decoder-only, generative), BART (Encoder-Decoder).
- **LLM Development:** GPT series from GPT-1 (2018) to GPT-4 (2023); powers ChatGPT and modern AI.
- **Applications:** QA, text generation, embeddings; ELMo (2018) as early contextual model.

### 3. Attention Mechanism
- **Seq2Seq Attention:** Decoder attends to relevant encoder hidden states weighted by importance.
- **Attention Weights:** Softmax over encoder states; focuses decoder on context like "Coffee" in "I Like Coffee".
- **Core Concept:** Query encoder states to generate context-aware representations.

### 4. Transformer Architecture
- **Encoder-Decoder Structure:** N=6 encoder/decoder blocks stacked; processes input/output embeddings.
- **Block Components:** Multi-Head Attention + Add&Norm + Feed Forward + Add&Norm.
- **Attention Variants:** Encoder Self-Attention, Decoder Masked Self-Attention, Encoder-Decoder Attention.

### 5. Key Components
- **Positional Encoding:** Adds position info to embeddings since no recurrence; sine/cosine functions.
- **Multi-Head Attention:** Multiple attention heads in parallel capture diverse relationships.
- **Masking:** Decoder uses masked attention to prevent future peeking during training.

### 6. Encoder and Decoder Details
- **Encoder Block:** Self-Attention → Feed Forward; processes full sequence bidirectionally.
- **Decoder Block:** Masked Self-Attention → Encoder-Decoder Attention → Feed Forward; autoregressive generation.
- **Output Flow:** Linear + Softmax for probability distribution over vocabulary.

</details>

<details open>
<summary>KOR (한국어 버전)</summary>

# Transformer 모델

### 1. Transformer 소개
- **RNN/LSTM 문제:** 시퀀스 처리에서 장기 의존성 문제; Transformer는 Self-Attention으로 해결.
- **기원:** Google 2017 논문 "Attention is All You Need"; 병렬 처리 가능, 순환/합성곱 불필요.
- **핵심 혁신:** Self-Attention으로 시퀀스 모든 위치 간 관계 동시 포착.

### 2. 발전과 응용
- **모델 계열:** BERT(인코더 전용, 양방향), GPT(디코더 전용, 생성), BART(인코더-디코더).
- **LLM 발전:** GPT-1(2018)부터 GPT-4(2023)까지; ChatGPT 등 현대 AI 동력.
- **응용:** QA, 텍스트 생성, 임베딩; ELMo(2018)가 초기 문맥 모델.

### 3. 어텐션 메커니즘
- **Seq2Seq 어텐션:** 디코더가 인코더 숨겨진 상태에 중요도 가중치로 주의 집중.
- **어텐션 가중치:** 인코더 상태에 소프트맥스; "I Like Coffee"에서 "Coffee"에 집중.
- **핵심 개념:** 쿼리로 인코더 상태 활용해 문맥 인식 표현 생성.

### 4. Transformer 구조
- **인코더-디코더 구조:** N=6 인코더/디코더 블록 적층; 입력/출력 임베딩 처리.
- **블록 구성요소:** Multi-Head Attention + Add&Norm + Feed Forward + Add&Norm.
- **어텐션 변형:** 인코더 Self-Attention, 디코더 Masked Self-Attention, 인코더-디코더 어텐션.

### 5. 주요 구성요소
- **위치 인코딩:** 순환 없어 임베딩에 위치 정보 추가; 사인/코사인 함수 사용.
- **멀티헤드 어텐션:** 병렬 다중 어텐션 헤드로 다양한 관계 포착.
- **마스킹:** 디코더 훈련 시 미래 정보 차단하는 마스크드 어텐션.

### 6. 인코더와 디코더 상세
- **인코더 블록:** Self-Attention → Feed Forward; 전체 시퀀스 양방향 처리.
- **디코더 블록:** Masked Self-Attention → Encoder-Decoder Attention → Feed Forward; 자동회귀 생성.
- **출력 흐름:** Linear + Softmax로 어휘 분포 확률 생성.

</details>
