# 학습내용

---

- Visual Instruction Tuning

---

## Visual Instruction Tuning

---

---

###  Abstract

---

  - Vision-Language Models (VLMs)를 위한 Instruction Tuning에 초점을 맞춤
  - OpenAI의 GPT-4와 같은 강력한 언어 모델을 활용하여 시각적 과제를 설명하는 대규모 시각적 지침 데이터를 생성, 이를 통해 VLM의 성능을 향상
  - 제안된 방식은 Instruction-following VLM을 효과적으로 훈련시키며, 모델이 더욱 풍부하고 실용적인 지능을 갖추도록 도움
  - 실험 결과, 이 접근 방식이 시각적 질문 응답 및 시각적 지침 수행에서 매우 우수한 성능을 발휘함

---

###  1 Introduction

---

  - Instruction Tuning은 LLM들이 사용자 명령을 따라 더 잘 응답하도록 만드는 핵심 기술로 자리 잡음
    - 이를 Vision-Language Models에 확장하는 것이 연구의 주된 동기
  - 그러나 VLM에 적합한 고품질 시각적 지침 데이터를 생성하는 데는 많은 비용이 듬
    - 이를 해결하기 위해 GPT-4를 활용하여 저비용으로 대규모 시각적 인스트럭션 데이터를 자동 생성하는 방식을 제안
  - 본 논문에서는 시각적 질문 생성, 응답 생성, 시각적 컨텍스트 강화의 자동화 프로세스를 포함
    - 이를 통해 훈련된 VLM은 시각적 지침을 더 잘 이해하고 다양한 사용자 요청에 더 적합하게 응답

---

###  3 GPT-assisted Visual Instruction Data Generation

---

  - GPT-4를 사용하여 고품질의 시각적 지침 데이터를 자동 생성하는 과정을 소개
    1. 시각적 컨텍스트 분석
       - 이미지와 텍스트 정보를 GPT-4로 전달하여, 이미지의 내용을 기반으로 자연스러운 질문과 답변을 생성
    2. 다양한 지침 형식
       - 단순 질문 응답뿐만 아니라, 설명, 요약, 비교와 같은 복합적인 지시를 포함한 다양한 데이터셋을 생성
    3. 자동화된 데이터 생성 프로세스
       - 사람이 데이터셋을 직접 생성하는 비용을 줄이고, 확장 가능성 확보
  - 이렇게 생성된 데이터는 기존의 시각적 데이터셋과 결합해 Instruction-tuned VLM을 효과적으로 훈련시키는 데 사용

---

### 4 Visual Instruction Tuning

---

---

#### 4.1 Architecture

---

  - Base Model
    - Vision-Language Foundation Model로 BLIP-2를 사용.
      - BLIP-2는 이미지를 입력으로 받아 텍스트를 생성할 수 있는 모델로, Vision Transformer (ViT)와 Language Model (LM)을 결합한 구조를 기반으로 함
  - Instruction-following Fine-tuning
    - 모델의 언어 생성 능력을 강화하기 위해, GPT-4로 생성된 Instruction-following Data로 추가 훈련을 수행
    - Vision Encoder와 Language Model 간의 Cross-Attention Mechanism을 활용하여, 이미지와 텍스트 간 상호작용을 최적화
  - Modular Design
    - Vision Encoder (이미지 처리)와 Language Model (텍스트 처리) 간의 분리된 설계로 유연성을 유지
    - Instruction Tuning 후에도 사전 훈련된 언어 모델의 잠재력을 유지할 수 있음

---

####  4.2 Training

---

  - Training Data
    - GPT-4를 활용해 자동 생성한 시각적 지침 데이터 사용
    - 다양한 태스크(예: 이미지 설명, 시각적 질문 응답, 이미지 비교 등)와 지침 유형을 포함함
  - Two-Stage Fine-Tuning
    1. Vision-Language Alignment Stage
       - 시각적 입력과 언어 출력 간의 정렬을 강화
       - Vision Encoder와 Language Model 간의 상호작용을 미세 조정
    2. Instruction-following Fine-Tuning Stage
       - GPT-4 기반의 고품질 Instruction Data로 추가적인 미세 조정 수행
       - 다양한 유형의 질문과 지침 수행 능력을 모델에 학습시킴
  - Optimization:
    - 학습 과정에서 Cross-Entropy Loss를 사용해 모델의 텍스트 생성 성능 최적화
    - Mixed Precision Training과 같은 기법을 활용해 학습 효율 극대화
  - Generalization:
    - 훈련되지 않은 새로운 지침이나 태스크에도 일반화할 수 있도록 모델 설계
    - Instruction Tuning 데이터의 다양성과 모델 구조의 유연성이 이를 가능하게 함
