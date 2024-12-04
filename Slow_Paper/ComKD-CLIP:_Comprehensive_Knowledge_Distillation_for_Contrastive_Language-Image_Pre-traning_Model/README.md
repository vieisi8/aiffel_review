# 학습 내용

---

- ComKD-CLIP: Comprehensive Knowledge Distillation for Contrastive Language-Image Pre-traning Model

---

## ComKD-CLIP: Comprehensive Knowledge Distillation for Contrastive Language-Image Pre-traning Model

---

### Abstract

---

  - 대조적 언어-이미지 사전 훈련(clip) 모델
    - 대조적 학습 기법을 통해 이미지와 텍스트 간의 의미 정보를 통합하는 데 사용됨
    - 다양한 멍티모달 작업에 주목할 만한 성과를 거둠
    - 그러나 리소스가 제한된 환경에서 대규모 clip 모델의 배포가 어려움
    - 소규모 모델은 실제 애플리케이션에 필요한 성능 벤치마크를 충족하지 못하는 경우가 빈번하게 발생
  - ComKD-CLIP: 대조 언어-이미지 사전 트랜싱 모델을 위한 포괄적 지식을 제안함
    - 대규모 teacher clip 모델의 지식을 더 작은 student 모델로 포괄적으로 추출해 매개변수를 크게 줄이면서도 비슷한 성능을 보장하는 새로운 접근 방식
    - 두가지 핵심 메커니즘으로 구성
      - Image Feature Alignment (IFAlign)
        - student 모델이 추출한 이미지 특징이 teacher 모델이 추출한 특징과 거의 일치하도록 만들어 학생이 교사의 이미지 특징 추출 지식을 학습할 수 있도록 함
      - Educational Attention (EduAttention)
        - teacher 모델에서 추출한 텍스트 특징과 student 모델에서 추출한 이미지 특징 간의 상호 관계를 탐색
        - student 모델에서 추출한 이미지 특징 간의 상호 관계를 탐색하여 student 모델이 어떻게 teacher 모델이 텍스트-이미지 특징을 통합하는지를 학습
    - teacher 모델에서 추출한 지식을 바탕으로 teacher 모델의 텍스트-이미지 기능 융합 결과를 활용하여 student 모델이 teacher의 지식을 정확하게 흡수할 수 있도록 함
    - 11개의 데이터 세트에 대한 광범위한 실험을 통해 제안된 방법의 우수성 입증

---

### Approach

---
---

#### Preliminaries

---

  - clip은 독립적인 이미지 및 텍스트 인코더 브랜치로 구성된 가장 일반적으로 사용되는 VLM 중 하나
    - 이미지와 텍스트를 정렬하고 융합해 공동의 멀티모달 이미지 공간을 학습
    - 이미지 인코딩 브랜치에서는 레이블이 지정된 시각 인식 데이터 세트 D = {xj , yj}^Mj=1이 입력으로 사용됨
      - 데이터 세트 D의 각 이미지 x는 이미지 인코더 fI에 의해 처리되어 정규화된 이미지 특징 u = fI (x)/||fI (x)||2 ∈ Rd를 얻음
      - 이미지 인식 데이터 세트 D에 해당하는 클래스 이름 c = {ci}^Ni=1이 있음
    - 텍스트 인코더 브랜치에서 입력 데이터는 “{ci}의 사진” 템플릿에서 생성된 텍스트 설명 ti임
      - 텍스트 인코더에 의해 인코딩된 각 ti는 정규화된 텍스트 피처 wi = fT (ti)/||fT (ti)||2 ∈ Rd를 산출하며, 여기서 d는 텍스트 피처의 차원
      - 모든 텍스트 특징의 모음 W = [w1, w2, ... , wN ] ∈ R^N×d는 분류 가중치 행렬로 사용
      - 이 데이터를 기반으로 분류 출력 확률은 다음과 같이 계산
        - ![image](https://github.com/user-attachments/assets/cc2c60e4-d1c5-43f7-8a35-cc08d07245a1)
        - 여기서 uw^T는 출력 로짓을 나타내고 τ는 온도 매개변수
  - KD는 원래 Hinton 등이 제안한 것으로(Hinton, Vinyals, Dean 2015), 사전 교육을 받은 대규모 teacher 모델에서 더 작고 가벼운 student 모델로 지식을 이전하는 방식입니다.
    - 이를 통해 student은 teacher의 지식을 흡수하여 효율적으로 배포할 수 있음
    - 이 프로세스에서는 KL divergence 손실을 사용하여 두 모델의 특징 분포를 정렬
    - KL divergence 손실은 다음과 같이 정의됨
      - ![image](https://github.com/user-attachments/assets/c284a642-fc74-480e-9477-0e4bea46ad50)
      - 여기서 q^t와 q^s는 각각 teacher 모델과 teacher 모델에서 예측한 로그를 나타냄
      - σ(-)는 소프트맥스 함수를, τ는 확률 분포의 평활성을 조정하는 온도 매개변수(Hinton, Vinyals, and Dean 2015; Li et al. 2023b)를 나타냄

---

#### Pipline

---

  - 그림 2에서 볼 수 있듯이, 우리가 제안한 ComKD-CLIP 프레임워크는 크게 두 가지 단계로 구성됨
    - 대규모 CLIP teacher 모델의 사전 교육과 소규모 CLIP student 모델의 후속 교육
    - 초기 단계에서는 그림 2(a)에 설명된 대로 대규모 CLIP teacher 모델을 레이블이 지정된 도메인 데이터 세트(Dlabeled = {xi, yi}^M i=1)로 사전 학습하여 성능을 향상시키고 PromptSRC(Khattak 외. 2023b) 및 PromptKD(Li 외. 2024b) 같은 최신 방법론과 연계
      - 혁신적으로, 연결 전략을 통해 학습 가능한 프롬프트를 teacher 모델의 이미지 및 텍스트 인코더 브랜치에 모두 통합합니다
      - 레이블이 지정된 도메인 데이터 세트의 이미지와 텍스트 데이터는 각각 이미지 인코더 f^tV와 텍스트 인코더 f^tT를 통해 처리되어 이미지 특징값 u^pt ∈ R^d와 텍스트 특징값 w^pt ∈ R^d를 생성
      - 최종 출력 로그 q^t는 식 1에 의해 계산됨
      - teacher 모델의 훈련에는 예측 확률 분포와 실제 레이블 사이의 교차 엔트로피 손실을 최소화하여 모델의 파라미터를 최적화하는 작업이 수반됨
      - 이 엄격한 사전 훈련 단계를 통해 teacher 모델은 프레임워크의 후반 단계에서 student 모델에 효과적으로 전달할 수 있는 강력한 지식을 습득할 수 있음
    - 그림 2(b)에서 볼 수 있듯이, student CLIP 모델은 teacher 모델에서 사전 학습된 텍스트 특징을 직접 활용하므로 텍스트 인코더 브랜치를 통한 학습 비용을 크게 절감할 수 있음
    - 동시에, student 모델 내에 경량 CLIP 이미지 인코더 분기가 설계되어 리소스 비용을 줄이면서도 경쟁력 있는 성능을 유지
    - student 모델의 이미지 인코딩에 의해 레이블이 지정되지 않은 도메인 데이터 세트의 입력 데이터를 처리하는 동안 IFAlign 모듈을 통합
      - 이 모듈은 student 모델의 이미지 특징값(u^ps ∈ R^d)을 teacher 모델의 이미지 특징값(u^pt ∈ R^d)과 정렬하여 student 모델이 teacher 모델이 두드러진 이미지 특징을 추출하는 방법에 대한 지식을 쉽게 흡수할 수 있도록 도와줌
    - student 모델이 추출한 이미지 특징과 teacher 모델이 제공한 텍스트 특징 간의 상호 관계를 탐색하기 위해 EduAttention 모듈이 도입
      - 이 탐색을 통해 student 모델은 텍스트-이미지 특징을 통합하기 위해 teacher 모델이 사용하는 미묘한 전략을 학습할 수 있음
      - 또한 KL divergence을 사용하여 teacher 모델과 student 모델이 생성한 로짓 간의 불일치를 최소화
    - 이러한 최적화를 통해 student 모델에서 추출된 지식이 teacher의 지식을 보다 정교하게 반영하여 student 모델이 teacher 모델의 지식을 정확하게 흡수할 수 있도록 개선
    - 마지막으로, 훈련된 student 모델의 추론 과정은 그림 2(d)에 나와 있음

![image](https://github.com/user-attachments/assets/24203f60-d00f-4e4f-abf7-528431b16871)


---

#### ComKD-CLIP

---

  - IFAlign
    - IFAlign의 개략도는 그림(c)에 나와 있음
    - student 모델에서 추출한 이미지 특징을 teacher 모델에서 추출한 특징과 거의 일치하도록 하기 위해 추출된 특징의 평균과 분산 통계 정렬
    - 계산 과정은 다음과 같이 공식화할 수 있음
      - ![image](https://github.com/user-attachments/assets/c86298b0-d31b-42a3-9277-27d5ea5d3846)
      - ![image](https://github.com/user-attachments/assets/57f27c4e-fc7b-4abb-aa11-d5217b199a30)
      - µs(us; p) 및 σ^2s(us; p)는 student 모델에서 추출한 이미지 특징의 평균과 분산을 나타냄
      - µt(ut; p) 및 σ^2t(ut; p)는 teacher 모델에서 추출한 특징에 해당
      - u^ps 및 u^pt는 각각 student 및 teacher 모델의 프롬프트가 있는 이미지 특징을 나타냄
      - student의 이미지 인코더 브랜치 내의 학습 가능한 프로젝터 P(-)
        - 특징 차원을 효율적이고 비용 효율적으로 조정하여 정확한 정렬을 보장하도록 설계됨
    - 그런 다음 L1 손실을 사용하여 student 모델에서 추출한 이미지 특징의 평균과 분산을 teacher 모델과 정렬
      - 이 정렬을 통해 student 모델은 teacher 모델이 두드러진 이미지 특징을 추출하는 방법에 대한 지식을 쉽게 흡수할 수 있음
      - 구체적인 정렬 과정은 다음과 같이 공식화할 수 있음
        - ![image](https://github.com/user-attachments/assets/f9c445de-9c83-4aed-a1d1-e2d4ad8e5c5a)
        - Lalign mean은 teacher 모델과 student 모델에서 추출한 이미지 특징의 평균값 차이를 나타냄
        - Lalign var는 teacher 모델과 student 모델에서 추출한 이미지 특징의 분산값 차이를 나타냄
        - Lalign mean과 Lalign var를 정렬 손실 Lalign으로 결합하면 student 모델이 teacher 모델이 이미지 특징을 추출하는 방법에 대한 지식을 완전히 흡수할 수 있음
  - EduAttention
    - EduAttention의 회로도는 그림 2(e)에 나와 있음
    - 이 모듈에서는 attention 메커니즘을 활용하여 student 모델이 추출한 이미지 특징과 teacher 모델이 제공한 텍스트 특징 간의 상호 관계를 탐색함으로써 student 모델이 텍스트-이미지 특징을 통합하기 위해 teacher 모델이 사용하는 미묘한 전략을 쉽게 학습할 수 있도록 함
      - 구체적인 계산 과정은 다음과 같이 공식화할 수 있음
        - ![image](https://github.com/user-attachments/assets/d2bff1d3-fef8-4cb6-a3a9-f06d32be0c42)
        - u^ps는 student 모델에서 추출한 이미지 특징을 나타냄
        - w^pt는 teacher 모델에서 추출한 텍스트 특징을 나타냄
        - fatt는 u^ps와 w^pt 간의 상호 관계를 나타냄
        - C는 하이퍼파라미터, F C(-)는 완전히 연결된 계층을 나타냄
    - teacher 모델에서 흡수한 지식을 IFAlign과 EduAttention에 통합하기 위해 fatt에 학습 가능한 파라미터 α를 곱하고 추출된 이미지 특징 텍스트 u^pt와 최종 이미지 특징 fe를 요소별로 합산하는 작업을 수행
      - 구체적인 계산 과정은 다음과 같이 공식화할 수 있음
        - ![image](https://github.com/user-attachments/assets/eb90b14d-d3c9-45d5-a8f4-4aec4e882bf6)
        - α는 0으로 초기화되고 점차적으로 더 많은 가중치를 할당하는 방법을 학습
  - Distilled Knowledge Refinement
    - student 모델이 teacher 모델이 이미지 특징을 추출하고 텍스트-이미지 특징을 결합하는 지식을 흡수한 후, teacher 모델의 특징 융합 결과를 바탕으로 흡수한 지식을 구체화하려고 함
    - 그림 2와 같이 teacher 모델과 student 모델에서 생성된 특징 분포의 불일치를 최소화하기 위해 KL divergence를 활용
      - 구체적인 프로세스는 다음과 같이 공식화할 수 있음
        - ![image](https://github.com/user-attachments/assets/35397248-d330-4d00-bc8f-62260ce52cdf)
        - q^t와 q^s는 각각 teacher 모델과 student 모델이 예측한 로짓을 나타냄
        - 식 1을 사용하여 해당 이미지 특징과 텍스트 특징으로 계산
        - τ는 확률 분포의 평활성을 조정하는 데 사용되는 온도 매개변수
    - 마지막으로 student 모델의 정렬 손실 Lstu와 특징 분포 손실 Lalign을 최종 손실 함수로 결합하여 작은 CLIP 모델의 파라미터를 훈련하며, 구체적인 손실 공식은 다음과 같음
      - ![image](https://github.com/user-attachments/assets/d77ca650-9adf-4cd7-a3f6-0cd2677da5a9)

---

### Experiments

---

![image](https://github.com/user-attachments/assets/cea2f422-cc61-45ec-b503-b22925d50bb8)

  - 표 3: ComKD-CLIP의 도메인 일반화 성능
    - 목적: ComKD-CLIP 모델의 도메인 일반화 성능을 평가
    - 사용된 데이터셋 (Target Dataset)
      - V2: ImageNet V2
      - -S: ImageNet Sketch
      - -A: ImageNet-A (Adversarial Examples)
      - -R: ImageNet-R (Rendition)
    - 결과
      - ComKD-CLIP의 평균 성능(Avg.)은 72.78로, 다른 방법보다 높은 성능을 보임
      - Δ: CLIP 대비 성능 향상 수치로, 평균적으로 +1.31의 향상이 있음
  - 표 4: ComKD-CLIP의 모듈 성능 분석 (Ablation Study)
    - 목적: ComKD-CLIP의 성능에 기여하는 각 모듈(IFAlign, EduAttention)의 중요성을 분석
    - 사용된 데이터셋
      - SUN397: 장면 분류(scene classification)를 위한 데이터셋
    - 모듈 구성
      - IFAlign 제거, EduAttention 제거, 또는 둘 다 제거한 경우 성능 비교
      - Base, Novel, HM (Harmonic Mean) 지표를 사용해 성능 평가
    - 결과
      - Full 모델(모든 모듈 포함)이 가장 높은 성능(Base: 84.19, HM: 82.93)을 보임
      - 각 모듈 제거 시 성능이 감소
