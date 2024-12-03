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

### Pipline

---



---

### ComKD-CLIP

---



---

### Experiments

---
