# 학습 내용

---

- attention is all you need

---

## attention is all you need

---

1. Introduction

- RNN의 문제점

  1. 병렬화 불가능
  2. 메모리 제약으로 인해 일괄 처리 제한(시퀀스 길이가 길어지면 문제가 심각해짐)

- Attention 메커니즘

  - 입출력 시퀀스의 거리에 관계없이 종속성을 모델링할 수 있게 해줌
  - 일부 사례를 제외한 모든 경우에서 RNN과 함께 사용

- Transformer 제안

  - 반복을 피하고 Attention 메커니즘에 전적으로 의존하여 입출력간의 글로벌 종속성을 도출하는 아키텍처 제안
  - 훨씬 더 많은 병렬화 가능

3. Model Architecture

![image](https://github.com/user-attachments/assets/1d249935-0db1-43c0-a3ec-3952dc671dbf)

    -> Transformer는 인코더와 디코더 모두에 대해 스택형  self-attention, point-wise, fully connected layers 사용

3.1 Encoder and Decoder Stacks

  1.인코더

  - N=6개의 layer 스택으로 구성
  - 각 layer에는 두개의 하위 layer 존재
    1. multi-head self-attention mechanism
    2. positionwise fully connected feed-forward network
  - 두개의 하위 layer 각각에 잔차 연결(residual connection), layer 정규화(normalization)
  - 각 서브 layer의 출력 (x + Sublayer(x))
  - Sublayer(x)는 서브 layer 자체에서 구현된 함수
  - 모델의 하위 layer와 embedding layer는 차원 dmodel = 512의 출력 생성

  2. 디코더

  - N=6개의 layer 스택으로 구성
  - 인코더 스택의 출력에 대한 multi-head self-attention을 수행하는 세 번째 하위 layer 존재
  - 각 하위 layer 각각에 잔차 연결(residual connection), layer 정규화(normalization)
  - 디코더 스택의 self-attention layer를 수정해 position이 후속 position에 영향을 주지 않도록 함(마스킹)
    - 출력 embedding이 한 위치만큼 offset된다는 사실과 결합
    - 위치 i에 대한 예측이 i보다 작은 위치의 알려진 출력에만 의존할 수 있도록 함
   
3.2 attention

  attention function

  - 쿼리와 키-값을 출력에 매핑하는 것으로 설명
    - 쿼리, 키, 출력은 모두 벡터
    - 출력
      - 값의 가중치의 합으로 계산
    - 각 값에 할당된 가중치
      - 해당 키와 쿼리의 compatibility function에 의해 계산

3.2.1  Scaled Dot-Product Attention

  - 입력
    - 쿼리, 키의 차원
      - dk
    - 값의 차원
      - dv
  - 가중치 계산
    - 모든 키로 쿼리의 dot products 계산
    - 각각 √dk로 나눔
    - softmax 함수 적용

![image](https://github.com/user-attachments/assets/323fc406-9795-4ae3-8c4f-c233a56f8554)

  - dot products attention을 사용한 이유
    - additive attention과 이론적 복잡성은 비슷
    - 최적화된 행렬 곱 코드를 사용해 훨씬 빠르고 공간 효율적
    - 작은 dk값
      - 두 메커니즘이 비슷하게 작동
    - 큰 dk값
      - additive attention이 성능이 뛰어남
      - dot products값의 크기가 커져 softmax 함수가 매우 작은 기울기를 갖게 되기 때문
      - 위와 같은 문제를 해결하기 위해 √dk으로 스케일링 함

3.2.2 Multi-Head Attention

  - dmodel - dk
    - 단일 attention function보단 학습된 다양한 선형 투영을 통해 쿼리, 키, 값을 각각 dk, dk, dv차원으로 선형 투영하는 것이 유리하다는 것을 알게 됨
  - dv 출력 값
    - 투영된 쿼리, 키, 값의 각 버전에 대해 attention function을 병렬로 수행해 산출
  - 최종 값
    - 위 출력값을 연결하고 다시 한번 투영해 도출

![image](https://github.com/user-attachments/assets/c908bf9d-92f1-4979-9375-1d0cd165cb6c)

  - 장점
    - 서로 다른 위치에서 서로 다른 표현 하위 공간의 정보에 서로 attention할 수 있음
  - head의 개수
    - h=8
    - dk = dv = dmodel/h = 64
    - 각 헤드의 크기가 줄었들었기 때문에 계산 비용은 단일 attention과 비슷함

3.2.3 Applications of Attention in our Model

  세가지 방식의 multi-head attention 사용
