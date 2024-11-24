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

  1. 인코더 - 디코더 attention
     - 쿼리
       - 디코더 계층
     - 키, 값
       - 인코더의 출력
     - 디코더의 모든 위치가 입력 시퀀스의 모든 위치에 attention 할 수 있음
  2. 인코더 self-attention
     -  모든 키, 값, 쿼이가 같은 위치에서 나옴
       - 인코더의 이전 계층의 출력 사용
     - 인코더의 이전 레이어에 있는 모든 위치를 확인 가능
  3. 디코더 self-attention
     - 각 위치가 해당 위치를 포함한 디코더의 모든 위치에 attenation 할 수 있음
     - 자동 회귀 속성을 유지하기 위해 디코더에거 왼쪽으로 정보가 흐르는 것을 방지해야 함
       - 해당하는 부분을 마스킹(−∞)으로 지정해 softmax 입력값으로 줌
       - scaled dot-product attention 내부에서 구현함

3.3 Position-wise Feed-Forward Networks

  - 인코더와 디코더의 각 레이어에는 attenation 하위 layer 외에도  fully connected feed-forward network가 포함되어 있음
  - 각 위치에 개별적으로 동일하게 적용됨
  - 두 개의 선형 변환과 ReLU 활성화로 구성

![image](https://github.com/user-attachments/assets/7862776b-3491-4167-9a1b-0e9d4cf31638)

  - 선형 변환
    - 여러 위치에서 동일하지만 레이어마다 다른 매개 변수를 사용
  - 입력과 출력의 차원
    - dmodel=512
  - 내부 layer의 차원
    - dff=2048

3.4 Embeddings and Softmax

  - 학습된 임베딩을 사용해 입력 토큰, 출력 토큰을 차원 dmodel의 벡터로 변환
  - 일반적인 학습된 선형 변환과 softmax 함수를 사용해 디코더 출력을 예측된 다음 토큰 확률로 변환
  - embedding layer와 학습된 softmax 선형 변환 간에 동일한 가중치 매트릭스를 공유
    - embedding layer에서는 이 가중치에 √dmodel을 곱함

3.5 Positional Encoding

  - recurrence와 convolution이 없기 때문에 시퀀스의 순서 활용 X
    - 시퀀스에서 토큰의 상대적 또는 절대적 위치에 대한 정보 주입
  - 인코더, 디코더 스택의 하단에 있는 입력 embedding에 위치 인코딩을 추가
  - embedding과 동일한 차원 dmodel를 가지므로 둘은 합산 가능
  - 위치 인코딩에는 학습 및 고정된 다양한 선택지 있음
    - 이 작업에서는 서로 다른 주파수의 사인, 코사인 함수 사용

![image](https://github.com/user-attachments/assets/dee45cd2-81af-46b2-a47c-459f7fdf1319)

  - pos, i
    - 각 위치, 차원을 나타냄
  - 위치 인코딩의 각 차원은 sinusoid에 해당함
  - 파장
    -  2π ~ 20000π까지의 기하학적 진행을 형성
  - 위 함수를 선택한 이유
    - 고정된 offset k에 대해 선형 함수로 표현할 수 있기 때문
    - 모델이 상대적인 위치로 참석하는 것을 쉽게 학습할 수 있을 것이라고 가정했음
    - 학습된 위치 임베딩과 거의 동일한 결과를 도출함
      - 하지만 모델이 훈련 중에 발생하는 것보다 더 긴 시퀀스 길이로 추청할 수 있음
     
4. Why Self-Attention

  - self-attention사용에 동기를 부여하기 위해 세 가지 필수 조건을 고려함
    1. layer당 총 계산 복잡도
    2. 병렬화할 수 있는 계산량
       - 필요한 최소 순차 연산 횟수로 측정
    3. 네트워크에서 장거리 종속성 사이의 경로 길이
       - 장거리 종속성을 학습하는 것은 많은 시퀀스 변환 작업에서 핵심 과제
       - 장거리 종속성 경로의 길이가 짧은수록 장거리 종속성을 학습하기가 더 쉬워짐
       - ∴ 두 입력과 출력 위치 사이의 최대 경로 길이 비교
      
![image](https://github.com/user-attachments/assets/b991304e-e901-47b6-8e01-9f074151843e)

  - 표에서 self-attention만 확인 해보겠음
    - Complexity per Layer
      - scaled dot-product attention 내부에서 QKT 행렬곱으로 인해 n^2
      - scaled dot-product attention 내부에서 값을 곱하기에 d
      - ∴ O(n^2*d)
    - Sequential Operations
      - 병렬화 가능하므로 O(1)
    - Maximum Path Length
      - self-attention으로 인해 시퀀스를 한번에 참조할 수 있으므로 O(1)
  - 부수적인 이점
    - self-attention으로 인해 더 해석하기 쉬운 모델을 만들 수 있음
   
5. Training

5.1 Training Data and Batching

  - WMT 2014 영어-독일어 데이터셋, WMT 2014 영어-프랑스어 데이터셋 사용
  - 각 훈련 배치
    - 소스 토큰
      - 25,000 토큰
    - 목표 토큰
      - 25,000 토큰

5.2 Hardware and Schedule

  - 8개의 NVIDIA P100 GPU가 장착된 컴퓨터 한대로 훈련
  - 기본 모델
    - 각 step
      - 0.4초 소요
    - 훈련시간(총 step)
      - 12시간(10만 step)
  - Big 모델
    - 각 step
      - 1.0초 소요
    - 훈련시간(총 step)
      - 3.5일(30만 step)

5.3 Optimizer

  -  β1 = 0.9, β2 = 0.98, ϵ = 10−9으로 Adam 옵티마이저 사용
    - 학습 속도를 공식에 따라 다양하게 변경

![image](https://github.com/user-attachments/assets/c43c75b1-9a3c-4712-9320-586a88bd9c1b)

  - warmup_steps 훈련 단계에 대해 학습 속도를 선형적으로 증가
    - warmup_steps=4000
  - 그 이후에는 단계 수의 역제곱근에 비례하여 감소

5.4 Regularization

  - 세가지 유형의 정규화 사용
    - Residual Dropout
      - 각 하위 계층의 출력에 드롭아웃 적용
      - 인코더와 디코더 스택 모두에서 임베딩과 위치 인코딩의 합에 드롭아웃 적용
      - Pdrop = 0.1
    - Label Smoothing
      - ϵls = 0.1값의 Label Smoothing 사용
      - 모델이 혼란스러워지지만 정확도와 BLUE 점수 향상상
