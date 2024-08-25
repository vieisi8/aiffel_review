# 학습 내용

---

- Regularizaion(정직화) Normalizaion(정규화)와 구분
- L1 Regularization
- L2 Regularization
- LP norm
- Dropout
- Batch Normalization

---

## Regularization(정직화)

---

	오버피팅(overfitting)을 해결하기 위한 방법 중의 하나

	종류 -> L1, L2 Regularization, Dropout, Batch normalization 등

	모델에 제약 조건을 걸어서 모델의 train loss를 증가, validation loss나 최종 test loss를 감소시키려는 목적

---

### Overfitting(과적합)

---

train set은 매우 잘 맞히지만, validation/test set은 맞히지 못하는 현상

---

	Regularization은 오버피팅을 막고자 하는 방법, Normalization은 트레이닝을 할 때에 서로 범위가 다른 데이터들을 같은 범위로 바꿔주는 전처리 과정

---

## L1 Refularization(Lasso)

---

N: 데이터의 개수, D: 데이터의 차원(feature의 개수)

![Alt text](![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/c09f8228-29c7-4dcb-8ca3-1de7d3988fab/c1a421c9-c928-4952-a6d2-ca9911a31734/Untitled.png))

	L1 regularization은 위와 같은 식으로 정의

![Alt text](![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/c09f8228-29c7-4dcb-8ca3-1de7d3988fab/2607bcff-e108-4c87-85c4-123e1797df9b/image.png))

	위 부분(L1 norm)이 없다면 linear regression과 동일

p=1이므로 L1 regularization라고 부르는 것

---

	from sklearn.linear_model import Lasso

---

L1 regularization에서는 총 13개 중 7개를 제외한 나머지의 값들이 모두 0임을 확인

linear regression과 L1, L2 regularization의 차이 중 하나는 하이퍼파라미터(수식에서는 λ)가 하나 더 들어간다는 것이고, 그 값에 따라 error에 영향을 미친다는 점입니다.

---

## L2 Regularization(Ridge)

---

![Alt text](![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/c09f8228-29c7-4dcb-8ca3-1de7d3988fab/dfeb5d90-3ecb-4314-87a9-e720643c6d7c/image.png))

	L2 regularization은 위와 같은 식으로 정의

![Alt text](![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/c09f8228-29c7-4dcb-8ca3-1de7d3988fab/77bf4ac9-921a-47ad-b2b4-c85d3e36c189/image.png))

	위 부분(L2 Norm)이 핵심 내용

---

	from sklearn.linear_model import Ridge

---

### L1 / L2 Regularization의 차이점

---

![Alt text](![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/c09f8228-29c7-4dcb-8ca3-1de7d3988fab/bd40bcf5-2155-44ed-af85-4069c861e999/image.png))

	- L1 regularization(왼쪽)은 |W1|+|W2|의 값을 최소화하는 것이 목적 -> 마름모 형태의 제약 조건이 생김
	- L2 regularization은 W1^2+W2^2값을 최소화하는 것이 목적 -> 제약 조건이 원의 형태로 나타나게 됨
	- 빨간색 직선 -> 문제의 해답이 될 수 있는 파라미터 (W1,W2)들의 집합

	L1 regularization -> 제약 조건이 마름모 모양 ∴ '정답 파라미터 집합'과 만나는 지점 역시 축 위에 있을 가능성이 높음

	L2 regularization -> 제약 조건이 원 모양 ∴ 직선과 만나는 지점이 축과 가까운 다른 곳에 있을 가능성이 높음

	__L2 norm(ridge)은 제곱이 들어가 있기 때문에 절댓값을 쓰는 L1 norm(Lasso)보다는 수렴이 빠르다는 장점 __

---

	L1 regularization은 가중치가 적은 벡터에 해당하는 계수를 0으로 보내면서 차원 축소와 비슷한 역할을 하는 것이 특징

	L2 regularization은 계수를 0으로 보내지는 않지만 제곱 텀이 있기 때문에 L1 regularization보다는 수렴 속도가 빠르다는 장점

---

## Lp norm

---

norm?

	벡터뿐만 아니라 함수, 행렬의 크기를 나타내는 개념

---

![Alt text](![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/c09f8228-29c7-4dcb-8ca3-1de7d3988fab/cd974831-546e-452c-8493-6bc52ad11b89/image.png))

	norm의 정의 

---

	p=∞ 인 infinity norm -> x에서 가장 큰 숫자를 출력

---

### Matrix norm

---

주로 p=1,∞ 인 경우만 알면 됨

	p=1 -> 컬럼(column)의 합이 가장 큰 값이 출력, p=∞ -> 로우(row)의 합이 가장 큰 값이 출력

---

## Dropout

---

	확률적으로 랜덤하게 몇 가지의 뉴런만 선택하여 정보를 전달하는 과정

	오버피팅을 막는 regularization layer 중 하나

---

	확률을 너무 높이면 (비활성화된 뉴런의 비중을 높이면) 모델 안에서 값들이 제대로 전달되지 않으므로 학습이 잘 되지 않음, 확률을 너무 낮추는 경우에는 fully connected layer와 같이 동작

---

	overfitting이 되는지 확인 -> train set과 validation set의 loss function을 그려보는 것

---

loss, accuracy 그래프로 overfitting을 어떻게 판단할 수 있을까?

	train loss, acc는 계속해서 감소/증가하는 데에 반해 val loss, acc는 train을 따라가지 못하거나 일정 범위 안에 수렴하는 것으로 보아 모델이 새로운 값에 대한 답을 제대로 내지 못하고 있다고 보인다. ⇒ overfitting

---

## Batch Normalization

---

### 경사 하강법 

[경사 하강법 종류](https://gooopy.tistory.com/69)

---

	- 데이터셋 전체를 본 다음 업데이트하는 'Batch Gradient Descent'
	- 데이터 하나를 관찰할 때마다 업데이트하는 'Stochastic Gradient Descent'
	- 데이터셋을 여러 개의 mini-batch로 쪼갠 다음 하나의 batch를 처리할 때마다 가중치를 업데이트하는 'Mini-batch Gradient Descent'

---

batch normalization?

![Alt text](![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/c09f8228-29c7-4dcb-8ca3-1de7d3988fab/4798f2d9-866a-4bee-839c-8e195ba5a8d1/image.png))

	각 mini-batch마다 평균과 분산을 계산하여 정규화(normalization)를 수행, scale and shift 변환을 적용하여 mini-batch들이 비슷한 데이터 분포를 가지도록 함

	mini-batch의 평균과 분산을 구해서 입력 데이터를 정규화(normalize)하고, 이 값에 scale(γ)과 shift(β)를 추가한 것

---

	- 중간에 ϵ이 붙은 이유는 분산이 0일 경우 나눗셈 오류가 발생하는 것을 방지하기 위함
	- γ와 β 값은 학습 파라미터로 모델 학습이 진행되면서 가중치와 함께 업데이트됨

