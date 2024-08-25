# 학습 내용

---

- 다양한 머신러닝 알고리즘
- 사이킷런에서 가이드하는 머신러닝 알고리즘
- 사이킷런 설치
- 사이킷런 주요 모듈
	- 데이터 표현법
	- 회귀 모델
	- Datasets 모듈
	- 분류 문제 실습
	- Estimator
- 훈련 데이터와 테스트 데이터 분리

---

## 머신러닝 알고리즘

---

- 지도학습
- 비지도학습
- 강화학습

---

### 지도학습과 비지도학습

---

[적절한 머신러닝 알고리즘을 선택하는 방법](https://modulabs.co.kr/blog/choosing-right-machine-learning-algorithm/)

---

Q. 위 링크에서 지도 학습에 속하는 대표적인 알고리즘에는 어떤것이 있나요?

	분류, 회귀, 예측

---

Q. 위 링크에서 비지도 학습에 속하는 대표적인 알고리즘에는 어떤 것이 있나요?

	군집화, 차원 축소

---

Q. 위 링크에서 특정 알고리즘에 대해서도 상세히 설명했는데요. 어떤 것들이 있었나요? (최소 5개 이상 적어보세요)

	선형 회귀, 로지스틱 회귀, 랜덤 포레스트, 그래디언트 부스팅, K-means

---

__정답 유무, 데이터의 종류, 특성, 문제 정의에 따라 머신러닝 알고리즘은 굉장히 복합적으로 사용됨__

---

### 강화학습

---

	학습하는 시스템을 에이전트라고 정의, 환경을 관찰해서 에이전트가 스스로 행동하게 함
	
	모델은 그 결과로 특정 보상을 받아 이 보상을 최대화하도록 학습

---

용어

- 에이전트(Agent): 학습 주체 (actor, controller)
- 환경(Environment): 에이전트에게 주어진 환경, 상황, 조건
- 행동(Action): 환경으로부터 주어진 정보를 바탕으로 에이전트가 판단한 행동
- 보상(Reward): 행동에 대한 보상을 머신러닝 엔지니어가 설계

---

대표적인 종류

- Monte Carlo methods
- Q-Learning
- Policy Gradient methods

---

## 사이킷런에서 가이드하는 머신러닝 알고리즘

---

[scikit-learn: Choosing the right estimator](https://scikit-learn.org/1.3/tutorial/machine_learning_map/index.html)

---

Q. 사이킷런에서 알고리즘의 Task는 몇 가지이며 각각 무엇인가요?

	4개
	
	classification
	
	regression
	
	clustering
	
	dimensionality reduction

---

Q. 사이킷런에서는 알고리즘을 크게 어떤 기준에 따라 나누었나요?

	samples > 50

	predicting a category

	have labeled data

	predicting a quantity

	just looking

---

Q. 사이킷런에서 소개하는 Classification용 알고리즘은 몇 개이며 그 종류에는 무엇이 있나요?

	7개 SGD Classifier, KNeighborsClassifier, LinearSVC, NaiveBayes, SVC, Kernel approximation, EnsembleClassifiers

---

Q. 사이킷런에서 소개하는 Regression용 알고리즘은 몇 개이며 그 종류에는 무엇이 있나요?

	7개 SGD Regressor, Lasso, ElasticNet, RidgeRegression, SVR(kernel='linear'), SVR(kernel='rbf'), EnsembelRegressor

---

Q. Ensemble 기법은 어느 Task 카테고리에 사용되었나요?

	Classification, Regression

---

Q. SGD 기법은 어느 Task 카테고리에 사용되었나요?

	Classification, Regression

---

## scikit-learn 설치

---

명령어

__$ pip install scikit-learn__

---

Q. scikit-Learn에서 scikit이란 이름은 어떻게 만들어 진 건가요?

	SciPy + Toolkit

---

Q. scikit-Learn에서 훈련 데이터와 테스트 데이터를 나누는 기능을 제공하는 함수의 이름은 무엇인가요?

	train_test_split

---

Q. 위 동영상을 보고 scikit-Learn에서 ETL(Extract Transform Load) 기능을 수행하는 함수가 무엇인지 찾아보세요

	transformer

---

Q. 위 동영상을 보고 scikit-Learn에서 모델(Model) 로 표현되는 클래스가 무엇인지 찾아보세요.

	Estimator

---

Q. 위 동영상에서 소개한 Estimator 클래스의 메소드에는 어떤 것이 있었나요?

	fit, predict

---

Q. 위 동영상에서 Estimator와 transformer() 2가지 기능을 수행하는 scikit-learn의 API는 무엇인가요?

	pipeline

---

## 데이터 표현법

---

주요 사이킷런 API

![Alt text](https://d3s0tskafalll9.cloudfront.net/media/images/Untitled_sP0AtFE.max-800x600.png)

---

데이터 표현법

	NumPy의 ndarray, Pandas의 DataFrame, SciPy의 Sparse Matrix

	특성 행렬(Feature Matrix) / 타겟 벡터(Target Vector) 2가지로 표현

---

![Alt text](https://d3s0tskafalll9.cloudfront.net/media/images/Untitled_1.max-800x600.png)

---

### 특성 행렬(Feature Matrix)

---

- 입력 데이터를 의미합니다.
- 특성(feature): 데이터에서 수치 값, 이산 값, 불리언 값으로 표현되는 개별 관측치를 의미합니다. 특성 행렬에서는 열에 해당하는 값입니다.
- 표본(sample): 각 입력 데이터, 특성 행렬에서는 행에 해당하는 값입니다.
- n_samples: 행의 개수(표본의 개수)
- n_features: 열의 개수(특성의 개수)
- X: 통상 특성 행렬은 변수명 X로 표기합니다.
- [n_samples, n_features]은 [행, 열] 형태의 2차원 배열 구조를 사용하며 이는 NumPy의 ndarray, Pandas의 DataFrame, SciPy의 Sparse Matrix를 사용하여 나타낼 수 있습니다.

---

### 타겟 벡터 (Target Vector)

---

- 입력 데이터의 라벨(정답) 을 의미합니다.
- 목표(Target): 라벨, 타겟값, 목표값이라고도 부르며 특성 행렬(Feature Matrix)로부터 예측하고자 하는 것을 말합니다.
- n_samples: 벡터의 길이(라벨의 개수)
- 타겟 벡터에서 n_features는 없습니다.
- y: 통상 타겟 벡터는 변수명 y로 표기합니다.
- 타겟 벡터는 보통 1차원 벡터로 나타내며, 이는 NumPy의 ndarray, Pandas의 Series를 사용하여 나타낼 수 있습니다.
- (단, 타겟 벡터는 경우에 따라 1차원으로 나타내지 않을 수도 있습니다. 이 노드에서 사용되는 예제는 모두 1차원 벡터입니다.)

---

	__특성 행렬 X의 n_samples와 타겟 벡터 y의 n_samples는 동일해야 합니다.__

---

Q. 결국 특성 행렬과 타겟 벡터를 한 문장으로 요약해볼까요?

	특성 행렬은 학습하고자 하는 데이터 셋이고, 타켓 벡터는 그 데이터 셋의 정답이다.

---

## 회귀 모델 실습

---

ex) y -> 라벨 / X -> 입력 데이터 100개씩 구성

	'''
	
	import numpy as np
	import matplotlib.pyplot as plt
	r = np.random.RandomState(10)
	x = 10 * r.rand(100)
	y = 2 * x - 3 * r.rand(100)
	plt.scatter(x,y)
	
	'''

x와 y의 모양은 (100,)으로 1차원 벡터

---

	'''

	from sklearn.linear_model import LinearRegression
	model = LinearRegression()
	model

	'''

LinearRegression 모델이 생성

---

위의 입력 데이터인 x를 그대로 넣으면, 에러가 발생 -> x를 행렬 바꿔줘야됨

	__reshape()를 사용__

	'''

	X = x.reshape(100,1)

	model.fit(X,y)

	'''

---

	잠깐 Tip!
	reshape() 함수에서 나머지 숫자를 -1로 넣으면 자동으로 남은 숫자를 계산해 줍니다. 즉, x_new의 인자의 개수가 100개이므로, (100, 1)의 형태나 (2, 50)의 형태 등으로 변환해 줄 수 있는데요. (2, -1)을 인자로 넣으면 (2, 50)의 형태로 자동으로 변환해 줍니다. 아래 코드를 통해 확인해 보세요.

	'''

	X_ = x_new.reshape(-1,1)
	X_.shape

	'''

---

회귀 모델의 경우 RMSE(Root Mean Square Error) 를 사용해 성능을 평가

	'''

	from sklearn.metrics import mean_squared_error
	
	y_true = y
	y_pred = y_new
	error = mean_squared_error(y_true, y_pred,squared=False)
	# error = np.sqrt(mean_squared_error(y_true, y_pred))
	print(error)

	'''

---

## Datasets 모듈

---

sklearn.datasets 모듈 = dataset loaders & dataset fetchers

Toy dataset과 Real World dataset을 제공

---

Toy dataset의 예시

- datasets.load_boston(): 회귀 문제, 미국 보스턴 집값 예측(version 1.2 이후 삭제 예정)
- datasets.load_breast_cancer(): 분류 문제, 유방암 판별
- datasets.load_digits(): 분류 문제, 0 ~ 9 숫자 분류
- datasets.load_iris(): 분류 문제, iris 품종 분류
- datasets.load_wine(): 분류 문제, 와인 분류

---

### datasets.load_wine() 뜯어보기

---

	Bunch -> 파이썬의 딕셔너리와 유사한 형태의 데이터 타입

	data.data.ndim -> 차원을 확인

	data.target -> 타겟 벡터
	
	data.feature_names -> 특성들의 이름 저장

	len() -> 개수 확인

	data.target_names -> 분류하고자 하는 대상

	data.DESCR -> 데이터에 대한 설명

---

분류 문제 실습

특성 행렬 -> DataFrame

특성 행렬 -> 통상 변수명 X에 저장하고, 타겟 벡터는 y에 저장

	'''

	X = data.data
	y = data.target

	'''

RandomForestClassifier를 사용해

	'''

	from sklearn.ensemble import RandomForestClassifier
	model = RandomForestClassifier()

	'''

훈련

	'''

	model.fit(X,y)

	'''

예측

	'''
	
	y_pred = model.predict(X)
	
	'''

평가

분류 문제의 경우 classification_report 와 accuracy_score를 이용

	'''
	
	from sklearn.metrics import accuracy_score
	from sklearn.metrics import classification_report
	
	#타겟 벡터 즉 라벨인 변수명 y와 예측값 y_pred을 각각 인자로 넣습니다. 
	print(classification_report(y, y_pred))
	#정확도를 출력합니다. 
	print("accuracy = ", accuracy_score(y, y_pred))

	'''

---

Q. 위 모델의 정확도가 어떻게 100%가 나왔을까요?

	학습과 테스트 셋을 구분하지 않고 학습 데이터로 정확도를 예측했기 때문

---

## Estimator

---

	데이터셋을 기반으로 머신러닝 모델의 파라미터를 추정하는 객체

	사이킷런의 모든 머신러닝 모델은 Estimator라는 파이썬 클래스로 구현

	훈련 fit() 메서드, 예측 predict() 메서드

	Estimator 객체는 LinearRegression() & RandomForestClassifier()

---

### 분류 Estimator

![Alt text](https://d3s0tskafalll9.cloudfront.net/media/images/Untitled_2_4s16x9i.max-800x600.png)

---

### 선형 회귀 Estimator

![Alt text](https://d3s0tskafalll9.cloudfront.net/media/images/Untitled_3.max-800x600.png)

---

### 비지도 Estimator

![Alt text](https://d3s0tskafalll9.cloudfront.net/media/images/Untitled_4_QrGQlgR.max-800x600.png)

---

## 훈련 데이터와 테스트 데이터 분리하기

![Alt text](https://d3s0tskafalll9.cloudfront.net/media/images/Untitled_6_uWp8wos.max-800x600.png)

---

	훈련에 쓰이는 데이터와 예측에 쓰이는 데이터는 다른 데이터를 사용

---

	사이킷런 API -> model_selection의 train_test_split()

	'''

	from sklearn.model_selection import train_test_split

	result = train_test_split(X, y, test_size=0.2, random_state=42)

	'''

	인자 -> 특성 행렬 X,  타겟 벡터 y, 테스트 데이터의 비율, random_state(랜덤하게 데이터를 섞어주는 기능)

	train_test_split()은 반환값 4개(list)

	'''

	ex)

	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

	'''

---
