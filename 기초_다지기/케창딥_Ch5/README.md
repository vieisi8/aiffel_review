# 학습 내용

---

- 일반화: 머신 러닝의 목표
- 머신 러닝 모델 평가
- 훈련 성능 향상하기
- 일반화 성능 향상하기

---

## 일반화: 머신 러닝의 목표

---

머신 러닝의 근본적인 이슈

	-> 최적화와 일반화 사이의 줄다리기

최적화?

	가능한 훈련 데이터에서 최고의 성능을 얻으려고 모델의 조정하는 과정

일반화?

	훈련된 모델이 이전에 본 적 없는 데이터에서 얼마나 잘 수행되는지 의미

---

목표

	좋은 일반화 성능을 얻는 것

---

### 과소적합과 과대적합

---

![Alt text](https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FmSvgt%2Fbtsn7vP4wi1%2FIUjWtkeATfqLYqBPUvQ1ck%2Fimg.png)

	전형적인 과대적합 진행 과정

과소적합

	훈련 데이터의 손실이 낮아질수록 테스트 데이터의 손실도 낮아짐

과대적합

	훈련 데이터에서 훈련을 특정 횟수만큼 반복하고 난 후에는 일반화 성능이 더 이상 높아지지 않으며 검증 세트의 성능이 멈추고 감소되기 시작

---

과대적합 이유

	1. 잡음 섞인 훈련 데이터
	
		ex) mnist 데이터에 이상한 훈련 샘플 / 레이블이 잘못된 훈련 샘플이 존재함

		    모델이 이런 이상치를 맞추려고 하면 

			-> 일반화 성능 감소

![Alt text](https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FbmqAly%2Fbtsn26wzOBJ%2F9QhJd4GFaSh22g5QOzxOQ1%2Fimg.png)

	2. 불확실한 특성

		불확실성과 모호성이 있다면

			-> 깔끔하게 레이블이 부여된 데이터라도 잡음이 생길 수 있음

		ex) 바나나 이미지 

			바나나가 덜 익었는지, 익었는지 또는 썩었는지 예측이 불가능

![Alt text](https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2Fcm0b9z%2Fbtsn3kaA091%2FcK1lnWNtgITLdfOTzszG4K%2Fimg.png)

	3. 드문 특성과 가짜 상관관계

		ex) 평생 두 마리의 주황색 얼룩무늬 고양이만 보았고 둘 다 사교성이 매우 없다면

			-> 주활색 얼룩무늬 고양이는 일반적으로 사교적이지 않다고 추측 가능

		중요한점!

			-> 가짜 상관관계를 만들어 내는 데 특성 값이 몇번만 등장할 필요가 없다는 것

				-> 과대적합의 가장 보편적인 원인 중 하나

---

3번 예제

	백색 잡음 픽셀과 0픽셀 추가

	'''
	
	from tensorflow.keras.datasets import mnist
	import numpy as np

	(train_images, train_labels),_=mnist.load_data()

	train_images=train_images.reshape((60000,28*28))
	train_images=train_images.astype("float32")/255
	train_images_with_noise_channels=np.concatenate(
	    [train_images,np.random.random((len(train_images),784))], axis=1
	)
	train_images_with_zeros_channels=np.concatenate(
	    [train_images,np.zeros((len(train_images),784))], axis=1
	)

	from tensorflow import keras
	from tensorflow.keras import layers

	def get_model():
	  model=keras.Sequential([
	      layers.Dense(512,activation="relu"),
	      layers.Dense(10,activation="softmax")
	  ])
	  model.compile(
	      optimizer="rmsprop",
	      loss="sparse_categorical_crossentropy",
	      metrics=["accuracy"])
	  return model

	model=get_model()
	history_noise=model.fit(
	    train_images_with_noise_channels,train_labels,batch_size=128,epochs=10,validation_split=0.2
	)

	model=get_model()
	history_zeros=model.fit(
	    train_images_with_zeros_channels,train_labels,batch_size=128,epochs=10,validation_split=0.2
	)

	import matplotlib.pyplot as plt

	val_acc_noise=history_noise.history["val_accuracy"]
	val_acc_zeros=history_zeros.history["val_accuracy"]
	epochs=range(1,11)
	plt.plot(epochs,val_acc_noise,"b-",label="Validation accuracy with noise channels")
	plt.plot(epochs,val_acc_zeros,"b--",label="Validation accuracy with zeros channels")
	plt.xlabel("Epochs")
	plt.ylabel("Accuracy")
	plt.legend()
	plt.show()

	'''

![Alt text](./a.png)

	잡음이 섞인 데이터에서 훈련된 모델의 검증 정확도 1% 포인트 낮음

		-> 가짜 상관관계의 영향 때문

잡음 특성

	특성 선택을 수행해 모델에 유익한 특성만 남겨야함

		1. 특성과 레이블 사이의 상호 의존 정보처럼 작업에 대해 특성이 얼마나 유익한지 측정(유용성 점수 계산)
		2. 임계 값을 넘긴 특성만 사용

			-> 백색 잡음이 걸러짐

---

### 딥러닝에서 일반화의 본질

---

표현 능력이 충분하다면

	-> 어떤 것에도 맞추도록 훈련 가능


랜덤하게 섞은 레이블로 MNIST 모델 훈련

	'''

	(trian_images,train_labels),_=mnist.load_data()
	train_images=train_images.reshape((60000,28*28))
	train_images=train_images.astype("float32")/255

	random_train_labels=train_labels[:]
	np.random.shuffle(random_train_labels)

	model=keras.Sequential([
	    layers.Dense(512,activation="relu"),
	    layers.Dense(10,activation="softmax")
	])

	model.compile(
	    optimizer="rmsprop",
	    loss="sparse_categorical_crossentropy",
	    metrics=["accuracy"]
	)

	model.fit(
	    train_images,
	    random_train_labels,
	    batch_size=128,
	    epochs=100,
	    validation_split=0.2
	)

	'''

		모델 파라미터가 충분하다면 모델을 훈련할 수 있음

---

딥러닝 모델의 일반화?

	딥러닝 모델 자체와 거의 관련이 없고,

		-> 실제 세상의 구조와 많은 관련이 있음

---

매니폴드 가설?


