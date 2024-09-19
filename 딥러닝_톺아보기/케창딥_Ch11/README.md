# 학습 내용

---

- 자연어 처리 소개
- 텍스트 데이터 준비
- 단어 그룹을 표현하는 두 가지 방법: 집합 / 시퀀스
- 트랜스포머 아키텍쳐
- 텍스트 분류를 넘어: 시쿼스-투-시퀀스 학습

---

## 자연어 처리 소개

---

자연어 처리 종류

	- 텍스트 분류
	- 콘텐츠 필터링
	- 감성 분석
	- 언어 모델링
	- 번역
	- 요약 등등

---

## 텍스트 데이터 준비

---

미분 가능한 함수인 딥러닝 델

	-> 수치 텐서만 처리 가능

텍스트 벡터화?

	텍스트를 수치 텐서로 바꾸는 과정

	1. 텍스트 표준화
	2. 토큰 단위로 분할 -> 토큰화
	3. 각 토큰 수치 벡터로 변환

![Alt text](https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FbPNKsQ%2FbtspsVtRl7O%2FQnDWdKD09VRUhdHmkgN8mK%2Fimg.png)

---

### 텍스트 표준화

---

텍스트 표준화?
	
	모델이 인코딩 차이를 고려하지 않도록 이를 제거하기 위한 기초적인 특성 공학의 한 형태

		-> 검색 엔진을 만들 때도 동일한 작업을 수행해야 함

가장 간단하고 널리 사용되는 표준화 방법

	소문자로 바꾸고 구두점 문자를 삭제

	ex) “Sunset came; I stared at the México sky. isn’t nature splendid?”

		-> "sunset came i stared at the méxico sky isnt nature splendid"

어간 추출?

	어형이 변형된 단어를 공통된 하나의 표현으로 바꾸는 것

	ex)

		"cats" -> "[cat]"

		"was staring", "stared" -> "[stare]"

	-> 머신 러닝에서 드물게 사용됨

장점

	모델에 필요한 훈련 데이터가 줄어들고 일반화가 더 잘됨

단점

	일정량의 정보를 삭제할 수도 있다는 점

---

### 텍스트 분할(토큰화)

---

텍스트 표준화 다음 스텝

	벡터화할 단위(토큰)로 나눔 -> 토큰화

세가지 방법

	- 단어 수준 토큰화 -> 토큰이 공백으로 (또는 구두점으로) 구분된 부분 문자열
		- ex) "staring" -> "star+ing" / "called" -> "call+ed"
	- N-그램 토큰화 -> 토큰 N개의 연속된 단어 그룹
		- ex) "the cat" / "he was" -> 2-그램(또는 바이그램(bigram)) 토큰
	- 문자 수준 토큰화 -> 각 문자가 하나의 토큰
		- 텍스트 생성 / 음성 인식 같은 특별한 작업에서만 사용

	-> 일반적으로 단어 수준 토큰화 / N-그램 토큰화를 항상 사용

---

N-그램과 BoW

ex) "the cat sat on the mat." 

	2-그램의 집합

	-> {"the", "the cat", "cat", "cat sat", "sat", "sat on", "on", "on the", "the mat", "mat"}

		-> 2-그램 가방(bag of 2-gram)
		
	3-그램의 집합

	-> {"the", "the cat", "cat", "cat sat", "the cat sat", "sat", "sat on", "on", "cat sat on", "on the", "sat on the", "the mat", "mat", "on the mat"}

		-> 2-그램 가방(bag of 2-gram)

	가방(bag)?

		-> 다루고자 하는 것이 토큰의 집합

		-> 특정한 순서 X

	이런 종류의 토큰화 방법

	-> BoW(또는 Bag-of-N-gram)

		-> 얕은 학습 방법의 언어 처리 모델에 사용되는 경향이 있음

	딥러닝 시퀀스 모델 -> 이런 수동적인 방식을 계층적인 특성 학습으로 대체

	1D 컨브넷, 순환 신경망, 트랜스포머 -> 단어와 문자 그룹에 대한 특성을 학습할 수 있음

	장점

		-> 그룹들을 명시적으로 알려 주지 않아도 연속된 단어나 문자의 시퀀스에 노출되면 자동으로 학습

---

### 어휘 사전 인덱싱

---

토큰화 다음 스텝

	각 토큰을 수치 표현으로 인코딩

어휘 사전 인덱싱?

	훈련 데이터에 있는 모든 토큰의 인덱스(어휘 사전)를 만들어 어휘 사전의 각 항목에 고유한 정수를 할당하는 방법 

	ex) 

	'''

	vocabulary = {}
	for text in dataset:
	    text = standardize(text)
	    tokens = tokenize(text)
	    for token in tokens:
	    	if token not in vocabulary:
	        	vocabulary[token] = len(bocabulary)

	'''

		-> 이 정수를 신경망이 처리할 수 있도록 원-핫 벡터 같은 벡터 인코딩으로 바꿀 수 있음

어휘 사전의 개수를 제한하는 이유?

	고유한 토큰이 굉장히 많음

		-> 드믄 토큰을 인덱싱하면 특성 공간이 과도하게 커지게 됨(거의 아무런 정보가 없을 것)

	일반적으로 2만개 / 3만개로 제한

간과해서는 안되는 중요한 사항

	어휘 사전 인덱스에서 새로운 토큰을 찾을 때 이 토큰이 항상 존재하지 않을 수 있음

		-> KeyError 에러 발생

	-> 예외 어휘 인덱스 사용

예외 어휘 인덱스(OOV 인댁스)?

	- 어휘 사전에 없는 모든 토큰에 대응됨
	- 일반적으로 1 할당
	- 디코딩 -> "[UNK]" 

왜 0이 아니고 1을 사용할까?

	->  0은 이미 사용되는 토큰(마스킹(masking) 토큰)!

마스킹(masking) 토큰?

	단어가 아니라 무시할 수 있는 토큰을 의미함

		-> 시퀀스 데이터를 패딩하기 위해 사용됨

패딩을 하는 이유?

	배치 데이터는 동일해야 하기 때문에 배치에 있는 모든 시퀀스는 길이가 같아야 함

		-> 길이가 짧은 시퀀스는 가장 긴 시퀀스 길이에 맞추어 패딩

![Alt text](https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FtGdpt%2FbtspxaqKLDN%2FGL7wJzK8Oe4jn3EkrzM6GK%2Fimg.png)

---

### TextVectorization 층

---

TextVectorization 층?

	케라스에서 제공

	'''

	from tensorflow.keras.layers import TextVectorization
	text_vectorization = TextVectorization(
	        output_mode="int",
	)

	'''

		-> 텍스트 표준화를 위해 소문자로 바꾸고 구두점을 제거, 토큰화를 위해 공백으로 나눔

 tf.string 텐서를 처리할 때?

	사용자 정의 함수를 사용해야 함!

	ex)

	'''

	import re
	import string
	import tensorflow as tf

	def custom_standardization_fn(string_tensor):
	    lowercase_string = tf.strings.lower(string_tensor) # 문자열을 소문자로 빠군다.
	    return tf.strings.regex_replace( # 구두점 문자를 빈 문자열로 바꾼다.
	        lowercase_string, f"[{re.escape(string.punctuation)}]", "")

	def custom_split_fn(string_tensor):
	    return tf.strings.split(string_tensor) # 공백을 기준으로 문자열을 나눈다.

	text_vectorization = TextVectorization(
	    output_mode="int",
	    standardize=custom_standardization_fn,
	    split=custom_split_fn,
	)

	'''

텍스트 말뭉치의 어휘 사전 인덱싱?

	adapt() 메서드 호출

		-> 매개변수 Dataset 객체 / 파이썬 문자열의 리스트를 사용 해야함

			-> 문자열을 반환해야 하기 때문

	ex)

	'''

	dataset = [
	    "I write, erase, rewrite",
	    "Erase again, and then",
	    "A poppy blooms.",
	]
	text_vectorization.adapt(dataset)

	'''

계산된 어휘 사전 추출?

	get_vocabulary() 메서드 사용

		-> 정수 시퀀스로 인코딩된 텍스트를 단어로 다시 변환할 때 유용

	ex)

	'''

	text_vectorization.get_vocabulary()

	'''

![Alt text](./a.png)

tmi

	어휘 사전의 항목 

		-> 빈도 순으로 정렬되어 있음

---

tf.data 파이프라인 또는 모델의 일부로 TextVectorization 층 사용

문제점

	대부분 딕셔너리 룩업(lookup) 연산이기 때문에 GPU(또는 TPU)에서 실행할 수 없고 CPU에서만 실행

		-> 모델을 GPU에서 훈련한다면 TextVecrorization 층이 CPU에서 실행된 후 그 출력을 GPU로 보낼 것

TextVectorization 층을 사용하는 방법이 두 가지

	- tf.data 파이프라인에 넣는 것
	- 모델의 일부로 만드는 것

1. tf.data 파이프라인

---

	'''

	int sequence_dataset = string_dataset.map( # string_dataset은 문자열 탠서를 반환하는 데이터셋
	text_vectorization,
	num_parallel_calls=4) # num_parallel_calls 매개변수를 사용하여 여러 개의 CPU 코어에서 map() 메서드를 병렬화

	'''
	
	- CPU에서 데이터 처리를 비동기적으로 수행할 수 있음
		-  GPU가 벡터화된 데이터 배치에서 모델을 실행할 때 CPU가 원시 문자열의 다음 배치를 벡터화
	- 모델을 GPU나 TPU에서 훈련한다면 최상의 성능을 얻을 수 있음

2. 모델의 일부

---

	'''

	# 문자열을 기대하는 심볼릭 입력을 만든다.
	text_input = keras.input(shape=(), dtype="string")

	# 텍스트 벡터화 층을 적용한다.
	vectorized_text = text_vectorization(text_input)

	# 일반적인 함수형 API 모델처럼 그 위에 새로운 층을 추가할 수 있다.
	embedded_input = keras.layers.Embedding(...)(vectorized_text)
	output = ...
	model = keras.Model(text_input, output)

	'''
	
	- 벡터화 단계가 모델의 일부이므로 모델의 나머지 부분과 동기적으로 수행
		- 훈련 단계마다 (GPU에 놓인) 모델의 나머지 부분이 실행되기 위해 (CPU에 놓인) TextVectorization 층의 출력이 준비되기를 기다린다는 의미
	- 모델을 제품 환경에 배포해야 한다면 좋은 방법

---

## 단어 그룹을 표현하는 두 가지 방법: 집합 / 시퀀스

---

중요한 것

	(단어를 문장으로 구성하는 방식인)단어 순서를 인코딩하는 방법

자연어에서 순서 문제?

	순서는 확실히 중요하지만 의미와 관계는 간단하지 않음

여러 NLP 아키텍쳐를 발생시킨 핵심 질문?

	어떻게 단어의 순서를 표현하는가??

		-> 1. 단어의 순서를 무시하고 텍스트를 단어의 (순서가 없는) 집합으로 처리하는 것 (BoW 모델)
		   2. 단어의 순서를 고려 (시퀀스 모델)

---

### IMDB 영화 리뷰 데이터 준비

---

1. 데이터 준비
	
	앤드류 마스(Andrew Maas)의 스탠포드 페이지에서 데이터셋을 내려받고 압출을 해제

	'''

	!curl -O https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz
	!tar -xf aclImdb_v1.tar.gz

	'''

![Alt text](./b.png)

	- train/pos/ 디렉터리에는 1만 2,500개의 텍스트 파일이 담겨 있음
	- 각 파일은 훈련 데이터로 사용할 긍정적인 영화 리뷰의 텍스트를 담고 있으며, 부정적인 리뷰는 "neg" 디렉터리에 담겨 있음
	- 모두 합해서 훈련용으로 2만 5,000개의 텍스트 파일이 있고 테스트를 위해 또 다른 2만 5,000개의 파일이 있음	

train/unsup 디렉터리 삭제

	필요하지 않으므로 디렉토리 삭제

	'''

	!rm -r aclImdb/train/unsup

	'''

2. 데이터 살펴보기

	실제 모델이 하는 작업에 대한 직관을 기를 수 있기에 데이터를 살펴보는 것이 좋음

	'''

	!cat aclImdb/train/pos/4077_10.txt

	'''

	결과: I first saw this back in the early 90s on UK TV, i did like it then but i missed the chance to tape it, many years passed but the film always stuck with me and i lost hope of seeing it TV again, the main thing that stuck with me was the end, the hole castle part really touched me, its easy to watch, has a great story, great music, the list goes on and on, its OK me saying how good it is but everyone will take there own best bits away with them once they have seen it, yes the animation is top notch and beautiful to watch, it does show its age in a very few parts but that has now become part of it beauty, i am so glad it has came out on DVD as it is one of my top 10 films of all time. Buy it or rent it just see it, best viewing is at night alone with drink and food in reach so you don't have to stop the film.<br /><br />Enjoy

3. 검증 세트 생성

---

	훈련 텍스트 파일에서 20%를 새로운 디렉터리 aclimdb/val로 덜어 내어 검증 세트 생성

	'''

	import os, pathlib, shutil, random

	base_dir = pathlib.Path("aclImdb")
	val_dir = base_dir / "val"
	train_dir = base_dir / "train"
	for category in ("neg", "pos"):
	    os.makedirs(val_dir / category)
	    files = os.listdir(train_dir / category)
	    random.Random(1337).shuffle(files) # 코드를 여러 번 실행해도 동일한 검증 세트가 만들어지도록 랜덤 시드를 지정하여 훈련 파일 목록을 섞는다.

	    # 훈련 파일 중 20%를 검증 세트로 덜어 낸다.
	    num_val_samples = int(0.2 * len(files))
	    val_files = files[-num_val_samples:]

	    # 파일을 aclimdb/val/neg 와 aclimdb/val/pos로 옮긴다.
	    for fname in val_files:
	        shutil.move(train_dir / category / fname,
	                    val_dir / category / fname)

	'''

4. Dataset 객체 생성

---

	image_dataset_from_directory() 메서드를 사용해 Dataset 객체 생성

	'''

	from tensorflow import keras
	batch_size = 32

	train_ds = keras.utils.text_dataset_from_directory(
	    "aclImdb/train", batch_size=batch_size
	)
	val_ds = keras.utils.text_dataset_from_directory(
	    "aclImdb/val", batch_size=batch_size
	)
	test_ds = keras.utils.text_dataset_from_directory(
	    "aclImdb/test", batch_size=batch_size
	)

	'''

		dataset 객체

			->  텐서플로의 tf.string 텐서인 입력과 "0" 또는 "1"로 인코딩된 int32텐서인 타깃을 반환

5. 첫 번째 배치의 크기와 dtype 출력

---

	'''

	for inputs, targets in train_ds:
	    print("inputs.shape:", inputs.shape)
	    print("inputs.dtype:", inputs.dtype)
	    print("targets.shape:", targets.shape)
	    print("targets.dtype:", targets.dtype)
	    print("inputs[0]:", inputs[0])
	    print("targets[0]:", targets[0])
	    break

	'''

![Alt text](./c.png)

---

### 단어를 집합으로 처리: BoW 방식

---

1. 이진 인코딩을 사용한 유니그램

개별 단어 == 유니그램

	ex) "the cat sat on the mat"

				↓

![Alt text](./d.png)

장점

	전체 텍스트를 하나의 벡터로 표현할 수 있다는 것

		-> 벡터의 각 원소는 한 단어의 존재 유무를 표시

- TextVectorization 층으로 데이터 전처리

---

	'''

	text_vectorization = TextVectorization(
	    # 가장 많이 등장하는 2만 개 단어로 어휘 사전을 제한한다.
	    # 그렇지 않으면 훈련 데이터에 있는 모든 단어를 인덱싱하게 된다.
	    # 아마도 수만 개의 단어가 한 번 또는 두 번만 등장하면 유용하지 않을 것이다.
	    # 일반적으로 텍스트 분류에서 2만 개는 적절한 어휘 사전 크기이다.
	    max_tokens=20000,

	    # 멀티-핫 이진 벡터로 출력 토큰을 인코딩한다.
	    output_mode="multi_hot",
	)
	# (레이블 없이) 원시 텍스트 입력만 반환하는 데이터셋을 준비한다.
	text_only_train_ds = train_ds.map(lambda x, y: x)

	# adapt() 메서드로 이 데이터셋의 어휘 사전을 인덱싱한다.
	text_vectorization.adapt(text_only_train_ds)

	# 훈련, 검증, 테스트 데이터셋을 전처리한다.
	# 다중 CPU 코어를 활용하기 위해 num_parallel_calls 매개변수를 지정한다.
	binary_1gram_train_ds = train_ds.map(
	    lambda x, y: (text_vectorization(x), y),
	    num_parallel_calls=4)
	binary_1gram_val_ds = val_ds.map(
	    lambda x, y: (text_vectorization(x), y),
	    num_parallel_calls=4)
	binary_1gram_test_ds = test_ds.map(
	    lambda x, y: (text_vectorization(x), y),
	    num_parallel_calls=4)

	'''

		멀티-핫 인코딩된 이진 단어 벡터

			-> 하나의 단어씩 처리(유니그램)

- 이진 유니그램 데이터셋의 출력 확인

---

	'''

	for inputs, targets in binary_1gram_train_ds:
	    print("inputs.shape:", inputs.shape)
	    print("inputs.dtype:", inputs.dtype)
	    print("targets.shape:", targets.shape)
	    print("targets.dtype:", targets.dtype)
	    print("inputs[0]:", inputs[0])
	    print("targets[0]:", targets[0])
	    break

	'''

![Alt text](./e.png)

-  모든 예제에서 사용할 모델 생성 함수 정의

---

	'''

	from tensorflow import keras
	from tensorflow.keras import layers

	def get_model(max_tokens=20000, hidden_dim=16):
	    inputs = keras.Input(shape=(max_tokens,))
	    x = layers.Dense(hidden_dim, activation="relu")(inputs)
	    x = layers.Dropout(0.5)(x)
	    outputs = layers.Dense(1, activation="sigmoid")(x)
	    model = keras.Model(inputs, outputs)
	    model.compile(optimizer="rmsprop",
	                  loss="binary_crossentropy",
	                  metrics=["accuracy"])
	    return model

	'''

- 이진 유니그램 모델 훈련, 테스트

---

	'''

	model = get_model()
	model.summary()
	callbacks = [
	    keras.callbacks.ModelCheckpoint("binary_1gram.keras",
	                                    save_best_only=True)
	]

	# 데이터셋의 cache() 메서드를 호출하여 메모리에 캐싱한다.
	# 이렇게 하면 첫 번째 에포크 에서 한 번만 전처리하고 이후 에포크에서는 전처리된 텍스트를 재사용한다.
	# 메모리에 들어갈 만큼 작은 데이터일 때 사용할 수 있다.
	model.fit(binary_1gram_train_ds.cache(),
	          validation_data=binary_1gram_val_ds.cache(),
	          epochs=10,
	          callbacks=callbacks)
	model = keras.models.load_model("binary_1gram.keras")
	print(f"테스트 정확도: {model.evaluate(binary_1gram_test_ds)[1]:.3f}")

	'''

	결과: 테스트 정확도 88.8%

---

2. 이진 인코딩을 사용한 바이그램

단어 순서를 무시?

	하나의 개념이 여러 단어로 표현 가능 하기에 단어 순서를 무시하는 것은 매우 파괴적!

해결책

	 N-그램을 사용

		-> 국부적인 순서 정보를 BoW 표현에 추가하게 됨

	ex) "the cat sat on the mat"

				↓

![Alt text](./f.png)

N-그램 반환 하는 방법

	ngrams=n 매개변수 전달

- 바이그램을 반환하는 TextVectorization 층 생성

---

	'''

	text_vectorization = TextVectorization(
	    ngrams=2,
	    max_tokens=20000,
	    output_mode="multi_hot",
	)

	'''

- 이진 바이그램 모델 훈련, 테스트

---

	'''

	text_vectorization.adapt(text_only_train_ds)
	binary_2gram_train_ds = train_ds.map(
	    lambda x, y: (text_vectorization(x), y),
	    num_parallel_calls=4)
	binary_2gram_val_ds = val_ds.map(
	    lambda x, y: (text_vectorization(x), y),
	    num_parallel_calls=4)
	binary_2gram_test_ds = test_ds.map(
	    lambda x, y: (text_vectorization(x), y),
	    num_parallel_calls=4)

	model = get_model()
	model.summary()
	callbacks = [
	    keras.callbacks.ModelCheckpoint("binary_2gram.keras",
	                                    save_best_only=True)
	]
	model.fit(binary_2gram_train_ds.cache(),
	          validation_data=binary_2gram_val_ds.cache(),
	          epochs=10,
	          callbacks=callbacks)
	model = keras.models.load_model("binary_2gram.keras")
	print(f"테스트 정확도: {model.evaluate(binary_2gram_test_ds)[1]:.3f}")

	'''

	결과: 테스트 정확도 89.6%

		-> 국부적인 순서가 매우 중요하다는 것

---

3. TF-IDF 인코딩을 사용한 바이그램

개별 단어나 N-그램의 등장 횟수를 카운트한 정보를 추가할 수 있음

	-> 텍스트에 대한 단어의 히스토그램(histogram)을 사용

![Alt text](./g.png)

장점

	텍스트 분류 작업 -> 한 샘플에 단어가 얼마나 많이 등장하는지가 중요

- 토큰 카운트를 반환하는 TextVectorization 층 생성

---

	'''

	text_vectorization = TextVectorization(
	    ngrams=2,
	    max_tokens=20000,
	    output_mode="count"
	)

	'''

문제 발생

	"the", "a", "is", "are" 단어는 분류 작업에 거의 쓸모없는 특성임에도 항상 단어 카운트 히스토그램을 압도하여 다른 단어의 카운트를 무색하게 만듦

해결 방법

	정규화!

		 -> 각 특성에서 평균을 빼면 희소성이 깨짐

			-> 나눗셈만 이용하는 정규화 방식을 사용해야 함

가장 좋은 정규화 방법

	TF-IDF 정규화(단어 빈도-역문서 빈도(Term Frequency-inverse Document Frequency))

---

TF-IDF 정규화 이해하기

	- 거의 모든 단어에 등장하는 ("the"나 "a" 같은) 단어는 별로 유용하지 않았음
	- 전체 텍스트 중 일부에서만 나타나는 ("Herzog" 같은) 단어는 매우 독특하므로 중요

		 -> TF-IDF는 두 아이디어를 합친 측정 방법

계산 방법

	1. 현재 문서에 단어가 등장하는 횟수인 '단어 빈도'로 해당 단어에 가중치 부여
	2. 데이터셋 전체에 단어가 등장하는 횟수인 '문서 빈도'로 나눔

코드화

	'''

	def tfidf(term, document, dataset):
	    term_freq = document.count(term)
	    doc_freq = math.log(sum(doc.count(term) for doc in dataset) + 1)
	    return term_freq / doc_freq

	'''

---

TF-IDF 구현

	널리 사용되는 방법이기 때문에 TextVectorization 층에 구현되어 있음

		-> utput_mode 매개변수를 "tf_idf"로 바꾸기만 하면 사용할 수 있음

- TF-IDF 가중치가 적용된 출력을 반환하는 TextVectorization 층 생성

---

	'''

	text_vectorization = TextVectorization(
	    ngrams=2,
	    max_tokens=20000,
	    output_mode="tf_idf",
	)

	'''

- TF-IDF 바이그램 모델 훈련, 테스트

---

	'''

	# 텐서플로 2.8.x 버전에서 TF-IDF 인코딩을 GPU에서 수행할 때 오류가 발생할 수 있습니다.
	# 텐서플로 2.9에서 이 이슈가 해결되었지만 코드를 테스트할 시점에 코랩의 텐서플로 버전은 2.8.2이기 때문에 
	# 에러를 피하기 위해 CPU를 사용하여 텍스트를 변환합니다.
	with tf.device("cpu"):
	    # adapt() 메서드를 호출하면 어휘 사전과 TF-IDF 가중치를 학습합니다.
	    text_vectorization.adapt(text_only_train_ds)

	tfidf_2gram_train_ds = train_ds.map(
	    lambda x, y: (text_vectorization(x), y),
	    num_parallel_calls=4)
	tfidf_2gram_val_ds = val_ds.map(
	    lambda x, y: (text_vectorization(x), y),
	    num_parallel_calls=4)
	tfidf_2gram_test_ds = test_ds.map(
	    lambda x, y: (text_vectorization(x), y),
	    num_parallel_calls=4)

	model = get_model()
	model.summary()
	callbacks = [
	    keras.callbacks.ModelCheckpoint("tfidf_2gram.keras",
	                                    save_best_only=True)
	]
	model.fit(tfidf_2gram_train_ds.cache(),
	          validation_data=tfidf_2gram_val_ds.cache(),
	          epochs=10,
	          callbacks=callbacks)
	model = keras.models.load_model("tfidf_2gram.keras")
	print(f"테스트 정확도: {model.evaluate(tfidf_2gram_test_ds)[1]:.3f}")

	'''

	결과: 테스트 정확도 88.8% 

		-> 많은 텍스트 분류 데이터셋에서 기본 이진 인코딩에 비해 TF-IDF를 사용했을 때 일반적으로 1퍼센트 포인트의 성능을 높일 수 있음

---

원시 문자열을 처리하는 모델 내보내기

문제점

	tf.data 파이프라인의 일부로 텍스트 표준화, 분할, 인덱싱을 수행

		-> 파이프라인과 독립적으로 실행되는 모델을 내보내야 한다면 자체적인 텍스트 전처리를 사용해야 함

해결 방법

	TextVectorization 층을 재사용하는 새로운 모델을 만들고 방금 훈련된 모델을 추가

	'''

	inputs = keras.Input(shape=(1,), dtype="string") # 하나의 입력 샘플은 하나의 문자열
	processed_inputs = text_vectorization(inputs) # 텍스트 전처리를 수행
	outputs = model(processed_inputs) # 이전에 훈련된 모델을 적용
	inference_model = keras.Model(inputs, outputs) # 엔드-투-엔드 모델을 만듦

	'''

테스트

	'''

	import tensorflow as tf
	raw_text_data = tf.convert_to_tensor([
	    ["That was an excellent movie, I loved it."],
	])
	predictions = inference_model(raw_text_data)
	print(f"긍정적인 리뷰일 확률: {float(predictions[0] * 100):.2f} 퍼센트")

	'''

![Alt text](./h.png)

---

### 단어를 시퀀스로 처리: 시퀀스 모델 방식

---

시퀀스 모델?

	원시 단어 시퀀스를 몯델에 전달해 스스로 이런 특성을 학습

시퀀스 모델 구현 방법

	1. 입력 샘플 -> 정수 인덱스의 시퀀스 변환
	2. 각 정수를 벡터로 매핑 -> 벡터 시퀀스 생성
	3. 인접한 벡터의 특징을 비교할 수 있는 층(RNN, 트랜스포머)에 전달

---

첫 번째 예제

- 정수 시퀀스 데이터셋 준비

--

	'''

	from tensorflow.keras import layers

	max_length = 600
	max_tokens = 20000
	text_vectorization = layers.TextVectorization(
	    max_tokens=max_tokens,
	    output_mode="int",

	    # 적당한 입력 크기를 유지하기 위해 입력에서 600개 단어 이후는 잘라 버린다.
	    # 평균 리뷰 길이가 233개의 단어고 600개의 단어보다 긴 리뷰는 5%뿐이므로 합리적인 선택이다.
	    output_sequence_length=max_length,
	)
	text_vectorization.adapt(text_only_train_ds)

	int_train_ds = train_ds.map(
	    lambda x, y: (text_vectorization(x), y),
	    num_parallel_calls=4)
	int_val_ds = val_ds.map(
	    lambda x, y: (text_vectorization(x), y),
	    num_parallel_calls=4)
	int_test_ds = test_ds.map(
	    lambda x, y: (text_vectorization(x), y),
	    num_parallel_calls=4)

	'''

- 원-핫 인코딩된 벡터 시퀀스로 시퀀스 모델 생성

---

	'''

	import tensorflow as tf

	# 입력은 정수 시퀀스이다.
	inputs = keras.Input(shape=(None,), dtype="int64")

	# 정수를 20,000차원의 이진 벡터로 인코딩한다.
	embedded = tf.one_hot(inputs, depth=max_tokens)

	# 양방향 LSTM 층을 추가
	x = layers.Bidirectional(layers.LSTM(32))(embedded)
	x = layers.Dropout(0.5)(x)

	# 마지막으로 분류 층을 추가한다.
	outputs = layers.Dense(1, activation="sigmoid")(x)
	model = keras.Model(inputs, outputs)
	model.compile(optimizer="rmsprop",
	              loss="binary_crossentropy",
	              metrics=["accuracy"])
	model.summary()

	'''

- 첫 번째 시퀀스 모델 훈련

---

	'''

	callbacks = [
	    keras.callbacks.ModelCheckpoint("one_hot_bidir_lstm.keras",
	                                    save_best_only=True)
	]
	model.fit(int_train_ds, validation_data=int_val_ds, epochs=10, callbacks=callbacks)
	model = keras.models.load_model("one_hot_bidir_lstm.keras")
	print(f"테스트 정확도: {model.evaluate(int_test_ds)[1]:.3f}")

	'''

	결과:
	
		- 모델의 훈련이 매우 느림
			- 각 입력 샘플은 (600, 20000) 크기의 행렬로 인코딩되어 있어 결국 하나의 영화 리뷰는 1,200만 개의 부동 소수점으로 이루어지게 됨
			- 양방향 LSTM이 해야 할 일이 많음
		- 모델의 테스트 정확도는 87%에 그침
			- 이진 유니그램 모델만큼 성능이 좋지 않음

	단점의 해결책

		단어 임베딩(word embedding)

---

단어 임베딩 이해하기

	- 원-핫 인코딩으로 무언가를 인코딩했다면 특성 공학을 수행한 것
		- 특성 공간의 구조에 대한 기초적인 가정을 모델에 주입한 것
	- 이 가정은 인코딩하는 토큰은 서로 독립적이라는 것을 의미
		- 사실 원-핫 벡터는 서로 모두 직교
	- 단어는 구조적인 공간을 형성
		- 단어에 공유되는 정보가 있음
	- 단어 "movie"와 "film"은 대부분의 문장에서 동일한 의미로 사용
		- 두 단의 벡터가 동일하거나 매우 가까워야 함
		- 두 단어 벡터 사이의 기하학적 관계는 단어 사이의 의미 관계를 반영해야 함
	- ex) 합리적인 단어 벡터 공간에서는 동의어가 비슷한 단어 벡터로 임베딩될 것이라고 기대
		- 이런 공간에서는 일반적으로 두 단어 벡터 사이의 기하학적 거리(예를 들어 코사인 거리나 L2 거리)가 단어 사이의 '의미 거리'에 연관되어 있다고 생각할 수 있음
		- 다른 의미를 가지는 단어는 서로 멀리 떨어져 있고 관련이 있는 단어는 가까이 놓여 있어야 함

단어 임베딩(word embedding)?

	정확히 위에서 설명한 이를 위한 단어의 벡터 표현

		- 사람의 언어를 구조적인 기하학적 공간에 매핑
		- 저차원의 부동 소수점 벡터(즉, 희소한 벡터가 아니라 밀집 벡터)
		- 많은 정보를 더 적은 차원으로 압축

![Alt text](https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2Fc121rE%2FbtspxaR89kg%2FeSJEZUqH38eYAQReuLFfmk%2Fimg.png)

		- 구조적인 표현이며 이 구조는 데이터로부터 학습
		- 비슷한 단어는 가까운 위치에 임베딩
		- 더 나아가 임베딩 공간의 특정 방향이 의미를 가질 수 있음

ex)

![Alt text](https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FcR9ZYX%2FbtspE35XOgD%2FzHbkkvIoGR9tTFyNPcCYjK%2Fimg.png)

	- 4개의 단어 cat, dog, wolf, tiger가 2D 평면에 임베딩되어 있음
	- 이 벡터 표현을 사용하여 단어 간의 의미 관계를 기하학적 변환으로 인코딩할 수 있음
	- ex) cat에서 tiger로 이동하는 것과 dog에서 wolf로 이동하는 것을 같은 벡터로 나타낼 수 있음
		- 애완동물에서 야생 동물로 이동하는 것으로 해석할 수 있음
	- ex) dog에서 cat으로 이동하는 것과 wolf에서 tiger로 이동하는 벡터
		- 개과에서 고양이과로 이동하는 벡터로 해석할 수 있음
	- 실제 단어 임베딩 공간에서 의미 있는 기하학적 변환의 일반적인 예 -> '성별' 벡터와 '복수(plural)' 벡터
		- 'king' 벡터에 'female' 벡터를 더하면 'queen' 벡터가 됨
		- 'plural' 벡터를 더하면 'kings'가 됨
	- 단어 임베딩 공간에는 일반적으로 이런 해석 가능하고 잠재적으로 유용한 수천 개의 벡터가 있음

단어 임베딩을 만드는 방법

	- 현재 작업(예를 들어 문서 분류나 감성 예측)과 함께 단어 임베딩을 학습 
		-  랜덤한 단어 벡터로 시작, 신경망의 가중치를 학습하는 것과 같은 방식으로 단어 벡터를 학습
	- 현재 풀어야 할 문제와 다른 머신 러닝 작업에서 미리 계산된 단어 임베딩을 모델에 로드
		- 사전 훈련된 단어 임베딩(pretrained word embedding)이라고 부름

---

Embedding 층으로 단어 임베딩 학습

좋은 단어 임베딩 공간을 만드는 것?

	문제에 따라 크게 달라짐

		영어로 된 영화 리뷰의 감성 분석 모델을 위한 완벽한 단어 임베딩 공간 != 영어 법률 문서 분류 모델을 위한 완벽한 단어 임베딩 공간

			-> 특정한 의미 관계의 중요성이 작업에 따라 다르기 때문임

	∴ 새로운 작업에는 새로운 임베딩을 학습하는 것이 타당

새로운 임베딩 학습하는 방법

	역전파를 사용하여 쉽게 만들 수 있고 케라스를 사용하면 더 쉬움

		-> Embedding 층의 가중치를 학습

- Embedding 층 만들기

--

	'''

	# Embedding 층은 적어도 2개의 매개변수가 필요하다. 가능한 토큰의 개수와 임배딩 차원(여기에서는 256)이다.
	embedding_layer = layers.Embedding(input_dim=max_tokens, output_dim=256)

	'''

		정수를 입력으로 받아 내부 딕셔너리에서 이 정수에 연관된 벡터를 찾아 반환

			-> 딕셔너리 룩업(lookup)

![Alt text](https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2Fr33rM%2FbtspmVBqFyT%2Fh3EI9dCVHmF0dPM9SukuH0%2Fimg.png)

Embedding 층 특징

	- 입력: 크기가 (batch_size, sequence_length)인 랭크-2 정수 텐서
		- 각 항목은 정수의 시퀀스
	- 반환: 크기가 (batch_size, sequence_length, embedding_dimensionality)인 랭크-3 부동 소수점 텐서
	- 가중치(토큰 벡터를 위한 내부 딕셔너리)는 다른 층과 마찬가지로 랜덤하게 초기화
	- 훈련 중 -> 이 단어 벡터는 역전파를 통해 점차 조정되고 후속 모델이 사용할 수 있도록 임베딩 공간을 구성
	- 훈련 종료 -> 임베딩 공간은 특정 문제에 전문화된 여러 가지 구조를 가지게 됨

- 밑바닥부터 훈련하는 Embedding 층을 사용한 모델

---

	'''

	inputs = keras.Input(shape=(None,), dtype="int64")
	embedded = layers.Embedding(input_dim=max_tokens, output_dim=256)(inputs)
	x = layers.Bidirectional(layers.LSTM(32))(embedded)
	x = layers.Dropout(0.5)(x)
	outputs = layers.Dense(1, activation="sigmoid")(x)
	model = keras.Model(inputs, outputs)
	model.compile(optimizer="rmsprop",
	              loss="binary_crossentropy",
	              metrics=["accuracy"])
	model.summary()

	callbacks = [
	    keras.callbacks.ModelCheckpoint("embeddings_bidir_lstm.keras",
	                                    save_best_only=True)
	]
	model.fit(int_train_ds, validation_data=int_val_ds, epochs=10, callbacks=callbacks)
	model = keras.models.load_model("embeddings_bidir_lstm.keras")
	print(f"테스트 정확도: {model.evaluate(int_test_ds)[1]:.3f}")

	'''

		- 원-핫 모델보다 훨씬 빠르고 테스트 정확도는 비슷함(86.6%)
			- LSTM이 256차원 벡터만 처리하기 때문에

		- 여전히 기본적인 바이그램 모델의 결과보다 차이가 남
			- 이 모델이 약간 적은 데이터를 사용하기 때문(600개의 단어 이후 시퀀스는 잘라 버림)

---

패딩과 마스킹 이해하기

입력 시퀀스가 0으로 가득 차 있으면?

	모델의 성능에 나쁜 영향을 미침

	TextVectorization 층에 output_sequence_length=max_length(600)옵션 사용했기 때문

		-> 600개의 토큰보다 긴 문장은 600개의 토큰 길이로 잘림 / 600개의 토큰보다 짧은 문장은 600개의 토큰이 되도록 끝에 0을 채움

두 RNN 층이 병렬로 실행되는 양방향 RNN을 사용

	원래 순서대로 토큰을 바라보는 RNN 층은 마지막에 패딩이 인코딩된 벡터만 처리하게 됨

		-> 원래 문장이 짧다면 수백 번 이를 반복할 수 있음

			-> 내부 상태에 저장된 정보는 이 의미 없는 입력을 처리하면서 점차 사라지게 될 것

	해결 방법

		-> 마스킹(masking)

마스킹(masking)?

	- 1과 0으로 이루어진 (batch_size, sequence_length) 크기의 텐서(또는 True/False 불리언)
	- mask[i,t] 원소는 샘플 i의 타임스텝 t를 건너뛰어야 할지 말아야 할지를 나타냄(mask[i,t]가 0 또는 False이면 이 타임스텝을 건너뛰고 그렇지 않으면 처리)
	- 활성화 -> Embedding 층에 mask_zero=True를 지정
	- compute_mask() 메서드로 어떤 입력에 대한 마스킹을 추출할 수 있음

![Alt text](./i.png)

	- 실전에서는 수동으로 마스킹을 관리할 필요가 거의 없음
	- 케라스가 마스킹을 처리할 수 있는 모든 층에 (시퀀스에 부착된 메타데이터(metadata)의 일부로) 자동으로 전달
	- 이 마스킹을 사용하여 RNN 층은 마스킹된 스텝을 건너뜀
	- 모델이 전체 시퀀스를 반환한다면 손실 함수도 마스킹을 사용하여 출력 시퀀스에서 마스킹된 스텝을 건너뛸 것

- 마스킹을 활용화한 Embedding 층 사용

---

	'''

	inputs = keras.Input(shape=(None,), dtype="int64")
	embedded = layers.Embedding(
	    input_dim=max_tokens, output_dim=256, mask_zero=True)(inputs)
	x = layers.Bidirectional(layers.LSTM(32))(embedded)
	x = layers.Dropout(0.5)(x)
	outputs = layers.Dense(1, activation="sigmoid")(x)
	model = keras.Model(inputs, outputs)
	model.compile(optimizer="rmsprop",
  	            loss="binary_crossentropy",
	            metrics=["accuracy"])
	model.summary()

	callbacks = [
	    keras.callbacks.ModelCheckpoint("embeddings_bidir_lstm_with_masking.keras",
	                                    save_best_only=True)
	]
	model.fit(int_train_ds, validation_data=int_val_ds, epochs=10, callbacks=callbacks)
	model = keras.models.load_model("embeddings_bidir_lstm_with_masking.keras")
	print(f"테스트 정확도: {model.evaluate(int_test_ds)[1]:.3f}")

	'''

	결과: 테스트 정확도 87.6%

---

사전 훈련된 단어 임베딩 사용

훈련 데이터가 부족

	작업에 맞는 단어 임베딩을 학습할 수 없음

		-> 미리 계산된 임베딩 공간의 임베딩 벡터를 로드

사전 훈련된 단어 임베딩을 사용하는 이유?

	충분한 데이터가 없어 자신만의 좋은 특성을 학습하지 못하지만 꽤 일반적인 특성이 필요할 때

		-> 학습한 특성을 재사용하는 것이 합리적

단어 임베딩의 계산

	일반적으로 (문장이나 문서에서 함께 등장하는 관어를 관찰하는) 단어 출현 통계를 사용해 계산

케라스의 Embedding 층을 위해 내려받을 수 있는 미리 계산된 단어 임베딩 데이터베이스

	- Word2vec
		- 성별처럼 구체적인 의미가 있는 속성을 잡아 냄
	- GloVe(Global Vectors for World Representation)
		- 단어의 동시 출현(co-occurrence) 통계를 기록한 행렬을 분해하는 기법을 사용
		- 위키피디아(Wickipidia) 데이터와 커먼 크롤(Commom Crawl) 데이터에서 가져온 수백만 개의 영어 토큰에 대해 임베딩을 미리 계산해 놓았음

- GloVe 파일 다운

---

	'''

	!wget http://nlp.stanford.edu/data/glove.6B.zip
	!unzip -q glove.6B.zip

	'''

- GloVe 단어 임베딩 파일 파싱하기

---

	'''

	import numpy as np
	path_to_glove_file = "glove.6B.100d.txt"

	embeddings_index = {}
	with open(path_to_glove_file) as f:
	    for line in f:
	        word, coefs = line.split(maxsplit=1)
	        coefs = np.fromstring(coefs, "f", sep=" ")
	        embeddings_index[word] = coefs

	print(f"단어 벡터 개수: {len(embeddings_index)}")

	'''

		-> 압축 해제한 파일(.txt 파일)을 파싱하여 단어(즉, 문자열)와 이에 상응하는 벡터 표현을 매핑하는 인덱스를 만듦

이후 스텝

	- GloVe 단어 임베딩 행렬 생성
	- Constant 초기화를 사용해 Embedding 층에 사전 훈련된 임베딩 로드
		- trainable=False -> 층을 동결해줘야 함
	- 모델 구축및 훈련

		지금까지 진행했던 예제와 스텝이 비슷하므로 생략

---

## 트랜스포머 아키텍쳐

---

트랜스포머?

	순환 층이나 합성곱 층을 사용하지 않고 '뉴럴 어텐션'이라는 간단한 메커니즘을 사용해 만든 강력한 시퀀스 모델

---

### 셀프 어텐션 이해하기

--- 

쉽게 설명하기

	모델이 어떤 특성에 '조금 더 주의'를 기울이고, 다른 특성에 '조금 덜 주의'를 기울이는것

어텐션 순서

	1. 일련의 특성에 대한 중요도 점수 계산
	2. 이를 사용해 새로운 입력 표현 형성

![Alt text](https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FbxrBXH%2FbtspFkmNLQP%2FIHAfbaUxcXX1FtcRRC3o3K%2Fimg.png)

	-> 어떤 특성을 강조하거나 삭제하는 것 이상을 위해 사용 가능

특징

	문맥 인식 특성을 만들 수 있음

이전 방식의 문제점

	단어의 의미가 일반적으로 문맥에 띠리 달라짐

		-> 스마트한 임베딩 공간이라면 주변 단어에 따라 단어의 벡터 표현이 달라져야 함

해결 방법

	셀프 어텐션 사용!

셀프 어텐션?

	시퀀스에 있는 관련된 토큰의 표현을 사용해 한 토큰의 표헌을 조절하는 것

		-> 문맥을 고려한 토큰 표현을 만듦

셀프 어텐션 작동 방식

![Alt text](https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FbC0MEv%2FbtspH1AhHje%2FhlwTpSfJopnSAmtuhcgopK%2Fimg.png)

	1. "station" 벡터와 문장에 있는 다른 모든 단어 사이의 관련성 점수를 계산 -> 어텐션 점수
		- 두 단어 벡터 사이의 점곱을 사용해 관계의 갈도 측정
	2. 관련성 점수로 가중치를 두어 문장에 있는 모든 단어 벡터의 합을 계산
		- "station"과 밀접하게 관련된 단어는 덧셈에 더 기여 / 관련이 없는 단어는 거의 기여 X
		- 만들어진 벡터 -> "station"의 새로운 표현 / 주변 문맥을 통합하는 표현
		- 특히 "train" 벡터 일부가 포함 되기에 사실상 "train station"이라는 것이 명확해짐

사용방법

	MultiHeadAttention층을 제공

	'''

	num_heads = 4
	embed_dim = 256
	mha_layer = MultiHeadAttention(num_heads = num_heads, key_dim = embed_dim)
	outputs = mha_layer(inputs, inputs, inputs)

	'''

---

일반화된 셀프 어텐션: 쿼리-키-값 모델

트랜스포머 아키텍쳐

	원래 기계 번역을 위해 개발됨

		-> 2개의 입력 시퀀스를 다루어야 함

			-> 시퀀스-투-시퀀스 모델
트랜스포머 입력 스퀀스

	- 소스 시퀀스(ex) "How's the weather today?")
	- 타킷 시퀀스(ex) “¿Qué tiempo hace hoy?”)

셀프 어텐션 메커니즘

![Alt text](https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FsSSV9%2FbtspCm6jQHZ%2FrnJS7O36xCUI4CTb1kyxHK%2Fimg.png)

	- inputs(A)에 있는 모든 토큰이 inputs(B)에 있는 모든 토큰에 얼마나 관련되어 있는지 계산
	- 이 점수를 사용하여 inputs(C)에 있는 모든 토큰의 가중치 합을 계산

		-> A, B, C가 동일한 입력 시퀀스여야 한다는 조건은 없음

셀프 어텐션 입력값들

	쿼리(query), 키(key), 값(value)

![Alt text](https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2Fcm77sg%2FbtspNQZhAXS%2FkRbsQAOGQJwUugHhckxYq1%2Fimg.png)

	- 쿼리에 있는 모든 원소가 키에 있는 모든 원소에 얼마나 관련되어 있는지 계산
	- 이 점수를 사용하여 값에 있는 모든 원소의 가중치 합을 계산

Query와 Key의 관련도 계산

ex) 사진 DB에서 사진 검색 

	Query -> "dogs on the beach"

	내부적으로 DB에 있는 사진마다 "cat", "dog", "party" 등과 같은 일련의 키워드가 연결되어 있음

		-> Key

	-------------------------------------------------------------------------------------------

	과정
	
	1. Query와 Key 비교
		- Query에 대한 "dog"의 매칭 점수: 1
		- Query에 대한 "cat"의 매칭 점수: 0
	2. 매칭(관련성) 강도에 따라 키의 순서를 매김
	3. 가장 잘 매칭된 상위 N개의 사진을 관련성이 높은 순서대로 반환

![Alt text](https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FdNP9UN%2FbtspFMXKWoZ%2F8jYYJDomFr9usN9WUPH2nk%2Fimg.png)

트랜스포머 스타일의 어텐션이 하는 일

	- 찾고 있는 것을 설명하는 참조 시퀀스 -> Query
	- 정보를 추출할 지식의 본체 -> Value
	- Query와 쉽게 비교할 수 있는 포맷으로 Value을 설명-> Key
		- 각 Value마다 Key가 할당되어 있음

	∴ Query와 Key를 매칭하기만 하면 됨

	반환: 값의 가중치 합

특징

	- 실제로 Key와 Value는 같은 시퀀스인 경우가 많음
		- 타깃의 각 원소(ex) "tiempo")에 대해 소스(ex) "How's the weather today?")에서 이와 관련된 원소를 찾음("tiempo"와 "weather"의 매칭 점수가 높아야 함)
			-시퀀스 분류라면 자연스럽게 쿼리, 키, 값이 모두 같음
	- 시퀀스가 자기 자신과 비교하여 각 토큰에 전체 시퀀스의 풍부한 맥락을 부여함
		- MultiHeadAttention 층에 inputs를 세 번 전달해야 하는 이유

---

### 멀티 헤드 어텐션

---

멀티 헤드 어텐션?

	셀프 어텐션 메커니즘의 변형

멀티 헤드??

	셀프 어텐션의 출력 공간이 독립적으로 학습되는 부분 공간으로 나뉘어진다는 사실을 의미

헤드???

	- Query, Key, Value가 독립적인 3개의 밀집 투영을 통과해 3개의 별개의 벡터가 됨
	- 각 벡터는 뉴럴 어텐션으로 처리, 출력이 하나의 시퀀스로 연결 됨

	∴ 이런 각 부분 공간을 의미

![Alt text](https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2Fcih3cC%2FbtspFMi7tXQ%2FtS8CVvMyuVRW2hQx2kssWk%2Fimg.png)

특징

	- 학습 가능한 밀집 투영 덕분에 이 층이 실제로 무언가 학습 가능
	- 독립적인 헤드가 존재하면 층이 토큰마다 다양한 특성 그룹을 학습 하는데 도움이 됨

Depthwise convolution과 비교

	작동 방식이 비슷함

---

### 트랜스포머 인코더

---

트랜스포머 아키텍쳐의 구성

	- 트랜스포머 인코더 -> 소스 시퀀스를 처리
	- 트랜스포머 디코더 -> 소스 시퀀스를 사용해 변환된 버전을 생성

트랜스포머 인코더의 구성

	멀티 헤드 어텐션층, 밀집 투영, 정규화, 잔차 연결을 연결해 구성

![Alt text](https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FbkWtFn%2FbtspK2y9Yub%2FomMRxreypMTNzNhve2PKlK%2Fimg.png)



트랜스포머 인코더의 사용?

	텍스트 분류에 사용할 수 있음

		-> 시퀀스를 주입하고 더 유용한 표현으로 바꾸는 것을 학습하는 일반적인 모듈

---

트랜스포머 인코더를 구현해 영화 리뷰 감성 분류 작업 

- Layer 층을 상속하여 구현한 트랜스포머 인코더

---

	'''

	import tensorflow as tf
	from tensorflow import keras
	from tensorflow.keras import layers

	class TransformerEncoder(layers.Layer):
	    def __init__(self, embed_dim, dense_dim, num_heads, **kwargs):
	        super().__init__(**kwargs)
	        self.embed_dim = embed_dim # 입력 토큰 벡터의 크기
	        self.dense_dim = dense_dim # 내부 밀집 층의 크기
	        self.num_heads = num_heads # 어텐션 헤드의 개수
	        self.attention = layers.MultiHeadAttention(
	            num_heads=num_heads, key_dim=embed_dim)
	        self.dense_proj = keras.Sequential(
	            [layers.Dense(dense_dim, activation="relu"),
	             layers.Dense(embed_dim),]
	        )
	        self.layernorm_1 = layers.LayerNormalization()
	        self.layernorm_2 = layers.LayerNormalization()

	    def call(self, inputs, mask=None): # call() 메서드에서 연산을 수행한다.
	        # Enbedding 층에서 생성하는 마스크는 2D이지만 어텐션 층은 3D 또는 4D를 기대하므로 랭크를 늘린다.
	        if mask is not None:
	            mask = mask[:, tf.newaxis, :]

        attention_output = self.attention(
	            inputs, inputs, attention_mask=mask)
	        proj_input = self.layernorm_1(inputs + attention_output)
	        proj_output = self.dense_proj(proj_input)
	        return self.layernorm_2(proj_input + proj_output)

	    def get_config(self): # 모델을 저장할 수 있도록 직렬화를 구현한다.
	        config = super().get_config()
	        config.update({
	            "embed_dim": self.embed_dim,
	            "num_heads": self.num_heads,
	            "dense_dim": self.dense_dim,
	        })
	        return config

	'''

---

tip

사용자 정의 층 저장하기

	- 사용자 정의 층을 만들 때 get_config 메서드를 구현해야 함
	- 이를 통해 설정 딕셔너리(config)로부터 층의 객체를 다시 만들 수 있어 모델 저장과 로딩에 사용
	- 이 메서드는 층을 만들 때 사용한 생성자 매개변수 값을 담은 파이썬 딕셔너리를 반환
	- 모든 케라스 층은 다음과 같이 직렬화 또는 역직렬화할 수 있음

	'''

	# config 딕셔너리는 가중치 값을 답고 있지 않기 때문에 층의 모든 가중치는 처음 상태로 초기화된다.
	config = layer.get_config()
	new_layer = layer.__class__.from_config(config)

	'''

ex)

	'''

	layer = PositionalEmbedding(sequence_length, input_dim, output_dim)
	config = layer.get_config()
	new_layer = PositionalEmbedding.from_config(config)

	'''

저장 및 로드

	- 사용자 정의 층이 포함되어 있는 모델을 저장할 때 저장 파일에 이런 설정 딕셔너리가 포함
	- 파일에서 모델을 로드할 때 설정 딕셔너리를 이해할 수 있도록 다음과 같이 사용자 정의 클래스를 알려 주어야 함

	'''

	model = keras.models.load_model(
		filename, custom_objects = {"PositionalEmbedding": PositionalEmbedding})

	'''

---

사용할 정규화 층

	BatchNormalization 층 X ->  시퀀스 데이터에는 잘 맞지 않기 때문

	->  LayerNormalization 층을 사용(배치에 있는 각 시퀀스를 독립적으로 정규화)

코드로 정의

	'''

	def layer_normalization(batch_normalization): # 입력 크기: (batch_size, sequence_length, embedding_dim)
		# 마지막 축(축 인덱스-1)을 따라 평균과 분산을 계산한다.
		mean = np.mean(batch_of_sequences, keepdims=True, axis=1)
	    variance = np.var(batch_of_sequences, keepdims=True, axis=-1)
	    return (batch_of_sequences - mean) / variance

	'''

		-> 특성의 분산에 대한 정확한 통계 값을 구하기 위해 각 시퀀스 안에서 데이터를 개별적으로 구하기 때문에 시퀀스 데이터에 더 적절

- 트랜스포머 인코더를 사용하여 텍스트 분류하기

---

	'''

	vocab_size = 20000
	embed_dim = 256
	num_heads = 2
	dense_dim = 32

	inputs = keras.Input(shape=(None,), dtype="int64")
	x = layers.Embedding(vocab_size, embed_dim)(inputs)
	x = TransformerEncoder(embed_dim, dense_dim, num_heads)(x)
	x = layers.GlobalMaxPooling1D()(x) # TransformerEncoder는 전체 시퀀스를 반환하기 때문에 분류 작업을 위해 전역 풀링 층으로 각 시퀀스를 하나의 벡터로 만든다.
	x = layers.Dropout(0.5)(x)
	outputs = layers.Dense(1, activation="sigmoid")(x)
	model = keras.Model(inputs, outputs)
	model.compile(optimizer="rmsprop",
	              loss="binary_crossentropy",
	              metrics=["accuracy"])
	callbacks = [
	    keras.callbacks.ModelCheckpoint("transformer_encoder.keras",
	                                    save_best_only=True)
	]
	model.fit(int_train_ds, validation_data=int_val_ds, epochs=20, callbacks=callbacks)
	model = keras.models.load_model(
	    "transformer_encoder.keras",
	    custom_objects={"TransformerEncoder": TransformerEncoder})
	print(f"테스트 정확도: {model.evaluate(int_test_ds)[1]:.3f}")

	'''

	결과: 87.5% 테스트 정확도

위에서 재현한 모델의 문제점

	- 트랜스포머는 원래 기계 번역을 위해 개발된 시퀀스 처리 아키텍처
		- 트랜스포머 인코더는 시퀀스 모델이 전혀 아님
		- 시퀀스 토큰을 독립적으로 처리하는 밀집 층과 토큰을 집합으로 바라보는 어텐션 층으로 구성
		- 시퀀스에 있는 토큰 순서를 바꾸어도 정확히 동일한 어텐션 점수와 (문맥이 고려된) 표현을 얻을 것
	- 셀프 어텐션은 시퀀스 원소 쌍 사이의 관계에 초점을 맞춘 집합 처리 메커니즘
		- 원소가 시퀀스 처음에 등장하는지, 마지막에 등장하는지, 중간에 등장하는지 알지 못함

![Alt text](https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2Fqk1BR%2FbtspOy5bWcl%2FFOMbQSbn1kX8UkRK368pV1%2Fimg.png)

해결책

	기술적으로 순서에 구애받지 않지만 모델이 처리하는 표현에 순서 정보를 수동으로 주입하는 하이브리드 방식을 사용!!

		-> 위치 인코딩(positional encoding)

---

위치 인코딩을 사용해 위치 정보 주입

위치 인코딩의 가장 간단한 방법

	단어 위치를 임베딩 벡터에 연결하는 것!

		-> 위치가 매우 큰 정수가 될 수 있어 임베딩 벡터 값의 범위를 넘어섬

위치 인코딩 방법!

	1. 단어 인덱스의 임베딩을 학습하는 것처럼 위치 임베딩 벡터를 학습
	2. 위치 임베딩을 이에 해당하는 단어 임베딩에 추가해 위치를 고려한 단어 임베딩 생성

		-> '위치 임베딩'이라고 부름

---
- 서브클래싱으로 위치 임베딩 구현하기
---

	'''

	class PositionalEmbedding(layers.Layer):
	    # 위치 임베딩의 단점은 시퀀스 길이를 미리 알아야 한다는 것이다.
	    def __init__(self, sequence_length, input_dim, output_dim, **kwargs):
	        super().__init__(**kwargs)
	        # 토큰 인덱스를 위한 Embedding 층을 준비한다.
	        self.token_embeddings = layers.Embedding(
	            input_dim=input_dim, output_dim=output_dim)
	        self.position_embeddings = layers.Embedding(
	            # 토큰 위치를 위한 Embedding 층을 준비한다.
	            input_dim=sequence_length, output_dim=output_dim)
	        self.sequence_length = sequence_length
	        self.input_dim = input_dim
	        self.output_dim = output_dim

	    def call(self, inputs):
	        length = tf.shape(inputs)[-1]
	        positions = tf.range(start=0, limit=length, delta=1)
	        embedded_tokens = self.token_embeddings(inputs)
	        embedded_positions = self.position_embeddings(positions)

	        # 두 임베딩 벡터를 더한다.
	        return embedded_tokens + embedded_positions

	    # Embedding 층처럼 이 층은 입력에 있는 0 패딩을 무시할 수 있도록 마스킹을 생성해야 한다.
	    # compute_mask 메서드는 프레임워크에 의해 자동으로 호출되고 만들어진 마스킹은 다음 층으로 전달된다.
	    def compute_mask(self, inputs, mask=None):
	        return tf.math.not_equal(inputs, 0)

	    # 모델 저장을 위한 직렬화를 구한다.
	    def get_config(self):
	        config = super().get_config()
	        config.update({
	            "output_dim": self.output_dim,
	            "sequence_length": self.sequence_length,
	            "input_dim": self.input_dim,
	        })
	        return config

	'''

		-> 일반적인 Embedding 층처럼 PositionEmbedding 층을 사용할 수 있음

---

텍스트 분류 트렌스포머

---
트랜스포머 인코더와 위치 임베딩 합치기
---

	'''

	vocab_size = 20000
	sequence_length = 600
	embed_dim = 256
	num_heads = 2
	dense_dim = 32

	inputs = keras.Input(shape=(None,), dtype="int64")
	x = PositionalEmbedding(sequence_length, vocab_size, embed_dim)(inputs) # 여기가 위치 임베딩
	x = TransformerEncoder(embed_dim, dense_dim, num_heads)(x)
	x = layers.GlobalMaxPooling1D()(x)
	x = layers.Dropout(0.5)(x)
	outputs = layers.Dense(1, activation="sigmoid")(x)
	model = keras.Model(inputs, outputs)
	model.compile(optimizer="rmsprop",
	              loss="binary_crossentropy",
	              metrics=["accuracy"])
	model.summary()

	callbacks = [
	    keras.callbacks.ModelCheckpoint("full_transformer_encoder.keras",
	                                    save_best_only=True)
	]
	model.fit(int_train_ds, validation_data=int_val_ds, epochs=20, callbacks=callbacks)
	model = keras.models.load_model(
	    "full_transformer_encoder.keras",
	    custom_objects={"TransformerEncoder": TransformerEncoder,
	                    "PositionalEmbedding": PositionalEmbedding})
	print(f"테스트 정확도: {model.evaluate(int_test_ds)[1]:.3f}")

	'''

	결과: 테스트 정확도 88.3% -> 텍스트 분류에 단어 순서가 중요하다는 것을 보여 줌

---

### BoW 모델 대신 언제 시퀀스 모델을 사용하나요?

---

BoW 방법 VS 트랜스포머 기반 시퀀스 모델

	- 바이그램 위에 몇 개의 Dense 층을 쌓은 모델이 많은 경우에 완전히 유효하고 타당한 접근 방법
	- 새로운 텍스트 분류 작업을 시도할 때 훈련 데이터에 있는 샘플 개수와 샘플에 있는 평균 단어 개수 사이의 비율에 주의를 기울여야 함
		- 이 비율이 1,500보다 작으면 BoW 모델의 성능이 더 나을 것(보너스로 훨씬 빠르게 훈련되고 많이 반복할 수 있음)
		- 이 비율이 1,500보다 크면 시퀀스 모델을 사용
	- 시퀀스 모델은 훈련 데이터가 많고 비교적 샘플의 길이가 짧은 경우에 잘 동작

![Alt text](https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FuZVsd%2FbtspTKdbHuR%2F66S7RC6NcTRvsQwygXj0ck%2Fimg.png)

	-> 이 법칙은 과학적 법칙이 아니라 경험 법칙이므로 대부분의 경우에 맞을 것이라고 예상하지만 모든 경우에 적용되지는 않을 것

---

## 텍스트 분류를 넘어: 시퀀스-투-시퀀스 학습

---

시퀀스-투-시퀀스 모델?

	시퀀스(종종 문장이나 문단)를 받아 이를 다른 시퀀스로 바꿈

종류

	- 기계 번역(machine translation): 소스 언어에 있는 문단을 타깃 언어의 문단으로 바꿈
	- 텍스트 요약(text summarization): 긴 문서를 대부분 중요한 정보를 유지한 짧은 버전으로 바꿈
	- 질문 답변(question answering): 입력 질문에 대한 답변을 생성
	- 챗봇(chatbot): 입력된 대화나 또는 대화 이력에서 다음 응답을 생성
	- 텍스트 생성: 시작 텍스트를 사용하여 하나의 문단을 완성

시퀀스-투-스퀀스 모델의 일반적인 구조

![Alt text](https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FbfRezm%2FbtspMJsUKEX%2F6ws2u2ztazpUPXVFTNKc20%2Fimg.png)

훈련 단계 순서
	- 인코더(encoder) 모델이 소스 시퀀스를 중간 표현으로 바꿈
	- 디코더(decoder)는 (0에서 i-1까지의) 이전 토큰과 인코딩된 소스 시퀀스를 보고 타깃 시퀀스에 있는 다음 토큰 i를 예측하도록 훈련

추론 단계 순서

	1. 인코더가 소스 시퀀스를 인코딩
	2. 디코더가 인코딩된 소스 시퀀스와 ("[start]" 문자열과 같은) 초기 "시드(seed)" 토큰을 사용하여 시퀀스의 첫 번째 토큰을 예측
	3. 지금까지 예측된 시퀀스를 디코더에 다시 주입하고 다음 토큰을 생성하는 식으로 ("[end]" 문자열과 같은) 종료 토큰이 생성될 때까지 반복

		- 추론에서는 타깃 시퀀스를 사용하지 못하므로 처음부터 예측해야 함
		- 한 번에 하나의 토큰을 생성

---

### 기계 번역 예제

---

시퀀스-투-시퀀스(트랜스포머)을 기계 번역 작업에 적용

---
1. 영어-스페인어 번역 데이터셋 다운 후 전처리
---

	1. [https://www.manythings.org/anki/](https://www.manythings.org/anki/)에 있는 영어-스페인어 번역 데이터셋을 사용

	'''

	!wget http://storage.googleapis.com/download.tensorflow.org/data/spa-eng.zip
	!unzip -q spa-eng.zip

	'''

	2. 파싱

	'''

	text_file = "spa-eng/spa.txt"
	with open(text_file) as f:
	    lines = f.read().split("\n")[:-1]
	text_pairs = []
	for line in lines: # 한 라인씩 처리한다.
	    english, spanish = line.split("\t") # 각 라인은 구절과 이에 해당하는 스페인 번역을 포함하여 텝으로 구성되어 있다.
	    spanish = "[start] " + spanish + " [end]" # 형식에 맞추기 위해 스페인 문장 처음에 [start]를 추가하고 끝에 [end]를 추가한다.
	    text_pairs.append((english, spanish))

	'''

![Alt text](./j.png)

	3. 셔플후 훈련, 검증, 테스트 세트로 나눔

	'''

	import random
	random.shuffle(text_pairs)
	num_val_samples = int(0.15 * len(text_pairs))
	num_train_samples = len(text_pairs) - 2 * num_val_samples
	train_pairs = text_pairs[:num_train_samples]
	val_pairs = text_pairs[num_train_samples:num_train_samples + num_val_samples]
	test_pairs = text_pairs[num_train_samples + num_val_samples:]

	'''

	4. (영어, 스페인어)2개의 TextVectorization 층 준비


	문자열을 전처리하는 방식을 커스터마이징해야 함

		- 앞에서 추가한 "[start]"와 "[end]" 토큰을 유지해야 함. 기본적으로 [와 ] 문자가 삭제되지만 여기에서는 "start" 단어와 "[start]"를 별개로 취급하기 위해 두 문자를 유지
		- 구두점은 언어마다다름! 스페인어 TextVectorization 층에서 구두점 문자를 삭제하려면 ¿ 문자도 삭제

	- 영어와 스페인어 텍스트 쌍을 벡터화하기

	'''

	import tensorflow as tf
	from tensorflow.keras import layers
	import string
	import re

	# 스페인어 TextVectorization 위해 사용자 정의 문자열 표준화 함수를 정의한다.
	# [와 ]문자는 유지하고 (strings.punctuation에 있는 문자를 포함하여) ¿ 문자를 삭제한다
	strip_chars = string.punctuation + "¿"
	strip_chars = strip_chars.replace("[", "")
	strip_chars = strip_chars.replace("]", "")

	def custom_standardization(input_string):
	    lowercase = tf.strings.lower(input_string)
	    return tf.strings.regex_replace(
	        lowercase, f"[{re.escape(strip_chars)}]", "")

	# 간단한 예를 위해 각 언어에서 가장 많이 등장하는 1만 5,000개의 단어만 사용하고 문장의 길이는 20개의 단어로 제한한다.
	vocab_size = 15000
	sequence_length = 20

	source_vectorization = layers.TextVectorization( # 영어 층
	    max_tokens=vocab_size,
	    output_mode="int",
	    output_sequence_length=sequence_length,
	)
	target_vectorization = layers.TextVectorization( # 스페인어 층
	    max_tokens=vocab_size,
	    output_mode="int",
                                                 
	    # 훈련하는 동안 한 스탭 앞서 문장이 필요하기 때문에 토큰 하나가 추가된 스페인어 문장을 생성한다.                                                 
	    output_sequence_length=sequence_length + 1,
	    standardize=custom_standardization,
	)
	train_english_texts = [pair[0] for pair in train_pairs]
	train_spanish_texts = [pair[1] for pair in train_pairs]

	# 각 언어의 어휘 사전을 만든다.
	source_vectorization.adapt(train_english_texts)
	target_vectorization.adapt(train_spanish_texts)

	'''

		- 마지막으로 데이터를 tf.data 파이프라인으로 변환할 수 있음
		- 이 데이터셋은 (inputs, target)의 튜플을 반환
		- inputs는 "encoder_inputs(영어 문장)"와 "decoder_inputs(스페인어 문장)"키 2개를 가진 딕셔너리
		- targets은 한 스텝 안의 스페인어 문장

	5. 번역 작업을 위한 데이터셋 준비하기

	'''

	batch_size = 64

	def format_dataset(eng, spa):
	    eng = source_vectorization(eng)
	    spa = target_vectorization(spa)
	    return ({
	        "english": eng,
	        "spanish": spa[:, :-1], # 입력 스페인어 문장은 마지막 토큰을 포함하지 않기 때문에 입력과 타깃 길이가 같다.
	    }, spa[:, 1:]) # 타깃 스페인어 문장은 한 스텝 앞의 문장이다. 길이는 입력과 같다(20개의 단어).

	def make_dataset(pairs):
	    eng_texts, spa_texts = zip(*pairs)
	    eng_texts = list(eng_texts)
	    spa_texts = list(spa_texts)
	    dataset = tf.data.Dataset.from_tensor_slices((eng_texts, spa_texts))
	    dataset = dataset.batch(batch_size)
	    dataset = dataset.map(format_dataset, num_parallel_calls=4)
	    return dataset.shuffle(2048).prefetch(16).cache() # 전처리 속도를 높이기 위해 메모리에 캐싱한다.

	train_ds = make_dataset(train_pairs)
	val_ds = make_dataset(val_pairs)

	'''

![Alt text](./k.png)

	->데이터셋의 크기를 확인

---

### RNN을 사용한 시퀀스-투-시퀀스 모델

---

가장 간단하고 쉬운 방법?

	각 타임스텝의 RNN출력을 그대로 유지하는 것

	'''

	inputs = keras.Input(shape=(sequence_length,), dtype='int64')
	x = layers.Embedding(input_dim=vocab_size, output_dim=128)(inputs)
	x = layers.LSTM(32, return_sequences=True)(x)
	outputs = layers.Dense(vocab_size, activation="softmax")(x)
	model = keras.Model(inputs, outputs)

	'''

위 방식은 두 가지 이슈 존재

	- 타깃 시퀀스가 항상 소스 시퀀스와 동일한 길이여야 함
		- 실제로는 이런 경우가 드묾
		- 기술적으로 치명적인 문제는 아님
		- 소스 시퀀스나 타깃 시퀀스에 패딩을 추가하여 길이를 맞출 수 있기 때문임
	- RNN의 스텝별 처리 특징 처리 때문에 모델이 타깃 시퀀스에 있는 토큰 N을 예측하기 위해 소스 시퀀스에 있는 토큰 0...N만 참조할 것
		- 이런 제약 때문에 이 방식이 대부분의 작업 특히 번역에 적합하지 않음
		- "The weather is nice today"를 프랑스어인 "Il fait beau aujourd’hui"로 번역한다고 가정해 보자
		- "The"에서 "Il"를 예측하고, "The weather"에서 "Il fait"를 예측해야 함. 이는 불가능

사람이라면?

	번역을 시작하기 전에 소스 문장 전체를 먼저 읽음

		-> 표준 시퀀스-투-시퀀스 모델이 하는 일

표준 시퀀스-투-시퀀스 모델일때?

	1. 적절한 시퀀스-투-시퀀스 구조에서는 먼저 RNN(인코더)을 사용하여 전체 소스 문장을 하나의 벡터로 (또는 벡터의 집합으로) 바꿈
		- 이 벡터는 RNN의 마지막 출력이거나 또는 마지막 상태 벡터일 수 있음
	2. 그다음 이 벡터(또는 벡터 집합)를 다른 RNN(디코더)의 초기 상태로 사용
	3. 이 RNN은 타깃 시퀀스에 있는 원소 0...N을 사용하여 스텝 N+1을 예측

![Alt text](https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FbFMQpt%2FbtspFjWb4Pf%2FHKdRo7IRkZh9ttGsWpbqCk%2Fimg.png)

---
2.  GRU 기반의 인코더와 디코더 구현
---

	1. GRU 기반 인코더

	'''

	from tensorflow import keras
	from tensorflow.keras import layers

	embed_dim = 256
	latent_dim = 1024

	# 영어 소스 문장이 여기에 입력된다.
	# 입력 이름을 지정하면 입력 딕셔너리로 모델을 훈련할 수 있다.
	source = keras.Input(shape=(None,), dtype="int64", name="english")

	# 마스킹을 잊지 말하야 한다. 이 방식에서 중요하다.
	x = layers.Embedding(vocab_size, embed_dim, mask_zero=True)(source)
	encoded_source = layers.Bidirectional(
	    # 인코딩된 소스 문장은 양방향 GRU의 마지막 출력이다.
	    layers.GRU(latent_dim), merge_mode="sum")(x)

	'''

	2. GRU 기반 디코더와 엔드-투-엔트 모델

	'''

	# 스페인어 타깃 시퀀스가 여기에 입력된다.
	past_target = keras.Input(shape=(None,), dtype="int64", name="spanish")

	# 마스킹을 잊지 말자
	x = layers.Embedding(vocab_size, embed_dim, mask_zero=True)(past_target)

	decoder_gru = layers.GRU(latent_dim, return_sequences=True)
	# 인코딩된 소스 시퀀스는 디코더 GRU의 초기 상태가 된다.
	x = decoder_gru(x, initial_state=encoded_source)

	x = layers.Dropout(0.5)(x)
	# 다음 토큰을 예측한다.
	target_next_step = layers.Dense(vocab_size, activation="softmax")(x)

	# 엔드-투-엔드 모델은 소스 시퀀스와 타깃 시퀀스를 한 스텝 앞의 타깃 시퀀스에 매핑한다.
	seq2seq_rnn = keras.Model([source, past_target], target_next_step)

	'''

		- 훈련하는 동안 디코더는 전체 타깃 시퀀스를 입력받음
		- RNN의 스텝별 처리 특징 덕분에 입력에 있는 토큰 0...N만 사용하여 타깃에 있는 토큰 N을 예측
			- 과거 정보만 사용해서 미래를 예측한다는 의미

	3. RNN기반 시퀀스-투-시퀀스 모델 훈련

	'''

	seq2seq_rnn.compile(
	    optimizer="rmsprop",
	    loss="sparse_categorical_crossentropy",
	    metrics=["accuracy"])
	seq2seq_rnn.fit(train_ds, epochs=15, validation_data=val_ds)

	'''

	결과: 64% 정확도

		- 다음 토큰의 정확도는 기계 번역 모델에서 좋은 척도가 아님
			- 토큰 N+1을 예측할 때 0에서 N까지 정확한 타깃 토큰을 알고 있다고 가정해야 하기 때문임
		- "BLEU 점수"를 사용해서 모델을 평가할 가능성이 높음

	4. RNN 인코더와 디코더로 새로운 문장 번역하기

		1. 시드 토큰 "[start]"와 인코딩된 영어 소스 문장을 디코더 모델에 주입
		2. 그다음 다음 토큰을 예측하고 이를 디코더에 반복적으로 다시 주입
		3. 이런 식으로 "[end]" 토큰이나 최대 문장 길이에 도달할 때까지 반복마다 새로운 타깃 토큰을 생성

	'''

	import numpy as np

	# 예측된 인덱스를 문자열 토큰으로 변환하기 위한 딕셔너리를 준비한다.
	spa_vocab = target_vectorization.get_vocabulary()
	spa_index_lookup = dict(zip(range(len(spa_vocab)), spa_vocab))
	max_decoded_sentence_length = 20

	def decode_sequence(input_sentence):
	    tokenized_input_sentence = source_vectorization([input_sentence])
	    decoded_sentence = "[start]" # 시드 토큰
	    for i in range(max_decoded_sentence_length):
	        tokenized_target_sentence = target_vectorization([decoded_sentence])

	        # 다음 토큰을 샘플링한다.
	        next_token_predictions = seq2seq_rnn.predict(
	            [tokenized_input_sentence, tokenized_target_sentence])
	        sampled_token_index = np.argmax(next_token_predictions[0, i, :])

	        # 다음 토큰 예측을 문자열로 바꾸고 생성된 문장에 추가한다.
	        sampled_token = spa_index_lookup[sampled_token_index]
	        decoded_sentence += " " + sampled_token

	        if sampled_token == "[end]": # 종료 조건: 최대 길이에 도달하거나 종료 문자가 생성된 경우
	            break
	    return decoded_sentence

	test_eng_texts = [pair[0] for pair in test_pairs]
	for _ in range(20):
	    input_sentence = random.choice(test_eng_texts)
	    print("-")
	    print(input_sentence)
	    print(decode_sequence(input_sentence))

	'''

		- 매우 간단하지만 효율적이지 않음
			- 전체 소스 시퀀스와 지금까지 생성된 전체 타깃 시퀀스를 새로운 단어를 샘플링할 때마다 모두 다시 처리해야 함
![Alt text](./l.png)

시퀀스-투-시퀀스 학습을 위한 RNN방식의 몇 가지 근본적인 제약

	- 소스 시퀀스가 인코더 상태 벡터(또는 벡터 집합)로 완전하게 표현되어야 함
		- 이는 번역할 수 있는 문장의 크기와 복잡도에 큰 제약이 됨
		- 사람이 번역할 때 소스 시퀀스를 두 번 보지 않고 완전히 기억만으로 문장을 번역하는 것과 같음
	- RNN은 오래된 과거를 점진적으로 잊어버리는 경향이 있기 때문에 매우 긴 문장을 처리하는 데 문제가 있음(그레이디언트 소실과 비슷함)
		- 어느 시퀀스에서든지 100번째 토큰에 도착하면 시작 부분에 대한 정보가 거의 남아 있지 않음
		- 이는 RNN기반 모델이 긴 문서를 번역하는 데 필수적인 넓은 범위의 문맥을 감지할 수 없다는 의미

---

### 트랜스포머를 사용한 시퀀스-투-시퀀스 모델

---

시퀀스-투-시퀀스 학습?

	트랜스포머가 진정 빛을 발하는 작업

		-> RNN이 다룰 수 있는 것보다 훨씬 길고 복잡한 시퀀스를 성공적으로 처리할 수 있음

특징

	- 소스 문장과 진행 중인 번역 사이를 왔다 갔다 할 것
		- 번역 문장의 부분 부분을 작성하면서 소스 문장에 있는 여러 단어에 주의를 기울이기 때문임
	- 트랜스포머 인코더 -> 소스 문장을 읽고 인코딩된 표현을 만듦
	- 트랜스포머 디코더(Transformer decoder)
		- 타깃 시퀀스에 있는 토큰 0...N을 읽고 토큰 N+1을 예측
		- 이를 수행하면서 뉴럴 어텐션을 사용하여 인코딩된 소스 문장에서 어떤 토큰이 현재 예측하려는 타깃 토큰에 가장 관련이 높은지 식별한다는 것
		- 쿼리-키-값 모델을 다시 떠올려 보면 트랜스포머 디코더에서 타깃 시퀀스는 소스 시퀀스에 있는 다른 부분에 더 주의를 집중하기 위해 사용하는 어텐션 "쿼리" 역할을 함(소스 시퀀스는 키와 값 역할을 함

트랜스포머 디코더

타깃 시퀀스에 적용되는 셀프 어텐션 블록과 마지막 블록의 밀집 층 사이에 추가적인 어텐션 블록이 들어가 있음

![Alt text](./m.png)

---
3.  트랜스포머 인코더와 디코더 구현
---

	1. Layer 클래스를 서브클래싱해 TransformerDecoder  구현

	'''

	class TransformerDecoder(layers.Layer):
	    def __init__(self, embed_dim, dense_dim, num_heads, **kwargs):
	        super().__init__(**kwargs)
	        self.embed_dim = embed_dim
	        self.dense_dim = dense_dim
	        self.num_heads = num_heads
	        self.attention_1 = layers.MultiHeadAttention(
	            num_heads=num_heads, key_dim=embed_dim)
	        self.attention_2 = layers.MultiHeadAttention(
	            num_heads=num_heads, key_dim=embed_dim)
	        self.dense_proj = keras.Sequential(
	            [layers.Dense(dense_dim, activation="relu"),
	             layers.Dense(embed_dim),]
	        )
	        self.layernorm_1 = layers.LayerNormalization()
	        self.layernorm_2 = layers.LayerNormalization()
	        self.layernorm_3 = layers.LayerNormalization()

	        # 이 속성은 층이 입력 마스킹을 출력으로 전달하도록 만든다.
	        # 케라스에서 마스팅을 사용하려면 명시적으로 설정을 해야 한다.
	        # compute_mask() 메서드를 구현하지 않으면서 supports_masking 속성을 제공하지 않는 층에 마스킹을 전달하면 에러가 발생한다.
	        self.supports_masking = True

	    def get_config(self):
	        config = super().get_config()
	        config.update({
	            "embed_dim": self.embed_dim,
	            "num_heads": self.num_heads,
	            "dense_dim": self.dense_dim,
	        })
	        return config

	'''

추가 고려 사항!

	코잘 패딩(casual padding)을 고려해야 함

코잘 패딩?

	시퀀스-투-시퀀스 트랜스포머를 성공적으로 훈련하는 데 매우 중요

		- TransformerDecoder는 순서에 구애받지 않고 한 번에 타깃 시퀀스 전체를 바라봄
		- 전체 입력을 사용하도록 둔다면 단순히 입력 스텝 N+1을 출력 위치 N에 복사하는 방법을 학습할 것

									↓

		- 모델이 미래에서 온 정보에 주의를 기울이지 못하도록 어텐션 행렬의 위쪽 절반을 마스킹
		- 즉, 타깃 토큰 N+1을 생성할 때 타깃 시퀀스에 있는 토큰 0...N에서 온 정보만 사용해야 함

---

	2. 코잘 마스킹을 생성하는 TransformerDecoder 메서드

	'''

	    def get_causal_attention_mask(self, inputs):
	        input_shape = tf.shape(inputs)
	        batch_size, sequence_length = input_shape[0], input_shape[1]
	        i = tf.range(sequence_length)[:, tf.newaxis]
	        j = tf.range(sequence_length)

	        # 절반은 1이고 나머지는 0인(sequence_length, sequence_length) 크기의 행렬을 만든다.
	        mask = tf.cast(i >= j, dtype="int32")

	        # 이를 배치 측에 반복하여 (batch_size, sequence_length, sequence_length) 크기의 행렬을 얻는다.
	        mask = tf.reshape(mask, (1, input_shape[1], input_shape[1]))
	        mult = tf.concat(
	            [tf.expand_dims(batch_size, -1),
	             tf.constant([1, 1], dtype=tf.int32)], axis=0)
	        return tf.tile(mask, mult)

	'''

	3. TransformerDecoder의 정방향 패스

	'''

	def call(self, inputs, encoder_outputs, mask=None):
	        # 코잘 마스킹을 추출한다.
	        causal_mask = self.get_causal_attention_mask(inputs)

	        # (타깃 시퀀스에 있는 패딩 위치를 나타내는) 입력 마스킹을 준비한다.
	        if mask is not None:
	            padding_mask = tf.cast(
	                mask[:, tf.newaxis, :], dtype="int32")
            
	            # 두 마스킹을 합친다.
	            padding_mask = tf.minimum(padding_mask, causal_mask)
        
	        # 코잘 마스킹을 타깃 시퀀스에 대해 셀프 어텐션을 수행하는 첫 번째 어텐션 층에 전달한다.
	        attention_output_1 = self.attention_1(
	            query=inputs,
	            value=inputs,
	            key=inputs,
	            attention_mask=causal_mask)
        
	        attention_output_1 = self.layernorm_1(inputs + attention_output_1)

	        # 합친 마스킹을 소스 시퀀스와 타깃 시퀀스를 연관시키는 두 번째 어텐션 층에 전달한다.
	        attention_output_2 = self.attention_2(
	            query=attention_output_1,
	            value=encoder_outputs,
	            key=encoder_outputs,
	            attention_mask=padding_mask,
	        )
        
	        attention_output_2 = self.layernorm_2(
	            attention_output_1 + attention_output_2)
	        proj_output = self.dense_proj(attention_output_2)
	        return self.layernorm_3(attention_output_2 + proj_output)

	'''

---

기계 번역을 위한 트랜스포머

---
- 엔드-투-엔드 트랜스포머
---

	'''

	embed_dim = 256
	dense_dim = 2048
	num_heads = 8

	encoder_inputs = keras.Input(shape=(None,), dtype="int64", name="english")
	x = PositionalEmbedding(sequence_length, vocab_size, embed_dim)(encoder_inputs)
	encoder_outputs = TransformerEncoder(embed_dim, dense_dim, num_heads)(x) # 소스 문장을 인코딩한다.

	decoder_inputs = keras.Input(shape=(None,), dtype="int64", name="spanish")
	x = PositionalEmbedding(sequence_length, vocab_size, embed_dim)(decoder_inputs)
	x = TransformerDecoder(embed_dim, dense_dim, num_heads)(x, encoder_outputs) # 타깃 시퀀스를 인코딩하고 인코딩된 소스 문장과 합친다.
	x = layers.Dropout(0.5)(x)

	decoder_outputs = layers.Dense(vocab_size, activation="softmax")(x) # 출력 위치마다 하나의 단어를 예측한다.
	transformer = keras.Model([encoder_inputs, decoder_inputs], decoder_outputs)

	'''

---
시퀀스-투-시퀀스 트랜스포머 훈련
---

	'''

	transformer.compile(
	    optimizer="rmsprop",
	    loss="sparse_categorical_crossentropy",
	    metrics=["accuracy"])
	transformer.fit(train_ds, epochs=30, validation_data=val_ds)

	'''

---
- 트랜스포머 모델을 사용하여 새로운 문장 번역하기
---

	'''

	import numpy as np
	spa_vocab = target_vectorization.get_vocabulary()
	spa_index_lookup = dict(zip(range(len(spa_vocab)), spa_vocab))
	max_decoded_sentence_length = 20

	def decode_sequence(input_sentence):
	    tokenized_input_sentence = source_vectorization([input_sentence])
	    decoded_sentence = "[start]"
	    for i in range(max_decoded_sentence_length):
	        tokenized_target_sentence = target_vectorization(
	            [decoded_sentence])[:, :-1]
	        predictions = transformer(
            
	            # 다음 토큰을샘플링한다.
	            [tokenized_input_sentence, tokenized_target_sentence])
	        sampled_token_index = np.argmax(predictions[0, i, :])

	        # 다음 토큰 예측을 문자열로 바꾸고 생성된 문장에 추가한다.
	        sampled_token = spa_index_lookup[sampled_token_index]
	        decoded_sentence += " " + sampled_token

	        # 종료조건
	        if sampled_token == "[end]":
	            break
	    return decoded_sentence

	test_eng_texts = [pair[0] for pair in test_pairs]
	for _ in range(20):
	    input_sentence = random.choice(test_eng_texts)
	    print("-")
	    print(input_sentence)
	    print(decode_sequence(input_sentence))

	'''

![Alt text](./n.png)
