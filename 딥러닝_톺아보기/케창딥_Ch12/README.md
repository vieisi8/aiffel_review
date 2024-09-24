# 학습 내용

---

- 텍스트 생성
- 딥드림
- 뉴럴 스타일 트랜스퍼
- 변이형 오토인코더 사용한 이미지 생성
- 생성적 적대 신경망 소개

---

인공지능 -> 우리 생활과 일에 지능을 더함

	여러 분야에서 특히 예술에서느 AI가 사람의 능력을 증가시키는 도구로 사용

		-> 인공적인 지능이 아니라 확장된 지능

예술 창작

	대부분 간단한 패턴 인식과 기교로 만들어짐

		-> 여기에 AI가 필요함

사람의 지각, 언어, 예술 작품

	모두 통계적 구조를 가짐

		-> 딥러닝 알고리즘은 이 구조를 학습하는 데 뛰어남

머신러닝 모델

	1. 이미지, 음악, 글의 통계적 잠재 공간을 학습
	2. 잠재 공간에서 샘플을 뽑아 새로운 예술 작품을 만듦

잠재 공간 샘플링

	- 예술가의 능력을 높이는 붓이 될 수 있음
	- 창작 가능성을 늘림
	- 상상의 공간을 확장시킴

---

## 텍스트 생성

---

### 시퀀스 데이터를 어떻게 생성할끼?

---

일반적인 방법

	이전 토큰을 입력으로 사용해 시퀀스의 다음 1개 또는 몇 개의 토큰을 (트랜스포머 / RNN으로) 예측하는 것

ex)	"the cat is on the"란 입력이 주어짐

	다음 타킥 "mat"을 예측하도록 모델을 훈련

언어 모델?

	이전 토큰들이 주어졌을 때 다음 토큰의 확률을 모델링할 수 있는 네트워크

		-> 언어의 통계적 구조인 잠재 공간을 탐색

언어 모델을 사용해 한 단어씩 텍스트를 생성하는 과정

	1. 초기 텍스트 문자열을 주입(조건 데이터)
	2. 새로운 글자나 단어를 생성
	3. 생성된 출력을 다시 입력 데이터로 추가

		1 ~ 3 번 과정을 여러번 반복

		-> 사람이 쓴 문장과 거의 비슷함

![Alt text](https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FbJ20lL%2FbtsqDdHuW0x%2FnOAl2jykKMoGlToL5nFVq0%2Fimg.png)

---

### 샘플링 전력의 중요성

---

텍스트 생성할 때 중요한 것

	다음 문자을 선택하는 방법

샘플링 방법

	- 탐욕적 샘플링 
		- 항상 가장 높은 확률을 가진 글자 선택하는 방법
	- 확률적 샘플링
		- 확률 분포에서 샘플링 하는 과정에 무작위성을 주입하는 방법

모델의 소프트맥스 출력

	확률적 샘플링을 사용하기 좋음

한 가지 문제점

	샘플링 과정에서 무작위성의 양을 조절할 방법 X

특징

	- 작은 엔트로피
		- 예상 가능한 구조를 가진 시퀀스 생성
	- 높은 엔트로피
		- 놀랍고 창의적인 시퀀스 생성

해결 가능한 파라미터

	소프트맥스 온도

		-> 샘플링에 사용되는 확률 분포의 엔트로피를 나타냄

![Alt text](https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FbZBq1L%2FbtsqJDk4QiD%2FuzLt3aoYmjGLwbaIiVXpU0%2Fimg.png)

---

### 케라스를 사용한 텍스트 생성 모델 구현

---


1. 데이터 준비
---

IMDB 영화 리뷰 데이터셋 사용

	'''

	!wget https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz
	!tar -xf aclImdb_v1.tar.gz

	'''

데이터셋 생성

	'''

	# TensorFlow와 Keras를 사용하여 텍스트 데이터셋을 생성하고 전처리
	import tensorflow as tf
	from tensorflow import keras

	'''
	keras.utils.text_dataset_from_dictionary: 이 함수를 사용해서 디렉토리에서 텍스트 데이터셋을 생성
	매개변수 역할
	dictionary: 데이터셋이 있는 디렉토리를 결정
	label_mode: 레이블 모드를 지정. 여기서는 None으로 설정하여 레이블 없이 데이터를 생성
	batch_size: 배치 크기를 지정
	'''
	dataset = keras.utils.text_dataset_from_directory(
	    directory="aclImdb", label_mode=None, batch_size=256)

	# 이 리뷰에 많이 등장하는 <br /> HTML태그를 제거.
	# 텍스트 분류에서 중요하지 않기 때문
	dataset = dataset.map(lambda x: tf.strings.regex_replace(x, "<br />", " "))

	'''

textvectorization을 사용해 벡터화

	'''

	# TensorFlow의 Keras 라이브러리 TextVectorization을 사용하여 텍스트 데이터를 벡터화하는 작업을 수행
	# 텍스트 데이터를 숫자로 변환하여 딥러닝 모델에 입력으로 사용할 수 있도록 전처리
	from tensorflow.keras.layers import TextVectorization

	sequence_length = 100

	# 가장 자주 등장하는 1만 5,000개 단어만 사용.
	# 그 외 단어는 모두 OOV 토큰인 "[UNK]"로 처리
	vocab_size = 15000

	text_vectorization = TextVectorization(
    
	    # 벡터화할 때 사용할 최대 단어 수를 제한하는 역할.
	    # 여기서는 가장 자주 등장하는 상위 1만 5,000개의 단어만 사용하도록 설정
	    max_tokens=vocab_size,

	    # 정수 단어 인덱스의 시퀀스를 반환하도록 설정
	    output_mode="int",

	    # 모델에 입력으로 사용할 시퀀스의 길이를 결정. 
	    # 길이가 100인 입력과 타깃을 사용(타깃을 한 스텝 차이가 나기 때문에 실제로 모델은 99개의 단어 시퀀스를 보게 된다.)
	    output_sequence_length=sequence_length,
	)

	# adapt메서드를 사용하여 데이터셋을 기반으로 벡터화 모델을 학습하고 적용
	# 이로써 모델은 데이터셋의 특성에 따라 단어를 인덱싱하고 벡터화하는데 필요한 정보를 얻게 됨
	text_vectorization.adapt(dataset)

	'''

언어 모델링 데이터셋 생성

	'''

	# 텍스트 데이터를 언어 모델을 학습하기 위한 데이터셋으로 변환하는 작업을 수행

	# prepare_lm_dataset: 입력된 텍스트 배치를 정수 시퀀스의 배치로 변환하는 작업을 수행
	# text_vectorization 모델을 사용하여 텍스트를 정수 시퀀스로 벡터화한 후, 시퀀스의 마지막 단어를 제외한 입력과 시퀀스의 첫 단어를 제외한 타깃을 생성
	def prepare_lm_dataset(text_batch):

	    # 텍스트(문자열)의 배치를 정수 시퀀스의 배치로 변환
	    vectorized_sequences = text_vectorization(text_batch)

	    # 시퀀스의 마지막 단어를 제외한 입력을 만든다.
	    x = vectorized_sequences[:, :-1]

	    # 시퀀스의 첫 단어를 제외한 타깃을 만든다.
	    y = vectorized_sequences[:, 1:]
	    return x, y

	# dataset.map 메서드를 사용하여 입렉 데이터셋에 'prepare_lm_dataset'함수를 적용하여 데이터셋을 변환한 결과
	lm_dataset = dataset.map(prepare_lm_dataset, num_parallel_calls=4)

	'''

---

트랜스포머 기반의 시퀀스-투-시퀀스 모델

일반 시퀀스 모델 사용시 발생하는 이슈 

	- N개 보다 적은 단어로 예측을 시작할 수 있어야 함
		- 그렇지 않으면 비교적 긴 시작 문장을 사용해야 하는 제약이 생김
	- 훈련에 사용하는 많은 시퀀스 
		- 중복되어 있음

2가지 이슈를 해결하기 위한 방법

	시퀀스-투-시퀀스 모델 사용

![Alt text](https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FbbNHmM%2FbtsqJg4MMZM%2FXa2kUWr6NgEGSqvKH12MO0%2Fimg.png)

텍스트 생성의 특징

	- 소스 시퀀스가 존재 X
		- 과거의 토큰이 주어지면 타킷 시퀀스에 있는 다음 토큰을 예측하는 것뿐이기 때문
	- 디코더만 사용해 수행 가능
		- 코잘 패딩 덕분에 N+1을 예측하기 위해 0 ~ N까지의 단어만 바라볼 것

2. 간단한 트랜스포머 기반 언어 모델 구성
---

PositionalEmbedding과 TransformerDecoder 정의

	'''

	import tensorflow as tf
	from tensorflow.keras import layers

	# 입력된 토큰 시퀀스에 위치 정보를 추가하는 레이어
	class PositionalEmbedding(layers.Layer):
	    def __init__(self, sequence_length, input_dim, output_dim, **kwargs):
	        super().__init__(**kwargs)

	        # 입력 토큰을 임베딩하는 레이어
	        self.token_embeddings = layers.Embedding(
	            input_dim=input_dim, output_dim=output_dim)
        
	        # 위치 정보를 임베딩하는 레이어
	        self.position_embeddings = layers.Embedding(
	            input_dim=sequence_length, output_dim=output_dim)
        
	        self.sequence_length = sequence_length
	        self.input_dim = input_dim
	        self.output_dim = output_dim

	    # 입력 시퀀스에 토큰 임베딩과 위치 임베딩을 더한 결과를 반환
	    def call(self, inputs):
	        length = tf.shape(inputs)[-1]
	        positions = tf.range(start=0, limit=length, delta=1)
	        embedded_tokens = self.token_embeddings(inputs)
	        embedded_positions = self.position_embeddings(positions)
	        return embedded_tokens + embedded_positions

	    # 입력 시퀀스에서 패딩을 마스킹하는 마스크를 생성
	    def compute_mask(self, inputs, mask=None):
	        return tf.math.not_equal(inputs, 0)

	    def get_config(self):
	        config = super(PositionalEmbedding, self).get_config()
	        config.update({
	            "output_dim": self.output_dim,
	            "sequence_length": self.sequence_length,
	            "input_dim": self.input_dim,
	        })
	        return config

	# 트랜스포머 디코더 계층을 정의하는 클래스
	class TransformerDecoder(layers.Layer):
	    def __init__(self, embed_dim, dense_dim, num_heads, **kwargs):
	        super().__init__(**kwargs)
	        self.embed_dim = embed_dim
	        self.dense_dim = dense_dim
	        self.num_heads = num_heads

	        # attention_1, attention_2는 각각 입력 시퀀스에 대한 셀프 어텐션과 인코더 출력에 대한 어텐션을 수행하는 MultiHeadAttention 레이어
	        self.attention_1 = layers.MultiHeadAttention(
	          num_heads=num_heads, key_dim=embed_dim)
	        self.attention_2 = layers.MultiHeadAttention(
	          num_heads=num_heads, key_dim=embed_dim)
        
	        # 어텐션 출력을 밀집 연산하는 Sequential 레이어
	        self.dense_proj = keras.Sequential(
	            [layers.Dense(dense_dim, activation="relu"),
	             layers.Dense(embed_dim),]
	        )

	        # layernorm_1, layernorm_2, layernorm3는 각각 레이어 정규화 레이어이다.
	        self.layernorm_1 = layers.LayerNormalization()
	        self.layernorm_2 = layers.LayerNormalization()
	        self.layernorm_3 = layers.LayerNormalization()
	        self.supports_masking = True

	    def get_config(self):
	        config = super(TransformerDecoder, self).get_config()
	        config.update({
	            "embed_dim": self.embed_dim,
	            "num_heads": self.num_heads,
	            "dense_dim": self.dense_dim,
	        })
	        return config

	    # 인과적 마스크(케이블러티 마스크)를 생성하는 함수
	    # 인과적 마스크: 현재 위치의 토큰이 미래의 위치에 영향을 미치지 않도록 하는 역할을 함
	    def get_causal_attention_mask(self, inputs):
	        input_shape = tf.shape(inputs)
	        batch_size, sequence_length = input_shape[0], input_shape[1]
	        i = tf.range(sequence_length)[:, tf.newaxis]
	        j = tf.range(sequence_length)
	        mask = tf.cast(i >= j, dtype="int32")
	        mask = tf.reshape(mask, (1, input_shape[1], input_shape[1]))
	        mult = tf.concat(
	            [tf.expand_dims(batch_size, -1),
	             tf.constant([1, 1], dtype=tf.int32)], axis=0)
	        return tf.tile(mask, mult)

	    # 입력, 인코더 출력, 마스크를 입력으로 받아 디코더 계층의 연산을 수행하고 결과를 반환
	    def call(self, inputs, encoder_outputs, mask=None):
	        causal_mask = self.get_causal_attention_mask(inputs)
	        if mask is not None:
	            padding_mask = tf.cast(
	                mask[:, tf.newaxis, :], dtype="int32")
	            padding_mask = tf.minimum(padding_mask, causal_mask)
	        attention_output_1 = self.attention_1(
	            query=inputs,
	            value=inputs,
	            key=inputs,
	            attention_mask=causal_mask)
	        attention_output_1 = self.layernorm_1(inputs + attention_output_1)
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

모델 정의

	'''

	from tensorflow.keras import layers

	# 임베딩 차원, 잠재 차원 및 어텐션 헤드의 수를 정의
	embed_dim = 256
	latent_dim = 2048
	num_heads = 2

	# 모델의 입력을 정의. 이 경우 가변 길이의 정수 시퀀스를 입력으로 사용
	inputs = keras.Input(shape=(None,), dtype="int64")

	# PositionalEmbedding클래스를 사용하여 입력 데이터에 위치 임베딩을 적용
	x = PositionalEmbedding(sequence_length, vocab_size, embed_dim)(inputs)

	# TransformerDecoder클래스를 사용하여 디코더 레이어를 생성. 인코더 출력을 입력으로 받고, 디코더의 출력을 계산한다.
	x = TransformerDecoder(embed_dim, latent_dim, num_heads)(x, x)

	# 출력 시퀀스 타임스텝마다 가능한 어휘 사전의 단어에 대해 소프트맥스 확률을 계산한다.
	outputs = layers.Dense(vocab_size, activation="softmax")(x)

	# 입력과 출력을 연결하여 최종 모델을 생성
	model = keras.Model(inputs, outputs)

	# 모델을 컴파일한다. 손실 함수로 'sparse_categorical_crossentropy'를 사용하며, 옵티마이저로 rmsprop을 사용
	model.compile(loss="sparse_categorical_crossentropy", optimizer="rmsprop")

	'''

---

### 가변 온도 샘플링을 사용한 텍스트 생성 콜백

---

다양한 온도로 텍스트 생성

	콜백을 사용해 에포크가 끝날 때마다 실행

3. 텍스트 생성 콜백 정의 및 모델 훈련
---

	'''

	import numpy as np

	# 단어 인덱스를 문자열로 매핑하는 딕셔너리. 텍스트 디코딩에 사용
	tokens_index = dict(enumerate(text_vectorization.get_vocabulary()))

	# 주어진 확률 분포를 기반으로 온도를 적용하여 다음 단어를 샘플링하는 함수. 온도가 높을수록 더 다양한 샘플링 결과가 나올 수 있다.
	def sample_next(predictions, temperature=1.0):
	    predictions = np.asarray(predictions).astype("float64")
	    predictions = np.log(predictions) / temperature
	    exp_preds = np.exp(predictions)
	    predictions = exp_preds / np.sum(exp_preds)
	    probas = np.random.multinomial(1, predictions, 1)
	    return np.argmax(probas)

	# 콜백 클래스로, 몯ㄹ의 에포크 종료 시마다 새로운 텍스트를 생성하는 역할을 한다. 생성할 텍스트의 길이, 모델 입력 길이, 샘플링 온도 등을 설정할 수 있다.
	class TextGenerator(keras.callbacks.Callback):
	    def __init__(self,
	                 prompt, # 텍스트 생성을 위한 시작 문장
	                 generate_length, # 생성할 단어 계수
	                 model_input_length,
	                 temperatures=(1.,), # 샘플링에 사용할 온도 범위
	                 print_freq=1):
	        self.prompt = prompt
	        self.generate_length = generate_length
	        self.model_input_length = model_input_length
	        self.temperatures = temperatures
	        self.print_freq = print_freq

	    # 에포크 종료 시 호출된다. 지정된 시작 문장으로부터 모델을 사용하여 텍스트를 생성하고 출력한다. 다양한 온도 값을 사용하여 텍스트를 생성하는 결과를 비교한다.
	    def on_epoch_end(self, epoch, logs=None):
	        if (epoch + 1) % self.print_freq != 0:
	            return
	        for temperature in self.temperatures:
	            print("== Generating with temperature", temperature)

	            # 시작 단어에서부터 텍스트를 생성한다.
	            sentence = self.prompt
	            for i in range(self.generate_length):
	                # 현재 시퀀스를 모델에 주입한다.
	                tokenized_sentence = text_vectorization([sentence])
	                predictions = self.model(tokenized_sentence)

	                # 마지막 타임스텝의 예측을 추출하여 다음 언어를 샘플링한다.
	                next_token = sample_next(predictions[0, i, :])
	                sampled_token = tokens_index[next_token]
	
	                # 새로운 단어를 현재 시퀀스에 추가하고 반복한다.
	                sentence += " " + sampled_token
	            print(sentence)

	prompt = "This movie"

	# TextGenerator 콜백 클래스의 인스턴스를 생성하여 설정한다. 이 콜백은 모델의 에포크 종료 시마다 주어진 시작 문장으로부터 다양한 온도 값에 대한 텍스트를 생성하고 출력한다.
	text_gen_callback = TextGenerator(
	    prompt,
	    generate_length=50,
	    model_input_length=sequence_length,

	    # 텍스트 샘플링에 다양한 온도를 사용하여 텍스트 생성에 미치는 온도의 영향을 확인
	    temperatures=(0.2, 0.5, 0.7, 1., 1.5))

	model.fit(lm_dataset, epochs=200, callbacks=[text_gen_callback])

	'''

	결과:

	'''

	temperature=0.2

	            “this movie is a [UNK] of the original movie and the first half hour of the 

	             movie is pretty good but it is a very good movie it is a good movie for the

	             time period”

	            “this movie is a [UNK] of the movie it is a movie that is so bad that it is a

	             [UNK] movie it is a movie that is so bad that it makes you laugh and cry at

	             the same time it is not a movie i dont think ive ever seen”

	temperature=0.5

	            “this movie is a [UNK] of the best genre movies of all time and it is not a

	             good movie it is the only good thing about this movie i have seen it for the

	             first time and i still remember it being a [UNK] movie i saw a lot of years”

	            “this movie is a waste of time and money i have to say that this movie was

	             a complete waste of time i was surprised to see that the movie was made

	             up of a good movie and the movie was not very good but it was a waste of

	             time and”

	temperature=0.7

	            “this movie is fun to watch and it is really funny to watch all the characters

	             are extremely hilarious also the cat is a bit like a [UNK] [UNK] and a hat

	             [UNK] the rules of the movie can be told in another scene saves it from

	             being in the back of ”

	            “this movie is about [UNK] and a couple of young people up on a small boat

	             in the middle of nowhere one might find themselves being exposed to a

	             [UNK] dentist they are killed by [UNK] i was a huge fan of the book and i

	             havent seen the original so it”

	temperature=1.0

	            “this movie was entertaining i felt the plot line was loud and touching but on

	              a whole watch a stark contrast to the artistic of the original we watched

	              the original version of england however whereas arc was a bit of a little too

	              ordinary the [UNK] were the present parent [UNK]”

	            “this movie was a masterpiece away from the storyline but this movie was

	              simply exciting and frustrating it really entertains friends like this the actors

	              in this movie try to go straight from the sub thats image and they make it a

	              really good tv show”

	temperature=1.5

	            “this movie was possibly the worst film about that 80 women its as weird

	             insightful actors like barker movies but in great buddies yes no decorated

	             shield even [UNK] land dinosaur ralph ian was must make a play happened

	             falls after miscast [UNK] bach not really not wrestlemania seriously sam

	             didnt exist”

	            “this movie could be so unbelievably lucas himself bringing our country

	             wildly funny things has is for the garish serious and strong performances

	             colin writing more detailed dominated but before and that images gears

	             burning the plate patriotism we you expected dyan bosses devotion to

	             must do your own duty and another”

	'''

		- 낮은 온도는 단조로운 반복적인 텍스트를 생성
			- 이로 인해 생성된 텍스트가 자주 반복되는 현상이 발생
		- 높은 온도에서 생성된 텍스트는 흥미롭고 다양한 결과를 만들어냄
			- 예상치 못한 내용이 등장하며 창의성을 발휘
			- 구조가 무너져 랜덤한 출력이 될 수 있
				- 무작위성이 강조되어 예측하기 어려운 텍스트가 생성
		- 0.7 정도의 온도가 적절한 결과를 만들어 내는 것으로 보임


---

## 딥드림

---
