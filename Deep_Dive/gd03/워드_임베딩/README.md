# 학습 내용

---

- 벡터화
	- 텍스트를 벡터화하는 방법 학습
- 벡터화 실습: 원-핫 인코딩 구현
	- 원-핫 인코딩 구현
- 워드 임베딩
	- 희소 벡터의 특징, 문제점 학습
- Word2Vec (1) 분포 가설
	- 분포 가설 학습
- Word2Vec (2) CBoW
	- CBoW(Continuous Bag of words) 학습
- Word2Vec (3) Skip-gram과 Negative Sampling
	- Skip-gram 학습
- Word2Vec (4) 영어 Word2Vec 실습과 OOV 문제
	- OOV(Out Of Vocabuary) 문제 학습
- 임베딩 벡터의 시각화
	- 임베딩 벡터들 시각화
- FastText
	- FastText 원리 학습
- GloVe
	- GloVe 원리 학습

---

## 벡터화

---

Bag of words?

	단어의 순서를 고려하지 않고, 단어의 등장 빈도(frequency)만을 고려해서 단어를 벡터화 방법

DTM(문서 단어 행렬, Document-Term Matrix)?

	Bag of words를 사용하여 문서 간 유사도를 비교하기 위해 만든 행렬

	ex) 문서1 : you know I want your love
	    문서2 : I like you
	    문서3 : what should I do

		-> 길이가 1인 단어를 제거

![image](https://d3s0tskafalll9.cloudfront.net/media/images/GN-3-L-1.max-800x600.jpg)

	->  대부분의 값이 0이라는 특징

		-> 희소 벡터(sparse vector)

단어장(vocabulary)?

	중복 카운트는 배제한 단어들의 집합(set)

DTM의 문제점

	문서의 유사도를 비교하는 경우

		-> 두 문서에서 공통적으로 등장하는 단어가 많으면 그 두 문서는 유사하다고 판단

			-> 별로 중요하지도 않은 단어인데도 모든 문서에서 공통적으로 등장하는 단어가 있다는 것

TF-IDF

	DTM의 문제점을 해결하고자 단어마다 중요 가중치를 다르게 주는 방법

		-> 여전히 희소 벡터

원-핫 인코딩(one-hot encoding)??

	모든 단어의 관계를 독립적으로 정의

	진행 방식

	1. 텍스트 데이터에서 단어들의 집합인 단어장(vocabulary) 생성
	2. 단어장에 있는 모든 단어에 대해서 1부터 V까지 고유한 정수 부여
		- 이 정수는 단어장에 있는 각 단어의 일종의 인덱스 역할
	3. 해당 단어의 인덱스 위치만 1이고 나머지는 전부 0의 값을 가지는 벡터 생성
		- 각 단어는 V(vocab_size)차원의 벡터로 표현

원-핫 인코딩 ex)

	문서 1 : 강아지, 고양이, 강아지			    	강아지: 1번	컴퓨터: 4번
	문서 2 : 애교, 고양이				->  	고양이: 2번	노트북: 5번
	문서 3 : 컴퓨터, 노트북				    	애교: 3 번

	∴  강아지 : [1, 0, 0, 0, 0]
	   고양이 : [0, 1, 0, 0, 0]
	   애교   : [0, 0, 1, 0, 0]
	   컴퓨터 : [0, 0, 0, 1, 0]
	   노트북 : [0, 0, 0, 0, 1]

TF(Term Frequency)?

	문장을 구성하는 단어들의 원-핫 벡터들을 모두 더해서 문장의 단어 개수로 나눈 것과 같음

---

## 벡터화 실습: 원-핫 인코딩 구현

---

한국어 실습을 위한 한국어 형태소 분석기 패키지 

	KoNLPy


	설치
	-> pip install konlpy

필요한 라이브러리 import 

	'''

	import re
	from konlpy.tag import Okt
	from collections import Counter

	'''

전처리

	정규 표현식을 사용해 특수문자들 제거 -> 한글과 공백을 제외 특수문자만 제거

		-> 자음의 범위는 'ㄱ ~ ㅎ', 모음의 범위는 'ㅏ ~ ㅣ', 완성형 한글의 범위는 '가 ~ 힣'와 같이 지정

			-> [^ㄱ-ㅎㅏ-ㅣ가-힣 ]

	'''

	text = "임금님 귀는 당나귀 귀! 임금님 귀는 당나귀 귀! 실컷~ 소리치고 나니 속이 확 뚫려 살 것 같았어."

	reg = re.compile("[^ㄱ-ㅎㅏ-ㅣ가-힣 ]")
	text = reg.sub('', text)
	print(text)

	'''

	결과: 임금님 귀는 당나귀 귀 임금님 귀는 당나귀 귀 실컷 소리치고 나니 속이 확 뚫려 살 것 같았어

토큰화

	한국어 -> 주로 형태소 분석기를 통해서 토큰 단위로 나눔

	'''

	okt=Okt()
	tokens = okt.morphs(text)
	print(tokens)

	'''

	결과: ['임금님', '귀', '는', '당나귀', '귀', '임금님', '귀', '는', '당나귀', '귀', '실컷', '소리', '치고', '나니', '속이', '확', '뚫려', '살', '것', '같았어']

단어장 생성

	빈도수가 높은 단어일수록 낮은 정수를 부여

	1. 각 단어의 빈도수 체크

		'''

		vocab = Counter(tokens)
		print(vocab)

		'''

		결과: Counter({'귀': 4, '임금님': 2, '는': 2, '당나귀': 2, '실컷': 1, '소리': 1, '치고': 1, '나니': 1, '속이': 1, '확': 1, '뚫려': 1, '살': 1, '것': 1, '같았어': 1})

	2. 등장 빈도수가 높은 상위 5개의 단어만 저장

		'''

		vocab_size = 5
		vocab = vocab.most_common(vocab_size)
		print(vocab)
	
		'''
	
		결과: [('귀', 4), ('임금님', 2), ('는', 2), ('당나귀', 2), ('실컷', 1)]

	3. 높은 빈도수를 가진 단어일수록 낮은 정수 인덱스 부여

		'''

		word2idx={word[0] : index+1 for index, word in enumerate(vocab)}
		print(word2idx)

		'''

		결과: {'귀': 1, '임금님': 2, '는': 3, '당나귀': 4, '실컷': 5}

원-핫 벡터 생성

	'''

	def one_hot_encoding(word, word2index):
	       one_hot_vector = [0]*(len(word2index))
	       index = word2index[word]
	       one_hot_vector[index-1] = 1
	       return one_hot_vector

	one_hot_encoding("임금님", word2idx)

	'''

	결과: [0, 1, 0, 0, 0]

케라스를 통한 원-핫 인코딩

	'''

	from tensorflow.keras.preprocessing.text import Tokenizer
	from tensorflow.keras.utils import to_categorical

	text = [['강아지', '고양이', '강아지'],['애교', '고양이'], ['컴퓨터', '노트북']]

	t = Tokenizer()
	t.fit_on_texts(text)

	vocab_size = len(t.word_index) + 1

	sub_text = ['강아지', '고양이', '강아지', '컴퓨터']
	encoded = t.texts_to_sequences([sub_text])

	one_hot = to_categorical(encoded, num_classes = vocab_size)
	print(one_hot)

	'''

	결과: [[[0. 1. 0. 0. 0. 0.]
	        [0. 0. 1. 0. 0. 0.]
	        [0. 1. 0. 0. 0. 0.]
	        [0. 0. 0. 0. 1. 0.]]]

---

## 워드 임베딩

---

희소 벡터(Sparse Vector)의 문제점

	차원의 저주(curse of dimensionality)

![image](https://d3s0tskafalll9.cloudfront.net/media/original_images/seukeurinsyas_2021-09-09_15-30-49.png)

	'강아지'와 '고양이'라는 두 단어의 의미적 유사성 VS  '강아지'와 '컴퓨터'라는 두 단어의 의미적 유사성

		->  '강아지'와 '고양이'는 귀여운 애완동물이고, '컴퓨터'는 데이터를 처리하는 전자기기라는 것을 반영 X

			-> 원-핫 벡터 간 내적 -> 서로 직교(orthogonal)해 그 값이 0

	∴  단어 벡터 간 유사도를 구할 수 없음

희소 벡터의 해결책

	임베딩 벡터

워드 임베딩?

	- 한 단어를 벡터로 바꿉니다. 그런데 그 벡터의 길이를 일정하게 정해줌
		- 벡터의 길이가 단어장 크기보다 매우 작기 때문에 각 벡터 값에 정보가 축약
		- 밀집 벡터(dense vector)
	- 밀집 벡터에서는 대부분 값이 0이 아님
		- 각 벡터 값의 의미가 파악하기 어려울 정도로 많은 의미를 함축

![image](https://d3s0tskafalll9.cloudfront.net/media/images/gn-3-l-3-2.max-800x600.jpg)

	- 단어가 갖는 특성을 계산 가능
	- 인공 신경망을 이용한 방법 사용
		- 인공 신경망을 학습해가는 과정을 이용해 벡터의 값을 조정해 가는 방법
		- 학습이 끝나면 단어가 들어가야 할 위치나 의미에 맞게 단어 벡터의 값이 결정됨
		ex) //      [둥근,빨간,단맛,신맛]
		    사과  : [0.8, 0.7, 0.7, 0.1] // 0.8만큼 둥글고, 0.7만큼 빨갛고, 0.7만큼 달고, 0.1만큼 신 것은 사과다
		    바나나: [0.1, 0.0, 0.8. 0.0] // 0.1만큼 둥글고, 0.0만큼 빨갛고, 0.8만큼 달고, 0.0만큼 신 것은 바나나다
		    귤    : [0.7, 0.5, 0.6, 0.5] // 0.7만큼 둥글고, 0.5만큼 빨갛고, 0.6만큼 달고, 0.5만큼 신 것은 귤이다

---

## Word2Vec (1) 분포 가설

---
