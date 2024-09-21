# 학습 내용

---

네이버 영화리뷰 감성분석 도전하기

1) 데이터 준비와 확인

	'''

	import pandas as pd

	# 데이터를 읽어봅시다. 
	train_data = pd.read_table('~/aiffel/sentiment_classification/data/ratings_train.txt')
	test_data = pd.read_table('~/aiffel/sentiment_classification/data/ratings_test.txt')

	train_data.head()

	'''

2) 데이터로더 구성

	- nsmc 데이터셋은 전혀 가공되지 않은 텍스트 파일로 이루어져 있음
	- 이것을 읽어서 imdb.data_loader()와 동일하게 동작하는 자신만의 data_loader를 만들어 보는 것으로 시작
	-  data_loader 안에서는 다음을 수행
		- 데이터의 중복 제거
		- NaN 결측치 제거
		- 한국어 토크나이저로 토큰화
		- 불용어(Stopwords) 제거
		- 사전word_to_index 구성
		- 텍스트 스트링을 사전 인덱스 스트링으로 변환
		- X_train, y_train, X_test, y_test, word_to_index 리턴

3) 모델 구성을 위한 데이터 분석 및 가공

	- 데이터셋 내 문장 길이 분포
	- 적절한 최대 문장 길이 지정
	- keras.preprocessing.sequence.pad_sequences 을 활용한 패딩 추가

4) 모델 구성 및 validation set 구성

	- 모델은 3가지 이상 다양하게 구성하여 실험해 보세요.

5) 모델 훈련 개시

6) Loss, Accuracy 그래프 시각화

7) 학습된 Embedding 레이어 분석

8) 한국어 Word2Vec 임베딩 활용하여 성능 개선

	- 한국어 Word2Vec은 /data 폴더 안에 있는 word2vec_ko.model을 활용
	- 한국어 Word2Vec을 활용할 때는 load_word2vec_format() 형태가 아닌 load() 형태로 모델 로드
	-  모델을 활용할 때에는 아래 예시와 같이 .wv를 붙여서 활용

	'''

	# 예시 코드
	from gensim.models.keyedvectors import Word2VecKeyedVectors
	word_vectors = Word2VecKeyedVectors.load(word2vec_file_path)
	vector = word_vectors.wv[‘끝’]

	'''
