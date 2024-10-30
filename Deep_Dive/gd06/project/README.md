# 학습 내용

---

- Step 1. 데이터 다운
- Step 2. 데이터 정제
- Step 3. 데이터 토큰화
- Step 4. Augmentation
- Step 5. 데이터 벡터화
- Step 6. 훈련
- Step 7. 성능 측정

---

## Step 1. 데이터 다운

---

[songys/Chatbot_data](https://github.com/songys/Chatbot_data)

	위 링크의 챗봇 데이터 사용

---

## Step 2. 데이터 정제

---

아래 조건을 만족하는 preprocess_sentence() 함수 구현

	1. 영문자의 경우, 모두 소문자로 변환
	2. 영문자와 한글, 숫자, 그리고 주요 특수문자를 제외하곤 정규식을 활용하여 모두 제거

---

## Step 3. 데이터 토큰화

---

토큰화 

	KoNLPy의 mecab 사용

아래 조건을 만족하는 build_corpus() 함수 구현

	1. 소스 문장 데이터와 타겟 문장 데이터를 입력으로 받음
	2. 데이터를 앞서 정의한 preprocess_sentence() 함수로 정제, 토큰화
	3. 토큰화는 전달받은 토크나이즈 함수를 사용
		- 이번엔 mecab.morphs 함수 전달
	4. 토큰의 개수가 일정 길이 이상인 문장은 데이터에서 제외
	5. 중복되는 문장은 데이터에서 제외
		-  소스 : 타겟 쌍을 비교하지 않고 소스는 소스대로 타겟은 타겟대로 검사
		- 중복 쌍이 흐트러지지 않도록 유의!

---

## Step 4. Augmentation

---

Lexical Substitution을 실제로 적용

[Kyubyong/wordvectors](https://github.com/Kyubyong/wordvectors)

	- 위 링크에서 한국어로 사전 훈련된 Embedding 모델을 다운
		- Korean (w) 가 Word2Vec으로 학습한 모델이며 용량도 적당하므로 사이트에서 Korean (w)를 찾아 다운해 ko.bin 파일을 얻음

	- 앞서 정의한 lexical_sub() 함수 참고해 정의
	- 전체 데이터가 원래의 3배가량으로 늘어나도록 적용
		- Augmentation된 que_corpus 와 원본 ans_corpus 병렬
		- 원본 que_corpus 와 Augmentation된 ans_corpus 병렬

---

## Step 5. 데이터 벡터화

---

1. 타겟 데이터 전체에 start 토큰과 end 토큰을 추가
2. ans_corpus, que_corpus 와 결합하고 전체 데이터에 대한 단어 사전을 구축해 벡터화하여 enc_train 과 dec_train 생성

---

## Step 6. 훈련

---

사용할 모델

	Transformer

---

## Step 7. 성능 측정

---

calculate_bleu() 함수 정의

	BLEU Score를 계산
