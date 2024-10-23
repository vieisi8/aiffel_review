# 학습 내용

---

- Step 1. 데이터 다운로드
- Step 2. 데이터 정제
- Step 3. 데이터 토큰화
- Step 5. 훈련

---

## Step 1. 데이터 다운로드

---

데이터셋 

[jungyeul/korean-parallel-corpora](https://github.com/jungyeul/korean-parallel-corpora/tree/master/korean-english-news-v1)

	위 링크의 korean-english-park.train.tar.gz 파일 사용

		-> 한영 병렬 데이터

---

## Step 2. 데이터 정제

---

1. set 데이터형이 중복을 허용하지 않는다는 것을 활용해 중복된 데이터를 제거
	-  데이터의 병렬 쌍이 흐트러지지 않게 주의
	- 중복을 제거한 데이터를 cleaned_corpus 에 저장
2. 한글에 적용할 수 있는 정규식을 추가해 preprocessing() 함수 재정의
3. 타겟 언어인 영문엔 <start> 토큰과 <end> 토큰을 추가하고 split() 함수를 이용하여 토큰화
	- 한글 토큰화는 KoNLPy의 mecab 사용

---

## Step 3. 데이터 토큰화

---

토큰화

	tokenize() 함수를 사용해 데이터를 텐서로 변환하고 각각의 tokenizer 생성

		-> 주의: 난이도에 비해 데이터가 많지 않아 훈련 데이터와 검증 데이터를 따로 나누지 않음

---

## Step 4. 모델 설계

---

Attention 기반 Seq2seq 모델을 설계

---

## Step 5. 훈련

---

	매 스텝 아래의 예문에 대한 번역을 생성

		## 예문 ##
		K1) 오바마는 대통령이다.
		K2) 시민들은 도시 속에 산다.
		K3) 커피는 필요 없다.
		K4) 일곱 명의 사망자가 발생했다.
