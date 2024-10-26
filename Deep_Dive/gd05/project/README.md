# 학습 내용

---

- Step 1. 데이터 다운로드
- Step 2. 데이터 정제 및 토큰화
- Step 3. 모델 설계
- Step 4. 훈련하기

---

## Step 1. 데이터 다운로드

---

사용할 데이터

[jungyeul/korean-parallel-corpora](https://github.com/jungyeul/korean-parallel-corpora/tree/master/korean-english-news-v1)

	korean-english-park.train.tar.gz 사용할 예정

---

## Step 2. 데이터 정제 및 토큰화

---

	1. set 데이터형이 중복을 허용하지 않는다는 것을 활용해 중복된 데이터를 제거
		- 데이터의 병렬 쌍이 흐트러지지 않게 주의
		- 중복을 제거한 데이터를 cleaned_corpus 에 저장
	2. 정제 함수를 아래 조건을 만족하게 정의
		- 모든 입력을 소문자로 변환
		- 알파벳, 문장부호, 한글만 남기고 모두 제거
		- 문장부호 양옆에 공백을 추가
		- 문장 앞뒤의 불필요한 공백을 제거
	3. 한글 말뭉치, 영문 말뭉치 각각 분리한 후, 정제해 토큰화 진행
		- 토큰화에 Sentencepiece 활용
		- 아래 조건을 만족하는 generate_tokenizer() 함수 정의
			- 단어 사전을 매개변수로 받아 원하는 크기의 사전을 정의할 수 있게 함 (기본: 20,000)
			- 학습 후 저장된 model 파일을 SentencePieceProcessor() 클래스에 Load()한 후 반환
			- 특수 토큰의 인덱스를 아래와 동일하게 지정
				- PAD : 0 / BOS : 1 / EOS : 2 / UNK : 3
	4. 토크나이저를 활용해 토큰의 길이가 50 이하인 데이터를 선별후 텐서로 변환
 
---

## Step 3. 모델 설계

---

사용할 모델

	Transformer 모델

---

## 훈련

---

	1. 2 Layer를 가지는 Transformer 선언
	2. 논문에서 사용한 것과 동일한 Learning Rate Scheduler 선언, 이를 포함하는 Adam Optimizer 선언
	3. Loss 함수를 정의
		- Sequence-to-sequence 모델에서 사용했던 Loss와 유사하되, Masking 되지 않은 입력의 개수로 Scaling하는 과정을 추가
	4. train_step 함수 정의
		- 입력 데이터에 알맞은 Mask를 생성, 이를 모델에 전달하여 연산에서 사용할 수 있게 함
	5. 학습 진행
		- 매 Epoch 마다 제시된 예문에 대한 번역 생성

	예문

		1. 오바마는 대통령이다.
		2. 시민들은 도시 속에 산다.
		3. 커피는 필요 없다.
		4. 일곱 명의 사망자가 발생했다.
