# 학습 내용 

---

- 1. Tokenizer 준비
- 2. 데이터 전처리 (1) MASK 생성
- 3. 데이터 전처리 (2) NSP pair 생성
- 4. 데이터 전처리 (3) 데이터셋 완성
- 5. BERT 모델 구현
- 6. pretrain 진행
- 7. 프로젝트 결과

---

## 1. Tokenizer 준비

---

SentencePiece 모델을 이용해 BERT의 MLM 학습용 데이터 생성

	한글 나무 위키 코퍼스로부터 8000의 vocab_size를 갖는 sentencepiece 모델 생성

		-> BERT에 사용되는 주요 특수문자가 vocab에 포함되어야 함

---

## 2. 데이터 전처리 (1) MASK 생성

---

MLM에 필요한 전체 토큰의 15% 정도 masking

	그 중 80%는 [MASK] 토큰, 10%는 랜덤한 토큰, 나머지 10%는 원래의 토큰을 그대로 사용

---

## 3. 데이터 전처리 (2) NSP pair 생성

---

pretrain task인 NSP

	- 2개의 문장을 짝지어 50%의 확률로 TRUE와 FALSE를 지정
	- 두 문장 사이에 segment 처리
		- 첫 번째 문장의 segment는 0, 두 번째 문장은 1로 채워준 후 둘 사이에 구분자인 [SEP]

---

## 4. 데이터 전처리 (3) 데이터셋 완성

---

BERT pretrain 데이터셋

	데이터셋을 생성해, json 포맷으로 저장

		-> np.memmap을 사용해 메모리 사용량을 최소화 load

---

## 5. BERT 모델 구현

---

사용할 모델

	BERT 모델
	
		-> pretraine용 BERT 모델 생성

---

## 6. pretrain 진행

---

- loss, accuracy 함수 정의
- Learning Rate 스케쥴링 구현
- 10 Epoch까지 모델 학습

---

## 7. 프로젝트 결과

---

학습된 모델, 학습과정 시각화
