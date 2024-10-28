# 학습 내용

---

- 번역의 흐름
	- 신경망 기반 번역기 이전의 역사는 어떠한지 확인
- 지적 생성을 위한 넓고 얕은 탐색 (1) Greedy Decoding
	- 신경망 기반 번역기의 원리가 되는 알고리즘 확인
- 지적 생성을 위한 넓고 얕은 탐색 (2) Beam Search
	- 좋은 문장을 생성하기 위한 방법에는 무엇인지 확인
- 지적 생성을 위한 넓고 얕은 탐색 (3) Sampling
	- 안정적으로 단어를 생성할 수 있는 sampling 공부
- 방과 후 번역 수업 (1) Data Augmentation
	- 데이터를 증강은 무엇이고 왜 사용할까요?
- 방과 후 번역 수업 (2) Lexical Substitution
	- 데이터 증강의 방법 중 lexical substitution 공부
- 방과 후 번역 수업 (3) Back Translation
	- 번역은 어떻게 데이터 증강법으로 활용될 수 있을까?
- 방과 후 번역 수업 (4) Random Noise Injection
	- 텍스트 데이터에서 노이즈는 무엇 인지?
- 채점은 어떻게?
	- 분류는 accuracy, 회귀는 MAE, 그렇다면 번역은?
- 실례지만, 어디 챗씨입니까? (1) 챗봇과 번역기
	- 챗봇과 번역기, 같으면서도 다른 두 개념 확인
- 실례지만, 어디 챗씨입니까? (2) 좋은 챗봇이 되려면
	- 좋은 챗봇과 나쁜 챗봇, 무엇으로 구분?
- 실례지만, 어디 챗씨입니까? (3) 대표적인 챗봇
	- 대표적인 챗봇 확인

---

## 번역의 흐름

---

규칙 기반 기계 번역(RBMT, Rule-Based Machine Translation)?

	번역할 때 경우의 수를 직접 정의해 주는 방식

		-> 수많은 규칙들은 모두 언어학을 기반으로 하기 때문에, 개발 과정에 언어학자가 동반되어야만 했음

규칙 기반 기계 번역의 한계

	- 규칙에 없는 문장이 들어올 경우 번역이 불가능
	- 유연성이 떨어짐
	- 모든 규칙을 정의하는 과정이 너무나도 복잡하고 오랜 시간이 필요

통계적 기계 번역?

	수많은 데이터로부터 통계적 확률을 구해 번역을 진행

		-> 통계적 언어 모델에서 파생된 확률에 위 모든 확률을 곱하여 학습하는 것

통계적 언어 모델?

[위키독스: 통계적 언어 모델](https://wikidocs.net/21687)

	-> "문장이 존재할 확률을 측정한다."

신경망 기계 번역?

	seq2seq | transformer

---

## 지적 생성을 위한 넓고 얕은 탐색 (1) Greedy Decoding

---

탐욕 알고리즘(Greedy Algorithm)?

	가장 높은 확률을 갖는 단어가 다음 단어로 결정

		-> Greedy Decoding이라 칭함

[Greedy Algorithm (탐욕 알고리즘)](https://janghw.tistory.com/entry/%EC%95%8C%EA%B3%A0%EB%A6%AC%EC%A6%98-Greedy-Algorithm-%ED%83%90%EC%9A%95-%EC%95%8C%EA%B3%A0%EB%A6%AC%EC%A6%98)

탐욕적인 방법의 한계

	효율적이지만 최적의 해를 구해준다는 보장이 없음

		-> 최고의 번역을 생성하고 있는 것 X

![image](https://d3s0tskafalll9.cloudfront.net/media/images/GN-6-L-03.max-800x600.png)

	-> 훈련 데이터가 실제 세계의 모든 데이터를 포함할 수는 없기 때문

		-> have 가 마시다 로 사용되는 경우가 훈련 데이터에 적거나 없었다면 탐욕적인 방법은 have 를 가장 높은 확률을 갖는 가지다 로 번역할 수밖에 없었을 것

해결 방법

	단어 사전으로 만들 수 있는 모든 문장을 만든 후, 실제 세계에 존재하는 우리가 직접 고르는 방법

		-> 1,000개의 단어를 갖는 사전으로 3개 단어 문장 하나를 만드는 데에 1,000,000,000개 문장이 서비스로 온다는 것

			-> 큰 문제가 됨

---

## 지적 생성을 위한 넓고 얕은 탐색 (2) Beam Search

---

Beam Search?

	지금 상황에서 가장 높은 확률을 갖는 Top-k 문장만 남기는 것

![image](https://d3s0tskafalll9.cloudfront.net/media/images/GN-6-L-04.max-800x600.png)

	-> Beam Size=2로 하는 Beam Search를 표현한 것

		-> Beam Size는 연산량과 성능 간의 Trade-off 관계를 가지고 있음

Beam Search ex)

	'''

	import math
	import numpy as np

	def beam_search_decoder(prob, beam_size):
	    sequences = [[[], 1.0]]  # 생성된 문장과 점수를 저장

	    for tok in prob:
	        all_candidates = []

	        for seq, score in sequences:
	            for idx, p in enumerate(tok): # 각 단어의 확률을 총점에 누적 곱
	                candidate = [seq + [idx], score * -math.log(-(p-1))]
	                all_candidates.append(candidate)

	        ordered = sorted(all_candidates,
	                         key=lambda tup:tup[1],
	                         reverse=True) # 총점 순 정렬
	        sequences = ordered[:beam_size] # Beam Size에 해당하는 문장만 저장 

	    return sequences

	vocab = {
	    0: "<pad>",
	    1: "까요?",
	    2: "커피",
	    3: "마셔",
	    4: "가져",
	    5: "될",
	    6: "를",
	    7: "한",
	    8: "잔",
	    9: "도",
	}

	# prob_seq은 문장의 각 위치에서 어떤 단어가 생성될지의 확률을 한 번에 정의해둔 것입니다.
	# 실제로는 각 단어에 대한 확률이 prob_seq처럼 한번에 정의되지 않기 때문에 실제 문장 생성과정과는 거리가 멉니다.
	# 하지만 Beam Search의 동작과정 이해를 돕기위해 이와 같은 예시를 보여드립니다.
	# prob_seq의 각 열은 위 vocab의 각 숫자(key)에 대응됩니다.
	prob_seq = [[0.01, 0.01, 0.60, 0.32, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01], # 커피 : 0.60
	            [0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.75, 0.01, 0.01, 0.17], # 를 : 0.75
	            [0.01, 0.01, 0.01, 0.35, 0.48, 0.10, 0.01, 0.01, 0.01, 0.01], # 가져 : 0.48
	            [0.24, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.68], # 도 : 0.68
	            [0.01, 0.01, 0.12, 0.01, 0.01, 0.80, 0.01, 0.01, 0.01, 0.01], # 될 : 0.80
	            [0.01, 0.81, 0.01, 0.01, 0.01, 0.01, 0.11, 0.01, 0.01, 0.01], # 까요? : 0.81
	            [0.70, 0.22, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01], # <pad> : 0.91
	            [0.91, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01], # <pad> : 0.91
	            [0.91, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01], # <pad> : 0.91
	            [0.91, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01]] # <pad> : 0.91

	prob_seq = np.array(prob_seq)
	beam_size = 3

	result = beam_search_decoder(prob_seq, beam_size)

	for seq, score in result:
	    sentence = ""

	    for word in seq:
	        sentence += vocab[word] + " "

	    print(sentence, "// Score: %.4f" % score)

	'''

	결과:   커피 를 가져 도 될 까요? <pad> <pad> <pad> <pad>  // Score: 42.5243
		커피 를 마셔 도 될 까요? <pad> <pad> <pad> <pad>  // Score: 28.0135
		마셔 를 가져 도 될 까요? <pad> <pad> <pad> <pad>  // Score: 17.8983

주의할 점!

	사람이 직접 좋은 번역을 고를 수 있게 상위 K개의 결과를 보여줄 뿐이라서 학습에 직접적으로 적용할 수는 없음

		-> 모델 학습 단계에서 Beam Search를 사용하지는 않음

---

## 지적 생성을 위한 넓고 얕은 탐색 (3) Sampling

---

Sampling?

	확률적으로 단어를 뽑는 방법

		-> 언어 모델은 반복적으로 다음 단어에 대한 확률 분포를 생성하기 때문에 그 확률 분포를 기반으로 랜덤하게 단어를 뽑아 보자는 것

![image](https://d3s0tskafalll9.cloudfront.net/media/images/GN-6-L-05.max-800x600.png)

---

## 방과 후 번역 수업 (1) Data Augmentation

---

Data Augmentation?

	훈련 데이터를 수십 배까지도 부풀리는 기술을 의미

[데이터 Preprocessing과 Augmentation](https://modulabs.co.kr/blog/data-processing-augmentation/)

Data Augmentation 장점

	어떤 변화를 시켜도 왜곡이 크지 않고 심지어 일괄적으로 처리할 수 있음

---

## 방과 후 번역 수업 (2) Lexical Substitution

---

Lexical Substitution?

	문장 데이터의 Data Augmentation 

		-> 어휘 대체

동의어 기반 대체?

	시소러스를 활용한 방법

![imgae](https://d3s0tskafalll9.cloudfront.net/media/original_images/GN-6-L-07.png)

시소러스(Thesaurus)?

	어떤 단어의 동의어나 유의어를 집중적으로 구축해놓은 사전

[시소러스를 활용한 단어 의미 파악](https://kh-kim.gitbook.io/natural-language-processing-with-pytorch/00-cover-4/03-wordnet)

한국어로 구축된 워드넷

	- KorLex
	- Korean WordNet(KWN)

한계점

	규칙 기반 기계 번역처럼 모든 것을 사람이 정의해야 한다는 것

Embedding 활용 대체?

	Pre-training Word Embedding을 활용하는 방법

		-> Embedding의 유사도를 기반으로 단어를 대체

![image](https://d3s0tskafalll9.cloudfront.net/media/images/GN-6-L-08.max-800x600.png)

TF-IDF 기반 대체?

	낮은 TF-IDF 값을 갖는 단어들은 핵심 단어가 아니기 때문에 다른 단어로 대체해도 문맥이 크게 변하지 않는다는 것에 주목한 아이디어

![image](https://d3s0tskafalll9.cloudfront.net/media/images/GN-6-L-09.max-800x600.png)

---

## 방과 후 번역 수업 (3) Back Translation

---

Back Translation?

	 번역 모델에 단일 언어 데이터를 학습시키는 방법

		->  Encoder에는 Source 언어로 된 문장을, Decoder에는 Target 언어로 된 문장을 좀 더 훈련

[Back Translation 정리](https://dev-sngwn.github.io/2020-01-07-back-translation/)

Back Translation의 특징

	- 데이터 수에 무관하게 효과적
	- 가장 좋은 성능을 보인 생성 기법
		- noise beam search

---

## 방과 후 번역 수업 (4) Random Noise Injection

---

Random Noise Injection?

	문장에 노이즈를 주는 것도 괜찮은 Augmentation 기법이 될 수 있음

오타 노이즈 추가

	타이핑을 할 때 주변 키가 눌려 발생하는 오타는 굉장히 자연스러움

		-> 오타 노이즈를 추가하는 것

오타 노이즈 추가 ex)

	올 때 아이스크림 사와 -> 놀 때 아이스크림 사와

		-> QWERTY 키보드 상에서 키의 거리를 기반으로 노이즈를 추가하는 방법

공백 노이즈 추가?

	문장의 일부 단어를 공백 토큰으로 치환

		-> _ 토큰을 활용

![image](https://d3s0tskafalll9.cloudfront.net/media/images/GN-6-L-11.max-800x600.png)

공백 노이즈 추가 장점

	- 학습의 과적합을 방지하는 데에 좋은 효과를 줌

랜덤 유의어 추가?

	주어진 문장에서 불용어(Stop word)가 아닌 단어를 랜덤하게 뽑은 후, 해당 단어와 유사한 단어를 골라 문장에 아무렇게나 삽입하는 방식

![image](https://d3s0tskafalll9.cloudfront.net/media/images/GN-6-L-12.max-800x600.png)

랜덤 유의어 추가의 장점

	- 원본 단어가 손실되지 않는다는 것
	- 모델의 Embedding 층을 더 견고하게 만들어 줄 수 있음

---

## 채점은 어떻게?

---

BLEU(Bilingual Evaluation Understudy) Score?

	'기계가 실제 번역을 얼마나 잘 재현했는가?' 를 평가하는 지표

[BLEU Score](https://donghwa-kim.github.io/BLEU.html)

[위키독스: BLEU Score](https://wikidocs.net/31695)

BLEU Score의 특징

	3가지 문제점을 방지한 N-Gram 지표라고 볼 수 있음

		- 같은 단어가 반복되어 높은 점수를 받는 것을 지양
		- 단어를 잘 재현하더라도 어순이 엉망인 번역을 지양
		- 지나치게 짧은 번역이 높은 점수를 받는 것을 지양

---

## 실례지만, 어디 챗씨입니까? (1) 챗봇과 번역기

---

챗봇의 종류

	- 검색기반 모델 
		- Closed Domain
		- 특정한 목표만을 수행하기 때문에 인풋과 아웃풋이 다소 제한적
	- 생성 모델
		- Open Domain
		- 유저는 아무렇게나 대화를 할 수 있음

번역기를 챗봇으로 사용하는 게 가능한 이유?

	- Encoder
		- Source 문장을 읽고 이해한 내용을 추상적인 고차원 문맥 벡터로 압축
	- Decoder
		- Encoder가 만든 문맥 벡터를 바탕으로 Target 문장을 생성

		-> Source 언어의 Embedding 공간 속 문장을 Target 언어의 Embedding 공간으로 매핑

![image](https://d3s0tskafalll9.cloudfront.net/media/images/GN-6-L-14.max-800x600.png)

	-> 사과는 무슨 색이야? 에 대한 답변도 빨간색입니다 에 수렴하기 때문!!

---

## 실례지만, 어디 챗씨입니까? (2) 좋은 챗봇이 되려면

---

좋은 챗봇이 되려면 신경 써야 할 부분

1. 200ms

	200ms는 대화가 자연스럽게 느껴지는 답변의 공백 마지노선

2. 시공간을 담은 질문

	오늘 무슨 요일이야? 와 같은 질문 

		-> 특정 시공간에 의해 결정되는 질문은 단순한 학습으로 답변 X

3. 페르소나

	학습에는 주로 많은 사람들의 채팅 데이터를 모아서 사용할 수밖에 없기 때문에 모델이 대답의 일관성을 갖는다는 것은 굉장히 도전적

		-> 이때의 일관성을 모델의 인격

			-> 페르소나라고 칭함

4. 대화의 일관성

	기존의 Source 질문을 Target 답변으로 매핑하는 훈련법

		-> 정답을 맞히게끔 학습하기 때문에 문제가 발생

![image](https://d3s0tskafalll9.cloudfront.net/media/original_images/GN-6-L-15.png)

	-> 강화학습을 이용한 방법으로 극복 가능

---

## 실례지만, 어디 챗씨입니까? (3) 대표적인 챗봇

---

멋진 성능을 보여주고 있는 챗봇?

	딥러닝 기반 챗봇

		-> 엄청난 규모의 데이터를 엄청난 크기의 모델에 학습시킨...

Meena?

	구글이 만든 챗봇

		-> GPT-2보다 2배가량 큰 모델을 9배 더 많은 데이터로 학습

	- 모델 구조
		- Evolved Transformer 사용
	- 자체적인 대화 평가 지표
		- SSA 제안

[무슨 대화든 할 수 있는 에이전트를 향하여](https://brunch.co.kr/@synabreu/35)

Blender?

	Facebook의 챗봇

	- 모델에 페르소나를 부여하고자 하는 시도
	- 자체적인 평가 지표
		- ACUTE-Eval 제안

[블렌더(Blender) - Facebook AI의 오픈 도메인 챗봇](https://littlefoxdiary.tistory.com/39)
