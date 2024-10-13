# 학습 내용

---

- SentencePiece 설치
- SentencePiece 모델 학습
- Tokenizer 함수 작성
- 네이버 영화리뷰 감정 분석 문제에 SentencePiece 적용
	- SentencePiece 모델 학습
	- sp_tokenize() 메소드 구현
	- 네이버 영화리뷰 감정 분석 모델 구현 및 학습
	- KoNLPy 형태소 분석기를 사용한 모델과 성능 비교
	- SentencePiece 모델의 model_type, vocab_size 등을 변경해 가면서 성능 개선 여부 확인

---

## SentencePiece 설치

---

SentencePiece??

	Google에서 제공하는 오픈소스 기반 Sentence Tokenizer/Detokenizer

		-> BPE와 unigram 2가지 subword 토크나이징 모델 중 하나를 선택해서 사용할 수 있도록 패키징한 것

	상세한 내용 -> [google/sentencepiece](https://github.com/google/sentencepiece)

SentencePiece의 최근 위상

	최근 pretrained model들이 거의 대부분 SentencePiece를 tokenizer로 채용

		-> 사실상 표준의 역할

설치 방법

	pip install sentencepiece

---

## SentencePiece 모델 학습

---

SentencePiece 모델 학습 과정

	'''

	import sentencepiece as spm
	import os
	temp_file = os.getenv('HOME')+'/aiffel/sp_tokenizer/data/korean-english-park.train.ko.temp'

	vocab_size = 8000

	with open(temp_file, 'w') as f:
	    for row in filtered_corpus:   # 이전에 나왔던 정제했던 corpus를 활용해서 진행해야 합니다.
	        f.write(str(row) + '\n')

	spm.SentencePieceTrainer.Train(
	    '--input={} --model_prefix=korean_spm --vocab_size={}'.format(temp_file, vocab_size)    
	)
	#위 Train에서  --model_type = unigram이 디폴트 적용되어 있습니다. --model_type = bpe로 옵션을 주어 변경할 수 있습니다.

	!ls -l korean_spm*

	'''

		-> korean_spm.model 파일과 korean_spm.vocab vocabulary 파일 생성

SentencePiece 모델 활용

	'''

	s = spm.SentencePieceProcessor()
	s.Load('korean_spm.model')

	# SentencePiece를 활용한 sentence -> encoding
	tokensIDs = s.EncodeAsIds('아버지가방에들어가신다.')
	print(tokensIDs)

	# SentencePiece를 활용한 sentence -> encoded pieces
	print(s.SampleEncodeAsPieces('아버지가방에들어가신다.',1, 0.0))

	# SentencePiece를 활용한 encoding -> sentence 복원
	print(s.DecodeIds(tokensIDs))

	'''

	결과:   [1398, 10, 382, 15, 1319, 10, 133, 17, 4]
		['▁아버지', '가', '방', '에', '들어', '가', '신', '다', '.']
		아버지가방에들어가신다.

---

## Tokenizer 함수 작성

---

만족해야 하는 조건

	1. 매개변수로 토큰화된 문장의 list를 전달하는 대신 온전한 문장의 list 를 전달해야 함

	2. 생성된 vocab 파일을 읽어와 { <word> : <idx> } 형태를 가지는 word_index 사전, { <idx> : <word>} 형태를 가지는 index_word 사전을 생성, 함께 반환

	3. 리턴값인 tensor 는 앞의 함수와 동일하게 토큰화한 후 Encoding된 문장

		-> 바로 학습에 사용할 수 있게 Padding 사용

코드

	'''

	def sp_tokenize(s, corpus): 

	    tensor = []

	    for sen in corpus:
	        tensor.append(s.EncodeAsIds(sen))

	    with open("./korean_spm.vocab", 'r') as f:
	        vocab = f.readlines()

	    word_index = {}
	    index_word = {}

	    for idx, line in enumerate(vocab):
	        word = line.split("\t")[0]

	        word_index.update({word:idx})
	        index_word.update({idx:word})

	    tensor = tf.keras.preprocessing.sequence.pad_sequences(tensor, padding='post')

	    return tensor, word_index, index_word

	'''
