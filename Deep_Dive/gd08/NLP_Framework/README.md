# 학습 내용

---

- 다양한 NLP Framework의 출현
	- 트랜스포머, 언제까지 구현해서 쓸 건가?
- Huggingface transformers 개요
	- 무엇이 Huggingface를 특별하게 만드나?
- Huggingface transformers (1) Model
	- Huggingface의 가장 핵심인 Model
- Huggingface transformers (2) Tokenizer
	정규표현식은 그만! 내장 Tokenizer 사용
- Huggingface transformers (4) Config
	- 모델을 손쉽게 구성
- Huggingface transformers (5) Trainer
	- 모델 학습을 위한 Trainer 클래스 학습

---

## 다양한 NLP Framework의 출현

---

General Framework for NLP

- AllenNLP

	- 제공자: Allen AI Institute
	- Backend : PyTorch
	- Allen Institute에서 만든 NLP framework
	- Glue dataset의 baseline 프로젝트 Starting Baseline를 제공하기도 함

![image](https://d3s0tskafalll9.cloudfront.net/media/images/GN-9-L-01.max-800x600.png)

	-> ELMO와 같은 pretrained model의 성공을 바탕으로 NLP framework를 완성해 나가려는 AllenNLP의 시도는 이후 많은 아이디어를 제공
		
		-> AllenNLP는 현재는 ELMO 이외에도 BERT 등 다양한 모델의 활용 가능

- Fairseq

	- 제공자 : Facebook AI Research
	- Backend : PyTorch
	- Facebook AI Research의 NLP Framework
	- CNN, LSTM 등 전통적인 모델로부터, 음성인식/합성 등 sequential한 데이터를 다루는 분야를 두루 다루는 다양한 pretrained model을 함께 제공

- Fast.ai

	- 제공자 : fast.ai
	- Backend : PyTorch
	- 빠르게 배우고 쉽게 모델을 구성할 수 있도록 하이레벨 API와 Application 블록까지 손쉽게 사용할 수 있도록 구성
	- NLP 분야 뿐 아니라 다양한 분야로 확장 가능

![image](https://d3s0tskafalll9.cloudfront.net/media/original_images/GN-9-L-02.png)

- tensor2tensor

	- 제공자 : Google Brain
	- Backend : Tensorflow
	- transformer를 중심으로 다양한 태스크와 다양한 모델을 하나의 framework에 통합하려는 시도
	- 2020년도부터는 tensor2tensor의 개발을 중단하고 관련 기능을 trax로 통합이관

 Preprocessing Libraries

	tokenization, tagging, parsing 등 특정 전처리 작업을 위해 설계된 라이브러리에 가까움

- Spacy
- NLTK
- TorchText
- KoNLPy

Transformer-based Framework(NLP Framework)

- Huggingface transformers

	- 제공자 : Huggingface.co
	- Backend : PyTorch and Tensorflow
	- 초기에는 BERT 등 다양한 transformer 기반의 pretrained model을 사용하기 위한 PyTorch 기반의 wrapper 형태로 시작
	- pretrained model 활용을 주로 지원하며, tokenizer 등 전처리 부분도 pretrained model들이 주로 사용하는 Subword tokenizer 기법에 집중되어 있는 특징

---

## Huggingface transformers 개요

---

Why Huggingface?

1. 광범위하고 신속한 NLP 모델 지원

	- 많은 사람들이 최신 NLP 모델들을 더욱 손쉽게 사용하는 것을 목표로 만들기 시작
		- 새로운 논문들이 발표될 때마다, 본인들의 framework에 흡수시키고 있음
	- pretrained model을 제공하고, dataset과 tokenizer를 더욱 쉽게 이용할 수 있도록 framework화
	- Huggingface의 지원 범위가 가장 광범위하고, 최신 논문을 지원하는 속도도 빠름

2. PyTorch와 Tensorflow 모두에서 사용 가능

	- 기본적으로 PyTorch를 기반으로 만들어져있음
		- Tensorflow로도 학습하고 사용할 수 있게끔 계속해서 framework를 확장하고 있는 중

3. 잘 설계된 framework 구조

	- 쉽고 빠르게 어떠한 환경에서도 NLP모델을 사용할 수 있도록 끊임없이 변화
		- 사용하기 쉽고 직관적일뿐더러 모델이나 태스크, 데이터셋이 달라지더라도 동일한 형태로 사용 가능하도록 잘 추상화되고 모듈화된 API 설계가 있기 때문에 가능한 것

Transformers 설치

	pip install transformers

	'''

	from transformers import pipeline

	classifier = pipeline('sentiment-analysis', framework='tf')
	classifier('We are very happy to include pipeline into the transformers repository.')

	'''

Huggingface transformers 설계구조 개요


	1. Task를 정의하고 그에 맞게 dataset을 가공
		- Processors
		- Tokenizer
	2. 적당한 model을 선택하고 이를 만듦 
		- Model
	3. model에 데이터들을 태워서 학습
		- Optimization
		- Trainer
	4. model의 weight와 설정(config)들을 저장 
	5. model의 checkpoint를 배포하거나, evaluation에 사용
		- Config

---

# Huggingface transformers (1) Model

---

model

	PretrainedModel 클래스를 상속받고 있음

		-> 모델을 불러오고 다운로드/저장하는 등의 작업에 활용하는 메소드는 부모 클래스의 것을 동일하게 활용할 수 있게 됨

PretrainedModel 클래스?

	학습된 모델을 불러오고, 다운로드하고, 저장하는 등 모델 전반에 걸쳐 적용되는 메소드들을 가지고 있음

model load 2가지 방식

	1. task에 적합한 모델을 직접 선택하여 import하고, 불러오는 방식

		from_pretrained라는 메소드 사용

		'''

		from transformers import TFBertForPreTraining
		model = TFBertForPreTraining.from_pretrained('bert-base-cased')

		'''

		- pretrained 모델이라면 모델의 이름을 string
		- 직접 학습시킨 모델이라면 config와 모델을 저장한 경로를 string으로 넘겨주면 됨

	2. AutoModel 사용

		모델에 관한 정보를 처음부터 명시하지 않아도 되어 조금 유용하게 사용 가능

		'''

		from transformers import TFAutoModel
		model = TFAutoModel.from_pretrained("bert-base-cased")

		'''

두 가지 방법의 차이

	Pretrain, Downstream Task 등 용도에 따라 모델의 Input이나 Output shape가 다를 수 있음

		- AutoModel을 활용한다면 모델의 상세정보를 확인할 필요 없이 Model ID만으로도 손쉽게 모델 구성이 가능

		- 정확한 용도에 맞게 사용하려면 모델별 상세 안내 페이지를 참고해서 최적의 모델을 선택하는 것이 좋음

---

## Huggingface transformers (2) Tokenizer

---

tokenizer

	transformers는 다양한 tokenizer를 각 모델에 맞추어 이미 구비

Pretrained model 기반의 NLP framework를 사용할 때 가장 중요한 두 가지 클래스

	- Model 
	- Tokenizer

		-> 파라미터 구조가 동일한 Model이라 하더라도 Tokenizer가 다르거나 Tokenizer 내의 Dictionary가 달라지면 사실상 완전히 다른 모델이 됨

tokenizer load

	직접 명시하여 내가 사용할 것을 지정해 주거나, AutoTokenizer를 사용

		-> model을 사용할 때 명시했던 것과 동일한 ID로 tokenizer를 생성해야 함

	'''

	from transformers import BertTokenizer
	tokenizer = BertTokenizer.from_pretrained('bert-base-cased')

	or

	from transformers import AutoTokenizer
	tokenizer = AutoTokenizer.from_pretrained('bert-base-cased')

	'''

tokenizer 자세히 살펴보기

	- tokenize()
		- 토큰 단위로 분할된 문장을 확인
	- batch 단위로 input을 받을 수 있음
	- padding, truncation 등 다양한 옵션 존재
	- 모델이 어떤 프레임워크를 사용하는가(Tensorflow 또는 PyTorch)에 따라 input 타입을 변경 시켜주는 return_tensors 존재

---

## Huggingface transformers (3) Config

---

config?

	모델을 학습시키기 위한 요소들을 명시한 json파일로 되어있음

		->  batch size, learning rate, weight_decay등 train에 필요한 요소들부터 tokenizer에 특수 토큰(special token eg.[MASK])들을 미리 설정하는 등 설정에 관한 전반적인 것들이 명시되어 있음

config 저장 방법

	PretrainedModel을 save_pretrained 메소드 사용

		-> 모델의 체크포인트와 함께 저장되도록 되어있음

config의 사용 방법

	pretrained model의 설정을 변경하고 싶거나 나만의 모델을 학습시킬 때에는 config파일을 직접 불러와야 함

config load

	'''

	from transformers import BertConfig

	config = BertConfig.from_pretrained("bert-base-cased")

	or

	from transformers import AutoConfig

	config = AutoConfig.from_pretrained("bert-base-cased")

	'''

	모델이 이미 생성된 상태일 때

		'''

		model = TFBertForPreTraining.from_pretrained('bert-base-cased')
		config = model.config

		'''

---

## Huggingface transformers (4) Trainer

---

trainer?

	모델을 학습시키기 위한 클래스

		-> TrainingArguments 를 통해 Huggingface 프레임워크에서 제공하는 기능들을 통합적으로 커스터마이징하여 모델을 손쉽게 학습시킬 수 있다는 장점 존재

TrainingArguments인스턴스

	trainer API를 사용하기 위해선 TrainingArguments인스턴스를 생성해야 함

	ex)

		'''

		training_args = TrainingArguments(
		    output_dir='./results',              # output이 저장될 경로
		    num_train_epochs=1,              # train 시킬 총 epochs
		    per_device_train_batch_size=16,  # 각 device 당 batch size
		    per_device_eval_batch_size=64,   # evaluation 시에 batch size
		    warmup_steps=500,                # learning rate scheduler에 따른 warmup_step 설정
		    weight_decay=0.01,                 # weight decay
		    logging_dir='./logs',                 # log가 저장될 경로
		    do_train=True,                        # train 수행여부
		    do_eval=True,                        # eval 수행여부
		    eval_steps=1000,
		    group_by_length=False,
		)

		'''

trainer 사용 방법

	'''

	trainer = Trainer(
	    model,                                                                    # 학습시킬 model
	    args=training_args,                                                # TrainingArguments을 통해 설정한 arguments
	    train_dataset=tokenized_datasets["train"],         # training dataset
	    eval_dataset=tokenized_datasets["validation"], # validation dataset
	    tokenizer=tokenizer,
	)

	# 모델 학습
	trainer.train()

	'''
