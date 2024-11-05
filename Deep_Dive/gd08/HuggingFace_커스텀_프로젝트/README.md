# 학습 내용

---

- GLUE dataset과 Huggingface
	- NLP를 대표하는 GLUE task 학습
- 커스텀 프로젝트 제작 (1) Dataset
	- 데이터를 task에 맞게 가공
- 커스텀 프로젝트 제작 (2) Tokenizer와 Model
	- Custom project를 위한 모델과 tokenizer load
- 커스텀 프로젝트 제작 (3) Train/Evaluation과 Test
	- Keras와 Huggingface 두 가지 방식으로 훈련

---

## GLUE dataset과 Huggingface

---

GLUE Benchmark Dataset?

	총 11가지 데이터셋 존재

	- CoLA : 문법에 맞는 문장인지 판단
	- MNLI : 두 문장의 관계 판단(entailment, contradiction, neutral)
	- MNLI-MM : 두 문장이 안 맞는지 판단
	- MRPC : 두 문장의 유사도 평가
	- SST-2 : 감정분석
	- STS-B : 두 문장의 유사도 평가
	- QQP : 두 질문의 유사도 평가
	- QNLI : 질문과 paragraph 내 한 문장이 함의 관계(entailment)인지 판단
	- RTE : 두 문장의 관계 판단(entailment, not_entailment)
	- WNLI : 원문장과 대명사로 치환한 문장 사이의 함의 관계 판단
	- Diagnostic Main : 자연어 추론 문제를 통한 문장 이해도 평가

		-> GLUE 홈페이지에는 위 11가지 task에 대한 상세한 설명, 그리고 Leaderboard를 운영하고 있음

---

## 커스텀 프로젝트 제작 (1) Datasets

---

GLUE MRPC 데이터셋 load

1. Huggingface dataset에서 불러오는 방법

	'''

	import datasets
	from datasets import load_dataset

	huggingface_mrpc_dataset = load_dataset('glue', 'mrpc')
	print(huggingface_mrpc_dataset)

	'''

	결과:   DatasetDict({
		    train: Dataset({
		        features: ['sentence1', 'sentence2', 'label', 'idx'],
		        num_rows: 3668
		    })
		    validation: Dataset({
		        features: ['sentence1', 'sentence2', 'label', 'idx'],
		        num_rows: 408
		    })
		    test: Dataset({
		        features: ['sentence1', 'sentence2', 'label', 'idx'],
		        num_rows: 1725
		    })
		})

2. 커스텀 데이터셋 생성

tensorflow_datasets에서 load

	'''

	import tensorflow_datasets as tfds
	from datasets import Dataset

	tf_dataset, tf_dataset_info = tfds.load('glue/mrpc', with_info=True)

	'''

이중 딕셔너리 내부에 데이터를 리스트 형태로 변환

	'''

	# Tensorflow dataset 구조를 python dict 형식으로 변경
	# Dataset이 train, validation, test로 나뉘도록 구성
	train_dataset = tfds.as_dataframe(tf_dataset['train'], tf_dataset_info)
	val_dataset = tfds.as_dataframe(tf_dataset['validation'], tf_dataset_info)
	test_dataset = tfds.as_dataframe(tf_dataset['test'], tf_dataset_info)

	# dataframe 데이터를 dict 내부에 list로 변경
	train_dataset = train_dataset.to_dict('list')
	val_dataset = val_dataset.to_dict('list')
	test_dataset = test_dataset.to_dict('list')

	# Huggingface dataset
	tf_train_dataset = Dataset.from_dict(train_dataset)
	tf_val_dataset = Dataset.from_dict(val_dataset)
	tf_test_dataset = Dataset.from_dict(test_dataset)

	'''

---

## 커스텀 프로젝트 제작 (2) Tokenizer와 Model

---

Huggingface에서 Model, Tokenizer load

	'''

	import transformers
	from transformers import AutoTokenizer, AutoModelForSequenceClassification

	huggingface_tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
	huggingface_model = AutoModelForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels = 2)

	'''

	AutoTokenizer와 AutoModel

		-> Huggingface에서 지원하는 Auto Class

Auto class?

	from_pretrained 메소드를 이용해 pretrained model의 경로 혹은 이름만 안다면 자동으로 생성하는 방법

		-> model의 경우 AutoModel을 그대로 사용하기보다 특정 task를 지정하는 방식인 AutoModelForSequenceClassification을 사용하는걸 권장

Tokenizer의 매개변수

	- truncation
		- 특정 문장이 길어 모델을 다루기 힘들어 질 수 있으므로 짧게 자르는 것을 의미
	- return_token_type_ids
		- 문장이 한개이상일 때 나뉘는걸 보여줌

데이터셋 토크나이징 함수 정의

	'''

	def transform(data):
	    return huggingface_tokenizer(
	        data['sentence1'],
	        data['sentence2'],
	        truncation = True,
	        padding = 'max_length',
	        return_token_type_ids = False,
	        )

	'''

토크나이징 적용 및 데이터셋 분리

	'''

	hf_dataset = huggingface_mrpc_dataset.map(transform, batched=True)

	# train & validation & test split
	hf_train_dataset = hf_dataset['train']
	hf_val_dataset = hf_dataset['validation']
	hf_test_dataset = hf_dataset['test']

	'''

		-> map을 사용해 토크나이징을 진행하기 때문에 batch를 적용해야 되므로 batched=True

tfds의 MRPC로 만든 커스텀 데이터셋 토크나이징 적용

	'''

	def transform_tf(batch):
	    sentence1 = [s.decode('utf-8') for s in batch['sentence1']]
	    sentence2 = [s.decode('utf-8') for s in batch['sentence2']]
	    return huggingface_tokenizer(
	        sentence1,
	        sentence2,
	        truncation=True,
	        padding='max_length',
	        return_token_type_ids=False,
	    )

	# 토큰화 및 패딩을 적용
	tf_train_dataset = tf_train_dataset.map(transform_tf, batched=True)
	tf_val_dataset = tf_val_dataset.map(transform_tf, batched=True)
	tf_test_dataset = tf_test_dataset.map(transform_tf, batched=True)

	'''

---

## 커스텀 프로젝트 제작 (3) Train/Evaluation과 Test

---

TrainingArguments 설정

	'''

	import os
	import numpy as np
	from transformers import Trainer, TrainingArguments

	output_dir = os.getenv('HOME')+'/aiffel/transformers'

	training_arguments = TrainingArguments(
	    output_dir,                                         # output이 저장될 경로
	    evaluation_strategy="epoch",           #evaluation하는 빈도
	    learning_rate = 2e-5,                         #learning_rate
	    per_device_train_batch_size = 8,   # 각 device 당 batch size
	    per_device_eval_batch_size = 8,    # evaluation 시에 batch size
	    num_train_epochs = 3,                     # train 시킬 총 epochs
	    weight_decay = 0.01,                        # weight decay
	)

	'''

compute_metrics 메소드?

	task가 classification인지 regression인지에 따라 모델의 출력 형태가 달라짐 

		-> task별로 적합한 출력 형식을 고려해 모델의 성능을 계산하는 방법을 미리 지정해 두는 것

	'''

	from datasets import load_metric
	metric = load_metric('glue', 'mrpc')

	def compute_metrics(eval_pred):    
	    predictions,labels = eval_pred
	    predictions = np.argmax(predictions, axis=1)
	    return metric.compute(predictions=predictions, references = labels)

	'''

train 진행

	Trainer에 model, arguments, train_dataset, eval_dataset, compute_metrics 전달해줌

	'''

	trainer = Trainer(
	    model=huggingface_model,           # 학습시킬 model
	    args=training_arguments,           # TrainingArguments을 통해 설정한 arguments
	    train_dataset=hf_train_dataset,    # training dataset
	    eval_dataset=hf_val_dataset,       # evaluation dataset
	    compute_metrics=compute_metrics,
	)
	trainer.train()

	'''

	결과:   Epoch	Training Loss	Validation Loss	  Accuracy	F1
		1	No log		0.418951	0.838235	0.890728
		2	0.501900	0.451517	0.843137	0.894040
		3	0.322900	0.535722	0.840686	0.888508
 

모델 평가

	'''

	trainer.evaluate(hf_test_dataset)

	'''

	결과:   {'eval_loss': 0.5647583603858948,
		 'eval_accuracy': 0.8342028985507246,
		 'eval_f1': 0.8792229729729729,
		 'eval_runtime': 30.017,
		 'eval_samples_per_second': 57.467,
		 'eval_steps_per_second': 7.196,
		 'epoch': 3.0}
