# 학습 내용

---

- STEP 1. NSMC 데이터 분석 및 Huggingface dataset 구성
- STEP 2. klue/bert-base model 및 tokenizer 불러오기
- STEP 3. 위에서 불러온 tokenizer으로 데이터셋을 전처리하고, model 학습 진행해 보기
- STEP 4. Fine-tuning을 통하여 모델 성능(accuarcy) 향상시키기
- STEP 5. Bucketing을 적용하여 학습시키고, STEP 4의 결과와의 비교

---

## STEP 5. Bucketing을 적용하여 학습시키고, STEP 4의 결과와의 비교

---

아래 링크를 바탕으로 bucketing과 dynamic padding이 무엇인지 알아보고, 이들을 적용하여 model을 학습

- [Data Collator](https://huggingface.co/docs/transformers/v4.30.0/en/main_classes/data_collator)
- [Trainer.TrainingArguments](https://huggingface.co/docs/transformers/main_classes/trainer#transformers.TrainingArguments) 의 group_by_length


	STEP 4에 학습한 결과와 bucketing을 적용하여 학습시킨 결과를 비교
