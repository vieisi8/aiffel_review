# 학습 내용

---

- STEP 1. 형태소 분석기를 이용하여 품사가 명사인 경우 해당 단어를 추출
- STEP 2. 추출된 결과로 embedding model 생성
- STEP 3. target, attribute 단어 셋 생성
	- 모든 장르를 사용해 attribute 셋 구성
	- attribute 셋 생성시 중복 제거
- STEP 4. WEAT score 계산과 시각화
