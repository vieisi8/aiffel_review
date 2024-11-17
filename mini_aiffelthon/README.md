# mini aiffelthon

---

- 주제
- 데이터
- CV Task
- NLP Task

---

## 주제

---

주제?

	- 칼로리 췍(CalorieCheck)
		- 가공식품의 이미지를 모델이 인식해 
			- 가공식품 이름
			- 가공식품의 설명
			- 가공식품의 영양성분을 확인할 수 있음
		- 장바구니 형태를 구현해 가공식품의 총 영양성분 표시

사용되는 모델

	CV 모델 + NLP 모델

		-> 멀티모달

---

## 데이터

---

사용되는 데이터?

[AI Hub 상품 이미지 데이터셋](https://aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=realm&dataSetSn=64)

	-> CV 모델의 학습 데이터

[식품영양성분 데이터베이스](https://various.foodsafetykorea.go.kr/nutrient/general/down/historyList.do)의 가공식품DB

	-> RAG의 검색 데이터로 활용

---

## CV Task

---




---

## NLP Task

---

기능

	1. 음료 이름을 rag기법을 통해 해당 음료의 맛이나 설명을 짧은 글로 생성
 	2. 음료 이름을 통해 음료 영양성분 DB에서 해당 음료 영양성분 검색후 Front-end로 반환
