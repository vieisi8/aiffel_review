# 학습 내용

---

- Big Query에 정형 데이터 및 비정형 데이터 upload
	- 이미지 데이터를 Google Cloud Storage에 올리고 Big Query로 관리하는 방법 학습

---

## Big Query에 정형 데이터 및 비정형 데이터 upload

---

첫 번째 실습 

	 BigQuery에 있는 데이터 Array를 저장해서 사용

	사용할 데이터셋

		-> MNIST 데이터셋

			-> png데이터로 이루어져있지 않고 ndarray구조로 이루어져있어서 BigQuery에 넣을 수 있음

[MLOps 2일차 첫번째 실습](https://colab.research.google.com/drive/1qWpqxq_NxFpFKxvIkw_Ohz4qdao7jVyP#scrollTo=0jlyxMCaN6BI)

두 번째 실습

	BigQuery와 Google Cloud Storage를 연동해서 사용하는 기법

Google Cloud Storage를 연동하는 이유?

	- 데이터를 이미지 그대로 저장이 가능하기 때문에 변환중 데이터 손상이 발생할 우려 X
	- 대용량 데이터를 처리할 때 계속 다운받고 불러오는 것은 비효율적이기에 이 저장방식이 효과적일 수 있음
	- 이미지 뿐만 아니라 모델까지 적재해서 사용이 가능

[MLOps 2일차 2번째 실습](https://colab.research.google.com/drive/1OLKbFOiY3a7XY_ejjNDtSYKirhXr0R0K#scrollTo=W6IdaTE-l-IX)

프로젝트

1. 텍스트 데이터셋을 Google BigQuery로 저장

[MLOps 2일차 첫번째 프로젝트](https://colab.research.google.com/drive/1lnx291Al2PZYc_298CrevYkYmcbAjLes#scrollTo=YeWKq6SnN6GF)

2. 이미지 데이터셋을 불러 와서 간단한 모델로 학습

[MLOps 2일차 2번째 프로젝트](https://colab.research.google.com/drive/13F7ZJIZsemiddKJtFneaeo_whyACsHKq#scrollTo=O1Ka-SDBl-Ks)
