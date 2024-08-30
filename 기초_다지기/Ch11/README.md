# 학습 내용

---

- 11-1. 프로젝트 : This is your playground! Leaderboard를 정복해 주세요!

---

## 11-1

---

### 튜닝해볼 수 있는 모델 클래스 인자

---

대표적으로 자주 튜닝하는 lightgbm 라이브러리의 인자는 다음과 같습니다.

	- max_depth : 의사 결정 나무의 깊이, 정수 사용
	- learning_rate : 한 스텝에 이동하는 양을 결정하는 파라미터, 보통 0.0001~0.1 사이의 실수 사용
	- n_estimators : 사용하는 개별 모델의 개수, 보통 50~100 이상의 정수 사용
	- num_leaves : 하나의 LightGBM 트리가 가질 수 있는 최대 잎의 수
	- boosting_type : 부스팅 방식, gbdt, rf 등의 문자열 입력

[lightGBM / XGBoost 파라미터 설명](https://machinelearningkorea.com/2019/09/29/lightgbm-%ED%8C%8C%EB%9D%BC%EB%AF%B8%ED%84%B0/)

[Chapter 4. 분류 - LightGBM](https://injo.tistory.com/48)

---

### 시도해볼 수 있는 방법

---

	- 기존에 있는 데이터의 피처를 모델을 보다 잘 표현할 수 있는 형태로 처리하기 (피처 엔지니어링)
	- LGBMRegressor, XGBRegressor, RandomForestRegressor 세 가지 이상의 다양한 모델에 대해 하이퍼 파라미터 튜닝하기
	- 다양한 하이퍼 파라미터에 대해 그리드 탐색을 시도해서 최적의 조합을 찾아보기
	- Baseline 커널에서 활용했던 블렌딩 방법 활용하기

