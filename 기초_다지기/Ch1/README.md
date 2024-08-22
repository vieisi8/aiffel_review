# 학습 내용

---

- 결측치(Missing Data)
- 중복된 데이터
- 이상치(Outlier)
- 정규화(Normalization)
- 원-핫 인코딩(One-Hot Encoding)
- 구간화(Binning)

---

## 결측치

---

-- trade.csv 파일 이용(관세청 무역 통계 )
-- trade = trade.csv 내용

---

결측치 처리 방법

1. 결측치 제거
2. 어떤 닶으로 대체

---

결측치 개수 파악 방법

1. len(trade)-trade.count
2. trade.isnull().sum()

---

trade의 기타사항을 보면 전부 결측치 이므로 컬럼 삭제
	trade=trade.drop('기타사항',axis=1)
		-- axis=0 -> 행을 의미
		-- axis=1 -> 열을 의미

---

결측치가 하나라도 존재하는 index를 찾을때
	trade[trade.isnull().any(axis=1)]

---

dropna -> 결측치를 삭제하는 메서드

옵션

subset -> 특정 컬럼 선택

how -> any(하나라도 결측치인 경우) / all(전부가 결측치인 경우)

inplace -> True(DataFrame에 바로 적용 O) / False(DataFrame에 바로 적용 X)

---

수치형 데이터 보안
1. 특정 값을 지정
2. 평균 중앙값 등으로 대체
3. 다른 데이터를 이용해 예측값으로 대체
4. (시계절) 앞 뒤 데이터를 통해 결측치 대체 -> 앞 뒤 데이터의 평균

---

pandas loc?

label이나 boolean array로 인덱싱하는 방법

컬럼명 / 특정 조건식을 인자로 받음

	ex) trade.loc[191,'수출금액'] -> index가 191이고, 컬럼명이 수출금액인 항목을 인덱싱

---

범주형 데이터 보안
1. 특정 값을 지정
2. 최빈값 등으로 대체
3. 다른 데이터를 이용해 예측값으로 대체
4. (시계절) 앞 뒤 데이터를 통해 대체

---

## 중복된 데이터

---

DataFrame.duplicated() -> 중복 여부를 True / False로 반환

trade[trade.duplicated()] -> trade에서 중복된 항목을 찾을 수 있음

trade.drop_duplicates(inplace=True) -> 중복된 데이터 삭제

옵션

subset -> 특정 컬럼 선택

keep -> first(중복된 데이터중 먼저 들어온 것을 남김) / last(중복된 데이터중 나중에 들어온 것을 남김)

---

## 이상치

---

이상치란?

대부분 값의 범위에서 벗아나 극단적으로 크거나 작은 값

---

이상치 판별

- Z-score
- IOR(사분위 범위수)

---



