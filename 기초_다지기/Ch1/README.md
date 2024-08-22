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

drop_duplicates 옵션

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

이상치 대처 방법
1. 이상치 제거
2. 다른 값으로 대체 ex) 최댓값, 최솟값 설정후 데이터 범위 제한
3. 다른 데이터를 활용하여 예측 모델을 만들어 예측값을 활용
4. binning(구간화)을 통해 수치형 데이터를 범주형으로 바꿈

---

### Z-score method

---

1. 데이터에서 평균을 빼고 절대값 취함
2. 위 값을 표준편차로 나눔
3. 값이 Z보다 큰 데이터의 인덱스 추출

'''

def outlier(df,col,z):
	
	return df[abs(df[col]-np.mean(df[col]))/np.std(df[col]>z)].index

'''

---

Z-score 단점 2개
- 정규분포 가정: z-score는 데이터가 정규 분포를 따른다는 가정을 전제로 합니다. 만약 데이터가 정규 분포를 따르지 않는다면 정확한 결과를 얻기 어려울 수 있습니다.
- 이상치 분포에 민감: 이상치가 많은 경우, 평균과 표준 편차에 영향을 미쳐 z-score가 과도하게 크거나 작아질 수 있습니다.

---

### IQR method

---

IQR?

제 3사분위에서 제 1사분위 값을 뺀 값 -> 약  데이터의 50%

판단 여부 -> Q3+1.5*IQR > data  > Q1-1.5*IQR

	ex)
	Q3,Q1 = np.percentile(data, [75,25]) -> 75% Q3 설정 / 25% Q1 설정
	data[(Q1-1.5*IQR > data)|(Q3+1.5*IQR < data)]

---

## 정규화(Normalization)

---

컬럼간 범위가 크게 차이 나는 경우 전처리 과정에서 정규화 함

- 표준화(Standardization)
- Min-Max Scaling

---

### Standardization(표준화)

---

- 데이터의 평균은 0, 분산은 1로 변환
- 데이터가 가우시안 분포를 따를  경우 유용함

---

	- data-data.mean()/data.std()
	- '''
	  from sklean.preprocessing import StandardScaler
	  scaler = StandardScaler
	  scaler.fit_transform(data)

	  '''

---

### Min-Max-Scaling

---

- 데이터의 최솟값은 0, 최댓값은 1로 변환
- 피처의 범위가 다를때 주로 사용
- 확률 분포를 모를때 유용함

---

	- (data-data.min())/(x.max()-x.min())
	- '''

	  from sklearn.preprocessing import MinMaxScaler
	  scaler = MinMaxScaler()
	  scaler.fit_transform(data)

	  '''
---

## 원-핫 인코딩(One-Hot Encoding)

---

카테고리별 이진 특성을 만들어 해당하는 특성만 1, 나머지는 0으로 만드는 방법

---

pandas에서 get_dummies 함수를 통해 손쉽게 원-핫 인코딩 가능

	pd.get_dummies(trade['국가명'])

---

## 구간화(Binning)

---

연속적인 데이터를 구간을 나눠 분석할 때 사용하는 방법

---

- cut (pd.cut(data,bins=구간))
		데이터와 구간을 입력하는 데이터를 구간별로 나눠줌
		
		bins 옵션에 구간이 아닌 정수를 입력하면 정수 만큼 구간을 나눠줌
- q cut (pd.qcut(data, q=정수))
		데이터의 분포를 비슷한 크기의 그룹으로 나눠줌
