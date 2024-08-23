# 학습내용 

---

- Matplotlib, Seaborn을 이용해 그래프 그리기
- 막대그래프, 선그래프, 산점도 히스토그램
- 시계열 데이터 시각화
- Heatmap

---


파이썬으로 그래프를 그린다는 건?

	도화지를 펼치고 축을 그리고 그 안에 데이터를 그리는 작업

---

과정

	1. 축 그리기
	'''

	fig = plt.figure() -> 그래프 객체 생성
	ax1 = fig.add_subplot(1,1,1) -> add_subplot 메서드를 이용해 축을 그려줌

	'''	

	figure 옵션 - figsize(a,b) -> 가로를 a, 세로를 b로 설정

	add_subplot (2,2,4) -> 2,2,4라는 인자는 최대 2행 2열로 그래프를 생성 할 수 있다는 의미이며, 4는 4번째 위치를 의미 한다.

	2. 그래프 그리기(bar 그래프 기준)

	ax1. bar(x축 data, y축 data)

	3. 그래프 요소 추가

	'''

	plt.xlabel('x축 data 이름')
	plt.ylabel('y축 data 이름')
	plt.title("타이틀")

	'''

	4. 보여주기

	plt.show()

---

os.getenv()?

	환경 변수를 가져오는 함수!

---

pandas Series 데이터 활용

	pandas Series -> 선 그래프를 그리기에 최적의 자료 구조

---

plt.xlim() / ylim() ??

	x축이나 y축의 범위를 지정

---

주석 달기

	annotate() 메서드 이용 
	

	ex)

	'''
	
	important_data = [(datetime(2019, 6, 3), "Low Price"), (datetime(2020, 2, 19), "Peak Price")]
	for d, label in important_data:
	    ax.annotate(label, xy=(d, price.asof(d)+10), # 주석을 달 좌표(x,y)
        	        xytext=(d, price.asof(d)+100), # 주석 텍스트가 위치할 좌표(x,y)
	                arrowprops=dict(facecolor='red')) # 화살표 추가 및 색 설정

	'''

---

그리드(격자눈금)

	grid() 메서드 사용

---

plt.plot()?

	가장 최근의 figure 객체와 서브플롯을 그림

	-- 만약 서브플롯이 없다면 새로운 서브플롯 생성

	인자 -> x data, y data, 마커, 색상 등

	ex)

		plt.plot(x, np.cos(x), '--', color='black') 

---

subplot()

	서브플롯 추가

---

plot 메서드 인자 종류

	- label: 그래프의 범례 이름
	- ax: 그래프를 그릴 matplotlib의 서브플롯 객체
	- style: matplotlib에 전달할 'ko--'같은 스타일의 문자열
	- alpha: 투명도 (0 ~1)
	- kind: 그래프의 종류: line, bar, barh, kde
	- logy: Y축에 대한 로그 스케일
	- use_index: 객체의 색인을 눈금 이름으로 사용할지의 여부
	- rot: 눈금 이름을 로테이션(0 ~ 360)
	- xticks, yticks: x축, y축으로 사용할 값
	- xlim, ylim: x축, y축 한계
	- grid: 축의 그리드 표시할지 여부

---

data가 DataFrame일 경우 plot 메서드 인자 종류

	- subplots: 각 DataFrame의 칼럼(column)을 독립된 서브플롯에 그립니다.
	- sharex: subplots=True면 같은 X축을 공유하고 축의 범위와 눈금을 연결합니다.
	- sharey: subplots=True면 같은 Y축을 공유합니다.
	- figsize: 그래프의 크기를 지정합니다. (튜플)
	- title: 그래프의 제목을 지정합니다. (문자열)
	- sort_columns: 칼럼을 알파벳 순서로 그립니다.

---

df['a'].value_count() -> a의 카테고리별 개수

---

## 범주형 데이터

---

### 막대그래프(bar graph)

---

pandas, matplot를 활용한 방법

	데이터 가공

		x축 -> series / list

		y축 -> list

---

df['a'].groupby(df['b']) -> b를 기준으로 a를 그룹별로 나눠줌

---

seaborn, matplotlib를 활용한 방법

	sns.barplot(data=df,x='a',y='b')

	matplot으로 figsize, title 등 다양한 옵션을 넣을 수 있음

---

### violin plot

---

	sns.violinplot(data=df,x='a',y='b')

---

palette 옵션으로 더 예쁜 색상을 사용 가능

---

### catplot

---

	sns.catplot(x="a,y="b",jitter=False,data=df)

---

## 수치형 데이터 

---

### 산점도(scatter plot)

---

hue 인자

	sns.scatterplot(data=df, x='a', y='b', hue='c')	

	c에 따른 a와 b의 관계를 시각화 가능함

---

### 선 그래프(line graph)

---

	plot의 기본 그래프

	seaborn 이용

		sns.lineplot(x=x, y=y))

---

### 히스토그램

---

용어

	x 축 -> 계급: 변수의 구간 bin(or bucket)

	y 축 -> 도수: 빈도수, frequency

	전체 총량: n

---

	'''

	fig = plt.figure()
	ax1 = fig.add_subplot(1,1,1)
	ax1.hist(data, bins=50, density=False) -> bins: 구간을 지정한 만큼 나눔

	'''

	plt.hist(df[a],bins=50)

	sns.histplot(df['a'], label="a")

---

### 밀도 그래프

---

밀도 그래프? 

	연속된 확률분포를 나타냄

	kde 커널 밀도 추정 그래프

	kind='kde'

---

## 시계열 데이터 시각화

---

	barplot, pointplot, lineplot, histplot으로 시각화 가능 

---

## Heatmap

---

Heatmap ?

	방대한 양의 데이터롸 현상을 수치에 따른 색상으로 나타낸 것

	2차원으로 시각화하여 표현됨

---

pivot ?

	어떤 축, 기준으로 바꾸다라는 의미

	df.pivot(index='a', columns='b', values='c') -> c를 a와 b를 기준으로 바꿈(pivot)

---

	sns.heatmap(pivot)

---

Heatmap 인자들 
	
	- linewidths -> 선을 나타내고 선의 굵기 조정
	- annot -> 셀의 값 표기
	- fmt -> 정수나 소숫점으로 표기
	- cmap -> 색상 지정

