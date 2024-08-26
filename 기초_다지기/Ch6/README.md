# 학습 내용

---

- 비지도 학습의 기본적인 개념
- 클러스터링 
	- K-means
	- DBSCAN
- 차원 축소
	- PCA
	- T-SNE

---

# 비지도 학습의 기본적인 개념

---

비지도 학습?

	- 지도학습과 달리 training data로 정답(labewl)이 없는 데이터가 주어지는 학습방법
	- 데이터셋의 특징(feature) 및 패턴을 기반으로 모델 스스로가 판단

---

## 클러스터링 - K-means

---

클러스터링?

	명확한 분류 기준이 없는 상황에서 데이터들을 분석 -> 가까운(유사한) 것들끼리 묶어 주는 작업

---

K-means 알고리즘?

	- k 값이 주어져 있을 때, 주어진 데이터들을 k 개의 클러스터로 묶는 알고리즘
	- 단점 -> 군집의 개수(K 값)를 미리 지정해야 하기 때문에 이를 알거나 예측하기 어려운 경우에는 사용하기 어렵다

---

K-mean 알고리즘의 순서

	1. 원하는 클러스터의 수(K)를 결정합니다.
	2. 무작위로 클러스터의 수와 같은 K개의 중심점(centroid)을 선정합니다. 이들은 각각의 클러스터를 대표합니다.
	3. 나머지 점들과 모든 중심점 간의 유클리드 거리를 계산한 후, 가장 가까운 거리를 가지는 중심점의 클러스터에 속하도록 합니다.
	4. 각 K개의 클러스터의 중심점을 재조정합니다. 특정 클러스터에 속하는 모든 점들의 평균값이 해당 클러스터 다음 iteration의 중심점이 됩니다.(이 중심점은 실제로 존재하는 데이터가 아니어도 상관없습니다.)
	5. 재조정된 중심점을 바탕으로 모든 점들과 새로 조정된 중심점 간의 유클리드 거리를 다시 계산한 후, 가장 가까운 거리를 가지는 클러스터에 해당 점을 재배정합니다.
	6. 4.번과 5.번을 반복 수행합니다. 반복의 횟수는 사용자가 적절히 조절하면 되고, 특정 iteration 이상이 되면 수렴(중심점이 더 이상 바뀌지 않음)하게 됩니다.

---

	'''

	from sklearn.cluster import KMeans
	
	kmeans_cluster = KMeans(n_clusters=5)
	kmeans_cluster.fit(points)

	'''

---

k-means 알고리즘이 잘 동작하지 않는 예시

	- 원형 분포 데이터
	- 초승달 분포 데이터
	- 대각선 방향 데이터

		이유 -> 유클리드 거리가 가까운 데이터끼리 군집이 형성되기 때문에 데이터의 분포에 따라 유클리드 거리가 멀면서 밀접하게 연관되어 있는 데이터들의 군집화를 성공적으로 수행하지 못할 수 있다.

---

## 클러스터링 - DBSCAN

---

DBSCAN?

	가장 널리 알려진 밀도(density) 기반의 군집 알고리즘

---

특징

	- 군집의 개수, 즉 K-means 알고리즘에서의 K 값을 미리 지정할 필요가 없다는 점
	- 조밀하게 몰려 있는 클러스터를 군집화하는 방식 사용 -> 불특정한 형태의 군집도 찾을 수 있음

---

용어 

	- epsilon: 클러스터의 반경
	- minPts: 클러스터를 이루는 개체의 최솟값
	- core point: 반경 epsilon 내에 minPts 개 이상의 점이 존재하는 중심점
	- border point: 군집의 중심이 되지는 못하지만, 군집에 속하는 점
	- noise point: 군집에 포함되지 못하는 점

---

DBSCAN 알고리즘의 순서

	1. 임의의 점 p를 설정하고, p를 포함하여 주어진 클러스터의 반경(elipson) 안에 포함되어 있는 점들의 개수를 세요.
	2. 만일 해당 원에 minPts 개 이상의 점이 포함되어 있으면, 해당 점 p를 core point로 간주하고 원에 포함된 점들을 하나의 클러스터로 묶어요.
	3. 해당 원에 minPts 개 미만의 점이 포함되어 있으면, 일단 pass 합시다.
	4. 모든 점에 대하여 돌아가면서 1~3 번의 과정을 반복하는데, 만일 새로운 점 p'가 core point가 되고 이 점이 기존의 클러스터(p를 core point로 하는)에 속한다면, 두 개의 클러스터는 연결되어 있다고 하며 하나의 클러스터로 묶어줘요.
	5. 모든 점에 대하여 클러스터링 과정을 끝냈는데, 어떤 점을 중심으로 하더라도 클러스터에 속하지 못하는 점이 있으면 이를 noise point로 간주해요. 또한, 특정 군집에는 속하지만 core point가 아닌 점들을 border point라고 칭해요.

---

	'''

	from sklearn.cluster import DBSCAN

	epsilon, minPts = 0.2, 3
	circle_dbscan = DBSCAN(eps=epsilon, min_samples=minPts)
	circle_dbscan.fit(circle_points)
	n_cluster = max(circle_dbscan.labels_)+1

	'''

---

단점

	- 데이터의 수가 적을 때 ->  수행시간: K-means >  DBSCAN / 데이터의 수가 많아질수록 -> 수행시간: DBSCAN의 수행 시간이 급격히 ↑
	- 데이터 분포에 맞는 epsilon과 minPts의 값을 지정

---

## 차원 축소 - PCA

---

차원 축소?

	- 수많은 정보 속에서 우리에게 더 중요한 요소가 무엇인지를 알게 해주는 방법
	- 특징 추출(feature extraction)의 용도

---

PCA?

	- 데이터들의 분산을 최대로 보존, 서로 직교(orthogonal)하는 기저(basis, 분산이 큰 방향벡터의 축)들을 찾아 고차원 공간을 저차원 공간으로 사영(projection)하는 방법
	- 기존의 feature를 선형 결합(linear combination)하는 방식
	- 가장 중요한 기저를 주성분(Principal Component) 방향, 또는 pc축 이라함

---

![Alt text](https://d3s0tskafalll9.cloudfront.net/media/original_images/F-44.3.png)

기저(basis)?

	위 그림에서 보이는 우상향 방향의 긴 화살표와 좌상향 방향의 짧은 화살표 방향을 좌표축(벡터)으로 만듦 -> 이것의 모음

	특징
		가장 분산이 긴 축을 첫 기저, 그 기저에 직교하는 축 중 가장 분산이 큰 값을 다음 기저 ∴  차원의 수를 최대로 줄이면서 데이터 분포의 분산을 그대로 유지 -> 차원 축소

![Alt text](https://d3s0tskafalll9.cloudfront.net/media/original_images/F-44-d-reduction.jpg)

차원 축소?

	X-Y, Y-Z 좌표축에 사영(projection) 했다는 것 -> 각각 Z, X 좌표축을 무시했다는 뜻
	
	특징

		무시한 데이터만큼의 정보손실

---

진행 방식

	차원축소를 시도하되, 주어진 좌표축 방향이 아니라, 가장 분산이 길게 나오는 기저(basis) 방향을 찾아서 그 방향의 기저만 남기고, 덜 중요한 기저 방향을 삭제하는 방식으로 진행

---

	'''

	from sklearn.preprocessing import StandardScaler
	from sklearn.decomposition import PCA

	scaler = StandardScaler() 
	train_X_ = scaler.fit_transform(train_X) # 불러온 데이터에 대한 정규화 -> 각 column의 range of value가 전부 다르기 때문에 정규화를 진행해 주어야 합니다.
	train_df = pd.DataFrame(train_X_, columns=cancer['feature_names'])
	pca = PCA(n_components=2) # 주성분의 수를 2개, 즉 기저가 되는 방향벡터를 2개로 하는 PCA 알고리즘 수행
	pc = pca.fit_transform(train_df)

	'''

---

## 차원 축소 - T-SNE

---

![Alt text](https://d3s0tskafalll9.cloudfront.net/media/original_images/F-32-6.png)

T-SNE?

	- 시각화에 많이 쓰이는 알고리즘
	- 방사형적, 비선형적 데이터에서는 많은 정보량을 담기 위한 주성분(Principal Component)으로 잡을 선형적인 축을 찾기 어려움 -> 멀리 있는 데이터가 가까이 있도록 차원축소가 이루어짐 -> 시각화하여 데이터를 이해한다는 목적에 맞지 않음
	- 기존 차원의 공간에서 가까운 점들은, 차원축소된 공간에서도 여전히 가깝게 유지되는 것을 목표

---

	'''

	from sklearn.manifold import TSNE

	n_dimension = 2
	tsne = TSNE(n_components=n_dimension)
	tsne_results = tsne.fit_transform(data_subset)

	'''

---

### PCA T-SNE 차이점

---

	- PCA: 정보 손실을 최소화하려는 관점
	- T-SNE:시각화에만 유리


