# 학습내용

---

- Loss & Metirc
- Confusion Matrix & Precision/Recall/F-score
- Threshold 변화에 따른 모델 성능 
- Precision-Recall 커브
- AUC(Area Under Curve)
- Roc 커브
- 다양한 머신러닝 모델의 평가척도

---

## Loss

---

	모델 학습시 학습데이터를 바탕으로 계산, 모델의 파라미터 업데이트에 활용되는 함수

---

## Metric

---

	모델 학습 종료 후 테스트데이터를 바탕으로 계산, 학습된 모델의 성능을 평가하는데 활용되는 함수

---

Q.  **Accuracy가 학습단계에 좋은 Loss가 될 수 없는지?**

	A. loss는 연속적 값으로 평가해야 하는데 acc는 맞냐.틀리냐로 평가하기 때문에 연속적 값으로 나타낼 수 없고, 계산 결과가 답에 얼마나 근접한지 평가하기 어려움. ⇒ Continuous vs Discrete

---

	분류모델의 성능을 평가 -> Accuracy가 더 우월한 Metric

	모델의 정확성을 향상시켜 궁극적으로 Accuracy가 높은 모델을 만듦 -> Loss: Cross Entropy

---

## Confusion Matrix

---

![Alt text](https://d3s0tskafalll9.cloudfront.net/media/images/F-38-1.max-800x600.png)

	분류 모델 측면에서 모델의 결과가 이진 분류 형태일때 모델의 예측 결과와 실제 정답셋을 비교해 아래 4가지 항목으로 표현

	- True Positive (TP) - 모델이 양성(Positive)을 양성으로 맞혔을 때
	- True Negative (TN) - 모델이 음성(Negative)을 음성으로 맞혔을 때
	- False Positive (FP) - 모델이 음성(Negative)을 양성(Positive)으로 잘못 예측했을 때
	- False Negative (FN) - 모델이 양성(Positive)을 음성(Negative)으로 잘못 예측했을 때

---

	- Accuracy(정확도) -> (TP+TN) / (TP+TN+FP+FN)
	- Precision(정밀도) -> TP / (TP+FP)

		모델이 판단한 양성만 중점을 둠 -> 정밀도 ↑ = FP ↓ ∴  음성을 양성으로 잘못 분류한것 ↓ = 정밀도 ↑
	
	- Recall(재현율) -> TP / (TP+FN)

		실제 양성만 중점을 둠 -> 재현율 ↑ = FN ↓ ∴  실제 양성을 분류해 내지 못한 경우 ↓ = 재현율 ↑

---

## F-score

---

![Alt text](https://prod-files-secure.s3.us-west-2.amazonaws.com/c09f8228-29c7-4dcb-8ca3-1de7d3988fab/05b84bd9-b315-4792-8d50-7ea5c75c6b9d/image.png)

	위의 F Score에서 β가 1이 될 때를 말함

		- F Score -> Precision과 Recall의 조화평균이 되는 값 ∴  Precision과 Recall 둘 다 고려 가능 

---

Q. 만약 Precision보다 Recall을 좀더 중요시하고 싶다면 F score에서 beta 값을 1보다 크게 하는게 좋을까요, 작게 하는게 좋을까요?

	beta=1 이라면 F-score는 위 두 경우의 값이 동일합니다. beta=2 라면 분자는 동일하지만 Case 1의 분모가 더 커지므로 Case 2의 F score가 더 큽니다. 이것은 beta가 1보다 클 때 recall이 더 큰 경우를 더 우대한다고 볼 수 있습니다. 따라서, Recall을 중요시하고 싶다면 beta 값을 1보다 크게 하는 것이 좋습니다.

---

## Threshold의 변화에 따른 모델 성능

---

	모델의 파라미터 등은 전혀 변한 것이 없는데, 모델의 출력값을 해석하는 방식만 다르게 해도 모델은 전혀 다른 성능을 가지게 됩니다. 
		
		∴  모델의 성능척도 값도 달라지게 될 것입니다.

---

-- 예시는 SVM 모델로 구성

	classifier.decision_function(X_test) -> 0보다 작으면 음성(label=0), 0보다 크면 양성(label=1)으로 분류
		              
			'''
				precision    recall  f1-score   support
		
			           0       0.77      0.83      0.80        24
			           1       0.83      0.77      0.80        26

			    accuracy                           0.80        50
			   macro avg       0.80      0.80      0.80        50
			weighted avg       0.80      0.80      0.80        50

			'''

	classifier.decision_function(X_test) > -0.1 -> -0.1보다 큰 것을 양성으로 분류

			'''
			
			              precision    recall  f1-score   support

				           0       0.78      0.75      0.77        24
				           1       0.78      0.81      0.79        26

				    accuracy                           0.78        50
				   macro avg       0.78      0.78      0.78        50
				weighted avg       0.78      0.78      0.78        50
	
			'''
	양성 분류 기준을 확대했기 때문에, Recall: 0.77 -> 0.81 ↑ F1-score: 0.80 -> 0.79 ↓

---

## Pricion-Recall 커브

---

	X축: Recall, Y축: Precision 설정후 Threshold 변화에 따른 두 값의 변화를 그래프 나타냄

---

	PrecisionRecallDisplay.from_estimator 메서드 이용

	'''

	from sklearn.metrics import PrecisionRecallDisplay
	import matplotlib.pyplot as plt

	disp = PrecisionRecallDisplay.from_estimator(classifier, X_test, y_test)
	disp.ax_.set_title('binary class Precision-Recall curve: '
                	'AP={0:0.2f}'.format(disp.average_precision))

	plt.show()

	'''

![Alt text](https://prod-files-secure.s3.us-west-2.amazonaws.com/c09f8228-29c7-4dcb-8ca3-1de7d3988fab/5b01e004-fd1d-4a19-93ef-eddd0cb6f260/image.png)

	Precision과 Recall 사이의 트레이드오프 관계를 확인 가능 

---

## AUC(Area Under Curve)

---

	Threshold 값에 무관하게 모델의 전체적인 성능을 평가하는 방법 -> 커브 아래쪽 면적을 계산하는 방법

---

### PR AUC 계산 방법

---

![Alt text](https://prod-files-secure.s3.us-west-2.amazonaws.com/c09f8228-29c7-4dcb-8ca3-1de7d3988fab/ad112ad6-0cdb-4924-b43c-7f89d9d622a6/image.png)

	n 값을 무한히 ↑ 한다면 아주 작은 Recall 구간에 대해 Pn 값을 적분하는 것과 같게 됨

		∴  PR 커브의 아래쪽 면적인 PR AUC와 같은 의미

	average_precision_score 메서드 사용

	'''

	from sklearn.metrics import average_precision_score
	average_precision = average_precision_score(y_test, y_score)

	print('평균 precision-recall score: {0:0.2f}'.format(
	      average_precision))

	'''

---

## ROC 커브

---

	수신자 조작 특성 곡선이라 표현 가능

![Alt text](https://prod-files-secure.s3.us-west-2.amazonaws.com/c09f8228-29c7-4dcb-8ca3-1de7d3988fab/c3071fbd-89cd-4d7f-b2ad-4265aa45840f/image.png)

	TP Rate(TPR) == Recall / FP Rate == 음성 샘플에 대한 Recall

---

### ROC AUC(Area Under the ROC Curve)

---

![Alt text](https://prod-files-secure.s3.us-west-2.amazonaws.com/c09f8228-29c7-4dcb-8ca3-1de7d3988fab/7816983a-acfd-49e3-95e2-6b532f0c197a/image.png)

	ROC AUC == 위 그림의  회색 영역

		∴  AUC ↑ -> 성능 ↑
	
	roc_curve, auc 메서드 사용 

	'''

	from sklearn.metrics import roc_curve, auc

	fpr, tpr, _ = roc_curve(y_test, y_score)
	roc_auc = auc(fpr, tpr)

	plt.figure()
	lw = 2
	plt.plot(fpr, tpr, color='darkorange',
        	 lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
	plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
	plt.xlim([0.0, 1.0])
	plt.ylim([0.0, 1.05])
	plt.xlabel('False Positive Rate')
	plt.ylabel('True Positive Rate')
	plt.title('Receiver operating characteristic example')
	plt.legend(loc="lower right")
	plt.show()
	print(f"auc result:{roc_auc}")

	'''

![Alt text](https://prod-files-secure.s3.us-west-2.amazonaws.com/c09f8228-29c7-4dcb-8ca3-1de7d3988fab/83cb108c-6e90-4e23-839a-aab4e517bc44/image.png)

	Precision과 Recall이 전부 1이 되는 완벽한 모델 -> ROC 커브에 (0, 1)

	파란 점선보다는 위쪽에 그려져야 하며, 가급적 (0, 1)에 가깝게 그려질 수록 우수한 분류기

		(0, 0)과 (1, 1)을 잇는 파란 점선 -> 극단적인 경우들만 모아놓은 경우

---

## 다양한 머신러닝 모델의 평가척도

---

### 회귀 모델

[회귀의 오류 지표 알아보기](https://modulabs.co.kr/blog/regression_metrics_mae_mse/)

---

Q. MSE, RMSE 등 square 계열 Metric과 MAE, MAPE 등 absolute value 계열 Metric이 특이값에 대해 어떤 차이를 보이는지 설명해 주세요.

	MSE와 RMSE는 큰 오차를 더 강조 ∴  이상치가 존재할 때 모델의 성능이 더 나쁘게 평가될 가능성 있음 / MAE는 모든 오차를 동등하게 처리 ∴   이상치의 영향을 상대적으로 적게 받음

---

	- 데이터에 이상치(Outlier) 가 많을 경우 MAE을 사용하기 적절
	- 이상치에 더 큰 가중치를 두어야 하거나 큰 오차가 작은 오차보다 더 심각한 문제인 경우 MSE, RMSE를 사용하는게 적절
	- NLL(Negative Logistic Likelihood) : 데이터의 분포를 보고 모델에 어떤 손실 함수 적용할지 결정함 / 가우시안이면 MSE, 라플라시안이면 MAE, 베르누이이면 BCE를 써야한다

---

### 랭킹 모델

[정보 검색(Information Retrieval) 평가 방법: MAP, MRR, DCG, NDCG](https://modulabs.co.kr/blog/information-retrieval-map-ndcg/)

---

Q. NDCG가 MRR, MAP 등 이전 평가척도들에 비해 어떤 부분에서 좋아졌다고 할 수 있을까요?

	랭킹을 매기기 위해 임의성을 배제하고 모든 콘텐츠 아이템에 대한 관련성을 계산하여 랭킹에 반영한다.

---

### 이미지 생성 모델

[이미지 간 유사성 측정하는 방법](https://modulabs.co.kr/blog/how-to-measure-similarity/)

---

Q. MSE나 PSNR 대비 SSIM이 가지는 가장 큰 차이점은 무엇인가요?

	MSE나 PSNR은 모두 픽셀 단위로 비교해서 거리를 측정한다. 그러나 이 방식은 이미지가 약간 평행이동해 있어도 두 차이를 크게 측정하는 단점이 있다. SSIM은 이와 달리 픽셀 단위 비교보다는 이미지 내의 구조적 차이에 집중하는 방식을 쓴다.

---

### 기계번역 모델

[BLEU : 기계번역에서 많이 사용하는 지표](https://modulabs.co.kr/blog/bleu-machine-translation/)

---

Q. BLEU score는 결국 두 텍스트가 얼마나 겹치는지를 측정하는 척도입니다. BLEU에서는 텍스트가 겹치는 정도를 어떻게 측정하나요?

	 1-gram, 2-gram, 3-gram, 4-gram이 두 문장 사이에 몇번이나 공통되게 출현하는지를 측정합니다.


