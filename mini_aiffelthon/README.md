# mini aiffelthon

---

- 주제
- 데이터
- CV Task
- NLP Task
- Prototype

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

	-> 음료 영양성분 DB 구축 데이터로 활용

---

## CV Task

---

Task

	음료를 인식해 해당 라벨을 반환

학습해본 모델

 	- 직접 구축한 CNN 모델
![image](https://github.com/user-attachments/assets/52329294-f475-4a07-a106-a7eec9e6ad82)

  	- YOLO v7
		- Model(
			  (model): Sequential(
			    (0): Conv(
			      (conv): Conv2d(3, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
			      (act): SiLU(inplace=True)
			    )
			    (1): Conv(
			      (conv): Conv2d(32, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
			      (act): SiLU(inplace=True)
			    )
			    (2): Conv(
			      (conv): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
			      (act): SiLU(inplace=True)
			    )
			    (3): Conv(
			      (conv): Conv2d(64, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
			      (act): SiLU(inplace=True)
			    )
			    (4): Conv(
			      (conv): Conv2d(128, 64, kernel_size=(1, 1), stride=(1, 1))
			      (act): SiLU(inplace=True)
			    )
			    (5): Conv(
			      (conv): Conv2d(128, 64, kernel_size=(1, 1), stride=(1, 1))
			      (act): SiLU(inplace=True)
			    )
			    (6): Conv(
			      (conv): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
			      (act): SiLU(inplace=True)
			    )
			    (7): Conv(
			      (conv): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
			      (act): SiLU(inplace=True)
			    )
			    (8): Conv(
			      (conv): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
			      (act): SiLU(inplace=True)
			    )
			    (9): Conv(
			      (conv): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
			      (act): SiLU(inplace=True)
			    )
			    (10): Concat()
			    (11): Conv(
			      (conv): Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1))
			      (act): SiLU(inplace=True)
			    )
			    (12): MP(
			      (m): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
			    )
			    (13): Conv(
			      (conv): Conv2d(256, 128, kernel_size=(1, 1), stride=(1, 1))
			      (act): SiLU(inplace=True)
			    )
			    (14): Conv(
			      (conv): Conv2d(256, 128, kernel_size=(1, 1), stride=(1, 1))
			      (act): SiLU(inplace=True)
			    )
			    (15): Conv(
			      (conv): Conv2d(128, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
			      (act): SiLU(inplace=True)
			    )
			    (16): Concat()
			    (17): Conv(
			      (conv): Conv2d(256, 128, kernel_size=(1, 1), stride=(1, 1))
			      (act): SiLU(inplace=True)
			    )
			    (18): Conv(
			      (conv): Conv2d(256, 128, kernel_size=(1, 1), stride=(1, 1))
			      (act): SiLU(inplace=True)
			    )
			    (19): Conv(
			      (conv): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
			      (act): SiLU(inplace=True)
			    )
			    (20): Conv(
			      (conv): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
			      (act): SiLU(inplace=True)
			    )
			    (21): Conv(
			      (conv): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
			      (act): SiLU(inplace=True)
			    )
			    (22): Conv(
			      (conv): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
			      (act): SiLU(inplace=True)
			    )
			    (23): Concat()
			    (24): Conv(
			      (conv): Conv2d(512, 512, kernel_size=(1, 1), stride=(1, 1))
			      (act): SiLU(inplace=True)
			    )
			    (25): MP(
			      (m): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
			    )
			    (26): Conv(
			      (conv): Conv2d(512, 256, kernel_size=(1, 1), stride=(1, 1))
			      (act): SiLU(inplace=True)
			    )
			    (27): Conv(
			      (conv): Conv2d(512, 256, kernel_size=(1, 1), stride=(1, 1))
			      (act): SiLU(inplace=True)
			    )
			    (28): Conv(
			      (conv): Conv2d(256, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
			      (act): SiLU(inplace=True)
			    )
			    (29): Concat()
			    (30): Conv(
			      (conv): Conv2d(512, 256, kernel_size=(1, 1), stride=(1, 1))
			      (act): SiLU(inplace=True)
			    )
			    (31): Conv(
			      (conv): Conv2d(512, 256, kernel_size=(1, 1), stride=(1, 1))
			      (act): SiLU(inplace=True)
			    )
			    (32): Conv(
			      (conv): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
			      (act): SiLU(inplace=True)
			    )
			    (33): Conv(
			      (conv): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
			      (act): SiLU(inplace=True)
			    )
			    (34): Conv(
			      (conv): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
			      (act): SiLU(inplace=True)
			    )
			    (35): Conv(
			      (conv): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
			      (act): SiLU(inplace=True)
			    )
			    (36): Concat()
			    (37): Conv(
			      (conv): Conv2d(1024, 1024, kernel_size=(1, 1), stride=(1, 1))
			      (act): SiLU(inplace=True)
			    )
			    (38): MP(
			      (m): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
			    )
			    (39): Conv(
			      (conv): Conv2d(1024, 512, kernel_size=(1, 1), stride=(1, 1))
			      (act): SiLU(inplace=True)
			    )
			    (40): Conv(
			      (conv): Conv2d(1024, 512, kernel_size=(1, 1), stride=(1, 1))
			      (act): SiLU(inplace=True)
			    )
			    (41): Conv(
			      (conv): Conv2d(512, 512, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
			      (act): SiLU(inplace=True)
			    )
			    (42): Concat()
			    (43): Conv(
			      (conv): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1))
			      (act): SiLU(inplace=True)
			    )
			    (44): Conv(
			      (conv): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1))
			      (act): SiLU(inplace=True)
			    )
			    (45): Conv(
			      (conv): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
			      (act): SiLU(inplace=True)
			    )
			    (46): Conv(
			      (conv): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
			      (act): SiLU(inplace=True)
			    )
			    (47): Conv(
			      (conv): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
			      (act): SiLU(inplace=True)
			    )
			    (48): Conv(
			      (conv): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
			      (act): SiLU(inplace=True)
			    )
			    (49): Concat()
			    (50): Conv(
			      (conv): Conv2d(1024, 1024, kernel_size=(1, 1), stride=(1, 1))
			      (act): SiLU(inplace=True)
			    )
			    (51): SPPCSPC(
			      (cv1): Conv(
			        (conv): Conv2d(1024, 512, kernel_size=(1, 1), stride=(1, 1))
			        (act): SiLU(inplace=True)
			      )
			      (cv2): Conv(
			        (conv): Conv2d(1024, 512, kernel_size=(1, 1), stride=(1, 1))
			        (act): SiLU(inplace=True)
			      )
			      (cv3): Conv(
			        (conv): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
			        (act): SiLU(inplace=True)
			      )
			      (cv4): Conv(
			        (conv): Conv2d(512, 512, kernel_size=(1, 1), stride=(1, 1))
			        (act): SiLU(inplace=True)
			      )
			      (m): ModuleList(
			        (0): MaxPool2d(kernel_size=5, stride=1, padding=2, dilation=1, ceil_mode=False)
			        (1): MaxPool2d(kernel_size=9, stride=1, padding=4, dilation=1, ceil_mode=False)
			        (2): MaxPool2d(kernel_size=13, stride=1, padding=6, dilation=1, ceil_mode=False)
			      )
			      (cv5): Conv(
			        (conv): Conv2d(2048, 512, kernel_size=(1, 1), stride=(1, 1))
			        (act): SiLU(inplace=True)
			      )
			      (cv6): Conv(
			        (conv): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
			        (act): SiLU(inplace=True)
			      )
			      (cv7): Conv(
			        (conv): Conv2d(1024, 512, kernel_size=(1, 1), stride=(1, 1))
			        (act): SiLU(inplace=True)
			      )
			    )
			    (52): Conv(
			      (conv): Conv2d(512, 256, kernel_size=(1, 1), stride=(1, 1))
			      (act): SiLU(inplace=True)
			    )
			    (53): Upsample(scale_factor=2.0, mode='nearest')
			    (54): Conv(
			      (conv): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1))
			      (act): SiLU(inplace=True)
			    )
			    (55): Concat()
			    (56): Conv(
			      (conv): Conv2d(512, 256, kernel_size=(1, 1), stride=(1, 1))
			      (act): SiLU(inplace=True)
			    )
			    (57): Conv(
			      (conv): Conv2d(512, 256, kernel_size=(1, 1), stride=(1, 1))
			      (act): SiLU(inplace=True)
			    )
			    (58): Conv(
			      (conv): Conv2d(256, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
			      (act): SiLU(inplace=True)
			    )
			    (59): Conv(
			      (conv): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
			      (act): SiLU(inplace=True)
			    )
			    (60): Conv(
			      (conv): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
			      (act): SiLU(inplace=True)
			    )
			    (61): Conv(
			      (conv): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
			      (act): SiLU(inplace=True)
			    )
			    (62): Concat()
			    (63): Conv(
			      (conv): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1))
			      (act): SiLU(inplace=True)
			    )
			    (64): Conv(
			      (conv): Conv2d(256, 128, kernel_size=(1, 1), stride=(1, 1))
			      (act): SiLU(inplace=True)
			    )
			    (65): Upsample(scale_factor=2.0, mode='nearest')
			    (66): Conv(
			      (conv): Conv2d(512, 128, kernel_size=(1, 1), stride=(1, 1))
			      (act): SiLU(inplace=True)
			    )
			    (67): Concat()
			    (68): Conv(
			      (conv): Conv2d(256, 128, kernel_size=(1, 1), stride=(1, 1))
			      (act): SiLU(inplace=True)
			    )
			    (69): Conv(
			      (conv): Conv2d(256, 128, kernel_size=(1, 1), stride=(1, 1))
			      (act): SiLU(inplace=True)
			    )
			    (70): Conv(
			      (conv): Conv2d(128, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
			      (act): SiLU(inplace=True)
			    )
			    (71): Conv(
			      (conv): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
			      (act): SiLU(inplace=True)
			    )
			    (72): Conv(
			      (conv): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
			      (act): SiLU(inplace=True)
			    )
			    (73): Conv(
			      (conv): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
			      (act): SiLU(inplace=True)
			    )
			    (74): Concat()
			    (75): Conv(
			      (conv): Conv2d(512, 128, kernel_size=(1, 1), stride=(1, 1))
			      (act): SiLU(inplace=True)
			    )
			    (76): MP(
			      (m): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
			    )
			    (77): Conv(
			      (conv): Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1))
			      (act): SiLU(inplace=True)
			    )
			    (78): Conv(
			      (conv): Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1))
			      (act): SiLU(inplace=True)
			    )
			    (79): Conv(
			      (conv): Conv2d(128, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
			      (act): SiLU(inplace=True)
			    )
			    (80): Concat()
			    (81): Conv(
			      (conv): Conv2d(512, 256, kernel_size=(1, 1), stride=(1, 1))
			      (act): SiLU(inplace=True)
			    )
			    (82): Conv(
			      (conv): Conv2d(512, 256, kernel_size=(1, 1), stride=(1, 1))
			      (act): SiLU(inplace=True)
			    )
			    (83): Conv(
			      (conv): Conv2d(256, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
			      (act): SiLU(inplace=True)
			    )
			    (84): Conv(
			      (conv): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
			      (act): SiLU(inplace=True)
			    )
			    (85): Conv(
			      (conv): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
			      (act): SiLU(inplace=True)
			    )
			    (86): Conv(
			      (conv): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
			      (act): SiLU(inplace=True)
			    )
			    (87): Concat()
			    (88): Conv(
			      (conv): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1))
			      (act): SiLU(inplace=True)
			    )
			    (89): MP(
			      (m): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
			    )
			    (90): Conv(
			      (conv): Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1))
			      (act): SiLU(inplace=True)
			    )
			    (91): Conv(
			      (conv): Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1))
			      (act): SiLU(inplace=True)
			    )
			    (92): Conv(
			      (conv): Conv2d(256, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
			      (act): SiLU(inplace=True)
			    )
			    (93): Concat()
			    (94): Conv(
			      (conv): Conv2d(1024, 512, kernel_size=(1, 1), stride=(1, 1))
			      (act): SiLU(inplace=True)
			    )
			    (95): Conv(
			      (conv): Conv2d(1024, 512, kernel_size=(1, 1), stride=(1, 1))
			      (act): SiLU(inplace=True)
			    )
			    (96): Conv(
			      (conv): Conv2d(512, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
			      (act): SiLU(inplace=True)
			    )
			    (97): Conv(
			      (conv): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
			      (act): SiLU(inplace=True)
			    )
			    (98): Conv(
			      (conv): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
			      (act): SiLU(inplace=True)
			    )
			    (99): Conv(
			      (conv): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
			      (act): SiLU(inplace=True)
			    )
			    (100): Concat()
			    (101): Conv(
			      (conv): Conv2d(2048, 512, kernel_size=(1, 1), stride=(1, 1))
			      (act): SiLU(inplace=True)
			    )
			    (102): RepConv(
			      (act): SiLU(inplace=True)
			      (rbr_reparam): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
			    )
			    (103): RepConv(
			      (act): SiLU(inplace=True)
			      (rbr_reparam): Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
			    )
			    (104): RepConv(
			      (act): SiLU(inplace=True)
			      (rbr_reparam): Conv2d(512, 1024, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
			    )
			    (105): Detect(
			      (m): ModuleList(
			        (0): Conv2d(256, 255, kernel_size=(1, 1), stride=(1, 1))
			        (1): Conv2d(512, 255, kernel_size=(1, 1), stride=(1, 1))
			        (2): Conv2d(1024, 255, kernel_size=(1, 1), stride=(1, 1))
			      )
			    )
			  )
			)

프로토타입에 쓰인 모델

	- CNN 모델

		-> 성능은 YOLO v7이 더 좋았으나 서빙과정에서 오류가 발생해 CNN모델 사용

---

## NLP Task

---

Task

	- 음료 이름을 통해 간략한 리뷰, 영양성분 출력

기능

	1. 음료 이름을 RAG 기법(검색 기반)을 통해 해당 음료의 맛이나 설명을 짧은 글로 생성
 	2. 음료 이름을 통해 음료 영양성분 DB에서 해당 음료 영양성분 검색후 Front-end로 반환

확습해본 모델

	- ko-gpt-trinity-1.2B-v0.5
 		- 모델 성능이 좋지 못함
![image](https://github.com/user-attachments/assets/202c4c18-4949-483d-8392-acc3812c8752)

 	- EEVE-Korean-Instruct-10.8B-v1.0
		- ollama를 통해 local로 구현해 학습 진행
![download](https://github.com/user-attachments/assets/3cfa6d7a-c031-45e8-ac65-c04cc8bd11c5)

RAG기법을 사용 전 후 결과 비교

	- RAG기법 사용 전
![download](https://github.com/user-attachments/assets/52dc3ee5-9cf6-44c3-9d56-ae362ad584c2)

	- RAG기법 사용 후
![download](https://github.com/user-attachments/assets/9826e39d-a6d4-44d1-bf8c-654f758d41b7)

		-> 음료에 대한 리뷰는 RAG기법 사용 후 더 좋음

langchain에서 제공하는 메서드 성능 비교

	- GoogleSerperAPIWrapper
![download](https://github.com/user-attachments/assets/b74ab58d-2328-4740-8714-9bbff6498388)

	- TavilySearchAPIRetriever
![download](https://github.com/user-attachments/assets/90859432-a61b-4ada-80b8-a6d13557f3cb)

		-> GoogleSerperAPIWrapper메서드가 성능이 더 좋음을 확인

GoogleSerperAPIWrapper .vs naver search api

	- 생성된 리뷰 완성도 비교
		- GoogleSerperAPIWrapper
	 		- 자연은 요거 상큼 복숭아는 달콤한 맛을 지닌 가벼운 칼로리의 복숭아 향 요거트로, 식사이나 간식으로 즐기기에 좋습니다. 1인분(340ml)당 185칼로리이며 지방이 거의 없고 단백질이 적당히 들어 있습니다. 이 제품은 저지방 우유에서 유래한 탈지 분유를 함유하고 있으며 건강한 식단에 적합합니다. 맛은 상큼한 복숭아 요거트 음료와 비슷하며, 식사 후나 아이 간식으로 즐기기 좋습니다
		- naver search api
	 		- 다양한 맛의 델몬트 오렌지 드링크는 상큼하고, 달콤하며, 청량한 맛으로 소비자들 사이에서 좋은 평가를 받고 있습니다.

프로토타입에서의 RAG

	- naver search api 사용
 		- api를 사용해 광고 필터링, 검색어 가중치 등등 설정 가능
   		- langchain에서 제공하는 메서드 보다 더 세밀한 결과 

음료 영양성분 DB

	- 데이터를 가지고 영양성분 DB 빠르게 구축
	- 간단한 SQL문으로 해당하는 음료 영양성분 검색
![image](https://github.com/user-attachments/assets/ac3fd18e-abc0-41f6-a607-4a565a8d255f)

 	- {'식품명': '델몬트오렌지드링크', '제조사명': '롯데칠성음료(주)', '영양성분함량기준량': '100ml', '에너지(kcal)': '32', '단백질(g)': '0.0', '지방(g)': '0.0', '탄수화물(g)': '7.89', '당류(g)': '7.89', '나트륨(mg)': '17.0', '콜레스테롤(mg)': '0.0', '포화지방산(g)': '0.0', '트랜스지방산(g)': '0.0', '식품중량': '6080ml'}

---

## Prototype

---

서빙 방법

	- Fast Api 사용

Prototype 결과

![스크린샷 2024-11-20 025908](https://github.com/user-attachments/assets/7189914f-6a72-475d-a8b1-74a70fd34b67)
![스크린샷 2024-11-20 025844](https://github.com/user-attachments/assets/566bf85d-845a-4b68-af64-69819693ac39)
![스크린샷 2024-11-20 035009](https://github.com/user-attachments/assets/f47c6cdf-792a-45ae-8fd8-35cef9dd8d65)

 


	

	


  	
