# 학습 내용

---

- KoChatGPT 업그레이드

---

## KoChatGPT 업그레이드

---

- 우리가 지난시간 살펴본 KoChatGPT 모델에 사용한 데이터셋은 아직 완벽히 정제 X
- Human Feedback이 반영된 데이터셋을 대체하기 위해 SFT와 RM 모델에 사용할 다양한 benchmark 데이터셋 검토
- 언어모델의 생성능력을 좌우하는 최선의 디코딩을 위한 하이퍼파라미터 서치 필요
- 생성된 답변에 대한 주관적인 평가를 보완할 수 있는 정량적인 메트릭 도입
- LLM Trend Note1에서 살펴본 다양한 Instruction Tuning 및 Prompting 기법들도 적용 가능
- 무엇보다 foundation model로 사용한 KoGPT-2는 Emergent abilities를 기대하기엔 다소 작은 사이즈의 모델
	- 더 큰 파라미터 스케일을 가진 모델을 사용
	- 더 효율적인 연산을 수행할 수 있는 LoRA의 적용
	- 새로운 Instruction Tuning 및 reward ranking 알고리즘을 도입
