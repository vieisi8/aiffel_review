# 학습 내용

---

- LLM의 Emergent Abilities
	- Foundation Model과 Emergent Abilities
- LLM + Emergent Abilities = AGI?
	- PaLM, FLAN, ChatGPT, LLaMA 등 최신 LLM 모델의 특징
- InstructGPT : RLHF, 언어모델과 강화학습의 만남
	- 최신 LLM 모델들의 주요 아키텍쳐 (Sparse Attention, RLHF)
- GPT-4 vs LLaMA
	- LLM을 더욱 효율적으로 학습시키는데 필요한 기술들 (LoRA, LLM.int8())

---

## LLM의 Emergent Abilities

---

Foundation Model?

	지금까지 나온 모든 Pre-trained LM(이하 PLM)들을 foundation model 이라는 개념으로 지칭하며 새로운 패러다임을 제시

foundation model의 특징

	- emergence
		- ?
	- homogenization
		- 거의 모든 최첨단 NLP 모델은 이제 BERT, RoBERTa, BART, T5 등과 같은 몇 가지 기본 모델 중 하나에서 채택
		- homogenization(균질화)가 의미하는 바
		- 매우 높은 레버리지를 가짐
			- 백본 모델로 쓰이는 몇가지 모델만 개선되면, NLP 전반에 즉각적으로 그 개선에 의한 이점이 퍼지게 됨
			- 모든 AI 시스템은 몇 가지 기본 모델의 동일한 문제(데이터의 편향 등)을 물려받을 수도 있음

LLM?

![image](https://d3s0tskafalll9.cloudfront.net/media/images/Screenshot_2023-06-09_at_4.52.00_PM.max-800x600.png)

	Large Language Model (이하 LLM) 에서 Large의 의미는 '크기의 상대적인 개념'에 구속되지 않음

2019년부터 2023년 초까지 발표된 모델 중 10B 이상의 파라미터를 가진 모델들만 추려놓은 그림

![image](https://d3s0tskafalll9.cloudfront.net/media/images/LLM_Map.max-800x600.png)

GPT-3?

	175B (1천 750억)개의 파라미터를 트랜스포머 디코더 기반 아키텍쳐에 집어 넣은 LLM

		-> 파라미터 수가 일정 수준을 초과한 LLM은 BERT와 같은 million 단위의 소규모 LM에 없는 성능을 나타냄

in-context learning?

	- zero-shot -> 적절한 instruction이 담긴 지시문을 모델에 던져서 원하는 답변을 이끌어 내는 것
	- one-shot
		   -> 단순히 지시문만 주는 게 아니라 구체적인 예제를 던져서 원하는 답변을 이끌어 내는 것
	- few-shot

	-> GPT-3를 개발한 OpenAI의 연구자들은 GPT-3를 개발한 OpenAI의 연구자들으로 위와 같은 현상을 정의했음

		-> 모델의 inference 단계에서 이뤄지는 작업

			-> gradient를 계산해 모델 파라미터 업데이트를 함으로써 학습하는 방법이 아니라는 뜻

	-> prompt learning 이라고도 불리게 됨

		-> 적절한 instruction 내지 prompt를 주었을 때 더 모델로부터 더 나은 답변을 얻어 낼 수 있게 하는 것

	-> 이후 instruction tuning, prompt engineering 같은 방법론으로 발전하게 됨
	

Emergent abilities?

	수백억개를 초과하는 LLM이 가진 특징이자, 바꿔 말해 소규모 LM에서는 발견되지 않는 특징

		1. 아주 많은 파라미터로 엄청난 양의 데이터를 단순히 auto-regressive 한 방법으로 학습
		2. 모델이 어떤 의미에서는 인간이 명시적으로 가르쳐주지 않았음에도 스스로 탁월한 언어이해(NLU)와 언어생성능력(NLG)을 가지게 됨

Instruction Tuning?

	Google이 발표한 FLAN 논문에 처음 등장

	- prompt engineering의 한 방법인 chain-of-thought prompting

		- Google이 발표한 PaLM 논문에 자세한 설명이 실려 있음

FLAN : Finetuned Language Models Are Zero-Shot Learners?

	FLAN 논문의 저자들은 zero-shot learning abilities를 개선하는 방법을 제안하는 것이 논문의 목적이라고 말함

![image](https://d3s0tskafalll9.cloudfront.net/media/images/FLAN_seongneung.max-800x600.png)

	- 연구진들은 LaMDA (이하 람다) 모델을 사용
		- GPT-3보다 좀 더 작은 137B 사이즈의 디코더 기반 트랜스포머 모델
	- instruction tuning을 활용해 GPT-3보다 더 나은 성능의 zero-shot learning abilities를 달성

		-> 모델에게 instruction following 능력을 직접적으로 부여

		-> fine-tuning 때 학습하지 않은 task에 대해서도 user의 instruction에 following하는 능력을 가지게 되었다는 것

	-> few-shot learning과 fine-tuning의 하이브리드라고 볼 수도 있음

instruction tuning의 핵심?

![image](https://d3s0tskafalll9.cloudfront.net/media/images/instruction_tuning.max-800x600.png)

	다양한 종류의 NLP task를 instruction과 그에 상응하는 label로 이뤄진 pair dataset으로 fine-tuning

		-> 한 번도 보지 못한 task에서 inference를 하여 만족할 만한 성능을 내는 튜닝방법

			-> user가 prompt engineering을 할 필요 없음

![image](https://d3s0tskafalll9.cloudfront.net/media/images/instruction_tuning_chai.max-800x600.png)

instruction tuning에서의 모델 fine-tuning

![image](https://d3s0tskafalll9.cloudfront.net/media/images/FLAN_dataset.max-800x600.png)

	1. 위 그림과 같이 기존에 공개된 데이터셋들을 총 12개 카테고리로 분류
	2. 각 데이터셋들을 instruction 형식으로 수정
		- 각 데이터셋 마다 10개의 instruction template 생성
	3. 전체 데이터 셋에서 무작위로 template을 뽑아 fine-tuning 수행

![image](https://d3s0tskafalll9.cloudfront.net/media/images/instruction_template.max-800x600.png)

Chain-of-Thought Prompting?

	PaLM 논문에서 LLM의 규모를 극한으로 몰아붙였을 때, few-shot 능력이 얼마나 상승하게 될지를 실험

		-> 540B에 달하는 파라미터를 탑재한 디코더 기반 트랜스포머 모델 사용, 자그마치 780B 개의 토큰을 가지고 pre-train

	-> Gopher가 어려움을 겪은 Multi-step reasoning에서 좋은 성능을 낼 수 있음을 보여줌

	-> 모델의 추론결과를 토대로 오류분석 가능, 모델이 왜 그렇게 추론했는지에 대한 해석가능성을 높일 수 있음

chain-of-thought prompting 의 원리?

![image](https://d3s0tskafalll9.cloudfront.net/media/images/chain_of_thought.max-800x600.png)

	문제를 푸는데 필요한 사고과정을 함께 준 뒤 유사한 문제를 풀게 시키면 놀랍게도 그 문제에 대한 답만 맞추는 게 아니라, 자신이 풀이한 과정까지 답변에 포함시켜 돌려주는 걸 볼 수 있음

---

## LLM + Emergent Abilities = AGI?

---

Emergent Abilities의 정의

	소규모 모델에는 없지만 대규모 모델에는 존재하는 능력

		-> 모델 파라미터 수로 측정된 모델 규모와 관련지어 짐

![image](https://d3s0tskafalll9.cloudfront.net/media/images/Emergence_model_scale_geulaepeu.max-800x600.png)

	-> task별 모델의 성능과 모델 파라미터 개수 사이의 관계에서 Emergence가 나타나는 패턴을 보여주는 그래프

		-> 모델 크기가 특정 임계값을 넘어서는 순간 모델 performance가 확연히 달라지는 걸 볼 수 있음

	-> few-shot prompting 에서의 Emergence를 나타냄

![image](https://d3s0tskafalll9.cloudfront.net/media/images/Emergence_model_scale_geulaepeu2.max-800x600.png)

	Chain of Thought(CoT) 같은 좀더 고급의 prompt engineering을 사용하거나 instruction tuning 같은 고급의 fine-tuning 기법을 썼을 때 

		-> task별 performance의 Emergence 패턴을 나타낸 그림

Deepmind의 논문 Training Compute-Optimal Large Language Models

![image](https://d3s0tskafalll9.cloudfront.net/media/images/less_wrong.max-800x600.png)

	데이터를 추가할 때 얻을 수 있는 이득은 엄청난 반면, 모델 크기를 키웠을 때 이득은 미미하다는 것

---

## InstructGPT : RLHF, 언어모델과 강화학습의 만남

---

RLHF 기원

	모델이 윤리적으로 적절한지, 유해한 정보를 창발해내고 있는지를 손실함수 단계에서 걸러낼 수 없다면

		-> 모델의 훈련루프에 인간이 직접 참여하지 않을 이유가 없다...

OpenAI는?

	라벨러들을 고용해 인간이 직접 그 씨앗을 모델에게 뿌리는 방법을 선택

RLHF?

	OpenAI와 Deepmind의 연구자들이 2017년에 발표한 Deep Reinforcement Learning from Human Preferences논문에서 처음 소개됨

		- Reinforce Learning을 사용해 가상환경에서 에이전트에게 backflip을 가르치는 과정
		- human feedback을 주는 게 목적
		- 피드백을 주는 인간에게는 agent(모델)의 action(행동)에 대한 두 가지 옵션이 제공되고 목표달성에 가장 가까운 옵션을 선택해 피드백을 줌

	-> 순전한 강화학습만으로 에이전트를 학습시킬 때보다 훨씬 효과적이라는 걸 입증한 논문

Fine-Tuning Language Models from Human Preferences논문

	강화 학습을 텍스트 요약task를 수행하는 PLM에 적용

		-> 강화학습 알고리즘인 PPO (Proximal Policy Optimization)를 적용할 수 있다는 걸 보여준 논문

Learning to summarize from human feedback논문

	앞선 논문에서 사용한 두 메트릭에 인간의 선호도가 충분히 반영될 수 없다는 문제점을 지적

		-> 인간이 직접 품질을 비교한 데이터셋을 바탕으로 인간의 선호도를 학습시킨 모델

			-> 강화학습의 보상함수(reward function)로 사용할 수 있다는 아이디어를 제시

일반적인 강화학습 시스템 도식화

![image](https://d3s0tskafalll9.cloudfront.net/media/images/RLHF1.max-800x600.png)

Learning to summarize from human feedback 논문의 LM

	- 보상함수가 envirionment/RL algorithm 에서 분리되어 있음
                - 보상함수는 human feedback을 학습한 별도의 모델로 존재
	- agent(모델)은 보상 모델이 주는 reward로 action을 개선해나가는 것

		-> 입력 데이터가 편향되었다 하더라도 알고리즘 편향없이 human feedback이 의도하는 품질의 요약을 수행하게 되는 것

Learning to summarize from human feedback 에서의 3단계 프로세스

![image](https://d3s0tskafalll9.cloudfront.net/media/images/RLHF2.max-800x600.png)

1. Collect human feedback 에서 원문과 복수의 요약 쌍으로 이뤄진 데이터셋에 대해, 인간이 직접 고품질 요약과 저품질 요약으로 분류
2. Train reward model 에서 원문과 그에 상응하는 분류된 요약쌍들을 RM (reward model)에 넣어 각각의 reward를 계산하고 RM이 각 요약의 (human feedback이 반영된)품질을 매길 수 있도록 학습
3. Train policy with PPO 에서 LM(agent)에은 새로은 원문을 입력받고, 훈련된 RM을 사용해 LM이 생성해낸 요약이 고품질 요약이 되도록 LM을 강화학습 진행

PPO algorithm에 관한 설명

- OpenAI에서 2017년에 발표한 [Proximal Policy Optimization Algorithms](https://arxiv.org/pdf/1707.06347) 논문
- 허깅페이스의 [블로그](https://huggingface.co/blog/deep-rl-ppo) 의 Proximal Policy Optimization (PPO) 내용

InstructGPT

	2022년에 발표된 Training language models to follow instructions with human feedback논문에서 발표됨

		-> LLM + RLHF의 프로토 타입이라고 볼 수 있음

InstructGPT 모델 학습 개요도

![image](https://d3s0tskafalll9.cloudfront.net/media/images/instructGPT.max-800x600.png)

	-> Learning to summarize from human feedback 에서의 3단계 프로세스와 매우 유사함

1. Pretraining a language model(SFT)

![image](https://d3s0tskafalll9.cloudfront.net/media/images/LM.max-800x600.png)

	- human annotator를 고용해 prompt에 대하여 사람의 선호를 반영한 label 쌍으로 이뤄진 고품질의 데이터셋을 직접 구축
	- 1.3B, 6B, 175B 3가지 사이즈의 GPT-3에 위 데이터셋으로 fine-tuning

2. gathering data and training a reward model(RM)

![image](https://d3s0tskafalll9.cloudfront.net/media/images/RM.max-800x600.png)

	- 1단계에서 준비한 SFT 모델을 약간 수정해 prompt와 그에 대한 response를 입력받아 scalar reward를 출력하는 RM 생성
	- RM의 사이즈는 6B짜리를 사용

	RM의 핵심

		-> 어떤 데이터를 어떻게 사용할 것인가라고 볼 수 있음

			-> InstructGPT의 경우 인간이 직접 LM이 생성한 답변에 rank를 매겨 이 순위를 RM이 학습

3. fine-tuning the LM with reinforcement learning(RL(PPO))

![image](https://d3s0tskafalll9.cloudfront.net/media/images/RL.max-800x600.png)

	- 회색 박스 안에 있는 Tuned Language Model이 RL과정에서 PPO로 학습될 모델(RL Policy)

		- 1단계에서 만든 SFT model 사용

	- 연두색 박스 안에 있는 Initial Language Model

		- RL Policy가 RM을 통해 업데이트될 때, 인간의 선호가 반영된 답변으로부터 너무 벗어난 문장을 생성하지 않게 제한을 걸어주는 페널티 역할

		- 1단계에서 만든 SFT model 사용

		- 페널티가 없으면 - RL Policy는 이상한 텍스트를 생성하기 쉬워지거나
				  - 반대로 RM으로부터 높은 보상을 받기 위해 RM을 속일 수도 있음

		- 페널티는 KL divergence를 활용해 RL Policy와 Initial model의 각 출력분포를 근사시키는 방법 사용

	- 실행 순서

		RL Policy가 입력받은 prompt에 대해 출력한 텍스트

		1. RM으로 들어가고 (회색박스에서 붉은색 박스로의 화살표) RM은 보상을 출력
		   동시에 Initial model 역시 똑같은 prompt를 입력받아 텍스트를 출력

		2. RL Policy와 Initial model의 각 출력으로 KL divergence값을 계산한 뒤 RM의 보상에서 빼준 값이 최종적인 보상이 됨

	- RM에 사용되는 손실함수와 RL에 사용되는 목적함수의 수식화

![image](https://d3s0tskafalll9.cloudfront.net/media/images/sonsilhamsu_mogjeoghamsu.max-800x600.png)

---

## GPT-4 vs LLaMA

---

LLaMA?

	- GPT-4와 비슷한 시기에 발표되어 GPT-3보다 더 작으면서 성능은 더 좋은 모델로 평가됨
	
		- 대부분의 zero-shot 벤치마크에서 GPT3(175B)를 능가하는 성능을 갖는 것으로 평가

	- 가장 작은 6.7B 모델은 V100 single machine에서도 실행 가능
	- 디코더 기반 트랜스포머 아키텍쳐 모델

		-> 오픈소스 LLM인 Vicuna가 발표되어 곧바로 LLaMA를 뛰어넘음

Alpaca?

	LLaMA를 instruction tuning한 모델

StackLLaMA?

	- StackExchange의 질문 답변 데이터셋과 Meta에서 공개한 LLaMA, 그리고 RLHF를 pytorch로 구현한 TRL 라이브러리를 사용해 만든 모델
	- 가장 작은 6.7B LLaMA를 initial LM으로 사용
	- RM의 경우 Anthropic의 PMP를 위해 사용한 [방식](https://huggingface.co/blog/stackllama#stack-exchange-dataset)을 따랐음

StackLLaMA의 훈련 단계

![image](https://d3s0tskafalll9.cloudfront.net/media/images/trl_loop.max-800x600.png)

1. SFT

	관심있는 도메인에 fine-tuning 하기 위해 LLaMA를 StackExchange 질문답변 dataset으로 causal language modeling

[dataset 링크](https://huggingface.co/datasets/lvwerra/stack-exchange-paired)
[supervised_finetuning 참고](https://github.com/huggingface/trl/blob/main/examples/research_projects/stack_llama/scripts/supervised_finetuning.py)

2. RM

	StackExchange 데이터셋을 활용해 upvotes를 기반으로 질문에 대한 답변들의 랭킹을 매김

		-> 더 높은 랭크의 답변을 인간의 피드백이 반영된(인간이 더 선호하는) 답변으로 간주하는 RM 학습

3. RLHF

	1. 1단계에서 준비한 LM(RL policy)으로 prompt와 response를 수집
	2. 2단계에서 준비한 RM에 넣어 reward를 계산
	3. LM의 카피본에도 같은 prompt와 response를 넣어 KL-divergence 값을 구해 최종 reward 계산
	4. StackLLaMA에서는 peft를 사용하므로 RL policy의 LoRA가중치를 PPO로 최적화

[rl_training 참고](https://github.com/huggingface/trl/blob/main/examples/research_projects/stack_llama/scripts/rl_training.py)
[ppo_trainer 참고](https://github.com/huggingface/trl/blob/3804a72e6ccedcbdf624efa38c424ea4232e894e/trl/trainer/ppo_trainer.py)

StackLLaMA와 ChatGPT의 차이점

	- StackExchange라는 특정 도메인에 국한된 데이터셋을 사용
	- ChatGPT와 같은 Human labeling 방식을 간접적으로 모방한 RM을 학습시켜 모델 전체를 업데이트하지 않고 head만 PPO로 업데이트하는 방식

저렴한 computing cost로 LLM을 fine-tuning 할 수 있는 방법

	StackLLaMA에서 사용된 방법 소개

	- 모델 가중치를 8bit로 로드
	- PLM의 가중치를 고정시키는 대신 어텐션 레이어에 훈련 가능한 rank decomposition 행렬을 주입해 연산비용을 획기적으로 낮춰주는 LoRA
	  (Low-Rank Adaptation of Large Language Models)

현재 시점에서 가장 강력한 Emergence를 보여주는 LLM?

	- instruction tuning된 LLM
	- RLHF가 적용된 LLM

sparse transformer(또는 sparse attention)?

	셀프 어텐션 연산은 일반적으로 O(n^2) 의 시간복잡도를 요구

		-> 이를 O(log n) 로 단축시키는 새로운 어텐션 메커니즘을 통칭

![image](https://d3s0tskafalll9.cloudfront.net/media/images/sparse_transformer.max-800x600.png)

	-> LLaMA의 경우에도 모델의 학습 속도를 개선하기 위해 위 기법들을 모델에 적용

분산학습 알고리즘?

	multiple GPU 사용이 가능한 환경

		-> LLM을 보다 효율적으로 분산학습시킬 수 있는 알고리즘

![image](https://d3s0tskafalll9.cloudfront.net/media/images/byeonglyeol.max-800x600.png)

	- 4x4 그리드
		- 16개의 코어를 나타냄
	- 음영처리된 사각형들
		- 해당 코어에 포함된 데이터(토큰들로 이뤄진 배치) 또는 모델 가중치
	- 첫번째 행의 다섯개 그림
		- 모델 가중치가 코어간에 분할되는 방식을 보여줌
	- 두번째 행의 그림
		- 데이터 배치가 코어 간에 분할되는 방식을 보여주는 그림
