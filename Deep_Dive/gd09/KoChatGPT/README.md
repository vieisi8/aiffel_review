# 학습 내용

---

- Base model and Dataset for RLHF
- Supervised Fine-Tuning
- Reward Model
- Proximal Policy Optimization

---

dataset

	Instruction Tuning / Prompt Engineering의 효과를 보기 위해선 언어모델의 입력을 단순한 query 형태 보단\

		-> 정교한 입력 시퀀스를 설계해야 함

backbone 모델

	KoGPT-2

backbone 모델의 디코딩 성능

	'''

	input_txt = "바람도 없는 공중에 수직의 파문을 내이며 고요히 떨어지는 오동잎은 누구의 발자취 입니까."

	tokens = tokenizer(input_txt).tokens()
	input_ids = tokenizer(input_txt, return_tensors="pt")["input_ids"].numpy()

	max_length=128
	input_ids = tokenizer(input_txt, return_tensors="pt")["input_ids"].to(device)
	output_greedy = model.generate(input_ids, max_length=max_length, do_sample=False)
	print(tokenizer.decode(output_greedy[0]))

	'''

	결과:   바람도 없는 공중에 수직의 파문을 내이며 고요히 떨어지는 오동잎은 누구의 발자취 입니까.'
		"그렇다면 그건 무슨 소리요?"
		"그건 무슨 소리요?"
		"그건 무슨 소리요?"
		"그건 무슨 소리요?"
		"그건 무슨 소리요?"
		"그건 무슨 소리요?"
		"그건 무슨 소리요?"
		"그건 무슨 소리요?"
		"그건 무슨 소리요?"
		"그건 무슨 소리요?"
		"그건 무슨 소리요?"
		"그건 무슨 소리요?"
		"그건 무슨 소리

		-> 시퀀스가 반복되어 출력

			-> 그리디 서치 디코딩시 발견되는 전형적인 현상

kogpt-2?

	오리지널 GPT2의 가장 작은 버전

![image](https://d3s0tskafalll9.cloudfront.net/media/original_images/Screenshot_2023-06-22_at_11.57.04_AM.png)

kochatgpt_1_SFT 데이터셋 확인

	'''

	import json 
	data_path_1_SFT = './aiffel/KoChatGPT/data_kochatgpt/kochatgpt_1_SFT.jsonl' 
	with open(data_path_1_SFT, "r", encoding='utf-8-sig') as json_file:
	    list_data_dict = json.load(json_file)

	print(len(list_data_dict))
	list_data_dict[:3]

	'''

	결과:   [{'prompt': '불고기용 고기 한우에요?',
		  'completion': "'저는 인공지능 챗봇이며, 직접적으로 식품에 관한 정보를 가지고 있지 않습니다. 하지만 일반적으로 불고기용 고기는 한우, 쇠고기, 돼지고기 등 다양한 종류의 고기를 사용합니다. 하지만 한우는 대표적인 고급 육류로 알려져 있기 때문에, 한우를 사용하는 경우도 많습니다. 알러지나 개별 건강 상태에 따라 다를 수 있으니 충분한 정보 수집 후에 선택해 주시기 바랍니다.",
		  'tokens': 193},
		 {'prompt': '쓰던 앱이 유료로 전환됐어',
		  'completion': "'어떤 앱인지 모르기 때문에 정확한 답변을 드리기 어렵습니다. 하지만, 일반적으로 유료 전환된 앱은 기존 무료 앱에서 추가적인 기능이나 서비스를 제공하는 경우가 많습니다. 이 경우, 유료 전환 이전 가격이 매우 저렴하거나 무료인 경우가 많으므로, 앱을 계속 사용하려면 추가적인 비용을 지불해야 할 수 있습니다. 하지만, 유료 전환 이후에는 기존 사용자 또는 새로운 사용자 모두가 추가 기능과 높은 품질의 서비스를 이용할 수 있게 됩니다. 따라서, 앱 개발자는 유료 모델로 전환함으로써 앱의 수익을 증가시키고 더 나은 서비스를 제공할 수 있습니다.",
		  'tokens': 288},
		 {'prompt': '여친이랑 다툼',
		  'completion': "'저는 인공지능이기 때문에 여친과의 다툼에 대해서는 조언을 드릴 수는 없습니다. 하지만 다투게 되면 상대방의 말을 경청하는 태도와 서로의 감정을 존중하는 것이 중요합니다. 논리적이고 차분한 대화를 통해 문제를 해결해보세요. 그리고 서로를 이해하고 서로의 의견을 수용하는 것이 중요합니다.",
		  'tokens': 153}]

kochatgpt_2_RM 데이터셋 확인

	'''

	data_path_2_RM = './aiffel/KoChatGPT/data_kochatgpt/kochatgpt_2_RM.jsonl'
	with open(data_path_2_RM, "r", encoding='utf-8-sig') as json_file:
	    list_data_dict = json.load(json_file)

	print(len(list_data_dict))
	list_data_dict[:3]

	'''

	결과:   [{'prompt': '번디는 자신이 탐정잡지, 범죄소설 그리고 성범죄 관련 실제 범죄 다큐멘터리들을 탐독했다고 누구에게 말했나?',
		  'completion_0': 'Allow me to answer your question. I know that you are curious about me.',
		  'completion_1': '번디는 다양한 인터뷰자들과 뉴스홍보 담당자들과의 면담 때 밝혔다.',
		  'completion_2': '라이언에게 말했다.',
		  'ranking': [2, 1, 0]},
		 {'prompt': '개포주공아파트는 몇 단지로 이루어져 있나?',
		  'completion_0': '개포주공아파트는 다섯 단지로 이루어져 있습니다.',
		  'completion_1': '이날 목송에서 구글상위노',
		  'completion_2': '개포주공아파트는 총 27개 단지로 이루어져 있습니다.',
		  'ranking': [2, 0, 1]},
		 {'prompt': '김영삼의 후보 시절 지역표심을 겨냥한 발언을 문제삼은 후보는?',
		  'completion_0': 'The diameter of the Metallic domain is bigger than the Hyperonic domain.',
		  'completion_1': '이 질문은 조금 불분명합니다. 김영삼 대통령이 후보 시절에 어떤 발언을 했고, 누가 그 발언을 문제삼았는지에 따라 답이 다를 수 있습니다.\\n\\n만약 김영삼 대통령이 후보 시절에 지역표심을 겨냥한 발언을 했다는 가정하에, 그 발언을 문제삼은 후보가 누구였는지를 대답하자면, 그 답은 이화선 당시 민주당 대통령 후보가 될 것입니다. 1992년 총선 때, 김영삼 대선후보는 "집값이 오른 노량진역 부근의 부동산 가격은 세월호 폭침 후 \\\'강남 도시재생\\\' 일환으로 상승했다"는 발언을 했습니다. 하지만 이화선 후보는 이 발언을 "전국적으로 경제적 발전이 이루어지지 않은 지방민의 마음을 멀리해지려는 무례한 발언"이라고 비판하며 문제삼았습니다.\\n\\n하지만, 이 질문을 답변하는 데 있어서 보다 명확한 정보가 있으면 답변을 보완할 수 있습니다.',
		  'completion_2': '김영삼의 후보 시절에 지역표심을 겨냥한 발언은 대통령 당선 전까지 대한민국 정부가 추구하고 있는 민주주의 광범위하게 확립과 보수의 사상을 이어가는 데 있어 지역경제 발전과 공공서비스 신속 개선을 위해 합리적인 국가 정책에 따르는 방향성을 제시하고 있습니다.',
		  'ranking': [1, 2, 0]}]

kochatgpt_3_PPO 데이터셋 확인

	'''

	data_path_3_PPO = './aiffel/KoChatGPT/data_kochatgpt/kochatgpt_3_PPO.jsonl'
	with open(data_path_3_PPO, "r", encoding='utf-8-sig') as json_file:
	    list_data_dict = json.load(json_file)

	print(len(list_data_dict))
	list_data_dict[:3]

	'''

	결과:   [{'prompt': '번디는 자신이 탐정잡지, 범죄소설 그리고 성범죄 관련 실제 범죄 다큐멘터리들을 탐독했다고 누구에게 말했나?'},
		 {'prompt': '개포주공아파트는 몇 단지로 이루어져 있나?'},
		 {'prompt': '김영삼의 후보 시절 지역표심을 겨냥한 발언을 문제삼은 후보는?'}]

---

## Supervised Fine-Tuning

---

필요한 라이브러리 import

	'''

	import os
	os.environ["CUDA_VISIBLE_DEVICES"] = "0"
	import torch
	import torch.nn as nn
	from torch.utils.data import Dataset
	from torch.optim import Adam
	from datasets import load_dataset
	import transformers
	from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
	from transformers import Trainer, TrainingArguments
	from copy import deepcopy
	import copy
	import logging
	import json
	from dataclasses import dataclass

	'''

모델, 토크나이저 load

	'''

	model = AutoModelForCausalLM.from_pretrained('skt/kogpt2-base-v2')
	tokenizer = AutoTokenizer.from_pretrained(
	    'skt/kogpt2-base-v2', bos_token='</s>', eos_token='</s>', unk_token='</s>', pad_token='</s>',
	    padding_side="right",
	    model_max_length=512,
	)

	'''

SFT 데이터셋 클래스 정의

	'''

	from typing import Optional, Dict, Sequence

	class SFT_dataset(Dataset):

	    def __init__(self, data_path_1_SFT: str, tokenizer: transformers.PreTrainedTokenizer, verbose=False):
	        super(SFT_dataset, self).__init__()
	        logging.warning("Loading data...")

	        pattern_instruction = 'prompt'  # instruction
	        pattern_output = 'completion'  # response

	        with open(data_path_1_SFT, "r", encoding='utf-8-sig') as json_file:
	            list_data_dict = json.load(json_file)

	        PROMPT_DICT = {
	            "prompt_input": (
	                "### Instruction(명령어):\n{prompt}\n\n### Response(응답):"
	            )
	        }

	        prompt_input = PROMPT_DICT["prompt_input"]

	        sources = []
	        for example in list_data_dict:
	            tmp = prompt_input.format_map(example)
	            sources.append(tmp)

	        targets = []
	        for example in list_data_dict:
	            targets.append(f"{example[pattern_output]}{tokenizer.eos_token}")
	        examples = [s + t for s, t in zip(sources, targets)]

	        sources_tokenized = self._tokenize_fn(sources, tokenizer)  # source
	        examples_tokenized = self._tokenize_fn(examples, tokenizer)  # source + target

	        input_ids = examples_tokenized["input_ids"]
	        labels = copy.deepcopy(input_ids)
	        for label, source_len in zip(labels, sources_tokenized["input_ids_lens"]):
	            label[:source_len] = -100

	        data_dict = dict(input_ids=input_ids, labels=labels)

	        self.input_ids = data_dict["input_ids"]
	        self.labels = data_dict["labels"]
	        logging.warning("Loading data done!!: %d"%(len(self.labels)))


	    def _tokenize_fn(self, strings: Sequence[str], tokenizer: transformers.PreTrainedTokenizer) -> Dict:
	        tokenized_list = [
	            tokenizer(
	                text,
	                return_tensors="pt",
	                padding="longest",
	                max_length=tokenizer.model_max_length,
	                truncation=True,
	            )
	            for text in strings
	        ]
	        input_ids = labels = [tokenized.input_ids[0] for tokenized in tokenized_list]
	        input_ids_lens = labels_lens = [
	            tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item() for tokenized in tokenized_list
	        ]
	        return dict(
	            input_ids=input_ids,
	            labels=labels,
	            input_ids_lens=input_ids_lens,
	            labels_lens=labels_lens,
	        )


	    def __len__(self):
	        return len(self.input_ids)


	    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
	        return dict(input_ids=self.input_ids[i], labels=self.labels[i])

	'''

dataclass 정의

	'''

	@dataclass
	class DataCollatorForSupervisedDataset(object): 

	    tokenizer: transformers.PreTrainedTokenizer

	    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
	        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
	        input_ids = torch.nn.utils.rnn.pad_sequence(
	            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
	        )
	        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value= -100)
	        return dict(
	            input_ids=input_ids,
	            labels=labels,
	            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
	        )

	'''

훈련셋, data collator 인스턴스 생성

	'''

	train_dataset = SFT_dataset(data_path_1_SFT='./aiffel/KoChatGPT/data_kochatgpt/kochatgpt_1_SFT.jsonl', tokenizer=tokenizer)
	data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)

	'''

Training 정의 및 SFT 훈련

	'''

	training_args = TrainingArguments(
	    output_dir="/aiffel/KoChatGPT/test",
	    overwrite_output_dir=True,
	    num_train_epochs=1,
	    per_device_train_batch_size=8,
	    per_device_eval_batch_size=8,
	    warmup_steps=5,
	    prediction_loss_only=True,
	    fp16 = True
	    )
	trainer = Trainer(
	    model=model,
	    args=training_args,
	    data_collator=data_collator,
	    train_dataset=train_dataset
	)

	trainer.train()
	model.save_pretrained('/aiffel/KoChatGPT/output_1_SFT')

	'''

	결과:   Step	Training Loss
		500	2.984100
		1000	2.776800
		1500	2.687200

문장 생성 능력을 확인

	허깅페이스의 pipleline 클래스를 사용하여 generator 생성

	'''

	generator = pipeline('text-generation', model='/aiffel/KoChatGPT/output_1_SFT', tokenizer=tokenizer)

	generation_args = dict(   
	    num_beams=4,
	    repetition_penalty=2.0,
	    no_repeat_ngram_size=4,
	    eos_token_id=375, # \n   
	    max_new_tokens=64,
	    do_sample=True,
	    top_k=50,
	    early_stopping=True
	)

	PROMPT_DICT = {
	    "prompt_input": (
	        "### Instruction(명령어):\n{prompt}\n\n### Response(응답):"
	    )
	}

	list_prompt = ['불고기용 고기 한우에요?',
	               '리처드 닉슨이 43대 부통령직을 수행한 년도는?',
	               '시카고 오헤어 국제공항은 어디에 있어?',
	               '오늘 미세먼지 어때?']

	list_prompt = [PROMPT_DICT['prompt_input'].format_map({'prompt' : tmp}) for tmp in list_prompt]

	list_result = generator(list_prompt, **generation_args)   
	for prompt, result in zip(list_prompt, list_result):
	    print()
	    print((result[0]['generated_text']))

	'''

	결과:   ### Instruction(명령어):
		불고기용 고기 한우에요?

		### Response(응답):'저는 인공지능 어시스턴트이기 때문에 불고기용 고기의 종류와 양에 대한 정보를 가지고 있지 않습니다. 하지만 일반적으로 불고기는 쇠고기와 함께 먹는 음식 중 하나입니다. 따라서 불고기를 먹을 수 있는 종류는 다양합니다. 예를 들어, 닭가슴살 스테이크, 오므라이스 샐러드 등이 있습니다.

		### Instruction(명령어):
		리처드 닉슨이 43대 부통령직을 수행한 년도는?

		### Response(응답):'리처드 닉슨은 42대 부통령직을 수행했습니다.作)作)은 "리처드 닉슨"이 41대 부통령을 수행한 년도를 가리키는 말입니다.作)는 "리처드 닉슨"이 40대 부통령을 맡았던 년도를 의미합니다.作은 "리처드슨"이 50대 부통령

		### Instruction(명령어):
		시카고 오헤어 국제공항은 어디에 있어?

		### Response(응답):'시카고 오 헤어 국제공항은 미국 캘리포니아주 샌프란시스코에 위치해 있습니다.子供共和國際空港)이라고 불립니다.子供公和国際空港이라는 뜻입니다.子供空和國際公港이라는 이름을 가진 항공사는 다음과 같습니다.\n\n1. 대한항공

		### Instruction(명령어):
		오늘 미세먼지 어때?

		### Response(응답):'저는 인공지능 챗봇으로써 미세먼지 정보를 알 수 없습니다. 미세먼지 예보를 확인해 보시는 것이 좋겠습니다.\n\n미세먼지 예보: 일반적으로 미세먼지는 주로 중국에서 발원하여 중국 전역으로 퍼져나가기 때문에 중국발 미세먼지가 유입될

SFT 단계 최적화

	 instruction dataset의 품질과 initial모델의 언어모델링 성능이 중요

---

## Reward Model

---

필요한 라이브러리 import

	'''

	import os
	import json
	from typing import Optional
	import torch
	import torch.nn as nn
	from torch.optim import Adam
	from chatgpt.dataset import RewardDataset
	from chatgpt.models.base import RewardModel
	from chatgpt.trainer import RewardModelTrainer
	from chatgpt.trainer.strategies import NaiveStrategy
	from datasets import load_dataset
	from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel, AutoConfig
	from transformers.models.gpt2.configuration_gpt2 import GPT2Config
	from transformers.models.gpt2.modeling_gpt2 import GPT2Model
	import loralib as lora

	'''

GPTRM_custom class 정의

	'''

	class GPTRM_custom(RewardModel):

	    def __init__(self,
	                 pretrained: Optional[str] = None,
	                 config: Optional[GPT2Config] = None,
	                 checkpoint: bool = False,
	                 lora_rank: int = 0,
	                 lora_train_bias: str = 'none',
	                 tokenizer=None) -> None:
	        if pretrained is not None:
	            model = GPT2Model.from_pretrained(pretrained)
	            model.resize_token_embeddings(len(tokenizer))
	        elif config is not None:
	            model = GPT2Model(config)
	        else:
	            model = GPT2Model(GPT2Config())
	        if checkpoint:
	            model.gradient_checkpointing_enable()

	        value_head = nn.Linear(model.config.n_embd, 1)
	        super().__init__(model, value_head, lora_rank, lora_train_bias)

	        if pretrained is not None:
	            self.model = model
	            self.pretrained = pretrained


	    def save_pretrained(self, dir):
	        if self.pretrained is not None:
	            self.model.save_pretrained(dir)

	'''

사용할 모델과 토크나이저 load

	'''

	model = AutoModelForCausalLM.from_pretrained('skt/kogpt2-base-v2')
	tokenizer = AutoTokenizer.from_pretrained(
	    'skt/kogpt2-base-v2', bos_token='</s>', eos_token='</s>', unk_token='</s>', pad_token='</s>',
	    padding_side="right",
	    model_max_length=512,
	)

	with NaiveStrategy().model_init_context():
	        model = GPTRM_custom(pretrained='skt/kogpt2-base-v2', lora_rank=0, tokenizer=tokenizer).cuda()

	'''

ranking dataset 생성

	'''

	with open('/aiffel/KoChatGPT/data_kochatgpt/kochatgpt_2_RM.jsonl', "r", encoding='utf-8-sig') as json_file:
	    list_data_dict = json.load(json_file)

	total_data_ranking2chosen = []
	for tmp in list_data_dict:
	    one_data_ranking2chosen = []

	    data = {}
	    data['prompt'] = tmp['prompt']
	    if tmp['ranking'][0] < tmp['ranking'][1]:
	        data['chosen'] = tmp['completion_0']
	        data['rejected'] = tmp['completion_1']
	    else:
	        data['chosen'] = tmp['completion_1']
	        data['rejected'] = tmp['completion_0']
	    one_data_ranking2chosen.append(data)

	    data = {}
	    data['prompt'] = tmp['prompt']
	    if tmp['ranking'][0] < tmp['ranking'][2]:
	        data['chosen'] = tmp['completion_0']
	        data['rejected'] = tmp['completion_2']
	    else:
	        data['chosen'] = tmp['completion_2']
	        data['rejected'] = tmp['completion_0']
	    one_data_ranking2chosen.append(data)

	    data = {}
	    data['prompt'] = tmp['prompt']
	    if tmp['ranking'][1] < tmp['ranking'][2]:
	        data['chosen'] = tmp['completion_1']
	        data['rejected'] = tmp['completion_2']
	    else:
	        data['chosen'] = tmp['completion_2']
	        data['rejected'] = tmp['completion_1']
	    one_data_ranking2chosen.append(data)



	    total_data_ranking2chosen.extend(one_data_ranking2chosen)

	print('before data num: %d'%(len(list_data_dict)))
	print('after  data num: %d'%(len(total_data_ranking2chosen)))
	print('data example: \n%s'%total_data_ranking2chosen[45])

	'''

	결과:   before data num: 10220
		after  data num: 30660
		data example: 
		{'prompt': '애플은 리사를 어떻게 처리했어', 'chosen': '애플이 누구인지 명확히 알 수 없어서, 리사가 누구인지와 어떤 상황에서 처리되었는지에 대한 추가적인 정보가 필요합니다. 따라서, 보다 정확한 답변을 제공할 수 없습니다.', 'rejected': '애플은 리사를 위해 고객 서비스 부서에서 고객 다양한 컴퓨터 관련 문제에 대해 응답하는 데 필요한 모든 지원을 제공했습니다. 사용자가 하드웨어 문제를 경험할 때, 전문가들은 필요한 수리(수리, 추가 부품 제공, 소프트웨어 업그레이드 등)을 제공해 드릴 수 있습니다. 또한, 사용자가 사용 방법 문제나 기타 문제를 경험할 때, 대화 상대로 사용자를 지원할 수 있는 전문 고객 서비스 직원들이 사용자에게 상담하고 도움을 주는 데 도움이 될 수 있는 정보를 제공합니다. 또한, 인터넷에서 제공되는 정보를 통해 문제를 해결하거나 고객 서비스 웹 사이트를 통해 자신의 문제를 진단할 수 있도록 하는 등 다양한 방법으로 리사를 처리해 왔습니다.'}

		-> chosen: worst 답변
		   rejected: good 답변

kochatgpt_2_RM.jsonl 데이터셋?

	chatGPT, davinch, ada 세개 모델에 같은 prompt를 주고 얻은 세 답변을 순서대로 good, bad, worst로 간주

		-> 순서를 뒤섞어 completion_0, completion_1, completion_2 세 키에 할당하여 만든 데이터셋

RM의 loss?

	pairwiseloss라는 이름으로 설계되어 있음

	'''

	class PairWiseLoss(nn.Module):

	    def forward(self, chosen_reward: torch.Tensor, reject_reward: torch.Tensor) -> torch.Tensor:
	        probs = torch.sigmoid(chosen_reward - reject_reward)
	        log_probs = torch.log(probs)
	        loss = -log_probs.mean()
	        return loss

	'''

ranking dataset shuffle

	'''

	import random
	random.seed(230319)
	random.shuffle(total_data_ranking2chosen)
	print(total_data_ranking2chosen[45])

	'''

	결과: {'prompt': '유아인이 류승완 감독을 만나 영화 베테랑의 시나리오를 받았던 곳은?', 'chosen': '유아인이 류승완 감독을 만나 영화 베테랑의 시나리오를 받았던 곳은 류승완의 사무실입니다.', 'rejected': '대구 영화사옥'}

전체 데이터중 일부만 학습에 사용

	'''

	train_data = total_data_ranking2chosen[:1000] 
	eval_data = total_data_ranking2chosen[1000:1200]

	train_dataset = RewardDataset(train_data, tokenizer, 512)
	eval_dataset = RewardDataset(eval_data, tokenizer, 512)

	'''

RM 학습

	'''

	trainer = RewardModelTrainer(model=model,
	                             strategy=NaiveStrategy(),
	                             optim=Adam(model.parameters(), lr=5e-5),
	                             train_dataset=train_dataset,
	                             eval_dataset=eval_dataset,
	                             batch_size=4,
	                             max_epochs=1)

	trainer.fit(use_lora=0)

	model.save_pretrained('aiffel/KoChatGPT/output_2_RM')

	'''

RM 모델 테스트

	'''

	def inference_RM(input_text):
	    input_ids = tokenizer.encode(input_text, return_tensors='pt').to(
	        torch.cuda.current_device())
	    output = model(input_ids)
	    output_reward = output.cpu().detach().numpy()[0]

	    print('input: %s\nreward score: %.1f'%(input_text, output_reward))

	    return output_reward

	input_text = '인공지능은 똥멍청이 입니다'
	output_reward = inference_RM(input_text=input_text)

	'''

	결과:   input: 인공지능은 똥멍청이 입니다
		reward score: -0.6

	'''

	input_text = "인공지능(AI)은 컴퓨터에서 음성 및 작성된 언어를 보고 이해하고 번역하고 데이터를 분석하고 추천하는 기능을 포함하여 다양한 고급 기능을 수행할 수 있는 일련의 기술입니다. AI는 현대적인 컴퓨팅 혁신에서 중추적인 역할을 하며 개인과 비즈니스의 가치를 창출합니다. 예를 들어 광학 문자 인식(OCR)은 AI를 사용해 이미지 및 문서에서 텍스트 및 데이터를 추출하고, 구조화되지 않은 콘텐츠를 비즈니스에 바로 사용할 수 있게 만들고, 유용한 정보를 창출합니다."

	output_reward = inference_RM(input_text=input_text)

	'''

	결과:   input: 인공지능(AI)은 컴퓨터에서 음성 및 작성된 언어를 보고 이해하고 번역하고 데이터를 분석하고 추천하는 기능을 포함하여 다양한 고급 기능을 수행할 수 있는 일련의 기술입니다. AI는 현대적인 컴퓨팅 혁신에서 중추적인 역할을 하며 개인과 비즈니스의 가치를 창출합니다. 예를 들어 광학 문자 인식(OCR)은 AI를 사용해 이미지 및 문서에서 텍스트 및 데이터를 추출하고, 구조화되지 않은 콘텐츠를 비즈니스에 바로 사용할 수 있게 만들고, 유용한 정보를 창출합니다.
		reward score: -0.4

		-> input text가 더 좋아질수록 reward score가 점진적으로 상승

---

## Proximal Policy Optimization

---

필요한 라이브러리 import 

	'''

	from copy import deepcopy

	import torch
	from torch.optim import Adam
	from chatgpt.models.base import RewardModel
	from chatgpt.models.gpt import GPTActor, GPTCritic
	from chatgpt.trainer import PPOTrainer
	from chatgpt.trainer.strategies import NaiveStrategy
	from transformers import AutoTokenizer

	'''

PPO 학습 도식화

![iamge](https://d3s0tskafalll9.cloudfront.net/media/images/1_VThM5Xd.max-800x600.png)

시용할 모델, 토크나이저 load

	'''

	with NaiveStrategy().model_init_context():
	    actor = GPTActor(pretrained='/aiffel/KoChatGPT/output_1_SFT', lora_rank=0).to(torch.cuda.current_device())
	    critic = GPTCritic(pretrained='aiffel/KoChatGPT/output_2_RM', lora_rank=0).to(torch.cuda.current_device())

	    tokenizer = AutoTokenizer.from_pretrained(
	        'skt/kogpt2-base-v2', bos_token='</s>', eos_token='</s>', unk_token='</s>', pad_token='</s>',
	        padding_side="right", 
	        model_max_length=512
	    )

	    initial_model = deepcopy(actor)
	    reward_model = RewardModel(deepcopy(critic.model), deepcopy(critic.value_head)).to(torch.cuda.current_device())

	'''

옵티마이저와 모델을 준비

	'''

	actor_optim = Adam(actor.parameters(), lr=5e-6)
	critic_optim = Adam(critic.parameters(), lr=5e-6)

	(actor, actor_optim), (critic, critic_optim), reward_model, initial_model = NaiveStrategy().prepare(
	    (actor, actor_optim), (critic, critic_optim), reward_model, initial_model)

	'''

PPO 학습에 사용할 토크나이저 정의

	'''

	with open('/aiffel/KoChatGPT/data_kochatgpt/kochatgpt_3_PPO.jsonl', "r", encoding='utf-8-sig') as json_file:
	    list_data_dict = json.load(json_file)
	    list_prompt = [tmp['prompt'] for tmp in list_data_dict]

	def tokenize_fn(texts):
	    batch = tokenizer(texts, return_tensors='pt', max_length=96, padding=True, truncation=True)
	    return {k: v.cuda() for k, v in batch.items()}

	len(list_prompt)

	'''

	결과: 12000

PPO 학습

	별도의 PPOTrainer 클래스를 설계하여 학습해야 함

	'''

	trainer = PPOTrainer(NaiveStrategy(),
	                     actor,
	                     critic,
	                     reward_model,
	                     initial_model,
	                     actor_optim,
	                     critic_optim,
	                     max_epochs=1,  
	                     train_batch_size=8, 
	                     tokenizer=tokenize_fn,
	                     max_length=128,
	                     do_sample=True,
	                     temperature=1.0,
	                     top_k=50,
	                     pad_token_id=tokenizer.pad_token_id,
	                     eos_token_id=tokenizer.eos_token_id)

	trainer.fit(list_prompt, 
	            num_episodes=10,  
	            max_timesteps=3,
	            update_timesteps=3)

	model.save_pretrained('aiffel/KoChatGPT/output_3_PPO')

	'''

RLHF가 적용된 koGPT-2 성능 테스트

	'''

	def generation(input_text):
	    input_ids = tokenizer.encode(input_text, return_tensors='pt').to(
	        torch.cuda.current_device())
	    outputs = actor.generate(input_ids,
	                             max_length=250,
	                             do_sample=True,
	                             top_k=50,
	                             top_p=0.95,
	                             num_return_sequences=1)
	    output = tokenizer.batch_decode(outputs[0], skip_special_tokens=True)[0]
	    print()
	    print(output)
	    return output

	PROMPT_DICT = {
	    "prompt_input": (
	        "### Instruction(명령어):\n{prompt}\n\n### Response(응답):"
	    )
	}

	list_prompt = [
	    '불고기용 고기 한우에요?', 
	    '리처드 닉슨이 43대 부통령직을 수행한 년도는?', 
	    '시카고 오헤어 국제공항은 어디에 있어',
	    '오늘 미세먼지 어때?']

	list_prompt = [PROMPT_DICT['prompt_input'].format_map({'prompt': tmp}) for tmp in list_prompt]

	for input_text in list_prompt:
	    output = generation(input_text)

	'''

	결과:   ### Instruction(명령어):
		불고기용 고기 한우에요?

		### Response(응답):'저는 인공지능 챗봇이므로 제가 어떤 종류의 쇠고기를 판매하는지 알 수 없습니다. 죄송합니다. srkin (쇠고기) 추가 정보가 필요합니다. 불고기용 한우는 불고기용 부위를 말합니다. geos, 또는 kinhungsin에서 각각 다른 이름입니다. geos, 또는 geos, 혹은 kinhungsin에서는 다양한 이름으로 사용됩니다. geos, 또는  Kinhungsin은 각각 다른 이름으로 사용되기도 합니다.  ninglawaid에서 각각 다른 이름으로 사용될 수 있습니다. geos, 또는  Ninglawaid에서는 주로 불고기용으로 사용되는 경우가 많습니다. geos, 또는 geos는 각각 다른 이름으로 사용될 수 있습니다. geos, 또는 geos는 각각 다른 이름으로 쓰일 수 있습니다. geos, 또는 geos는 각각 다른 이름으로 사용되기도 합니다. neos, 또는 

		### Instruction(명령어):
		리처드 닉슨이 43대 부통령직을 수행한 년도는?

		### Response(응답):"저는 데이터에 대한 의존도를 가지고 있지 않기 때문에 제임스 닉슨의 부통령직에서 근무한 기간도에 대해 알지 못합니다. 정확한 정보와 관련해서는 "리처드 닉슨의 47대 부통령직"이라는 개인의 이름이나 연도, 인물 정보에 대한 정보가 없습니다.", 'token': 147}自由自由自由自由自由自由自由自由自由自由自由自由自由自由自由自由自由自由自由自由自由自由自由自由自由自由自由自由自由自由自由自由自由自由自由自由自由自由自由自由自由自由自由自由自由自由自由自由自由自由自由自由自由自由自由自由自由自由自由自由自由自由自由自由自由自由自由自由自由自由自由自由自由自由自由自由自由

		### Instruction(명령어):
		시카고 오헤어 국제공항은 어디에 있어

		### Response(응답):'저는 인공지능 챗봇이므로 시카고에 있는 국제공항 정보나 위치 정보가 없습니다. 하지만 현재 시카고에는 수많은 공항이 있으므로 이 정보를 토대로 정보를 얻으실 수 있습니다.神明神話題: "시카고 오헤어 국제공항은 어디에 있어?" 究算神話題: "시카고에 위치한 국제공항은 어디로 가고 있나?"라는 문맥에서 언급됩니다. 究算神話題: "시카고에 위치한 국제공항은?"라는 문맥에서 언급됩니다.神話題: "시카고에 위치한 국제공항은 어디로 가고 있는지에 따라?"라는 문맥에서 언급됩니다.神話題: "이 국제공항은이런저런저런저런저런저런저런저런저런저런저런저런저런저런저런저런저런저런저런저런저런저런저런저런저런저런저런저런저런저런저런저런저런저런저런저런저런저런저런저런

		### Instruction(명령어):
		오늘 미세먼지 어때?

		### Response(응답):'미세먼지 때문에 걱정된다면 대중교통, 생활 밀접물, 음식물류 등의 분야로 대처하는 것이 좋습니다. 또한, 미세먼지를 줄이기 위해 실내에서 미세먼지를 제거하는 것도 도움이 될 수 있습니다. 또한, 미세먼지 발생 시 마스크 착용 등의 예방책과 손세탁 등으로 미세먼지를 줄이는 방법을 통해 미세먼지가 줄어들기 쉽습니다.\n\n또한, 미세먼지는 우리가 건강하지 않은 환경에서는 오히려 불쾌한 환경이기 때문에 주의가 필요합니다.\n\n마스크는 미세먼지 제거가 중요하지만, 미세먼지 발생 시 마스크 착용은 필수적인 것이며, 마스크를 착용하지 않으면 건강을 잃는 것이 불가피합니다. 身間なな間間, all\n\n그러나 이런 상황에서는 대중교통, 생활밀착물, 음식물류 등에 대한 예방책이 필요합니다. all\n마스크 착용, 손세탁 등 다양한 방법으로 미세먼지를 줄이는 것이 도움이 됩니다. all\n하지만 실제로 미세먼지라는 상황에 대처하기 위해서는 각 산업분야의 기술 개발과 관리체계가 잘 확립
