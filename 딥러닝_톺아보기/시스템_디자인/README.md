# 학습 내용

---

- 시스템 디자인 분석

---

## 시스템 디자인 분석

---

[Paperswithcode](https://paperswithcode.com/)에서 관심있는 태스크의 모델을 찾아 분석

- 디렉토리 구조 : 레포에서 확인하셔도 좋습니다
- GPT를 활용한 파일 분석
- GPT를 활용한 코드 분석
- 다이어그래밍 툴 [Mermaid](https://mermaid.js.org/intro/)를 활용한 차트 제작

---

- **LeakGAN** → 텍스트 생성 모델
    - 디렉토리 구조
        - [Image COCO](https://github.com/CR-Gjx/LeakGAN/tree/master/Image%20COCO) (Image COCO 데이터 훈련 모델)
            - ckpts (모델 체크포인트)
                - [checkpoint](https://github.com/CR-Gjx/LeakGAN/blob/master/Image%20COCO/ckpts/checkpoint)
            - save (저장된 가중치들)
                - [experiment-log.txt](https://github.com/CR-Gjx/LeakGAN/blob/master/Image%20COCO/save/experiment-log.txt)
                - [generator_sample.txt](https://github.com/CR-Gjx/LeakGAN/blob/master/Image%20COCO/save/generator_sample.txt)
                - [realtest_coco.txt](https://github.com/CR-Gjx/LeakGAN/blob/master/Image%20COCO/save/realtest_coco.txt)
                - [realtrain_cotra.txt](https://github.com/CR-Gjx/LeakGAN/blob/master/Image%20COCO/save/realtrain_cotra.txt)
                - [significance_test_sample.pkl](https://github.com/CR-Gjx/LeakGAN/blob/master/Image%20COCO/save/significance_test_sample.pkl)
                - [vocab_cotra.pkl](https://github.com/CR-Gjx/LeakGAN/blob/master/Image%20COCO/save/vocab_cotra.pkl)
            - speech (생성된 텍스트 예시)
                
                [generated_coco_examples.txt](https://github.com/CR-Gjx/LeakGAN/blob/master/Image%20COCO/speech/generated_coco_examples.txt)
                
            - [Discriminator.py](https://github.com/CR-Gjx/LeakGAN/blob/master/Image%20COCO/Discriminator.py) (판별자 정의)
            - [LeakGANModel.py](https://github.com/CR-Gjx/LeakGAN/blob/master/Image%20COCO/LeakGANModel.py) (LeakGAN model 정의)
            - [Main.py](https://github.com/CR-Gjx/LeakGAN/blob/master/Image%20COCO/Main.py) (훈련에 필요한 메서드 정의)
            - [convert.py](https://github.com/CR-Gjx/LeakGAN/blob/master/Image%20COCO/convert.py) (생성된 텍스트 저장 메서드 정의)
            - [dataloader.py](https://github.com/CR-Gjx/LeakGAN/blob/master/Image%20COCO/dataloader.py) (데이터 전처리 메서드 정의)
            - [eval_bleu.py](https://github.com/CR-Gjx/LeakGAN/blob/master/Image%20COCO/eval_bleu.py) (BLEU 점수 평가)
        - [No Temperature](https://github.com/CR-Gjx/LeakGAN/tree/master/No%20Temperature) (확률적 샘플링을 사용하지 않은 모델)
            - [Image COCO](https://github.com/CR-Gjx/LeakGAN/tree/master/No%20Temperature/Image%20COCO) (Image COCO 데이터 훈련 모델)
                - ckpts (모델 체크포인트)
                    - [checkpoint](https://github.com/CR-Gjx/LeakGAN/blob/master/No%20Temperature/Image%20COCO/ckpts/checkpoint)
                - save (저장된 가중치들)
                    - [experiment-log.txt](https://github.com/CR-Gjx/LeakGAN/blob/master/No%20Temperature/Image%20COCO/save/experiment-log.txt)
                    - [generator_sample.txt](https://github.com/CR-Gjx/LeakGAN/blob/master/No%20Temperature/Image%20COCO/save/generator_sample.txt)
                    - [realtest_coco.txt](https://github.com/CR-Gjx/LeakGAN/blob/master/No%20Temperature/Image%20COCO/save/realtest_coco.txt)
                    - [realtrain_cotra.txt](https://github.com/CR-Gjx/LeakGAN/blob/master/No%20Temperature/Image%20COCO/save/realtrain_cotra.txt)
                    - [significance_test_sample.pkl](https://github.com/CR-Gjx/LeakGAN/blob/master/No%20Temperature/Image%20COCO/save/significance_test_sample.pkl)
                    - [vocab_cotra.pkl](https://github.com/CR-Gjx/LeakGAN/blob/master/No%20Temperature/Image%20COCO/save/vocab_cotra.pkl)
                - speech (생성된 텍스트 예시)
                    
                    [generated_coco_examples.txt](https://github.com/CR-Gjx/LeakGAN/blob/master/No%20Temperature/Image%20COCO/speech/generated_coco_examples.txt)
                    
                - [Discriminator.py](https://github.com/CR-Gjx/LeakGAN/blob/master/No%20Temperature/Image%20COCO/Discriminator.py) (판별자 정의)
                - [LeakGANModel.py](https://github.com/CR-Gjx/LeakGAN/blob/master/No%20Temperature/Image%20COCO/LeakGANModel.py) (LeakGAN model 정의)
                - [Main.py](https://github.com/CR-Gjx/LeakGAN/blob/master/No%20Temperature/Image%20COCO/Main.py) (훈련에 필요한 메서드 정의)
                - [convert.py](https://github.com/CR-Gjx/LeakGAN/blob/master/No%20Temperature/Image%20COCO/convert.py) (생성된 텍스트 저장 메서드 정의)
                - [dataloader.py](https://github.com/CR-Gjx/LeakGAN/blob/master/No%20Temperature/Image%20COCO/dataloader.py) (데이터 전처리 메서드 정의)
                - [eval_bleu.py](https://github.com/CR-Gjx/LeakGAN/blob/master/No%20Temperature/Image%20COCO/eval_bleu.py) (BLEU 점수 평가)
            - [Synthetic Data](https://github.com/CR-Gjx/LeakGAN/tree/master/No%20Temperature/Synthetic%20Data) (합성 데이터 훈련 모델)
                - ckpts (모델 체크포인트)
                    - [checkpoint](https://github.com/CR-Gjx/LeakGAN/blob/master/No%20Temperature/Synthetic%20Data/ckpts/checkpoint)
                - save (저장된 가중치들)
                    - [eval_file.txt](https://github.com/CR-Gjx/LeakGAN/blob/master/No%20Temperature/Synthetic%20Data/save/eval_file.txt)
                    - [experiment-log.txt](https://github.com/CR-Gjx/LeakGAN/blob/master/No%20Temperature/Synthetic%20Data/save/experiment-log.txt)
                    - [experiment-log40.txt](https://github.com/CR-Gjx/LeakGAN/blob/master/No%20Temperature/Synthetic%20Data/save/experiment-log40.txt)
                    - [generator_sample.txt](https://github.com/CR-Gjx/LeakGAN/blob/master/No%20Temperature/Synthetic%20Data/save/generator_sample.txt)
                    - [real_data.txt](https://github.com/CR-Gjx/LeakGAN/blob/master/No%20Temperature/Synthetic%20Data/save/real_data.txt)
                    - [target_params.pkl](https://github.com/CR-Gjx/LeakGAN/blob/master/No%20Temperature/Synthetic%20Data/save/target_params.pkl)
                - [Discriminator.py](https://github.com/CR-Gjx/LeakGAN/blob/master/No%20Temperature/Synthetic%20Data/Discriminator.py) (판별자 정의)
                - [LeakGANModel.py](https://github.com/CR-Gjx/LeakGAN/blob/master/No%20Temperature/Synthetic%20Data/LeakGANModel.py) (LeakGAN model 정의)
                - [Main.py](https://github.com/CR-Gjx/LeakGAN/blob/master/No%20Temperature/Synthetic%20Data/Main.py) (훈련에 필요한 메서드 정의)
                - [__init__.py](https://github.com/CR-Gjx/LeakGAN/blob/master/No%20Temperature/Synthetic%20Data/__init__.py) (내용 X)
                - [dataloader.py](https://github.com/CR-Gjx/LeakGAN/blob/master/No%20Temperature/Synthetic%20Data/dataloader.py) (데이터 전처리 메서드 정의)
                - [target_lstm.py](https://github.com/CR-Gjx/LeakGAN/blob/master/No%20Temperature/Synthetic%20Data/target_lstm.py) (길이 40인 LSTM 기반 텍스트 생성 정의)
                - [target_lstm20.py](https://github.com/CR-Gjx/LeakGAN/blob/master/No%20Temperature/Synthetic%20Data/target_lstm20.py) (길이 20인 LSTM 기반 텍스트 생성 정의)
        - [Synthetic Data](https://github.com/CR-Gjx/LeakGAN/tree/master/Synthetic%20Data) (합성 데이터 훈련 모델)
            - ckpts (모델 체크포인트)
                
                [checkpoint](https://github.com/CR-Gjx/LeakGAN/blob/master/Synthetic%20Data/ckpts/checkpoint) 
                
            - save (저장된 가중치들)
                - [eval_file.txt](https://github.com/CR-Gjx/LeakGAN/blob/master/Synthetic%20Data/save/eval_file.txt)
                - [experiment-log.txt](https://github.com/CR-Gjx/LeakGAN/blob/master/Synthetic%20Data/save/experiment-log.txt)
                - [experiment-log40.txt](https://github.com/CR-Gjx/LeakGAN/blob/master/Synthetic%20Data/save/experiment-log40.txt)
                - [generator_sample.txt](https://github.com/CR-Gjx/LeakGAN/blob/master/Synthetic%20Data/save/generator_sample.txt)
                - [real_data.txt](https://github.com/CR-Gjx/LeakGAN/blob/master/Synthetic%20Data/save/real_data.txt)
                - [target_params.pkl](https://github.com/CR-Gjx/LeakGAN/blob/master/Synthetic%20Data/save/target_params.pkl)
            - [Discriminator.py](https://github.com/CR-Gjx/LeakGAN/blob/master/Synthetic%20Data/Discriminator.py) (판별자 정의)
            - [LeakGANModel.py](https://github.com/CR-Gjx/LeakGAN/blob/master/Synthetic%20Data/LeakGANModel.py) (LeakGAN model 정의)
            - [Main.py](https://github.com/CR-Gjx/LeakGAN/blob/master/Synthetic%20Data/Main.py) (훈련에 필요한 메서드 정의)
            - [__init__.py](https://github.com/CR-Gjx/LeakGAN/blob/master/Synthetic%20Data/__init__.py) (내용 X)
            - [dataloader.py](https://github.com/CR-Gjx/LeakGAN/blob/master/Synthetic%20Data/dataloader.py) (데이터 전처리 메서드 정의)
            - [target_lstm.py](https://github.com/CR-Gjx/LeakGAN/blob/master/Synthetic%20Data/target_lstm.py) (길이 40인 LSTM 기반 텍스트 생성 정의)
            - [target_lstm20.py](https://github.com/CR-Gjx/LeakGAN/blob/master/Synthetic%20Data/target_lstm20.py) (길이 20인 LSTM 기반 텍스트 생성 정의)
    - GPT를 활용한 파일 분석
        - [Image COCO](https://github.com/CR-Gjx/LeakGAN/tree/master/Image%20COCO)
            - **`ckpts/`**: 모델 체크포인트 저장 폴더.
            - **`save/`**: 학습된 모델을 저장하는 폴더.
            - **`speech/`**: 스피치 데이터 관련 폴더.
            - **`Discriminator.py`**: Discriminator 모델 코드.
            - **`LeakGANModel.py`**: LeakGAN 모델의 주요 코드.
            - **`Main.py`**: 메인 학습 실행 코드.
            - **`convert.py`**: 데이터 변환 코드.
            - **`dataloader.py`**: 데이터 로더 코드.
            - **`eval_bleu.py`**: BLEU 점수 평가 코드.
        - [Synthetic Data](https://github.com/CR-Gjx/LeakGAN/tree/master/Synthetic%20Data)
            - **`ckpts/`**: 모델 체크포인트 저장.
            - **`save/`**: 모델 결과 저장 폴더.
            - **`Discriminator.py`**: Discriminator(구분자) 모델 구현.
            - **`LeakGANModel.py`**: LeakGAN 모델 핵심 코드.
            - **`Main.py`**: 모델 학습 실행 파일.
            - **`__init__.py`**: 폴더를 패키지로 인식하게 하는 초기화 파일.
            - **`dataloader.py`**: 데이터 로딩 관련 코드.
            - **`target_lstm.py` 및 `target_lstm20.py`**: LSTM 기반 텍스트 생성 관련 코드.
        - [No Temperature](https://github.com/CR-Gjx/LeakGAN/tree/master/No%20Temperature)
            
            위 Image COCO 모델, 합성 데이터 모델 디렉토리가 동일하게 존재 하므로 설명 생략
            
    - GPT를 활용한 코드 분석
        - Discriminator.py
            
            ### 1. **cosine_similarity 함수**
            
            - 두 벡터 `a`와 `b`의 **코사인 유사도(cosine similarity)**를 계산합니다.
            - 두 벡터를 정규화한 후, 각각의 요소를 곱하여 코사인 유사도를 계산합니다.
            
            ### 2. **linear 함수**
            
            - 주어진 입력을 선형 변환하는 함수로, 출력은 `input_` 텐서에 가중치 행렬 `W`를 곱하고, 편향 `b`를 더하는 식으로 계산됩니다.
            - 이 함수는 일반적으로 신경망 레이어에서 사용되는 연산을 수행합니다. 여기서는 `input_`이 2차원(batch 크기 x 입력 크기)이어야 합니다.
            
            ### 3. **highway 함수**
            
            - 이 함수는 **하이웨이 네트워크**를 구현한 것입니다. 하이웨이 네트워크는 **transform gate**와 **carry gate**를 사용해 정보가 입력에서 출력으로 직접 전달될지 여부를 제어합니다.
            - 입력을 여러 층에 걸쳐서 처리하며, 각 층에서 `t`라는 게이트를 이용하여 정보를 얼마나 유지하고 변형할지 결정합니다.
            
            ### 4. **Discriminator 클래스**
            
            - 이 클래스는 판별자의 전체 구조를 정의합니다.
            - LeakGAN의 판별자는 문장을 입력으로 받아, 해당 문장이 실제 문장(진짜)인지 생성자가 만든 가짜 문장인지 구분하는 역할을 합니다.
            
            ### 4.1 **초기화 함수 (`__init__`)**
            
            - `sequence_length`, `num_classes`, `vocab_size` 등 다양한 하이퍼파라미터들을 받아 초기화합니다.
            - 문장과 레이블 데이터를 담는 플레이스홀더(`input_x`, `input_y`)와 드롭아웃 확률을 위한 플레이스홀더를 정의합니다.
            - `FeatureExtractor_unit`를 통해 문장에서 특징(feature)을 추출하는 모듈을 정의하고, 이를 기반으로 문장을 분류(classification)하는 연산을 정의합니다.
            
            ### 4.2 **FeatureExtractor 함수**
            
            - 이 함수는 문장에서 특징을 추출하는 역할을 합니다.
            - 문장의 단어를 임베딩(embedding)한 후, 다양한 크기의 필터를 사용하는 합성곱층(CNN)을 통해 중요한 특징들을 추출합니다.
            - 추출된 특징은 하이웨이 네트워크와 드롭아웃을 거쳐 최종적으로 문장의 특징 벡터로 변환됩니다.
            
            ### 4.3 **classification 함수**
            
            - `FeatureExtractor`에서 추출된 특징 벡터를 입력으로 받아 문장이 어느 클래스(진짜 또는 가짜)에 속하는지 예측합니다.
            - 가중치 행렬 `W_d`와 편향 `b_d`를 통해 예측 스코어를 계산하고, `softmax`를 통해 확률로 변환합니다.
            - 여기서 사용된 `l2_loss`는 모델의 가중치를 규제(regularization)하는 역할을 하며, 오버피팅을 방지하는 데 사용됩니다.
        - LeakGANModel.py
            
            ### `__init__` 메소드:
            
            1. **입력 파라미터:**
                - `sequence_length`: 생성할 시퀀스(문장)의 길이.
                - `num_classes`: 분류 클래스의 개수.
                - `vocab_size`: 사용되는 단어 사전의 크기(어휘 크기).
                - `emb_dim`: 임베딩 벡터의 차원.
                - `dis_emb_dim`: 판별자의 임베딩 차원.
                - `filter_sizes`: CNN 필터의 크기 목록.
                - `num_filters`: CNN 필터의 개수.
                - `batch_size`: 한 번에 처리할 배치(batch)의 크기.
                - `hidden_dim`: LSTM에서 사용할 은닉층의 크기.
                - `start_token`: 생성 시퀀스의 시작 토큰.
                - `goal_out_size`: 목표 출력 크기.
                - `goal_size`: 목표 크기.
                - `step_size`: 단계 크기.
                - `D_model`: 판별자 모델(`Discriminator`).
                - `LSTMlayer_num`: LSTM 레이어의 수.
                - `l2_reg_lambda`: L2 정규화의 람다 값.
                - `learning_rate`: 학습률.
            2. **주요 초기화 변수:**
                - `self.sequence_length`부터 `self.learning_rate`까지는 입력 파라미터를 클래스 내부 변수로 저장.
                - `self.start_token`: 시작 토큰을 설정.
                - `self.worker_params`와 `self.manager_params`: 각각 워커와 매니저에 대한 파라미터.
                - `self.scope`: 판별자의 스코프.
                - `self.epis`와 `self.tem`: `LeakGAN`의 에피소드와 온도 관련 하이퍼파라미터.
            3. **`tf.placeholder` 정의:**
                - `self.x`: 생성된 토큰 시퀀스를 입력받는 자리표시자.
                - `self.reward`: 생성된 시퀀스에 대한 보상값을 받는 자리표시자.
                - `self.given_num`: 주어진 토큰의 개수를 받는 자리표시자.
                - `self.drop_out`: 드롭아웃 확률을 설정하는 자리표시자.
                - `self.train`: 훈련 여부를 결정하는 자리표시자.
            
            ### `Worker` 및 `Manager` 관련 블록:
            
            1. **`with tf.variable_scope('Worker')`:**
                - 이 블록은 워커 네트워크(Worker Network)에 해당합니다. 텍스트 생성을 주도하는 부분입니다.
                - **`self.g_embeddings`**: 어휘 크기와 임베딩 차원을 갖는 임베딩 행렬을 정의합니다. 이 행렬은 단어를 고유한 벡터로 변환하는 데 사용됩니다.
                - **`self.g_worker_recurrent_unit` 및 `self.g_worker_output_unit`**: 워커의 순환 유닛과 출력 유닛을 생성합니다. 순환 유닛은 이전 상태에서 새로운 상태로 업데이트하는 역할을 하고, 출력 유닛은 단어 생성에 필요한 출력을 만듭니다.
                - **`self.W_workerOut_change` 및 `self.g_change`**: 워커의 출력값을 목표 공간으로 변환하는 데 필요한 가중치 행렬입니다.
                - **`self.h0_worker`**: 워커 네트워크의 초기 은닉 상태를 0으로 설정합니다.
            2. **`with tf.variable_scope('Manager')`:**
                - 매니저 네트워크는 워커의 목표(goal)를 설정하고 이를 전달합니다.
                - **`self.g_manager_recurrent_unit` 및 `self.g_manager_output_unit`**: 매니저의 순환 유닛과 출력 유닛을 생성합니다. 이 유닛들은 매니저의 목표 상태를 관리하는 데 사용됩니다.
                - **`self.h0_manager`**: 매니저 네트워크의 초기 은닉 상태를 0으로 설정합니다.
                - **`self.goal_init`**: 초기 목표값을 설정하는 변수입니다.
            3. **`self.padding_array`**:
                - 패딩을 위한 배열로, 배치(batch) 처리 시 시퀀스의 길이가 다를 경우 부족한 부분을 패딩하여 균일한 길이를 맞춰줍니다.
            
            ### `rollout` 함수:
            
            - 이 부분은 시퀀스의 미래 보상을 예측하기 위한 롤아웃(Rollout) 기술을 적용합니다. `self.gen_for_reward`는 생성된 시퀀스에 대해 보상을 계산하는데 사용됩니다.
            
            ### `processed_x`:
            
            - **`tf.nn.embedding_lookup(self.g_embeddings, self.x)`**:
                - `self.x`에 포함된 인덱스를 사용하여 임베딩 벡터를 가져오는 함수입니다. 이 벡터는 워커와 매니저가 문장을 생성하는 데 사용하는 입력으로 변환됩니다.
            
            ### `tensor_array_ops.TensorArray` 사용:
            
            - `TensorArray`는 텐서플로우에서 동적인 텐서 배열을 다루기 위한 자료 구조입니다. 생성 중간 결과를 저장하고, 각 시퀀스의 상태를 관리하는 데 사용됩니다.
                - **`gen_o`**: 생성된 출력값을 저장하는 배열입니다.
                - **`gen_x`**: 생성된 시퀀스(토큰)를 저장하는 배열입니다.
                - **`goal`**: 매니저가 설정한 목표(goal)를 저장하는 배열입니다.
                - **`feature_array`**: 각 시퀀스의 특징을 저장하는 배열입니다.
                - **`real_goal_array` 및 `gen_real_goal_array`**: 실제 목표와 생성된 목표를 저장하는 배열입니다.
                - **`gen_o_worker_array`**: 워커가 생성한 출력값을 저장하는 배열입니다.
            
            ### `_g_recurrence` 함수:
            
            - 이 함수는 `LeakGAN`에서 순환(recurrence)적으로 동작하는 핵심적인 부분으로, 각 시퀀스의 상태와 목표를 업데이트합니다.
                - **`i`**: 현재 시퀀스의 인덱스.
                - **`x_t`**: 현재 시퀀스에서의 입력 토큰.
                - **`h_tm1`, `h_tm1_manager`**: 이전 단계에서의 워커와 매니저의 숨겨진 상태(hidden state).
                - **`gen_o`, `gen_x`**: 각각 생성된 출력과 시퀀스를 저장하는 배열.
                - **`goal`, `last_goal`**: 매니저가 설정한 목표와 이전 목표.
                - **`step_size`**: 단계 크기.
                - **`gen_real_goal_array`, `gen_o_worker_array`**: 각 단계에서 생성된 목표와 워커의 출력을 저장하는 배열.
            - **핵심 작업**:
                - **`cur_sen`**: 시퀀스의 현재 상태를 가져옵니다. 패딩이 필요한 경우 패딩 처리된 문장을 사용합니다.
                - **`self.FeatureExtractor_unit(cur_sen, self.drop_out)`**: 판별자가 현재 시퀀스에서 특징을 추출합니다.
                - **`h_t_Worker`**: 워커의 순환 유닛을 통해 현재 상태를 업데이트합니다.
    - 다이어그래밍 툴 [Mermaid](https://mermaid.js.org/intro/)를 활용한 차트 제작
        - Discriminator.py
            - https://www.mermaidchart.com/raw/5e0d4787-1126-462a-ac0f-b41b67e5a499?theme=light&version=v0.1&format=svg
        - LeakGANModel.py
            - https://www.mermaidchart.com/raw/12f983e3-46f1-4737-a99e-67cb02d49a9d?theme=light&version=v0.1&format=svg
