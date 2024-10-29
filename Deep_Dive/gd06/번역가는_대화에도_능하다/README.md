# 학습 내용

---

- 번역 데이터 준비
	- 번역을 위해 영어-스페인어 데이터셋 사용
- 번역 모델 만들기
	- 번역엔 뭐다? Transformere!
- 번역 성능 측정하기 (1) BLEU Score
	- 몇 점이면 훌륭한 번역기라고 할 수 있을까?
- 번역 성능 측정하기 (2) Beam Search Decoder
	- Beam search + BLEU = ?
- 데이터 부풀리기
	- 내 모델을 강하고 똑똑하게 만들기!

---

## 번역 데이터 준비

---

필요한 라이브러리 import

	'''

	import numpy as np
	import pandas as pd
	import tensorflow as tf
	import sentencepiece as spm
	from nltk.translate.bleu_score import sentence_bleu
	from nltk.translate.bleu_score import SmoothingFunction

	import re
	import os
	import random
	import math

	from tqdm.notebook import tqdm
	import matplotlib.pyplot as plt

	'''

영어-스페인어 데이터 다운

	'''

	zip_path = tf.keras.utils.get_file(
	    'spa-eng.zip',
	    origin='http://storage.googleapis.com/download.tensorflow.org/data/spa-eng.zip',
	    extract=True
	)

	'''

중복된 데이터 제거

	'''

	file_path = os.path.dirname(zip_path)+"/spa-eng/spa.txt"

	with open(file_path, "r") as f:
	    spa_eng_sentences = f.read().splitlines()

	spa_eng_sentences = list(set(spa_eng_sentences)) 
	total_sentence_count = len(spa_eng_sentences)
	print("Example:", total_sentence_count)

	'''

	결과: Example: 118964

전처리

	'''

	def preprocess_sentence(sentence):
    
	    # 소문자로 치환, 앞뒤 공백 제거
	    sentence = sentence.lower().strip()
	    # 다중 공백 -> 단일 공백
	    sentence = re.sub(r'[" "]+', " ", sentence)
    
	    return sentence

	spa_eng_sentences = list(map(preprocess_sentence, spa_eng_sentences))

	'''

데이터셋 분리

	'''

	test_sentence_count = total_sentence_count

	train_spa_eng_sentences = spa_eng_sentences[:-test_sentence_count]
	test_spa_eng_sentences = spa_eng_sentences[-test_sentence_count:]

	print("Train Example:", len(train_spa_eng_sentences))
	print("Test Example:", len(test_spa_eng_sentences))

	'''

	결과:   Train Example: 118370
		Test Example: 594

영어와 스페인어 분리

'''

	def split_spa_eng_sentences(spa_eng_sentences):
	    spa_sentences = []
	    eng_sentences = []
	    for spa_eng_sentence in tqdm(spa_eng_sentences):
	        eng_sentence, spa_sentence = spa_eng_sentence.split('\t')
	        spa_sentences.append(spa_sentence)
	        eng_sentences.append(eng_sentence)
	    return eng_sentences, spa_sentences

	train_eng_sentences, train_spa_sentences = split_spa_eng_sentences(train_spa_eng_sentences)

	test_eng_sentences, test_spa_sentences = split_spa_eng_sentences(test_spa_eng_sentences)

	'''

Sentencepiece 기반의 토크나이저 생성

	'''

	def generate_tokenizer(corpus,
	                       vocab_size,
	                       lang="spa-eng",
	                       pad_id=0,   # pad token의 일련번호
	                       bos_id=1,  # 문장의 시작을 의미하는 bos token(<s>)의 일련번호
	                       eos_id=2,  # 문장의 끝을 의미하는 eos token(</s>)의 일련번호
	                       unk_id=3):   # unk token의 일련번호
	    file = "./%s_corpus.txt" % lang
	    model = "%s_spm" % lang

	    with open(file, 'w') as f:
	        for row in corpus: f.write(str(row) + '\n')

	    import sentencepiece as spm
	    spm.SentencePieceTrainer.Train(
	        '--input=./%s --model_prefix=%s --vocab_size=%d'\
	        % (file, model, vocab_size) + \
	        '--pad_id==%d --bos_id=%d --eos_id=%d --unk_id=%d'\
	        % (pad_id, bos_id, eos_id, unk_id)
	    )

	    tokenizer = spm.SentencePieceProcessor()
	    tokenizer.Load('%s.model' % model)

	    return tokenizer

	VOCAB_SIZE = 20000
	tokenizer = generate_tokenizer(train_eng_sentences + train_spa_sentences, VOCAB_SIZE, 'spa-eng')
	tokenizer.set_encode_extra_options("bos:eos")  # 문장 양 끝에 <s> , </s>>추가

	'''

토큰화

	'''

	def make_corpus(sentences, tokenizer):
	    corpus = []
	    for sentence in tqdm(sentences):
	        tokens = tokenizer.encode_as_ids(sentence)
	        corpus.append(tokens)
	    return corpus

	eng_corpus = make_corpus(train_eng_sentences, tokenizer)
	spa_corpus = make_corpus(train_spa_sentences, tokenizer)

	'''

토큰 길이가 50이 되도록 패딩

	'''

	MAX_LEN = 50
	enc_ndarray = tf.keras.preprocessing.sequence.pad_sequences(eng_corpus, maxlen=MAX_LEN, padding='post')
	dec_ndarray = tf.keras.preprocessing.sequence.pad_sequences(spa_corpus, maxlen=MAX_LEN, padding='post')

	'''

데이터 셋 객체로 생성

	'''

	BATCH_SIZE = 64
	train_dataset = tf.data.Dataset.from_tensor_slices((enc_ndarray, dec_ndarray)).batch(batch_size=BATCH_SIZE)

	'''

---

## 번역 모델 만들기

---

트랜스포머 구현

Positional Encoding

	'''

	# Positional Encoding 구현
	def positional_encoding(pos, d_model):
	    def cal_angle(position, i):
	        return position / np.power(10000, (2*(i//2)) / np.float32(d_model))

	    def get_posi_angle_vec(position):
	        return [cal_angle(position, i) for i in range(d_model)]

	    sinusoid_table = np.array([get_posi_angle_vec(pos_i) for pos_i in range(pos)])

	    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])
	    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])

	    return sinusoid_table

	'''

마스크 생성

	'''

	# Mask  생성하기
	def generate_padding_mask(seq):
	    seq = tf.cast(tf.math.equal(seq, 0), tf.float32)
	    return seq[:, tf.newaxis, tf.newaxis, :]

	def generate_lookahead_mask(size):
	    mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
	    return mask

	def generate_masks(src, tgt):
	    enc_mask = generate_padding_mask(src)
	    dec_enc_mask = generate_padding_mask(src)

	    dec_lookahead_mask = generate_lookahead_mask(tgt.shape[1])
	    dec_tgt_padding_mask = generate_padding_mask(tgt)
	    dec_mask = tf.maximum(dec_tgt_padding_mask, dec_lookahead_mask)

	    return enc_mask, dec_enc_mask, dec_mask

	'''

Multi-head Attention

	'''

	# Multi Head Attention 구현
	class MultiHeadAttention(tf.keras.layers.Layer):
	    def __init__(self, d_model, num_heads):
	        super(MultiHeadAttention, self).__init__()
	        self.num_heads = num_heads
	        self.d_model = d_model
        
	        self.depth = d_model // self.num_heads
        
	        self.W_q = tf.keras.layers.Dense(d_model)
	        self.W_k = tf.keras.layers.Dense(d_model)
	        self.W_v = tf.keras.layers.Dense(d_model)
        
	        self.linear = tf.keras.layers.Dense(d_model)

	    def scaled_dot_product_attention(self, Q, K, V, mask):
	        d_k = tf.cast(K.shape[-1], tf.float32)
	        QK = tf.matmul(Q, K, transpose_b=True)

	        scaled_qk = QK / tf.math.sqrt(d_k)

	        if mask is not None: scaled_qk += (mask * -1e9)  

	        attentions = tf.nn.softmax(scaled_qk, axis=-1)
	        out = tf.matmul(attentions, V)

	        return out, attentions
        

	    def split_heads(self, x):
	        bsz = x.shape[0]
	        split_x = tf.reshape(x, (bsz, -1, self.num_heads, self.depth))
	        split_x = tf.transpose(split_x, perm=[0, 2, 1, 3])

	        return split_x

	    def combine_heads(self, x):
	        bsz = x.shape[0]
	        combined_x = tf.transpose(x, perm=[0, 2, 1, 3])
	        combined_x = tf.reshape(combined_x, (bsz, -1, self.d_model))

	        return combined_x

    
	    def call(self, Q, K, V, mask):
	        WQ = self.W_q(Q)
	        WK = self.W_k(K)
	        WV = self.W_v(V)
        
	        WQ_splits = self.split_heads(WQ)
	        WK_splits = self.split_heads(WK)
	        WV_splits = self.split_heads(WV)
        
	        out, attention_weights = self.scaled_dot_product_attention(
	            WQ_splits, WK_splits, WV_splits, mask)
                        
	        out = self.combine_heads(out)
	        out = self.linear(out)
            
	        return out, attention_weights

	'''

Position-wise Feed Forward Network

	'''

	# Position-wise Feed Forward Network 구현
	class PoswiseFeedForwardNet(tf.keras.layers.Layer):
	    def __init__(self, d_model, d_ff):
	        super(PoswiseFeedForwardNet, self).__init__()
	        self.d_model = d_model
	        self.d_ff = d_ff

	        self.fc1 = tf.keras.layers.Dense(d_ff, activation='relu')
	        self.fc2 = tf.keras.layers.Dense(d_model)

	    def call(self, x):
	        out = self.fc1(x)
	        out = self.fc2(out)
            
	        return out

	'''

Encoder Layer

	'''

	# Encoder의 레이어 구현
	class EncoderLayer(tf.keras.layers.Layer):
	    def __init__(self, d_model, n_heads, d_ff, dropout):
	        super(EncoderLayer, self).__init__()

	        self.enc_self_attn = MultiHeadAttention(d_model, n_heads)
	        self.ffn = PoswiseFeedForwardNet(d_model, d_ff)

	        self.norm_1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
	        self.norm_2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

	        self.do = tf.keras.layers.Dropout(dropout)
        
	    def call(self, x, mask):
	        '''
	        Multi-Head Attention
	        '''
	        residual = x
	        out = self.norm_1(x)
	        out, enc_attn = self.enc_self_attn(out, out, out, mask)
	        out = self.do(out)
	        out += residual
        
	        '''
	        Position-Wise Feed Forward Network
	        '''
	        residual = out
	        out = self.norm_2(out)
	        out = self.ffn(out)
	        out = self.do(out)
	        out += residual
        
	        return out, enc_attn

	'''

Decoder Layer

	'''

	# Decoder 레이어 구현
	class DecoderLayer(tf.keras.layers.Layer):
	    def __init__(self, d_model, num_heads, d_ff, dropout):
	        super(DecoderLayer, self).__init__()

	        self.dec_self_attn = MultiHeadAttention(d_model, num_heads)
	        self.enc_dec_attn = MultiHeadAttention(d_model, num_heads)

	        self.ffn = PoswiseFeedForwardNet(d_model, d_ff)

	        self.norm_1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
	        self.norm_2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
	        self.norm_3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

	        self.do = tf.keras.layers.Dropout(dropout)
    
	    def call(self, x, enc_out, dec_enc_mask, padding_mask):
	        '''
	        Masked Multi-Head Attention
	        '''
	        residual = x
	        out = self.norm_1(x)
	        out, dec_attn = self.dec_self_attn(out, out, out, padding_mask)
	        out = self.do(out)
	        out += residual

	        '''
	        Multi-Head Attention
	        '''
	        residual = out
	        out = self.norm_2(out)
	        # Q, K, V 순서에 주의하세요!
	        out, dec_enc_attn = self.enc_dec_attn(Q=out, K=enc_out, V=enc_out, mask=dec_enc_mask)
	        out = self.do(out)
	        out += residual
        
	        '''
	        Position-Wise Feed Forward Network
	        '''
	        residual = out
	        out = self.norm_3(out)
	        out = self.ffn(out)
	        out = self.do(out)
	        out += residual

	        return out, dec_attn, dec_enc_attn

	'''

Encoder

	'''

	# Encoder 구현
	class Encoder(tf.keras.Model):
	    def __init__(self,
	                    n_layers,
	                    d_model,
	                    n_heads,
	                    d_ff,
	                    dropout):
	        super(Encoder, self).__init__()
	        self.n_layers = n_layers
	        self.enc_layers = [EncoderLayer(d_model, n_heads, d_ff, dropout) 
	                        for _ in range(n_layers)]
    
	        self.do = tf.keras.layers.Dropout(dropout)
        
	    def call(self, x, mask):
	        out = x
    
	        enc_attns = list()
	        for i in range(self.n_layers):
	            out, enc_attn = self.enc_layers[i](out, mask)
	            enc_attns.append(enc_attn)
        
	        return out, enc_attns

	'''

Decoder

	'''

	# Decoder 구현
	class Decoder(tf.keras.Model):
	    def __init__(self,
	                    n_layers,
	                    d_model,
	                    n_heads,
	                    d_ff,
	                    dropout):
	        super(Decoder, self).__init__()
	        self.n_layers = n_layers
	        self.dec_layers = [DecoderLayer(d_model, n_heads, d_ff, dropout) 
	                            for _ in range(n_layers)]
                            
	    def call(self, x, enc_out, dec_enc_mask, padding_mask):
	        out = x
    
	        dec_attns = list()
	        dec_enc_attns = list()
	        for i in range(self.n_layers):
	            out, dec_attn, dec_enc_attn = \
	            self.dec_layers[i](out, enc_out, dec_enc_mask, padding_mask)

	            dec_attns.append(dec_attn)
	            dec_enc_attns.append(dec_enc_attn)

	        return out, dec_attns, dec_enc_attns

	'''

Transformer 전체 모델 조립

	'''

	class Transformer(tf.keras.Model):
	    def __init__(self,
	                    n_layers,
	                    d_model,
	                    n_heads,
	                    d_ff,
	                    src_vocab_size,
	                    tgt_vocab_size,
	                    pos_len,
	                    dropout=0.2,
	                    shared_fc=True,
	                    shared_emb=False):
	        super(Transformer, self).__init__()
        
	        self.d_model = tf.cast(d_model, tf.float32)

	        if shared_emb:
	            self.enc_emb = self.dec_emb = \
	            tf.keras.layers.Embedding(src_vocab_size, d_model)
	        else:
	            self.enc_emb = tf.keras.layers.Embedding(src_vocab_size, d_model)
	            self.dec_emb = tf.keras.layers.Embedding(tgt_vocab_size, d_model)

	        self.pos_encoding = positional_encoding(pos_len, d_model)
	        self.do = tf.keras.layers.Dropout(dropout)

	        self.encoder = Encoder(n_layers, d_model, n_heads, d_ff, dropout)
	        self.decoder = Decoder(n_layers, d_model, n_heads, d_ff, dropout)

	        self.fc = tf.keras.layers.Dense(tgt_vocab_size)

	        self.shared_fc = shared_fc

	        if shared_fc:
	            self.fc.set_weights(tf.transpose(self.dec_emb.weights))

	    def embedding(self, emb, x):
	        seq_len = x.shape[1]

	        out = emb(x)

	        if self.shared_fc: out *= tf.math.sqrt(self.d_model)

	        out += self.pos_encoding[np.newaxis, ...][:, :seq_len, :]
	        out = self.do(out)

	        return out

        
	    def call(self, enc_in, dec_in, enc_mask, dec_enc_mask, dec_mask):
	        enc_in = self.embedding(self.enc_emb, enc_in)
	        dec_in = self.embedding(self.dec_emb, dec_in)

	        enc_out, enc_attns = self.encoder(enc_in, enc_mask)
        
	        dec_out, dec_attns, dec_enc_attns = \
	        self.decoder(dec_in, enc_out, dec_enc_mask, dec_mask)
        
	        logits = self.fc(dec_out)
        
	        return logits, enc_attns, dec_attns, dec_enc_attns

	'''

모델 인스턴스 생성

	'''

	# 주어진 하이퍼파라미터로 Transformer 인스턴스 생성
	transformer = Transformer(
	    n_layers=2,
	    d_model=512,
	    n_heads=8,
	    d_ff=2048,
	    src_vocab_size=VOCAB_SIZE,
	    tgt_vocab_size=VOCAB_SIZE,
	    pos_len=200,
	    dropout=0.3,
	    shared_fc=True,
	    shared_emb=True)
		
	d_model = 512

	'''

Learning Rate Scheduler

	'''

	# Learning Rate Scheduler 구현
	class LearningRateScheduler(tf.keras.optimizers.schedules.LearningRateSchedule):
	    def __init__(self, d_model, warmup_steps=4000):
	        super(LearningRateScheduler, self).__init__()
        
	        self.d_model = d_model
	        self.warmup_steps = warmup_steps
    
	    def __call__(self, step):
	        arg1 = step ** -0.5
	        arg2 = step * (self.warmup_steps ** -1.5)
        
	        return (self.d_model ** -0.5) * tf.math.minimum(arg1, arg2)

	'''

Learning Rate & Optimizer

	'''

	# Learning Rate 인스턴스 선언 & Optimizer 구현
	learning_rate = LearningRateScheduler(d_model)

	optimizer = tf.keras.optimizers.Adam(learning_rate,
	                                        beta_1=0.9,
	                                        beta_2=0.98, 
	                                        epsilon=1e-9)

	'''

Loss Function 정의

	'''

	# Loss Function 정의
	loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
	    from_logits=True, reduction='none')

	def loss_function(real, pred):
	    mask = tf.math.logical_not(tf.math.equal(real, 0))
	    loss_ = loss_object(real, pred)

	    mask = tf.cast(mask, dtype=loss_.dtype)
	    loss_ *= mask

	    return tf.reduce_sum(loss_)/tf.reduce_sum(mask)

	'''

Train Step 정의

	'''

	# Train Step 정의
	@tf.function()
	def train_step(src, tgt, model, optimizer):
	    tgt_in = tgt[:, :-1]  # Decoder의 input
	    gold = tgt[:, 1:]     # Decoder의 output과 비교하기 위해 right shift를 통해 생성한 최종 타겟

	    enc_mask, dec_enc_mask, dec_mask = generate_masks(src, tgt_in)

	    with tf.GradientTape() as tape:
	        predictions, enc_attns, dec_attns, dec_enc_attns = \
	        model(src, tgt_in, enc_mask, dec_enc_mask, dec_mask)
	        loss = loss_function(gold, predictions)

	    gradients = tape.gradient(loss, model.trainable_variables)    
	    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

	    return loss, enc_attns, dec_attns, dec_enc_attns

	'''

훈련 진행

	'''

	EPOCHS = 3

	for epoch in range(EPOCHS):
	    total_loss = 0
    
	    dataset_count = tf.data.experimental.cardinality(train_dataset).numpy()
	    tqdm_bar = tqdm(enumerate(train_dataset), total=dataset_count, desc=f'Epoch {epoch+1}/{EPOCHS}')

	    for batch, (src, tgt) in tqdm_bar:
	        batch_loss, enc_attns, dec_attns, dec_enc_attns = \
	        train_step(src, tgt, transformer, optimizer)  

	        total_loss += batch_loss
        
	        tqdm_bar.set_description_str('Epoch %2d' % (epoch + 1))
	        tqdm_bar.set_postfix_str('Loss %.4f' % (total_loss.numpy() / (batch + 1))) 

	'''

---

## 번역 성능 측정하기 (1) BLEU Score

---

NLTK를 활용한 BLEU Score

	NLTK

		-> 자연어 처리에 큰 도움이 될 것 같은 라이브러리

	'''

	reference = "많 은 자연어 처리 연구자 들 이 트랜스포머 를 선호 한다".split()
	candidate = "적 은 자연어 학 개발자 들 가 트랜스포머 을 선호 한다 요".split()

	print("원문:", reference)
	print("번역문:", candidate)
	print("BLEU Score:", sentence_bleu([reference], candidate))

	'''

	결과:   원문: ['많', '은', '자연어', '처리', '연구자', '들', '이', '트랜스포머', '를', '선호', '한다']
		번역문: ['적', '은', '자연어', '학', '개발자', '들', '가', '트랜스포머', '을', '선호', '한다', '요']
		BLEU Score: 8.190757052088229e-155

BLEU Score?

	0~1 사이의 값을 가지지만, 100을 곱한 백분율 값으로 표기하는 경우도 많음음


BLEU Score의 점수대별 해석

[BLEU 점수 이해](https://cloud.google.com/translate/automl/docs/evaluate?hl=ko#bleu)

BLEU Score 수식

	N-gram으로 점수를 측정

![image](./a.png)

	-> 1-gram부터 4-gram까지의 점수(Precision)를 모두 곱한 후, 루트를 두 번 씌우면(1/4 제곱) BLEU Score가 됨

위 예시의 각 N-gram이 점수 확인

	weights의 디폴트 값은 [0.25, 0.25, 0.25, 0.25]로 1-gram부터 4-gram까지의 점수에 가중치를 동일하게 주는 것

		-> 이 값을 [1, 0, 0, 0]으로 바꿔주면 BLEU Score에 1-gram의 점수만 반영

	'''

	print("1-gram:", sentence_bleu([reference], candidate, weights=[1, 0, 0, 0]))
	print("2-gram:", sentence_bleu([reference], candidate, weights=[0, 1, 0, 0]))
	print("3-gram:", sentence_bleu([reference], candidate, weights=[0, 0, 1, 0]))
	print("4-gram:", sentence_bleu([reference], candidate, weights=[0, 0, 0, 1]))

	'''

	결과:   1-gram: 0.5
		2-gram: 0.18181818181818182
		3-gram: 2.2250738585072626e-308
		4-gram: 2.2250738585072626e-308

예전 버전 nltk의 BLEU Score 단점

	3-gram, 4-gram 점수가 1이 나와서, 전체적인 BLEU 점수가 50점 이상으로 매우 높게 나오게 될 수 있음

![image](./a.png)

	-> 위 수식에서 어떤 N-gram이 0의 값을 갖는다면 그 하위 N-gram 점수들이 곱했을 때 모두 소멸해버리기 때문

		-> 일치하는 N-gram이 없더라도 점수를 1.0 으로 유지하여 하위 점수를 보존하게끔 구현되어 있음

			-> 1.0 은 모든 번역을 완벽히 재현했음을 의미하기 때문에 총점이 의도치 않게 높아질 수 있음

SmoothingFunction()으로 BLEU Score 보정

	 BLEU 계산시 특정 N-gram이 0점이 나와서 BLEU가 너무 커지거나 작아지는 쪽으로 왜곡되는 문제

		->  보완하기 위해 SmoothingFunction() 을 사용

Smoothing 함수?

	모든 Precision에 아주 작은 epsilon 값을 더해주는 역할

		-> 이로써 0점이 부여된 Precision도 완전한 0이 되지 않으니 점수를 1.0 으로 대체할 필요가 없어짐

			-> 우리의 의도대로 점수가 계산 가능

SmoothingFunction() 종류

	각 method들의 상세한 설명은 아래 링크 방문

[nltk의 bleu_score 소스코드](https://www.nltk.org/_modules/nltk/translate/bleu_score.html)

	sentence_bleu() 함수에 smoothing_function=None을 적용하면 method0가 기본 적용됨을 알 수 있음

SmoothingFunction() 구현

	method1 사용

	'''

	def calculate_bleu(reference, candidate, weights=[0.25, 0.25, 0.25, 0.25]):
	    return sentence_bleu([reference],
	                         candidate,
	                         weights=weights,
	                         smoothing_function=SmoothingFunction().method1)  # smoothing_function 적용

	print("BLEU-1:", calculate_bleu(reference, candidate, weights=[1, 0, 0, 0]))
	print("BLEU-2:", calculate_bleu(reference, candidate, weights=[0, 1, 0, 0]))
	print("BLEU-3:", calculate_bleu(reference, candidate, weights=[0, 0, 1, 0]))
	print("BLEU-4:", calculate_bleu(reference, candidate, weights=[0, 0, 0, 1]))

	print("\nBLEU-Total:", calculate_bleu(reference, candidate))

	'''

	결과:   BLEU-1: 0.5
		BLEU-2: 0.18181818181818182
		BLEU-3: 0.010000000000000004
		BLEU-4: 0.011111111111111112

		BLEU-Total: 0.05637560315259291

		-> 거의 의미 없는 번역

BLEU Score를 측정하는 함수 eval_bleu() 구현

	'''

	def translate(tokens, model, src_tokenizer, tgt_tokenizer):
	    padded_tokens = tf.keras.preprocessing.sequence.pad_sequences([tokens],
	                                                           maxlen=MAX_LEN,
	                                                           padding='post')
	    ids = []
	    output = tf.expand_dims([tgt_tokenizer.bos_id()], 0)   
	    for i in range(MAX_LEN):
	        enc_padding_mask, combined_mask, dec_padding_mask = \
	        generate_masks(padded_tokens, output)
	
        	predictions, _, _, _ = model(padded_tokens, 
	                                      output,
	                                      enc_padding_mask,
	                                      combined_mask,
	                                      dec_padding_mask)

	        predicted_id = \
	        tf.argmax(tf.math.softmax(predictions, axis=-1)[0, -1]).numpy().item()

	        if tgt_tokenizer.eos_id() == predicted_id:
	            result = tgt_tokenizer.decode_ids(ids)  
	            return result

	        ids.append(predicted_id)
	        output = tf.concat([output, tf.expand_dims([predicted_id], 0)], axis=-1)

	    result = tgt_tokenizer.decode_ids(ids)  
	    return result

	'''

한 문장만 평가하는 eval_bleu_single() 정의
	
	'''

	def eval_bleu_single(model, src_sentence, tgt_sentence, src_tokenizer, tgt_tokenizer, verbose=True):
	    src_tokens = src_tokenizer.encode_as_ids(src_sentence)
	    tgt_tokens = tgt_tokenizer.encode_as_ids(tgt_sentence)

	    if (len(src_tokens) > MAX_LEN): return None
	    if (len(tgt_tokens) > MAX_LEN): return None
	
	    reference = tgt_sentence.split()
	    candidate = translate(src_tokens, model, src_tokenizer, tgt_tokenizer).split()

	    score = sentence_bleu([reference], candidate,
	                          smoothing_function=SmoothingFunction().method1)

	    if verbose:
	        print("Source Sentence: ", src_sentence)
	        print("Model Prediction: ", candidate)
	        print("Real: ", reference)
	        print("Score: %lf\n" % score)
        
	    return score
	
	'''

테스트 데이터 중에 하나를 골라 평가

	'''

	test_idx = 0

	eval_bleu_single(transformer, 
	                 test_eng_sentences[test_idx], 
	                 test_spa_sentences[test_idx], 
	                 tokenizer, 
	                 tokenizer)

	'''

	결과:   Source Sentence:  whose side are you?
		Model Prediction:  ['¿de', 'quién', 'eres?']
		Real:  ['¿del', 'lado', 'de', 'quién', 'estás?']
		Score: 0.058335

		0.05833544737207805

전체 테스트 데이터 평가하는 eval_bleu() 정의

	'''

	def eval_bleu(model, src_sentences, tgt_sentence, src_tokenizer, tgt_tokenizer, verbose=True):
	    total_score = 0.0
	    sample_size = len(src_sentences)
    
	    for idx in tqdm(range(sample_size)):
	        score = eval_bleu_single(model, src_sentences[idx], tgt_sentence[idx], src_tokenizer, tgt_tokenizer, verbose)
	        if not score: continue
        
	        total_score += score
    
	    print("Num of Sample:", sample_size)
	    print("Total Score:", total_score / sample_size)

	'''

평가

	'''

	eval_bleu(transformer, test_eng_sentences, test_spa_sentences, tokenizer, tokenizer, verbose=False)

	'''

	결과:   Num of Sample: 594
		Total Score: 0.11322484850753045

---

## 번역 성능 측정하기 (2) Beam Search Decoder

---

Beam Search 코드

	'''

	def beam_search_decoder(prob, beam_size):
	    sequences = [[[], 1.0]]  # 생성된 문장과 점수를 저장

	    for tok in prob:
	        all_candidates = []

	        for seq, score in sequences:
	            for idx, p in enumerate(tok): # 각 단어의 확률을 총점에 누적 곱
	                candidate = [seq + [idx], score * -math.log(-(p-1))]
	                all_candidates.append(candidate)

	        ordered = sorted(all_candidates,
	                         key=lambda tup:tup[1],
	                         reverse=True) # 총점 순 정렬
	        sequences = ordered[:beam_size] # Beam Size에 해당하는 문장만 저장 

	    return sequences

	'''

Beam Search를 생성 기법으로 구현할 때 주의점

	분기를 잘 나눠줘야 함!

		Beam Size가 5라고 가정하면 맨 첫 단어로 적합한 5개의 단어를 생성하고, 두 번째 단어로 각 첫 단어(5개 단어)에 대해 5순위까지 확률을 구하여 총 25개의 문장을 생성

			-> 그 25개의 문장들은 점수(존재 확률) 를 가지고 있으니 각각의 순위 매겨 상위 5개의 표본만 살아남아 세 번째 단어를 구할 자격을 얻게 됨


각 단어의 확률값을 계산하는 calc_prob() 정의

	'''

	def calc_prob(src_ids, tgt_ids, model):
	    enc_padding_mask, combined_mask, dec_padding_mask = \
	    generate_masks(src_ids, tgt_ids)

	    predictions, enc_attns, dec_attns, dec_enc_attns =\
	    model(src_ids, 
	            tgt_ids,
	            enc_padding_mask,
	            combined_mask,
	            dec_padding_mask)
    
	    return tf.math.softmax(predictions, axis=-1)

	'''

Beam Search를 기반으로 동작하는 beam_search_decoder() 정의

	'''

	# beam_search_decoder() 구현
	def beam_search_decoder(sentence, 
	                        src_len,
	                        tgt_len,
	                        model,
	                        src_tokenizer,
	                        tgt_tokenizer,
	                        beam_size):
	    tokens = src_tokenizer.encode_as_ids(sentence)
    
	    src_in = tf.keras.preprocessing.sequence.pad_sequences([tokens],
	                                                            maxlen=src_len,
	                                                            padding='post')

	    pred_cache = np.zeros((beam_size * beam_size, tgt_len), dtype=np.int64)
	    pred_tmp = np.zeros((beam_size, tgt_len), dtype=np.int64)

	    eos_flag = np.zeros((beam_size, ), dtype=np.int64)
	    scores = np.ones((beam_size, ))

	    pred_tmp[:, 0] = tgt_tokenizer.bos_id()

	    dec_in = tf.expand_dims(pred_tmp[0, :1], 0)
	    prob = calc_prob(src_in, dec_in, model)[0, -1].numpy()

	    for seq_pos in range(1, tgt_len):
	        score_cache = np.ones((beam_size * beam_size, ))

	        # init
	        for branch_idx in range(beam_size):
	            cache_pos = branch_idx*beam_size

	            score_cache[cache_pos:cache_pos+beam_size] = scores[branch_idx]
	            pred_cache[cache_pos:cache_pos+beam_size, :seq_pos] = \
	            pred_tmp[branch_idx, :seq_pos]

	        for branch_idx in range(beam_size):
	            cache_pos = branch_idx*beam_size

	            if seq_pos != 1:   # 모든 Branch를 로 시작하는 경우를 방지
	                dec_in = pred_cache[branch_idx, :seq_pos]
	                dec_in = tf.expand_dims(dec_in, 0)

	                prob = calc_prob(src_in, dec_in, model)[0, -1].numpy()

	            for beam_idx in range(beam_size):
	                max_idx = np.argmax(prob)

	                score_cache[cache_pos+beam_idx] *= prob[max_idx]
	                pred_cache[cache_pos+beam_idx, seq_pos] = max_idx

	                prob[max_idx] = -1

	        for beam_idx in range(beam_size):
	            if eos_flag[beam_idx] == -1: continue

	            max_idx = np.argmax(score_cache)
	            prediction = pred_cache[max_idx, :seq_pos+1]

	            pred_tmp[beam_idx, :seq_pos+1] = prediction
	            scores[beam_idx] = score_cache[max_idx]
	            score_cache[max_idx] = -1

	            if prediction[-1] == tgt_tokenizer.eos_id():
	                eos_flag[beam_idx] = -1

	    pred = []
	    for long_pred in pred_tmp:
	        zero_idx = long_pred.tolist().index(tgt_tokenizer.eos_id())
	        short_pred = long_pred[:zero_idx+1]
	        pred.append(short_pred)
	    return pred

	'''

문장에 대해 BLEU Score를 출력하는 beam_bleu() 정의

	'''

	def calculate_bleu(reference, candidate, weights=[0.25, 0.25, 0.25, 0.25]):
	    return sentence_bleu([reference],
	                            candidate,
	                            weights=weights,
	                            smoothing_function=SmoothingFunction().method1)

	def beam_bleu(reference, ids, tokenizer):
	    reference = reference.split()

	    total_score = 0.0
	    for _id in ids:
	        candidate = tokenizer.decode_ids(_id.tolist()).split()
	        score = calculate_bleu(reference, candidate)

	        print("Reference:", reference)
	        print("Candidate:", candidate)
	        print("BLEU:", calculate_bleu(reference, candidate))

	        total_score += score
        
	    return total_score / len(ids)

	'''

평가

	'''

	test_idx = 1

	ids = \
	beam_search_decoder(test_eng_sentences[test_idx],
	                    MAX_LEN,
	                    MAX_LEN,
	                    transformer,
	                    tokenizer,
	                    tokenizer,
	                    beam_size=5)

	bleu = beam_bleu(test_spa_sentences[test_idx], ids, tokenizer)
	print(bleu)

	'''

	결과:   Reference: ['usaré', 'el', 'vestido', 'azul.']
		Candidate: ['me', 'puse', 'el', 'vestido', 'azul', 'de', 'azul', 'azules', 'azules', 'azules', 'azules', 'azules']
		BLEU: 0.03602080288207364
		Reference: ['usaré', 'el', 'vestido', 'azul.']
		Candidate: ['me', 'el', 'vestido', 'azul', 'de', 'azul', 'azules', 'azules', 'azules', 'azules', 'azules']
		BLEU: 0.03986357128268015
		Reference: ['usaré', 'el', 'vestido', 'azul.']
		Candidate: ['me', 'puse', 'el', 'vestido', 'azul', 'de', 'azul', 'azules', 'azules', 'azules', 'azules']
		BLEU: 0.03986357128268015
		Reference: ['usaré', 'el', 'vestido', 'azul.']
		Candidate: ['me', 'puse', 'el', 'vestido', 'azul', 'de', 'azul', 'azules', 'azules', 'azules', 'azules']
		BLEU: 0.03986357128268015
		Reference: ['usaré', 'el', 'vestido', 'azul.']
		Candidate: ['me', 'el', 'vestido', 'azul', 'de', 'azul', 'azules', 'azules', 'azules', 'azules']
		BLEU: 0.0446323613785333
		0.040048775621729475

---

## 데이터 부풀리기

---

Data Augmentation

	Embedding을 활용한 Lexical Substitution으로 Data Augmentation

gensim 에 사전 훈련된 Embedding 모델을 불러오는 방법

	1. 직접 모델을 다운로드해 load 하는 방법
	2. gensim 이 자체적으로 지원하는 downloader 를 활용해 모델을 load 하는 방법

Embedding 모델 load

	glove-wiki-gigaword-300 모델 사용

	'''

	import gensim.downloader as api

	wv = api.load('glove-wiki-gigaword-300')

	'''

Lexical Substitution 구현

입력된 문장을 Embedding 유사도를 기반으로 Augmentation 하여 반환하는 lexical_sub() 정의

	'''

	def lexical_sub(sentence, model, topn=5):
	    new_sentence = []
    
	    for word in sentence:
	        # 모델에 해당 단어가 있는지 확인
	        if word in model:
	            # 해당 단어와 유사한 단어 리스트에서 무작위로 선택
	            similar_words = model.most_similar(word, topn=topn)
	            substitute_word = random.choice(similar_words)[0]
	            new_sentence.append(substitute_word)
	        else:
	            # 모델에 단어가 없으면 원본 단어 사용
	            new_sentence.append(word)
    
	    return new_sentence

	'''

테스트 데이터의 Augmentation 생성

	'''

	new_corpus = []

	for old_src in tqdm(test_eng_sentences):
	    new_src = lexical_sub(old_src, wv)
	    if new_src is not None: 
	        new_corpus.append(new_src)
	    # Augmentation이 없더라도 원본 문장을 포함시킵니다
	    new_corpus.append(old_src)

	print(new_corpus[:10])

	'''

	결과: [['f', 'j', 'y', 'l', 'mail', ' ', 'wrldcom', 'really', 'sen.', 'mail', ' ', 'this', 'sen.', 'messages', ' ', 'en', 'n', '}', 'why'], 'whose side are you?', ["'m", '--', 'h', ' ', 'h', 'mailing', 'is', 'sen.', ' ', 'shirts', 'g', 'mailing', ' ', 'j', 'w', ']', 'mails', ' ', 'f', 'reps.', 'messages', 'wrldcom', 'nokiacorp', 'but'], "i'l wear the blue dress.", ['w', 'te', 'metre', ' ', "'m", 'nokiacorp', ' ', 'l', 'shirt', "'d", 'f', 'f', ' ', "'ve", 'h', ' ', 'l', 'w', 'mail', ' ', 'k', 'te', 'wrldcom', 'k', "'d", 'w', 'one', 'k', 'there'], 'tom is still in the hospital.', ['k', 'mails', ' ', 'p', "'d", 'h', 'f', ' ', 'h', 'en', ' ', 'l', 'y', ' ', 'h', 'n', 'mails', ' ', 'metres', 'mailing', 'mailing', 'shirt', "'ve", 'h', 'c', ' ', 'really', 'k', 'sunmicro', 'l', 'email', 'is', 'r', ' ', 'te', 'n', ' ', 'metre', 'messages', ','], 'he will go to the meeting instead of me.', ['c', 'mailing', 'l', ' ', 'ft', 'messages', ' ', 'is', ' ', 'l', 'k', 'another', 'wrldcom', 'nasdaq100tr', ' ', 'n', 'k', ' ', 'metres', "'d", 'k', 'h', 'there'], 'get me a glass of milk.']
