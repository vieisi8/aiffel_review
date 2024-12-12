# 학습 내용

---

- EXAONE-3.5, langchain를 활용한 RAG 시스템 구현
  - LLM 로컬 구현
  - langserve를 통해 모델 서빙
  - RAG 시스템 구현

---

## EXAONE-3.5, langchain를 활용한 RAG 시스템 구현

---

---

### LLM 로컬 구현

---

  - [EXAONE-3.5-32B-Instruct](https://huggingface.co/LGAI-EXAONE/EXAONE-3.5-32B-Instruct)
    - EXAONE-3.5
      - 최근 LG에서 공개한 한국 언어 모델
  - Ollama를 통해 EXAONE-3.5을 로컬로 구현
    - 가장 큰 모델인 32B Instruct 모델 사용
    1. EXAONE-3.5 gguf 다운
    2. Modelfile 생성
    3. Ollama에서 EXAONE-3.5 모델 생성

---

### langserve를 통해 모델 서빙

---

  - Fast api와 langserve를 활용해 Chatbot 및 LLM 모델 서빙
  - Colab 환경 이기에 local 서버를 ngrok를 통해 외부에서 통신 가능 하도록 설정
    - 기본 엔드포인트
      - Chatbot으로 설정
      - langserve로 구현 되어 Chatbot을 누구나 쉽게 사용 가능
      - ![image](https://github.com/user-attachments/assets/df31bc38-919a-4114-883e-c872fbe8d697)

    - 엔드포인트(/llm)
      - EXAONE-3.5 모델을 반환

---

### RAG 시스템 구현

---

  - langchain 라이브러를 통해 RAG 시스템 구현
    - Embedding 모델
      - HuggingFace의 임베딩 모델인 BAAI/bge-m3 사용
    - LLM
      - langserve의 RemoteRunnable을 통해 위 서버에서 LLM을 가져옴
    - RAG 시스템
      1. 문서를 Embedding 모델이 임베딩화
      2. Cahin 생성(프롬프트, LLM 포함)
      3. 문서에 대한 질의
      - Attention Is All You Need paper를 업로드후 질문
        - ![image](https://github.com/user-attachments/assets/9665df0b-6481-4b09-9c0e-ba4b396c4709)
