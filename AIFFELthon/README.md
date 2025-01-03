# AIFFELthon

---

  - 기업과 협업하는 프로젝트이기 때문에 당일 무엇을 했는지 일지 정도만 작성 예정

---

## 일지

---

  - 2024.12.26
    - 맡은 Tool 구현
      - 처음으로 다시 돌아가 PDF -> Text 실헝
        - 전에 구현 했던 프로토 타입으로 PDF를 실험 했을 때 엉망으로 IUPAC 이름이 추출됨
        - 그래서 아예 처음으로 돌아가 PDF -> Text 실헝 진행
          - OCR 기술 없이 PDF에서 Text를 추출하는 모델 검색및 실험
          - PDF -> Markdown 변환후 Text 추출 실험
          - 기존 OCR 기반, 텍스트 기반 실험도 진행
  - 2024.12.27
    - 맡은 Tool 구현
      - IUPAC 이름 패턴을 통해 추출
    - 기업과 진행 상황 공유를 위한 미팅
 - 2024.12.28
   - 맡은 Tool 구현
     - PDF -> 텍스트
       - Google cloud vision 과 pdfplumber 성능 차이 줄여보기
 - 2024.12.29
   - 맡은 Tool 구현
     - 28일과 동일한 실험 진행
 - 2024.12.30
   - 맡은 Tool 구현
     - 추출된 텍스트나 추출된 IUPAC을 수정하기 위한 모델 탐색
       - Open Ai API를 사용하면 손쉽게 가능하나 지속적인 비용이 발생하기 때문
       - Ollama를 활용해 Local LLM 구현해볼 예정
       - 컴퓨팅 파워가 제한적이므로 상대적으로 작은 모델로 진행해볼 예정
 - 2024.12.31
   - 맡은 Tool 구현
     - 모델 탐색(Qwen2.5, EXANONE, ChemLLM)을 진행
       - 본문에서 IUPAC 추출 불가
       - 추출된 IUPAC 수정 불가
     - OPEN AI API를 사용해야 될 것 같음
 - 2025.1.1
   - 맡은 Tool 구현
     - 추출된 텍스트에서 더 정교한 IUPAC 추출 실험
 - 2025.1.2
   - 맡은 Tool 구현
     - IUPAC 추출후 OPEN AI API를 통해 원래의 IUPAC으로 수정
       - API KEY를 기업측에서 받지 못해 진행하지 못하고 있음
   - 다른 Tool 백업
     - PDF -> 테이블 추출
       - layout parser 모델을 사용해 실험 진행
         - label이 테이블인 경우만 추출해 원본 이미지에 레이아웃 표시
 - 2025.1.3
   - 다른 Tool 백업
     - table 검출 모델을 서용해 실험 진행
       - 테이블만 추출해 원본 이미지에 레이아웃 표시
