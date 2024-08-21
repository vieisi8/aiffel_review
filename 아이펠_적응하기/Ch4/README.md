# 학습 내용
---

- 우분투, 파이썬, 텐서 플로우에 대해 알아보기
- 터미널에 대해 알아보기
- 터미널에서 사용 가능한 명령어 학습
- 패키지를 관리하는 명령어(apt-get) 학습
- 아나콘다 설치후 가상환경 실행 방법 학습

---

우분투(ubuntu)?

리눅스 기반으로 만들어진 운영체제를 뜻함
-- OS는 OPERATING SYSTEM의 줄임말!
-- Ubuntu의 로고는 부족끼리 손을 잡고 원으로 그린 모양이며, 정밀도와 신뢰를 뜻함
-- ex) Ubuntu 22.04는 22년도 4월 출시된 버전을 뜻함

---

Python?

프로그래밍 언어!

---

TensorFlow?

머신러닝/딥러닝에 특화된 라이브러리

-- 라이브러리란? 특정 기능을 위한 여러 함수 또는 클래스를 담고 있는 보따리!

---

![Alt text](https://d3s0tskafalll9.cloudfront.net/media/images/Untitled_19.max-800x600_laxgilq.png)

위 화면의 공식 이름은 터미널!!(터미널 환경 -> CLI)

---

터미널과 셸의 차이

- 터미널 위에 셸이 실행되는 것
- 터미널은 명령을 입력하는 셸을 실행하기 위한 토대!

---

- whoami -> 유저 이름 출력
- pwd -> 현재 디렉토리 위치 출력
- ls -> 현재 디랙토리 내의 모든 파일 또는 하위 디렉토리 목록 출력
-- ls -al -> a는 숨김 파일까지 출력 / l은 자세한 정보 출력
- cd a-> a 디렉토리 이동
-- cd .. -> 상위 디렉토리로 이동
-- cd ~ -> Home으로 이동
- mkdir a -> a 새디렉토리 생성
- rm -r a -> a 디렉토리 삭제
- mv a b -> a를 b로 이동
- cp a b -> a를 b로 복사
-- -r 옵션 -> 하위 디렉토리까지 함께 복사 가능

---

패키지? 
특정 기능을 하는 작은 프로그램 단위
-- 패키지 > 라이브러리

sudo -> 다른 사용자 권한으로 실행 한다는 의미

apt list --installed

	↓

지금까지 설치된 패키지 리스트를 확인하는 명령어
-- grep 옵션을 통해 원하는 패키지만 검색 가능

sudo apt-get update / sudo apt-get upgrade

		↓

패키지 업데이트(둘 다 실행하는게 좋음)

sudo apt-get remove a

	↓

패키지 a를 삭제

---

가상환경? 

독립된 공간을 만들어주는 기능 

필요한 이유? 

- 프로젝트마다 특정 패키지의 서로 다른 버전 필요
- 패키지 간 충돌 위험 방지

---

which a

   ↓

프로그램 a의 설치 경로를 확인하는 명령어

---

__아나콘다 기준__

conda create -n a python3.11.2 

		↓

이름이 a인 가상환경 생성(python 버전은 3.11.2임)

conda env list

	↓

가상환경 리스트 확인

conda activate a

	↓

이름이 a인 가상환경 활성화(오류 발생시 conda init 실행)

conda deactivate a

	↓

이름이 a인 가상환경 비활성화

conda env remove -n a

	↓

이름이 a인 가상환경 삭제


