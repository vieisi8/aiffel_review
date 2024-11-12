# 학습 내용

---

- 로컬 환경 가상세팅
- Airflow로 파이프라인 생성
- W&B로 하이퍼파라미터 튜닝

---

##로컬 환경 가상세팅

---

윈도우기준 가상세팅

1. WSL2 설치

	- 시스템 요구사항 확인
		- Windows 10 버전 2004 이상 또는 Windows 11
		- 64비트 프로세서
		- 4GB 이상의 RAM (8GB 권장)
	- Windows 기능 활성화
		1. Windows 검색창에서 "Windows 기능 켜기/끄기" 검색
		2. 다음 두 항목 체크:
		    - Windows Subsystem for Linux(Linux용 Windows 하위 시스템)
		    - Virtual Machine Platform(가상 머신 플랫폼)
		3. 확인 버튼 클릭 후 컴퓨터 재시작
	- WSL2를 기본 버전으로 설정
		- '''wsl --set-default-version 2'''

2. Ubuntu 설치

	- Ubuntu 22.04 설치
		1. Microsoft Store 실행
		2. "Ubuntu 22.04 LTS" 검색
		3. "받기" 또는 "설치" 버튼 클릭
		4. 설치 완료 후 "실행" 클릭
	- Ubuntu 업데이트 및 업그레이드
		- sudo apt update
		- sudo apt upgrade -y

3. 파이썬 가상환경

	- venv 패키지 설치
		- sudo apt install python3-venv
	- 가상환경 생성
		- python3 -m venv aiffel
	- 가상환경 활성화
		- source aiffel/bin/activate

4. 사용할 패키지 설치

	- pip 업그레이드
		- pip install --upgrade pip
	- 필요한 패키지 설치
		- pip install numpy tensorflow wandb apache-airflow notebook ipykernel pydot pillow

5. VSCode IDE

	- WSL 연동
		- 확장에서 WSL 검색
		- WSL 설치
		- "ctrl + shift + p" 명령 입력 후 "WSL: WSL에 연결" 선택
		- "폴더 열기"를 선택하여 작업할 폴더 선택

---

## Airflow로 파이프라인 생성 / W&B로 하이퍼파라미터 튜닝

---

첫번째 실습과 두번째 실습 진행

1. 첫번째 실습

	1. MNIST 데이터셋을 이용해 데이터 전처리 및 데이터 훈련 파트를 Apache Airflow로 생성
	2. 실험기록을 W&B로 기록
	3. 동시에 하이퍼 파라미터 튜닝 진행

		-> dag test가 잘 실행만 되도 문제없는것으로 간주

2. 두번째 실습

[Airflow 튜토리얼](https://velog.io/@clueless_coder/Airflow-%EC%97%84%EC%B2%AD-%EC%9E%90%EC%84%B8%ED%95%9C-%ED%8A%9C%ED%86%A0%EB%A6%AC%EC%96%BC-%EC%99%95%EC%B4%88%EC%8B%AC%EC%9E%90%EC%9A%A9)

	위 튜토리얼을 실습으로 진행
