# 학습 내용

---

- MLOps
- 머신러닝 시스템 디자인 로드맵
- MLOps 도구 소개

---

## MLOps

---

MLOps?

	머신러닝의 개발과 운영을 통합하는 혁신적인 방법론

전통적인 ML 개발 방식의 한계

	- 느린 업데이트 주기
		- 전통적인 방식에서는 모델 업데이트에 많은 시간이 소요됨
		- 빠르게 변화하는 비즈니스 환경에 대응하기 어려움
	- 환경 불일치
		- 개발 환경과 운영 환경의 차이로 인해 예상치 못한 문제가 발생
		- 모델의 성능 저하와 안정성 문제
	- 수동적 관리
		- 효율성을 떨어뜨림
		- 모델의 성능 저하를 늦게 발견하게 만들어 비즈니스에 악영향을 줌

MLOps의 중요성

	- 개발과 운영의 통합
		- 머신러닝 모델의 개발과 운영을 seamless하게 통합
		- 모델의 지속적인 개선과 배포를 가능하게 함
	- 비즈니스 가치 실현
		- 모델 개발부터 실제 비즈니스 가치 창출까지의 시간 단축
		- 기업의 AI 투자 효율성을 크게 향상시킴
	- 안정성과 확장성
		- 안정적이고 확장 가능한 AI 시스템 구축 지원
		- 더 큰 규모의 AI 프로젝트를 수행할 수 있음

MLOps가 해결하는 문제들

	- 자동화된 CI/CD 파이프라인
		- 모델 개발부터 배포까지의 과정을 자동화
		- 개발 주기를 크게 단축
	- 일관된 환경
		- 개발과 운영 환경의 일관성 보장
		- "works on mymachine" 문제 해결
	- 지속적 모니터링
		- 실시간 모델 성능 모니터링 제공
		-  문제를 조기에 발견하고 신속하게 대응할 수 있음

MLOps의 핵심 개념

	- 버전 관리
		- 코드, 데이터, 모델의 버전을 체계적으로 관리
		- 실험의 재현성과 추적성 보장
	- 자동화된 테스팅
		- 모델의 성능과 품질을 자동으로 검증
		- 배포 전 잠재적 문제를 사전에 발견
	- 지속적 통합 및 배포
		- 코드 변경사항을 자동으로 통합하고 배포
		- 개발 속도를 높이고 오류를 줄임
	- 모니터링 및 로깅
		- 모델의 성능과 시스템 상태를 지속적으로 추적
		- 문제 해결과 최적화에 필수적임

머신러닝 시스템 디자인 로드맵

![image](./a.png)

---

## 머신러닝 시스템 디자인 로드맵

---

ML 프로젝트 생명주기(문제 정의)

	1. 비즈니스 목표 설정
		- 명확한 목표로 프로젝트 방향 결정
	2. ML 문제 변환
		- 비즈니스 문제를 ML 과제로 구체화
	3. 평가 지표 정의
		- 성공 기준과 측정 가능한 지표 설정

데이터 수집 및 전처리(소스 식별)

	- 내부 데이터
		- 기업 내 축적된 데이터 활용
	- 외부 데이터
		- 공개 데이터셋 또는 파트너십 통한 획득
	- 실시간 데이터
		- IoT 기기나 센서로부터 수집

데이터 품질 검사 및 클렌징

	- 결측치 처리
		- 누락된 데이터 식별 및 대체 방법 적용
	- 이상치 탐지
		- 통계적 방법으로 비정상 데이터 확인
	- 중복 제거
		- 동일 정보의 중복 엔트리 제거
	- 일관성 확보
		- 데이터 형식과 단위의 통일성 검증

특성 엔지니어링 및 선택

	- 특성 생성
		- 도메인 지식 기반 새로운 특성 도출
	- 차원 축소
		- PCA, t-SNE 등 기법으로 차원 감소
	- 특성 선택
		- 중요도 기반 최적 특성 집합 선정

데이터 분할(훈련, 검증, 테스트)

	- 훈련 세트
		- 모델 학습에 사용되는 주요 데이터
	- 검증 세트
		- 하이퍼파라미터 튜닝 및 모델 선택용
	- 테스트 세트
		- 최종 모델 성능 평가를 위한 데이터

모델 개발(알고리즘 선택)

	지도학습 	비지도학습	 강화학습

	분류		 군집화 	  Q-학습
	회귀 		차원 축소 	정책 기반

하이퍼파라미터 튜닝

	- 그리드 탐색
		- 모든 가능한 조합 탐색
	- 랜덤 탐색
		- 무작위 조합으로 효율적 탐색
	- 베이지안 최적화
		- 확률 모델 기반 지능적 탐색

모델 훈련 및 최적화

	1. 초기 훈련
		- 기본 설정으로 모델 학습
	2. 성능 분석
		- 학습 곡선 및 오차 분석
	3. 정규화 적용
		- 과적합 방지 위한 정규화 기법 도입
	4. 재훈련 및 미세조정
		- 최적화된 설정으로 모델 재학습

앙상블 방법 고려

	- 랜덤 포레스트
		- 다수의 결정 트리 결합
	- 그래디언트 부스팅
		- 순차적 약학습기 개선
	- 스태킹
		- 다양한 모델 예측 결합

모델 평가(성능 지표)

	- 분류 평가
		- 정확도, 정밀도, 재현율, F1 점수
	- 회귀 평가
		- MSE, MAE, R-squared
	- 랭킹 평가
		- NDCG, MAP, MRR

교차 검증 기법

	- K-폴드 교차 검증
		- 데이터를 K개 부분집합으로 나누어 평가
	- 계층화 K-폴드
		- 클래스 분포 유지하며 K-폴드 적용
	- 시계열 교차 검증
		- 시간 종속성 고려한 분할 평가

A/B 테스팅

	1. 가설 설정
		- 테스트할 변경사항과 예상 효과 정의
	2. 실험 설계
		- 대조군과 실험군 설정, 샘플 크기 결정
	3. 데이터 수집
		- 충분한 기간 동안 사용자 반응 수집
	4. 결과 분석
		- 통계적 유의성 검정 및 결론 도출

편향성 및 공정성 검사

	1. 데이터 편향 검사
		- 훈련 데이터의 대표성 및 다양성 확인
	2. 모델 예측 분석
		- 서브그룹 간 성능 차이 평가
	3. 공정성 지표 측정
		- 통계적 형평성, 기회 균등성 등 계산

모델 서빙 전략

	- 서버리스 배포
		- AWS Lambda, Azure Functions 활용
	- 컨테이너화
		- Docker 기반 모델 패키징 및 배포
	- 엣지 컴퓨팅
		- 로컬 디바이스에서 추론 실행

배포 파이프라인 구축

	1. 코드 버전 관리
		- Git 기반 소스 코드 관리
	2. CI/CD 구성
		- Jenkins, GitLab CI 활용 자동화
	3. 모델 레지스트리
		- AirFlow로 모델 버전 및 메타데이터 관리
	4. 모니터링 설정
		- Google Cloud, W&B로 성능 추적

성능 모니터링 및 로깅

	- 예측 성능 추적
		- 실시간 정확도, 지연 시간 모니터링
	- 리소스 사용량 관찰
		- CPU, 메모리, 네트워크 사용률 확인
	- 이상 탐지
		- 비정상적인 패턴이나 오류 자동 감지

피드백 루프 구현

	1. 사용자 피드백 수집
		- 예측 결과에 대한 사용자 평가 획득
	2. 데이터 드리프트 감지
		- 입력 데이터 분포 변화 모니터링
	3. 모델 재훈련 트리거
		- 성능 저하 시 자동 재학습 시작
	4. A/B 테스트 자동화
		- 새 모델과 기존 모델 비교 실험

배치 예측 시스템

	- 정기적 실행
		- 일일, 주간 단위 대량 예측 처리
	- 리소스 효율성
		- 컴퓨팅 자원 최적 활용
	- 지연 허용
		- 즉각적 응답이 불필요한 상황에 적합


실시간 예측 시스템

	- 즉각적 응답
		- 밀리초 단위의 빠른 예측 제공
	- 개별 요청 처리
		- 각 요청마다 독립적 예측 수행
	- 스케일링 중요성
		- 트래픽 변동에 대응 가능한 구조

스트리밍 예측 시스템

	1. 데이터 수집
		- 연속적인 데이터 스트림 입력
	2. 실시간 처리
		- 스트림 처리 엔진으로 데이터 분석
	3. 예측 생성
		- 지속적으로 업데이트되는 예측 결과 제공

온라인 학습 시스템

	- 점진적 학습
		- 새로운 데이터로 모델 지속 업데이트
	- 적응형 모델
		- 변화하는 환경에 빠르게 대응
	- concept drift 대응
		- 데이터 분포 변화 자동 감지 및 조정

마이크로서비스 아키텍처 개요

	- 독립적 서비스
		- 각 기능을 독립된 서비스로 구현
	- 유연한 확장
		- 개별 서비스 단위로 확장 가능
	- 기술 다양성
		- 서비스별 최적 기술 스택 선택


ML 시스템의 마이크로서비스

	- 데이터 수집 서비스
		- 다양한 소스에서 데이터 수집 및 저장
	- 모델 훈련 서비스
		- 주기적 또는 온디맨드 모델 재 훈련
	- 예측 서비스
		 - 실시간 추론 요청 처리
	- 모니터링 서비스
		- 시스템 성능 및 건강 상태 추적

마이크로서비스 간 통신: REST API

	- HTTP 기반
		- 표준 HTTP 메서드 활용 (GET, POST등)
	- 상태 비저장
		- 각 요청은 독립적으로 처리
	- 범용성
		- 다양한 클라이언트와 쉽게 통합

마이크로서비스 간 통신: gRPC

	- 프로토콜 버퍼
		- 효율적인 데이터 직렬화
	- 양방향 스트리밍
		- 실시간 데이터 교환에 적합
	- 강력한 타입 체크
		- 컴파일 시점 오류 검출

마이크로서비스 간 통신: 메시지 큐

	1. 메시지 생성
		- 서비스가 메시지를 큐에 발행
	2. 메시지 저장
		- 큐가 메시지를 임시 보관
	3. 메시지 소비
		- 다른 서비스가 메시지를 처리


포트포워딩?

	네트워크 포트 간 데이터 전달 메커니즘

	- 용도
		- 방화벽 우회, 내부 서비스 외부 노출
	- 구현
		- NAT, 리버스 프록시 등 다양한 방식


MLOps에서의 포트포워딩 활용

	- 원격 개발
		- 클라우드 자원에 로컬 환경처럼 접근
	- 디버깅
		- 운영 환경의 서비스를 로컬에서 검사
	- 보안 접근
		- 제한된 네트워크 환경에서 서비스 이용

포트포워딩 보안 고려사항

	- 접근 제어
		- 인증된 사용자만 포트포워딩 허용
	- 암호화
		- SSH 터널링으로 데이터 보호
	- 모니터링
		- 비정상적인 포트 사용 감지 및 차단

---

## MLOps 도구 소개

---

Git을 활용한 코드 버전 관리

	- 저장소
		- 프로젝트 파일들의 중앙 집중식 저장소
	- 커밋
		- 변경사항을 영구적으로 기록
	- 브랜치
		- 독립적인 개발 라인 생성
	- 병합
		- 다른 브랜치의 변경사항을 통합

MLOps에서의 Git 활용

	- 실험 관리
		- 다양한 모델 구성과 하이퍼파라미터 추적
	- 협업
		- 팀원 간 코드 공유 및 리뷰 용이
	- 롤백
		- 문제 발생 시 이전 버전으로 신속 복구

데이터 버전 관리의 중요성

	- 재현성
		- 동일한 데이터로 실험 결과 재현 가능
	- 협업
		- 팀원 간 데이터셋 공유 및 동기화
	- 감사
		- 데이터 변경 이력 추적 및 문제 해결
	- 롤백
		- 이전 버전의 데이터셋으로 쉽게 복원

DVC (Data Version Control)?

	- 데이터 추적
		- 대용량 데이터 파일의 변경사항 효율적 관리
	- 저장소 연동
		- 원격 스토리지와 연동하여 데이터 공유
	- 파이프라인 관리
		- 데이터 처리 및 모델 훈련 과정 버전화

DVC의 작동 원리

	1. 데이터 해시
		- 파일 내용 기반 고유 식별자 생성
	2. 메타데이터 저장
		- Git에 데이터 파일 대신 메타데이터 저장
	3. 캐시 관리
		- 로컬 캐시에 실제 데이터 파일 보관
	4. 원격 동기화
		- 원격 저장소와 데이터 동기화

Git과 DVC의 통합

	- Git
		- 코드와 설정 파일 버전 관리
	- DVC
		- 대용량 데이터 및 모델 파일 추적
	- 통합
		- 코드와 데이터의 일관된 버전 관리 가능

ML 파이프라인 자동화

	- 데이터 처리
		- 데이터 수집, 정제, 변환 자동화
	- 모델 훈련
		- 다양한 모델 구성으로 자동 학습
	- 평가
		- 성능 지표 계산 및 모델 선택
	- 배포
		- 선택된 모델의 자동 배포 및 서빙

파이프라인 정의 및 실행 방법

	- DAG 정의
		- 작업 간 의존성을 그래프로 표현
	- 컴포넌트 개발
		- 재사용 가능한 작업 단위 구현
	- 파라미터 설정
		- 실행 시 필요한 설정 값 정의
	- 스케줄링
		- 주기적 또는 트리거 기반 실행 설정

파이프라인 자동화 도구

	- Kubeflow
		- 쿠버네티스 기반의 확장 가능한 ML 워크플로우
	- MLflow
		- 유연한 워크플로우 스케줄링 및 모니터링

AirFlow Tracking

	- 실험 로깅
		- 파라미터, 메트릭, 아티팩트 자동 기록
	- 실험 비교
		- 여러 실험 결과를 시각적으로 비교 분석
	- 재현성
		- 실험 환경과 결과를 정확히 재현 가능

AirFlow & Google Cloud Platform

	- 중앙 저장소
		- 모델 버전을 중앙에서 관리
	- 단계 관리
		- 모델의 개발, 스테이징, 프로덕션 단계 추적
	- 협업
		- 팀 간 모델 공유 및 승인 프로세스
	- 배포 자동화
		- 승인된 모델의 자동 배포 지원

모델 버전 관리의 필요성

	- 버전 추적
		- 모델 개발 과정의 모든 변경사항 기록
	- 롤백 용이성
		- 문제 발생 시 이전 버전으로 신속 복구
	- 협업 강화
		- 팀원 간 모델 공유 및 리뷰 프로세스
	- 규정 준수
		- 모델 개발 과정의 투명성과 감사 용이성

모델 메타데이터 관리

	항목 				설명

	버전 			모델의 고유 식별자
	생성 			일시 모델이 훈련된 시간
	입력 			데이터 훈련에 사용된 데이터셋 정보
	하이퍼파라미터 		모델 구성에 사용된 주요 파라미터
	성능 			지표 정확도, F1 스코어 등 주요 평가 지표

모델 성능 모니터링

	지표 				설명

	정확도 			예측의 정확성
	지연 시간 		요청부터 응답까지의 시간
	처리량 			단위 시간당 처리 가능한 요청 수
	드리프트 		데이터 분포 변화 감지

단위 테스트

	- 함수 테스트
		- 개별 함수의 입출력 검증
	- 모듈 테스트
		- 관련 함수 그룹의 동작 확인
	- 예외 처리
		- 에러 상황에 대한 적절한 대응 검증

ML 특화 테스트

	- 데이터 품질
		- 결측치, 이상치, 분포 검증
	- 모델 성능
		- 정확도, 재현율 등 주요 지표 검증
	- 편향성 테스트
		- 모델의 공정성과 편향 여부 확인

테스트 프레임워크

	- pytest
		- 파이썬 기반의 유연한 테스트 프레임워크
	- unittest
		- 파이썬 표준 라이브러리의 테스트 모듈
	- Great Expectations
		- 데이터 검증에 특화된 라이브러리

모델 서빙 방식

	- RESTful API
		- HTTP 기반의 범용적인 인터페이스
	- gRPC
		- 고성능 RPC 프레임워크
	- 배치 추론
		- 대량의 데이터를 일괄 처리
	- 실시간 추론
		- 낮은 지연시간의 온디맨드 예측

롤백 및 카나리 배포 전략

	1. 롤백
		- 문제 발생 시 이전 버전으로 신속 복귀
	2. 블루/그린 배포
		- 두 버전을 동시에 운영하며 전환
	3. 카나리 배포
		- 일부 트래픽으로 새 버전 테스트

컨테이너화 (Docker)

	- 이미지
		- 애플리케이션과 의존성을 포함한 실행 단위
	- 컨테이너
		- 이미지의 실행 인스턴스
	- Dockerfile
		- 이미지 빌드 명세서
	- Docker Compose
		- 다중 컨테이너 애플리케이션 정의 및 실행

오케스트레이션 (Kubernetes)

	- 파드
		- 하나 이상의 컨테이너 그룹
	- 서비스
		- 파드 집합에 대한 네트워크 추상화
	- 배포
		- 파드의 선언적 업데이트 관리
	- 스케일링
		- 워크로드에 따른 자동 확장/축소

클라우드 서비스 활용

	서비스 			특징
	
	AWS 		광범위한 서비스와 강력한 확장성
	GCP 		데이터 분석과 ML에 강점
	Azure 		기업 환경 통합에 유리
