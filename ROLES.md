# 디렉터리별 담당 안내

기존 `README.md`와 `CODE_INTERFACE.md`의 역할 분리를 구현에 연결하는 작업 안내다. 원래 설계 문서를 대체하지 않는다. 현재 구현의 담당 범위와 `forward`–gradient 실행 계약은 아래 문서에서 확인한다.

| 경로 | 담당자 | 상세 안내 |
|---|---|---|
| `train/` | 실험 통합 연구자 | [학습 절차의 조립과 선택](train/ROLES.md) |
| `src/query_expansion/data_loader/` | 데이터 연구자 | [스키마·adapter·sampling](src/query_expansion/data_loader/ROLES.md) |
| `src/query_expansion/agents/` | 모델·정책 연구자 | [모델 구성·생성·log-probability](src/query_expansion/agents/ROLES.md) |
| `src/query_expansion/rewards/` | 목적함수·보상 연구자 | [보상·advantage·loss](src/query_expansion/rewards/ROLES.md) |
| `src/query_expansion/retriever/` | 검색 연구자 | [고정 인덱스와 검색 API](src/query_expansion/retriever/ROLES.md) |
| `src/query_expansion/utils/` | 학습 기반 엔지니어 | [LossTerm·backward·갱신·재개 계약](src/query_expansion/utils/ROLES.md) |
| `src/query_expansion/evaluation/` | 평가 연구자 | [검색 지표와 집계](src/query_expansion/evaluation/ROLES.md) |
| `src/query_expansion/`의 CLI 파일 | 실험 통합 담당자 | [CLI와 패키지 경계](src/query_expansion/ROLES.md) |
| `configs/` | 실험 통합 연구자 | [실험 설정과 재현성](configs/ROLES.md) |
| `tests/` | 각 모듈 담당자 + 통합 검증 담당자 | [계약 및 회귀 검증](tests/ROLES.md) |
| `papers/` | 문헌 조사 담당자 | [논문과 구현의 근거 관리](papers/ROLES.md) |
| `tmp/` | 해당 임시 작업 수행자 | [전처리 보조 작업](tmp/ROLES.md) |

루트의 `pyproject.toml`은 통합 담당자가 설치·패키징 책임으로 관리한다. `runs/`와 `exports/`는 학습·평가 프로그램이 생성하는 산출물이며 코드 모듈이 아니다. 산출물을 수동 변경해 정상 checkpoint나 검증 완료 모델처럼 만들지 않는다. `.venv/`, cache, build 디렉터리도 담당자의 소스 편집 영역에 포함하지 않는다.

## 계약 변경을 조율하는 방법

구현 세부 변경은 해당 모듈 내부에서 처리한다. 공개 입력·출력이나 수식의 의미가 바뀌면 생산자와 소비자 담당자가 먼저 합의하고, 통합 담당자가 `train/` 연결과 회귀 테스트를 확인한다. 다른 모듈의 내부 구조를 import해 임시로 연결하지 않는다.

특히 `forward`와 decorator 사이 계약의 정의·검증 구현은 `utils` 담당자가 소유한다. 실험 통합 담당자는 **어떤 loss를 어떤 단위로 평균할지** 결정한다. 모델 담당자나 `utils` 담당자가 실험의 그룹 수·가중치를 묵시적으로 바꾸지 않는다.

현재 연구 흐름을 읽으려면 `train/sft.py` 또는 `train/rl.py`부터 시작한다. 실행 명령은 `RUNNING.md`를 참고한다. 이번 추상화로 옮겨진 공통 실행 처리는 `utils/iteration.py`, 산출물 관리는 `utils/experiment.py`에 있으며, `train/common.py`는 데이터 준비·provenance·평가 연결만 맡는다.
