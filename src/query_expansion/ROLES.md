# 패키지 및 CLI 담당

이 경로 바로 아래의 `train.py`, `evaluate.py`, `export.py`는 실험 통합 담당자가 관리한다. CLI 옵션을 해석하고 적절한 실험 또는 공개 API를 연결한다. CLI에 새로운 학습 알고리즘이나 모델·검색 구현을 넣지 않는다.

학습 알고리즘은 저장소의 `train/`, 각각의 구현은 아래 모듈이 소유한다.

- [data_loader](data_loader/ROLES.md): 데이터 계약과 sampling.
- [agents](agents/ROLES.md): 정책과 생성.
- [rewards](rewards/ROLES.md): 보상과 목적함수.
- [retriever](retriever/ROLES.md): 고정 검색 환경.
- [utils](utils/ROLES.md): 실행 계약, 갱신과 재개.
- [evaluation](evaluation/ROLES.md): 측정과 보고.

모듈의 `__init__.py`는 해당 모듈 담당자가 공개 API로 관리한다. 서로의 내부 구현을 연결하기 위한 별도 공용 contracts 패키지는 두지 않는다. 모듈 사이 연결은 entrypoint에서 주입하고, `rewards → retriever`는 검색 공개 인터페이스만 사용한다.
