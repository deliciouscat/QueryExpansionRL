# Training utilities module

**담당:** 학습 기반 전문가 한 명 또는 한 역할 담당자. 변경 범위는 이 디렉터리다.

`utils.backward.gradient` decorator와 공통 optimizer 갱신, gradient accumulation, checkpoint 저장·재개를 관리한다. 모델별 구조나 보상 로직은 이 디렉터리에 넣지 않는다.

optimizer 갱신 경계와 저장 가능한 상태의 공개 동작을 이 모듈 안에서 문서화한다. 실험 entrypoint는 학습 절차와 저장 시점을 지정하고, 공통 실행 동작은 이 모듈을 사용한다.
