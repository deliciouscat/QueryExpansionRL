# 실험 설정 담당자

실험 통합 연구자가 데이터·모델·학습 전략을 고른 뒤 해당 모듈 담당자와 수치를 확인한다. YAML은 코드에 존재하는 기능을 선택하는 설정이며 새로운 알고리즘을 몰래 구현하는 장소가 아니다.

- `smoke_sft.yaml`, `smoke_rl.yaml`: 다운로드 없는 기능 검증. 실제 모델 품질을 증명하지 않는다.
- `qwen_sft.yaml`, `qwen_rl.yaml`: 고정 model revision·데이터 provenance·container digest가 필요한 GPU 실행 템플릿.

SFT의 token 평균과 RL의 query-group 평균은 실험 코드에 명시되어 있다. `accumulation_steps`는 optimizer 갱신 전 처리할 micro-batch 수이며 rollout 후보 수 `group_size`와 다르다. RL의 batch_size를 바꾸면 window당 query 그룹 수도 달라진다.

데이터·모델·인덱스·설정·코드 identity가 달라진 실행을 동일 학습의 resume로 취급하지 않는다. 새 실험에는 새 output_dir을 쓰고, 가중치만 이어 학습할 때는 init_from을 사용한다. 모델 변경으로 위반되는 동결 정책이나 GPU 메모리 가정을 config만으로 숨기지 않는다.
