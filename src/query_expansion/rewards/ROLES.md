# 목적함수·보상 담당자

## 소유 범위

`objectives.py`에서 전략 registry와 SFT/RL loss 수식을, `retrieval.py`에서 검색 보상·환경/candidate 선택·group advantage를 관리한다. 모델 생성이나 backward, optimizer와 checkpoint를 실행하지 않는다.

## 공개 API

- `Rewards(strategy)`: 등록한 목적함수 선택. 지원하지 않는 전략은 오류.
- `target_token_count(labels)`: causal completion-only SFT에서 평균할 유효 target+EOS 수.
- `group_advantages(rewards)`: population standard deviation을 사용한 그룹 정규화.
- `RetrievalReward(retrievers, ...)`: 주입받은 검색 공개 API로 rank-gain reward 계산.
- `prepare(record, rng)`: 그룹이 공유할 환경·후보 문서·원문 baseline 생성.
- `score_group(record, rollouts, rng)`: 하나의 공통 context로 모든 후보를 평가하고 `RewardGroup(advantages, metrics)` 반환.

rollout 입력은 `completion.tokens`와 `expansion.query/invalid/fallback` 필드가 있는 객체다. `agents` 내부를 import하지 않는다. `RewardGroup.active`는 적어도 하나의 nonzero advantage가 있는지 나타낸다. metrics는 query 그룹당 한 관측이며 loss Tensor를 포함하지 않는다.

## 반드시 보장할 것

같은 query 그룹의 모든 후보를 같은 환경·문서 집합에서 비교한다. IDF와 평균 문서 길이를 query별 후보 집합으로 다시 계산하지 않는다. reward의 길이/invalid penalty와 실제 retrieval 평가 지표를 구분한다.

어떤 loss와 denominator로 학습할지는 통합 담당자가 조립한다. `score_group`이 `utils.LossTerm`까지 만들어 optimizer 정책을 결정하지 않는다. 보상 담당자는 상수 보상 그룹의 zero advantage를 알리고, `train`은 해당 query를 분모에 포함할지 명시하며 현재 실험은 포함한다.

새 수식은 해당 모듈과 테스트에서 검증한 후 통합 담당자가 실험 파일에 연결한다. 검색 구현의 내부 필드 대신 `Retriever` 공개 API만 사용한다.

검증: `tests/test_objectives.py`, `tests/test_retrieval_rl.py`. graded qrels, 동률, missing positive, 상수 보상과 gradient 방향을 확인한다.
