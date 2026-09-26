# 실험 통합 담당자

## 소유 범위

`train/<실험명>.py`에서 데이터·모델·검색 환경·목적함수를 선택하고 연결한다. 연구자가 바꿔야 하는 알고리즘은 `forward`와 눈에 보이는 batch 반복문에 둔다. 모델 내부, BM25 구현, backward 반복 동작을 이 디렉터리에 복제하지 않는다.

- `sft.py`: completion-only SFT의 모델 호출·loss와 target-token 평균 단위를 지정한다.
- `rl.py`: 후보 생성 → 보상·advantage → 후보별 log-probability → policy loss 순서를 지정한다.
- `common.py`: 재현 seed를 모델 생성 전에 적용하고 데이터와 provenance, dev 평가 callback을 연결한다. 평가 결과 중 어떤 값을 best score로 쓸지, 언제 early-stop할지도 통합 담당자의 선택이다.

`Agents`, `DatasetStream`, `Rewards`, retriever 선택을 실험 파일에서 직접 확인할 수 있다. 공통 `configure`가 새로운 모델·데이터·목적함수를 몰래 선택하지 않는다.

## forward가 지킬 계약

`utils.backward.LossTerm`을 반환하거나 하나씩 yield한다. 단일 scalar Tensor 반환도 지원하지만, 서로 다른 길이의 SFT나 RL에는 명시적 `LossTerm`을 사용한다.

- SFT: `LossTerm.mean(loss, count=target_token_count(labels))`.
- RL: query마다 먼저 `LossTerm(normalizer=1, metrics=...)`를 yield한다. 그 뒤 후보마다 `LossTerm(loss, weight=1/G)`를 yield한다.
- advantage가 전부 0인 그룹도 query 수에는 포함한다. 그 그룹의 loss 계산은 생략한다.
- 여러 후보의 계산 그래프를 list에 모으지 않는다. 모델 생성 API가 반환하는 token ID·문자열의 목록과 미분 가능한 loss 목록은 다르다.
- `with experiment.steps() as batches:` 안에서 각 batch마다 decorated `forward`를 정확히 한 번 호출한다. 함수 안의 여러 yield는 한 호출이다.

`experiment.steps()`는 batch 진행과 accumulation 경계만 관리하며 모델을 호출하거나 loss를 선택하지 않는다. 저장 요청 `save_flag`는 iterator가 모아 완료된 window 경계에서 처리한다. 실험별 추가 저장 정책은 `Experiment`의 boundary callback을 조립하는 곳에서 지정하며, gradient가 남아 있는 동안 직접 저장하지 않는다.

## 다른 담당자와의 연결

`forward`의 denominator·weight 의미는 통합 담당자가 정하고 `utils` 담당자가 실행을 보장한다. 새 rollout 방식은 `agents`, 새 보상식은 `rewards`, 새 데이터 형식은 `data_loader`에 요청한다. 공개 API 변경이 필요하면 해당 담당자와 합의한다. 기능을 넣기 위해 `train`에서 다른 모듈의 비공개 필드를 뒤지지 않는다.

검증: `tests/test_public_loop.py`, `tests/test_cli.py`, `tests/test_streaming_contract.py`. 모델 선택부터 checkpoint까지 학습 절차가 여전히 읽히는지 코드 리뷰한다.
