# 학습 기반 담당자

## 소유 범위

`backward.py`의 실행 계약, `iteration.py`의 accumulation 진행, `experiment.py`의 실행 산출물, `checkpoint.py`의 저장·재개, `config.py`의 설정·provenance를 관리한다. 모델·데이터·검색·보상 구현을 import하지 않는다. 실제 평가 실행과 모델 채택 판단은 통합 담당자가 callback으로 주입한다.

## 공개 계약: LossTerm과 gradient

`LossTerm`의 정의는 이 디렉터리가 소유한다. 필드를 바꾸거나 평균 단위의 해석을 바꾸려면 실험 통합 담당자와 합의한다.

| 필드 | 의미 |
|---|---|
| `loss` | 미분 가능한 scalar Tensor. `None`이면 분모·통계만 전달하는 관측치 |
| `weight` | 해당 loss의 분자 가중치. 실제 loss가 있을 때 유한한 양수 |
| `normalizer` | window 전체 분모에 더할 유한한 0 이상의 값 |
| `metrics` | 그래프가 없는 Python 관측값. 숫자는 관측 횟수로 평균, list는 이어 붙임 |

한 window의 목적함수는 `sum(weight * loss) / sum(normalizer)`다. 분모는 미분하지 않는다. SFT에서 평균할 token 수, RL에서 평균할 query 수는 실험 담당자가 명시한다. `utils`가 label이나 reward를 살펴 자동 추정하지 않는다.

`@gradient(engine)`은 `forward`의 Tensor/LossTerm 반환 또는 LossTerm generator를 소비한다. generator를 list로 변환하지 않고, **한 항의 backward가 끝난 후에 다음 항을 요청**한다. loss Tensor나 계산 그래프를 로그에 저장하지 않는다. 미리 계산한 loss list 및 아무 관측도 없는 generator는 거부한다. skip은 `LossTerm(loss=None, normalizer=1)`처럼 명시한다.

학습 코드에서 의도적으로 Tensor를 보관하면 그 참조 자체까지 제거해 주는 것은 아니다. `forward`는 이전 후보 그래프를 재사용하거나 `retain_graph=True`를 요구하지 않아야 한다.

## 업데이트 경계

`TrainingLoop`는 사용자 반복문을 통해 전달한 batch마다 decorated forward가 정확히 한 번 완료됐는지 확인한다. 누락·중복 호출·잡아서 숨긴 forward 예외도 정상 갱신으로 처리하지 않는다.

1. window 시작에 gradient를 비운다.
2. 각 LossTerm을 즉시 backward하여 gradient만 누적한다.
3. 지정한 micro-batch 수, epoch 마지막 batch 또는 정상적인 조기 break에서 window를 끝낸다.
4. 전달된 전체 분모로 gradient를 나누고 global clipping한다.
5. AdamW와 SGD를 각각 한 번 step하고 각 scheduler도 한 번 진행한다.
6. gradient를 비운 후에만 저장·평가 callback을 실행한다.

window 전체에 loss가 없으면 두 optimizer, weight decay, scheduler와 global optimizer step을 모두 유지한다. 부분 window의 분모는 실제 처리한 데이터에서 산출한다. clean break는 이미 소비한 batch만 반영한다.

`GradientEngine.window(normalizer=...)`는 분모를 미리 아는 저수준 실험·테스트용 API로 유지한다. 일반 실험에서는 `TrainingLoop`가 동적 분모를 관리한다.

## 실패·재개

forward/generator/backward/gradient 유한성 검사가 실패하면 해당 window의 gradient를 폐기하며 완료 callback을 실행하지 않는다. generator는 닫는다. 데이터 stream의 메모리상 cursor와 RNG는 이미 진행됐을 수 있으므로 실패한 실험은 마지막 완료 checkpoint에서 재개한다. 같은 객체로 실패 window를 임의로 건너뛰어 저장하지 않는다.

optimizer 내부에서 step 도중 오류가 나는 경우 이미 변경된 파라미터까지 되돌리는 transaction은 아니다. 이런 경우도 마지막 완료 checkpoint에서 모델과 양쪽 optimizer를 함께 복원한다.

checkpoint에는 전체 학습 상태를 저장하고 완료 marker 및 hash를 검증한다. active gradient window 중 저장은 거부한다. artifact 관리가 학습 알고리즘을 선택하거나 모델 forward를 호출해서는 안 된다.

검증: `tests/test_streaming_contract.py`에서 즉시 backward·activation 해제·분모·실패·저장 경계를, `tests/test_cli.py`에서 SFT/RL 저장·재개의 완전 일치를 검사한다.
