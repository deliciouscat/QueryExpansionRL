# 모델·정책 담당자

## 소유 범위

모델·tokenizer 로딩, prompt와 출력 후처리, 모델에 맞는 Tensor encoding, 생성 정책, log-probability 재계산, 학습 파라미터 분리 및 export를 관리한다. 데이터 source 선택, relevance 판정, advantage 계산, optimizer step은 맡지 않는다.

## 공개 API

- `Agents(...)`: tiny 또는 고정 revision의 Qwen 정책과 trainable parameter groups 구성.
- `supervised(inputs, targets)`: query/language와 검수된 target을 encoding해 inputs와 masked labels 반환. prompt/padding에는 loss가 걸리지 않는다.
- `generate(query, language, sample=...)`: `Completion(tokens, text, ended)` 반환.
- `generate_group(inputs, size)`: 후보를 순차 생성하고 후처리한 `Rollout(completion, expansion)` 목록 반환. gradient 없이 실행하며 token ID·문자열·판정 결과만 보관한다.
- `completion_log_prob(inputs, tokens)`: 지정된 생성 토큰과 EOS의 log-probability 합을 미분 가능한 scalar로 반환.
- `postprocess`: 학습·평가·추론에서 공통으로 쓰는 결합 질의와 invalid/fallback 판정.

`Rollout` 형식은 `agents/text.py`가 소유한다. `rewards.score_group`은 필요한 필드를 구조적으로 읽으며 agents 구현을 import하지 않는다. 형식을 바꾸면 보상 담당자와 통합 담당자에게 변경을 알린다.

## 반드시 보장할 것

policy 입력은 query와 language다. answers·qrels·정답 문서를 prompt에 넣지 않는다. RL의 sampling 분포와 재계산 분포를 일치시키고 dropout을 비활성화한다. 생성에 필요한 모델 mode 변경은 복원하며 gradient checkpointing이 필요한 학습 forward를 무조건 eval 모드로 바꾸지 않는다.

하위 LoRA와 상위 전체 학습 파라미터는 서로 겹치지 않고 전체 trainable 집합을 정확히 덮어야 한다. embedding·공유 head·최종 norm 등 동결 정책을 지킨다. decoder 하위 전체를 `no_grad`로 감싸 LoRA gradient를 끊지 않는다.

후보별 계산 그래프의 소비·backward 시점은 `utils`가 담당한다. 모델 담당자는 미분 가능한 결과를 제공하며 optimizer를 직접 호출하지 않는다.

검증: `tests/test_training.py`, `tests/test_qwen_integration.py`, export 관련 CLI 테스트. 실제 A5000 메모리·성능 검증은 CPU tiny 테스트와 구분해 기록한다.
