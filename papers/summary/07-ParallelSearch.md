# 7번째 논문 요약

논문: [ParallelSearch: Train your LLMs to Decompose Query and Search Sub-queries in Parallel with Reinforcement Learning](</Users/deliciouscat/projects/QueryExpansion/papers/parsed/2508.09303v1.md>)

저자: Shu Zhao 외 (NVIDIA)

## 한 줄 요약

기존 search agent가 독립적인 sub-query도 순서대로 검색하는 비효율을 해결하기 위해, LLM이 질문의 병렬 가능성을 판단하고 여러 검색어를 한 번에 실행하도록 강화학습한다.

## 문제 설정

예를 들어 “Claude Monet과 Camille Pissarro 중 누가 더 나이가 많은가?”는 두 사람의 생년월일을 각각 찾은 뒤 비교하면 된다. 두 검색은 서로 의존하지 않지만 기존 Search-R1은 다음처럼 순차 처리한다.

1. Monet 생년월일 검색
2. 결과를 읽고 Pissarro 검색
3. 두 결과 비교

ParallelSearch는 한 번에 다음을 만든다.

`Claude Monet birthday ## Camille Pissarro birthday`

두 검색을 병렬 실행하고 결과를 한 번에 context에 넣는다.

## 검색·추론 과정

모델은 `<think>`, `<search>`, `<information>`, `<answer>` 구조를 사용한다. `<search>` 안에서 `##`로 연결된 질의를 분리하고 각 질의를 동시에 검색한다.

알고리즘은 다음과 같다.

1. 질문을 보고 reasoning
2. 검색이 필요하면 하나 또는 여러 sub-query 생성
3. `##`로 sub-query 분리
4. 모든 검색을 병렬 실행
5. 검색 결과를 합쳐 다시 reasoning
6. 충분하면 answer, 아니면 다음 검색 라운드

## 보상 설계

단순히 답만 맞히면 순차 검색을 해도 높은 보상을 받기 때문에, 네 가지 보상을 결합한다.

- Outcome reward: 최종 답 Exact Match
- Decomposition reward: 병렬화 가능한 질문을 올바르게 분해했는지 평가
- Search count reward: 불필요한 검색 라운드·호출을 줄였는지 평가
- Format reward: 출력 태그와 sub-query 형식 준수

HotpotQA의 comparison 질문은 두 엔티티에 대한 검색으로 나눌 수 있지만, bridge 질문은 첫 번째 검색 결과가 다음 질문을 결정하므로 병렬화하면 안 된다. 따라서 모델은 모든 multi-hop 질문을 무조건 분해하는 것이 아니라, parallel/compositional 구조를 구분해야 한다.

## 비교군과 실험

모델은 Qwen2.5-7B Base/Instruct이고, Wikipedia 또는 MultiHopRAG corpus를 검색한다. 검색 결과는 기본 top-3개를 사용한다.

비교군:

- R1: 검색 없이 reasoning
- Search-R1: 순차 검색 RL
- ZeroSearch: 검색 엔진을 simulator로 학습
- StepSearch: 검색 action의 정보 이득·중복을 token-level reward로 평가
- OTC: 긴 trajectory에 penalty
- Search-R1 + Parallel Prompt: prompt만으로 병렬화를 유도

데이터셋은 NQ, TriviaQA, PopQA, HotpotQA, 2WikiMultiHopQA, MuSiQue, Bamboogle이며, 별도로 HotpotQA-par/seq, 2wiki-par/seq, MultihopRAG-par/seq subset을 만들었다.

## 주요 결과

전체 일곱 QA benchmark에서 평균 2.9% 성능 개선을 얻었다. 병렬화 가능한 질문에서는 평균 12.7% 개선을 보였고, LLM 호출 수는 순차 방식의 69.6% 수준으로 줄었다. 즉 호출 약 30.4%를 절약했다.

병렬 질문 subset의 예:

| 데이터셋 / 모델 | Search-R1 EM | ParallelSearch EM | 병렬 분해 비율 |
|---|---:|---:|---:|
| HotpotQA-par Instruct | 0.580 | **0.656** | 94.30% |
| HotpotQA-par Base | 0.641 | **0.673** | 97.03% |
| 2wiki-par Instruct | 0.476 | **0.650** | 99.75% |
| 2wiki-par Base | 0.624 | **0.691** | 99.86% |
| MultihopRAG-par Instruct | 0.509 | **0.641** | 88.67% |

ParallelSearch는 병렬 질문에서 대체로 2회 정도의 turn으로 끝나는 반면 Search-R1은 3~6회가 필요했다. 응답 길이도 더 짧아졌다.

## 순차 질문에서의 안전성

병렬화만 강조하면 순차 질문을 망칠 수 있다. 그러나 ParallelSearch는 강화학습으로 의존성을 학습해 순차 질문에서는 필요할 때 순서대로 검색한다.

| 데이터셋 | Search-R1 | ParallelSearch | EM 개선 |
|---|---:|---:|---:|
| HotpotQA-seq Instruct | 31.08 | **37.42** | +6.34 |
| 2wiki-seq Instruct | 21.15 | **22.45** | +1.30 |
| MultihopRAG-seq Instruct | 77.34 | **86.20** | +8.86 |

Base 모델에서도 HotpotQA-seq 37.86에서 40.61, MultihopRAG-seq 87.76에서 91.15로 개선됐다. 반면 Search-R1에 병렬 prompt만 추가한 baseline은 turn 수가 늘거나 성능이 하락했다. 즉 prompt 지시만으로는 병렬 검색을 안정적으로 학습하기 어렵다.

## 보상 ablation

분해 보상 `lambda_d`나 검색 수 보상 `lambda_s`를 제거하면:

- 분해 비율이 낮아짐
- 검색 turn 수가 증가
- 불필요한 sequential behavior가 늘어남

HotpotQA-par에서는 `lambda_d=0.15`, `lambda_s=0.35`가 EM 0.656, 분해율 94.30%, 평균 2.08 turn으로 가장 좋은 균형을 보였다.

## 효율성 분석

- top-k를 1에서 3으로 올리면 EM이 크게 개선된다.
- 5 이상에서는 개선폭이 작아진다.
- ParallelSearch는 검색 결과를 더 짧은 reasoning trajectory로 처리한다.
- parallel prompt baseline은 병렬 지시를 따라도 실제 turn 수를 줄이지 못한다.

## 한계

- 병렬/순차 구조 라벨을 만들기 위한 질문 유형 분류가 필요하다.
- 검색 결과가 서로 독립적이라는 가정이 틀리면 정보 의존성을 잃을 수 있다.
- 병렬 검색은 API rate limit과 동시성 인프라를 요구한다.
- 실험은 주로 text Wikipedia와 정형 QA에 집중되어 있다.
- 검색 결과를 모두 한 번에 context에 넣으면 context length와 noise가 증가할 수 있다.

## 최종 평가

이 논문은 search agent의 문제를 “더 많이 검색할 것인가”에서 “어떤 검색을 동시에 할 수 있는가”로 확장한다. 핵심은 병렬화 자체가 아니라, 강화학습으로 독립성과 의존성을 구분하도록 만든 점이다. 따라서 inference 비용과 latency가 중요한 multi-hop RAG 서비스에서 실용적인 개선 방향을 제시한다.

