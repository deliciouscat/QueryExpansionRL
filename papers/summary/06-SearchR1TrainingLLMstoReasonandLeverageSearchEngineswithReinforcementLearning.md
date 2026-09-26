# 6번째 논문 요약

논문: [Search-R1: Training LLMs to Reason and Leverage Search Engines with Reinforcement Learning](</Users/deliciouscat/projects/QueryExpansion/papers/parsed/2503.09516v5.md>)

## 한 줄 요약

LLM이 추론 중 `<search>` 토큰으로 직접 검색어를 만들고, 검색 결과를 읽으며 여러 차례 추론하도록 PPO/GRPO로 학습한다. 검색 행동 자체에 별도 라벨을 주지 않고 최종 정답의 Exact Match만 보상으로 사용한다.

## 문제의식

일반 RAG는 질문당 한 번 검색하고 문서를 붙인다. 하지만 복잡한 질문은 어떤 정보를 검색해야 하는지 먼저 추론해야 하고, 첫 검색 결과를 본 뒤 후속 검색어를 바꿔야 한다.

기존 ReAct·IRCoT는 prompt만으로 이 행동을 유도하고, Toolformer는 검색 trajectory의 지도 데이터가 필요하다. Search-R1은 검색 엔진을 RL 환경에 넣어 모델이 시행착오로 검색 시점과 질의를 학습한다.

## 구조와 학습

모델 출력은 다음 토큰 구조를 따른다.

- `<think>...</think>`: 현재 정보에 대한 추론
- `<search>query</search>`: 검색 엔진 호출
- `<information>...</information>`: 검색 결과 삽입
- `<answer>...</answer>`: 최종 답

검색이 끝날 때마다 retrieved passage가 rollout sequence에 붙고, 모델은 그 문맥을 보고 다시 think/search하거나 answer를 낸다.

### 보상

최종 답을 추출해 정답과 exact match를 계산한다. 별도의 process reward, format reward, neural reward model은 사용하지 않는다.

### Retrieved token masking

검색 결과는 모델이 생성한 토큰이 아니라 외부 환경이 넣은 토큰이다. 이 토큰까지 loss/KL 계산에 포함하면 모델이 검색 내용 자체를 최적화하는 이상한 경로가 생길 수 있다. Search-R1은 검색 결과 토큰에 대한 loss를 mask하고 모델이 생성한 reasoning/search/answer 토큰만 학습한다.

## 실험 설정

모델:

- Qwen2.5-3B Base/Instruct
- Qwen2.5-7B Base/Instruct

검색:

- 2018 Wikipedia dump
- E5 retriever
- 모든 retrieval baseline에서 top-3 passage

데이터셋:

- 일반 QA: NQ, TriviaQA, PopQA
- Multi-hop QA: HotpotQA, 2WikiMultiHopQA, MuSiQue, Bamboogle

훈련은 NQ와 HotpotQA를 합친 데이터로 하고, 나머지는 OOD 평가로 사용한다.

비교군:

- Direct inference, CoT
- RAG
- IRCoT, Search-o1
- SFT
- 검색 없는 R1 RL
- 정답 trajectory를 고르는 rejection sampling

## 주요 결과

7B 모델의 Search-R1은 일곱 데이터셋 평균 EM 0.431(base), 0.385(instruct)를 기록했다. 같은 조건의 일반 RAG 평균은 0.304, rejection sampling은 0.348이었다. 논문은 RAG 대비 7B에서 평균 상대 24%, 3B에서 20% 개선으로 보고한다.

7B Search-R1-base의 세부 결과:

| 데이터셋 | RAG | Search-R1-base |
|---|---:|---:|
| NQ | 0.349 | **0.480** |
| TriviaQA | 0.585 | **0.638** |
| PopQA | 0.392 | **0.457** |
| HotpotQA | 0.299 | **0.433** |
| 2Wiki | 0.235 | **0.382** |
| MuSiQue | 0.058 | **0.196** |
| Bamboogle | 0.208 | **0.432** |

특히 multi-hop과 OOD 데이터에서 검색을 여러 번 사용하면서 개선폭이 컸다. 검색 없는 R1보다도 Search-R1이 높아, RL reasoning에 외부 검색을 결합하는 이점이 확인됐다.

## PPO와 GRPO 비교

- GRPO: critic 없이 그룹 내 reward 평균으로 baseline을 만들기 때문에 빠르게 수렴
- PPO: critic이 필요해 느리지만 학습이 더 안정적

실험에서 GRPO는 초기에 빠르지만 오래 학습하면 reward collapse가 발생했고, PPO는 느려도 안정적으로 유지됐다. 최종 성능은 비슷했으므로 이 환경에서는 PPO가 더 실용적인 기본 선택으로 평가된다.

## Base와 Instruct 모델

Instruct 모델은 초기 성능이 높고 더 빨리 수렴한다. 그러나 충분히 학습하면 Base 모델도 비슷한 최종 보상에 도달했다. 즉 instruction tuning은 탐색을 쉽게 하지만 search reasoning 능력을 RL로 새로 학습할 수 있다.

## 검색 행동과 응답 길이

학습 초기에는 불필요한 말이 줄어 응답 길이가 감소한다. 이후 모델이 검색을 더 자주 호출하고 검색 결과를 답변에 반영하면서 응답 길이가 다시 증가한다. 유효 검색 호출 수도 학습과 함께 늘었다.

이는 모델이 단순히 길게 생각하는 것이 아니라, 외부 정보가 필요할 때 검색을 호출하는 정책을 학습했음을 시사한다.

## Masking ablation

Qwen2.5-7B base, PPO 결과:

| 설정 | 평균 EM |
|---|---:|
| retrieved token masking 사용 | **0.431** |
| masking 미사용 | 0.343 |

일곱 데이터셋 모두 masking을 사용한 모델이 높았다. 검색 결과 토큰의 gradient를 막는 것이 안정화에 매우 중요하다.

## 한계

- Wikipedia와 E5라는 한정된 검색 환경에 의존한다.
- 최종 EM 하나만 보상으로 사용해 검색 과정의 효율·근거 품질·중복을 직접 최적화하지 않는다.
- 검색 API 비용과 multi-turn latency가 크다.
- exact-match 정답이 없는 생성형·주관적 질문에는 적용이 어렵다.
- 검색 결과가 틀리거나 편향되면 모델이 이를 추론에 사용한다.

## 최종 평가

Search-R1의 핵심 기여는 search-as-a-tool을 별도 지도 trajectory 없이 outcome RL로 학습할 수 있음을 보인 것이다. 이전 query reformulation 연구가 검색어만 최적화했다면, 이 논문은 검색 시점·다중 검색·검색 결과를 활용한 reasoning 전체를 최적화한다. 이후 ParallelSearch나 retriever-aware RL의 직접적인 기준선 역할을 한다.

