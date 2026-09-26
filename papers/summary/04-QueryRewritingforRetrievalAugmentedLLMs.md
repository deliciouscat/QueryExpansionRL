# 4번째 논문 요약

논문: [Query Rewriting for Retrieval-Augmented Large Language Models](</Users/deliciouscat/projects/QueryExpansion/papers/parsed/2305.14283v3.md>)

저자: Xinbei Ma 외

## 한 줄 요약

고정된 검색 엔진과 black-box LLM reader 앞에 작은 T5 질문 재작성기를 추가하고, 최종 LLM 답변의 정확도를 보상으로 PPO 학습해 RAG 검색어를 최적화한다.

## 문제의식

일반적인 RAG는 원래 질문을 그대로 retriever에 넣고, 검색 문서를 LLM에 전달한다. 그러나 원래 질문은 답에 필요한 지식을 검색하기에는 너무 길거나, 여러 단계의 의도를 섞고 있거나, 검색 엔진이 잘 처리하지 못하는 표현일 수 있다.

기존 연구는 retriever 또는 reader를 학습하는 경우가 많지만, ChatGPT처럼 내부 파라미터에 접근할 수 없는 LLM을 reader로 쓰면 어렵다. 이 논문은 검색어만 학습해 고정된 retriever와 reader 사이의 간극을 줄인다.

## Rewrite-Retrieve-Read 구조

1. 원래 입력 `x`를 작은 rewriter가 검색 질의 `x~`로 변환
2. Bing 검색 엔진이 관련 문서나 snippet을 검색
3. frozen LLM reader가 원래 질문과 검색 문서를 보고 답변
4. 답변의 EM/F1와 검색 결과의 answer hit를 합쳐 rewriter 보상으로 사용

두 가지 형태를 비교한다.

- Frozen rewriter: ChatGPT를 few-shot prompting해 검색어를 생성
- Trainable rewriter: T5-large(약 770M)를 학습

## 학습 방법

### Warm-up

먼저 ChatGPT에게 원래 질문을 검색어로 바꾸게 하고, 그 검색어로 LLM이 정답을 맞힌 사례만 pseudo label로 남긴다. 이 데이터로 T5를 지도학습한다.

### 강화학습

Warm-up 이후 PPO로 계속 학습한다. 한 episode에서 T5가 검색어를 생성하고 Bing 검색과 reader 호출이 끝나면 최종 보상을 계산한다.

보상은 다음을 결합한다.

- `EM`: 최종 답이 정확히 일치하면 +1
- `F1`: 예측 답과 정답 답의 단어 중첩
- `Hit`: 검색 문서에 정답이 포함되면 +1, 아니면 -1

다중선택 QA에서는 최종 EM만 사용한다.

## 실험 설정

### 데이터셋

- HotpotQA: 여러 문서를 연결해야 하는 multi-hop QA
- AmbigNQ: 모호한 질문을 구체화한 open-domain QA
- PopQA: long-tail entity 중심 QA
- MMLU: Humanities, STEM, Social Science, Other의 객관식 QA

reader는 ChatGPT를 기본으로 사용하고, rate limit 때문에 Vicuna-13B도 평가했다. 검색기는 Bing이다. 검색 결과는 snippet을 직접 사용하거나 BM25로 URL 본문에서 관련 문단을 고른다.

### 비교군

- Direct: 검색 없이 LLM이 직접 답함
- Retrieve-then-read: 원래 질문을 그대로 검색
- LLM rewriter: frozen ChatGPT가 검색어 생성
- Trainable rewriter: warm-up + PPO로 학습한 T5

## Open-domain QA 결과

| 데이터셋 / 방법 | EM | F1 |
|---|---:|---:|
| HotpotQA Direct | 32.36 | 43.05 |
| HotpotQA Retrieve-then-read | 30.47 | 41.34 |
| HotpotQA LLM rewriter | 32.80 | 43.85 |
| HotpotQA Trainable rewriter | **34.38** | **45.97** |
| AmbigNQ Direct | 42.10 | 53.05 |
| AmbigNQ Retrieve-then-read | 45.80 | 58.50 |
| AmbigNQ LLM rewriter | 46.40 | 58.74 |
| AmbigNQ Trainable rewriter | **47.80** | **60.71** |
| PopQA Direct | 41.94 | 44.61 |
| PopQA Retrieve-then-read | 43.20 | 47.53 |
| PopQA LLM rewriter | **46.00** | **49.74** |
| PopQA Trainable rewriter | 45.72 | 49.51 |

분석 포인트:

- HotpotQA에서는 원 질문을 그대로 검색하면 오히려 성능이 하락한다. 복합 질문 전체가 검색어로는 부적합해 불필요한 문서를 가져오기 때문이다.
- 재작성은 HotpotQA의 multi-hop 단서를 분리해 성능을 회복하고, T5 rewriter가 가장 좋았다.
- AmbigNQ에서는 검색 자체가 유용하며, 질문 재작성은 그 효과를 추가로 높인다.
- PopQA에서는 ChatGPT rewriter가 T5보다 약간 좋다. “검색하지 않아도 되는 질문”을 구분하는 adaptive retrieval이 어렵기 때문이다.

## MMLU 결과

ChatGPT reader에서 LLM rewriter는 대체로 direct/retrieve-then-read보다 좋았지만 Social Science에서는 하락했다.

Vicuna-13B reader에서는 다음과 같이 재작성 효과가 더 뚜렷했다.

| Reader / 방법 | Humanities | STEM | Other | Social |
|---|---:|---:|---:|---:|
| Vicuna Direct | 39.8 | 34.9 | 50.2 | 46.6 |
| Vicuna Retrieve | 40.2 | 39.8 | 55.2 | 50.6 |
| Vicuna LLM rewriter | 42.0 | 41.5 | 57.1 | 52.2 |
| Vicuna Trainable rewriter | **43.2** | 40.9 | **59.3** | 51.2 |

매우 강한 reader일수록 이미 내부 지식이 많아 검색의 추가 이득이 작아지는 경향도 관찰됐다.

## 추가 분석

- RL은 warm-up 직후보다 3~4 iteration 뒤에 AmbigNQ와 PopQA의 baseline을 넘었다. pseudo label만으로는 부족하고 최종 QA 보상이 추가로 필요하다.
- AmbigNQ에서 정답이 검색 문서에 포함된 경우의 reader 성능 상한은 EM 58.40, F1 69.45였다. Trainable rewriter의 EM 47.80은 검색 hit와 reader 추론 모두에 개선 여지가 있음을 의미한다.
- BM25로 검색 문단을 고르는 방식이 단순 Bing snippet보다 answer hit가 높았다.
- 사례 분석에서 T5는 `2000 movie`처럼 검색 엔진이 잘 처리하는 단어 결합을 유지해 ChatGPT가 만든 모호한 질의보다 좋은 문서를 가져오기도 했다.

## 한계

- Bing API와 black-box LLM 호출 비용·속도에 의존한다.
- 웹 문서는 일관성·중복·유해성 문제가 있고 검색 결과를 통제하기 어렵다.
- 하나의 rewriter를 여러 downstream task에 직접 전이하면 일반화와 특화 사이 trade-off가 있다.
- 여러 번 검색하며 읽고 후속 질의를 만드는 agent 방식보다 한 번의 rewrite에 집중하므로 복잡한 장기 reasoning에는 제한이 있다.
- dense retriever와 통제된 지식베이스가 더 안정적일 수 있다.

## 최종 평가

이 논문은 RAG의 개선 지점을 retriever나 reader의 내부가 아니라 “검색어”로 옮긴다. 특히 ChatGPT처럼 frozen reader를 활용해야 하는 환경에서, 작은 학습 모델만 추가해 시스템을 최적화할 수 있다는 점이 실용적이다. 다만 질문 재작성의 품질은 절대적이지 않고, 어떤 reader·검색 엔진·데이터셋을 고정했는지에 따라 달라진다.

