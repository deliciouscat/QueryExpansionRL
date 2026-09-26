# 8번째 논문 요약

논문: [Understanding the Behaviors of Environment-aware Information Retrieval](</Users/deliciouscat/projects/QueryExpansion/papers/parsed/2606.16817v1.md>)

## 한 줄 요약

retriever마다 잘 맞는 검색어 문체가 다르므로, 하나의 query rewriting policy를 모든 retriever에 공유하면 안 된다. 이 논문은 RL로 rewriter를 retriever별로 적응시키고, 서로 다른 retriever 사이의 전략 전이 실패를 “Structural Drift”로 분석한다.

## 문제의식

RAG·agentic retrieval 연구는 검색기를 일반적인 tool처럼 취급하는 경우가 많다. 그러나 retriever의 학습 방식과 점수 함수가 다르면 같은 질문에도 최적 검색어가 달라진다.

예:

- BM25: 짧고 희귀한 keyword 중심 query
- Contriever: 검색 대상 문서와 비슷한 서술형·가상 문서 query
- dense embedding retriever: 질문형 또는 의미적으로 풍부한 문장

따라서 한 retriever에서 학습한 rewriting policy를 다른 retriever에 그대로 적용하면 성능이 떨어질 수 있다.

## 방법

### RL 환경

사용자 질문을 입력받은 LLM rewriter가 query를 생성한다. retriever가 문서를 검색하고, gold passage에 대한 `nDCG@10`을 보상으로 준다.

Recall만 쓰지 않고 nDCG를 쓴 이유는 정답 문서를 찾는 것뿐 아니라 상위에 잘 배치하는 것도 중요하기 때문이다.

### Single-turn

한 번 rewrite한 query의 검색 성능으로 GRPO를 학습한다. 여러 출력의 reward 평균과 표준편차로 advantage를 계산하므로 별도 critic이 필요 없다.

### Multi-turn과 branching rollout

여러 검색 turn에서는 첫 query의 품질과 두 번째 query의 품질이 섞여 credit assignment가 불안정해진다. 이를 해결하기 위해 `4 × 4 branching`을 사용한다.

- 첫 turn query 4개 생성
- 각 첫 query마다 두 번째 turn query 4개 생성
- 같은 첫 query에서 나온 4개 결과의 평균으로 첫 turn 가치를 추정
- 두 번째 turn은 같은 history 안에서 비교

이 구조는 중간 reward의 분산을 줄이고, 후속 query가 앞선 검색 결과를 잘 활용했는지를 분리해 학습한다.

## retriever 환경

- BM25: sparse keyword 검색
- Contriever: Wikipedia span contrastive training 기반 unsupervised dense retriever
- all-MiniLM-L6-v2: 작은 supervised dense embedding 모델
- Qwen3-Embedding: 대형 decoder 계열 embedding 모델

학습은 RAGBench의 10만 개 이상 sample로 구성한 retrieval 환경에서 하고, BEIR와 FinAgentBench로 전이한다.

## 주요 결과

### RAGBench

Contriever 기준 평균 nDCG@10:

| 방법 | 평균 |
|---|---:|
| Contriever | 45.5 |
| 일반 rewrite | 54.5 |
| retriever-specific direct rewrite | **56.3** |
| general + 2-turn | 55.2 |
| direct + 2-turn | **56.5** |

BM25도 rewrite 적용 시 평균이 45.0에서 53~54대까지 상승했다. 즉 RL rewriter가 retriever의 특성에 맞는 “dialect”를 학습한다.

### BEIR zero-shot 전이

일반 데이터에서 학습한 rewriter를 BEIR에 추가 학습 없이 적용했을 때 Contriever 평균 nDCG@10이 28.84에서 34.98로 상승했다. 여러 retriever에서 전반적인 개선이 관찰되지만, source retriever의 전략을 target retriever에 그대로 복사하는 것은 안정적이지 않다.

### 금융 도메인 전이

FinAgentBench를 금융 문서·질문·graded relevance 구조로 변환해 평가했다. Contriever는 nDCG@10 6.43에서 rewrite 적용 후 7.39로, BM25는 8.17에서 9.02로 개선됐다. 앞 절의 45.5, 54.5, 56.3은 RAGBench 결과다.

## Structural Drift 분석

저자들은 retriever 간 전략 차이를 Retrieval Environment MMD(RE-MMD)로 측정한다. 두 가지 drift를 분리한다.

- Semantic drift: query가 가리키는 정보 의도가 달라지는 정도
- Structural drift: 단어 선택·문장 길이·문체가 달라지는 정도

결과는 semantic RE-MMD가 대체로 2.0보다 낮은 반면, structural RE-MMD는 특히 BM25와 dense retriever 사이에서 20 이상으로 크게 나타났다.

즉 retriever를 바꾸면 모델이 다른 정보를 찾으려는 것이 아니라, 같은 정보를 찾기 위해 다른 “표현 언어”를 써야 한다. 전략 전이 실패의 주원인이 의미가 아니라 스타일이라는 해석이다.

## 인간 지식(prompt)과 모델 크기

세 종류의 prompt를 비교했다.

- General: 일반적인 query rewriting 지시
- Exploratory: retriever의 학습 방식과 행동을 알려 주고 스스로 전략 탐색
- Direct expert: hypothetical document generation이나 keyword query처럼 사람이 정한 전략을 명시

Contriever에서는 retriever 특화 지시가 높은 성능을 보였고, BM25에서는 exploratory prompt도 keyword 스타일을 빠르게 발견했다. 14B 모델은 4B·8B가 빠진 local optimum을 벗어나 사람이 직접 지시하지 않은 전략도 발견했다.

## 사례

질문 `Is it possible to be white and latino?`에서 Contriever용 14B rewriter는 질문을 그대로 반복하는 대신 `Being white and Latino`처럼 문서형 표현으로 바꾸었다. nDCG@10은 0.965까지 올라갔다.

반대로 BM25에는 `Dwyane Wade current team`처럼 짧고 keyword 중심인 query가 적합했다. 이 차이는 동일한 LLM이라도 backend 환경을 알려 주거나 직접 feedback을 주어야 한다는 점을 보여준다.

## 최종 평가

이 논문의 핵심은 query rewriting을 “질문을 더 자연스럽게 고치는 작업”이 아니라 “특정 retrieval environment에 맞춰 query language를 번역하는 작업”으로 보는 것이다. 앞으로 RAG 시스템은 retriever 이름만 바꾸는 것이 아니라, retriever별 rewriting policy와 평가를 함께 관리해야 한다.

## 한계

- text-only retriever와 문서만 다뤄 이미지·오디오 multimodal retrieval은 제외했다.
- multi-turn 실험은 주로 1~2 turn이다.
- nDCG reward가 최종 생성 답변의 factuality까지 직접 보장하지는 않는다.
- API 환경·corpus 변화가 있으면 학습된 query dialect가 다시 drift할 수 있다.
