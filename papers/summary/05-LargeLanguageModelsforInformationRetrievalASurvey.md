# 5번째 논문 요약

논문: [Large Language Models for Information Retrieval: A Survey](</Users/deliciouscat/projects/QueryExpansion/papers/parsed/2308.07107v5.md>)

## 한 줄 요약

LLM이 정보검색 시스템의 query rewriter, retriever, reranker, reader, search agent에 어떻게 사용되는지를 체계적으로 정리하고, 각 모듈의 장점·한계·향후 연구 방향을 제시하는 조사 논문이다.

## 논문의 범위

전통적인 IR은 Boolean retrieval, vector-space model, language model, BM25를 거쳐 neural retrieval로 발전했다. LLM 시대에는 검색 시스템 전체를 한 번에 대체하기보다 각 모듈에 LLM을 배치하는 연구가 빠르게 늘었다.

논문은 다음 네 모듈과 search agent를 다룬다.

1. Query rewriter: 검색 의도를 더 정확한 검색어로 표현
2. Retriever: 질문과 문서를 매칭해 후보 문서 수집
3. Reranker: 후보 문서의 순서를 정밀하게 재배치
4. Reader: 검색 결과를 읽고 답·요약·근거 생성
5. Search agent: 여러 검색·추론·도구 사용을 계획

## 1. Query rewriter

### 역할

짧고 모호한 사용자 질문, 대화 맥락에 의존하는 질문, 여러 의도를 섞은 질문을 검색 엔진이 처리하기 좋은 형태로 바꾼다.

대표 작업은 다음과 같다.

- query expansion: 동의어·상위어·하위어 추가
- conversational rewriting: 이전 대화의 생략된 엔티티 복원
- query decomposition: 복잡한 질문을 여러 sub-query로 분해
- paraphrasing: retriever가 잘 이해하는 표현으로 재서술
- hypothetical document generation: 정답 문서처럼 보이는 가상 문서 생성

### 학습 방식

- Zero-shot/few-shot prompting: 유연하고 데이터가 필요 없지만 결과 변동성이 크다.
- Supervised fine-tuning: 도메인에 맞는 재작성기를 만들 수 있지만 양질의 query-document 또는 rewrite 라벨이 필요하다.
- Reinforcement learning: 최종 retrieval·QA 결과를 보상으로 사용해 문자열 유사도보다 downstream 성능을 직접 최적화한다.

### 대표 접근

Query2Doc처럼 LLM에게 질문에 대한 가상 문서를 생성하게 하고, 그 텍스트를 검색에 활용하는 방식은 짧은 질문과 긴 문서 사이의 표현 차이를 줄인다. 반면 생성 내용이 틀리거나 과도하게 구체적이면 concept drift가 발생해 원래 의도에서 벗어날 수 있다.

### 주요 한계

- 원래 질문에 없는 정보를 추가하는 hallucination/concept drift
- 강한 LLM은 불필요한 expansion이 오히려 성능을 낮출 수 있음
- 긴 문서·대화 맥락을 처리하는 비용
- 사람이 보기에는 자연스러워도 실제 retriever에는 유용하지 않을 수 있음

## 2. Retriever

### LLM이 만드는 검색 데이터

검색 데이터가 부족한 도메인에서는 LLM이 문서에서 pseudo query를 생성하거나, query-positive-negative triplet과 relevance label을 만들 수 있다.

세 가지 생성 방식이 정리된다.

- pseudo query generation: 문서에 대응하는 질문 생성
- relevance label generation: 질문-문서 관련도 또는 soft label 생성
- complete example generation: query, positive document, negative document를 함께 생성

이런 synthetic data는 도메인 확장과 zero-shot 전이에 유용하지만, 생성된 오류가 retriever에 누적될 수 있다.

### LLM 자체를 retriever로 사용하는 경우

LLM의 encoder/decoder 표현으로 dense embedding을 만들거나, 생성 확률을 문서 관련도 신호로 사용한다. 일반적으로 dense retriever는 의미적 유사성을 잘 포착하지만, exact entity, 수치, 희귀 용어, 최신 정보에는 sparse retrieval이 더 강할 수 있다.

### 핵심 trade-off

- Sparse/BM25: 빠르고 해석 가능하지만 단어 불일치에 약함
- Dense: 의미적 불일치에 강하지만 인덱스와 embedding 비용이 큼
- LLM 기반: 복잡한 의미·추론을 이해하지만 계산량과 latency가 큼

따라서 실제 시스템에서는 BM25와 dense retriever의 hybrid, 또는 sparse first-stage + neural reranker 구성이 자주 사용된다.

## 3. Reranker

retriever가 수십~수천 개 후보를 빠르게 모으면 reranker가 query-document 쌍을 더 정밀하게 비교한다.

LLM reranker는 다음 전략으로 분류된다.

- Pointwise: 문서 하나의 relevance score 또는 yes/no 예측
- Pairwise: 두 문서 중 어느 것이 더 관련 있는지 비교
- Listwise: 전체 후보 목록의 순서를 직접 생성·평가

LLM reranker의 장점은 복잡한 조건, 다중 제약, 문서 간 비교를 다룰 수 있다는 점이다. 단점은 후보마다 LLM을 호출해야 하므로 latency가 증가하고, 점수의 일관성과 calibration이 부족할 수 있다는 점이다.

## 4. Reader와 RAG

reader는 검색 문서를 읽어 답변, 요약, 근거를 생성한다. LLM은 긴 문서의 여러 단서를 종합하고 자연어 답변을 생성하는 데 강하지만, 검색 문서가 부정확하거나 너무 많으면 hallucination과 context distraction이 생긴다.

대표 설계는 다음과 같다.

- retrieve-then-read: 검색 후 한 번 읽기
- iterative retrieval: 추론 중 필요한 정보를 다시 검색
- self-reflection/verification: 답변의 근거를 다시 확인
- black-box LLM + trainable adapter: reader는 고정하고 앞단 rewriter/retriever만 학습

## 5. Search agent

LLM이 검색어를 만들고 검색 결과를 읽은 뒤 다음 행동을 계획하는 agent 방식이다. ReAct, IRCoT, Toolformer, DSP 등은 reasoning과 tool call을 결합한다.

이 방식은 multi-hop 질문과 실시간 정보에 강하지만 다음 비용이 있다.

- 한 질문당 여러 LLM·검색 호출
- 긴 trajectory의 credit assignment 문제
- 잘못된 초기 검색이 후속 추론을 오염
- 검색을 지나치게 하거나 필요한데도 생략하는 reward hacking

강화학습은 최종 답 정확도, 검색 hit, 정보 이득, 중복 검색 패널티를 보상으로 넣어 search policy를 학습하는 방향으로 활용된다.

## 논문이 강조하는 공통 한계

### 데이터와 평가

사람이 만든 relevance label은 비싸고 도메인마다 다르다. LLM synthetic data는 규모를 늘리지만 factuality와 bias를 보장하기 어렵다. 또한 retrieval metric 개선이 최종 QA·생성 품질 개선으로 항상 이어지는 것도 아니다.

### 효율성

LLM을 query rewriting, reranking, reading에 모두 사용하면 검색 품질은 오를 수 있으나 서비스 latency와 비용이 급격히 증가한다. 따라서 빠른 BM25/dense first-stage와 제한된 neural reranking의 조합이 현실적인 절충이다.

### 신뢰성과 해석 가능성

LLM은 그럴듯하지만 틀린 query와 답변을 생성할 수 있다. 검색 근거, citation, uncertainty estimation, provenance 추적이 필요하다.

## 향후 연구 방향

- 검색 모듈별 비용을 고려한 end-to-end 최적화
- query rewriter와 retriever·reader의 공동 학습
- multi-turn search에서 안정적인 RL과 credit assignment
- multilingual·도메인 특화·long-tail retrieval
- 개인화·최신성·안전성을 반영한 검색
- text뿐 아니라 표·이미지·오디오를 다루는 multimodal IR
- 검색 결과를 근거로 한 faithful answer와 자동 검증

## 최종 평가

이 논문은 특정 모델 하나의 성능을 보고하는 논문이라기보다, LLM이 IR pipeline의 어느 위치에서 어떤 방식으로 기여하는지를 지도처럼 정리한다. 핵심 메시지는 LLM의 의미 이해 능력과 BM25의 속도·안정성을 대립시키기보다, query reformulation과 reranking에는 LLM을 선택적으로 사용하고 first-stage retrieval과 인덱싱은 효율적인 전통 방식으로 유지하는 hybrid 설계가 중요하다는 것이다.

