# 3번째 논문 요약

논문: [BM25 Query Augmentation Learned End-to-End](</Users/deliciouscat/projects/QueryExpansion/papers/parsed/2305.14087v1.md>)

저자: Xiaoyin Chen, Sam Wiseman

## 한 줄 요약

BM25의 빠른 inverted-index 검색은 유지하면서, 작은 BERT 계열 모델이 질문마다 sparse augmentation과 term re-weighting을 예측하도록 학습해 기존 BM25의 성능을 높인다.

## 문제의식

BM25는 단어가 정확히 일치해야 하는 sparse retrieval 방식이지만, 여전히 빠르고 강력한 기준선이다. 반면 dense retriever는 의미적 불일치를 완화하지만 문서 임베딩 저장과 벡터 검색 비용이 크다.

저자들은 문서 전체를 다시 인코딩하지 않고 검색어만 바꾸면 BM25의 장점을 유지하면서 lexical mismatch를 줄일 수 있다고 본다.

## 제안 기법

### BM25를 미분 가능한 형태로 재구성

BM25 점수를 다음과 같이 표현한다.

`query IDF × query term vector × document BM25 term-frequency vector`

이때 BERT-like encoder가 질문 `q`를 입력받아 두 벡터를 예측한다.

- `a(q)`: 질문에 추가할 sparse augmentation 벡터
- `w(q)`: 기존 단어와 추가 단어의 가중치를 조절하는 벡터

최종 점수는 다음 개념으로 볼 수 있다.

`(w(q) ⊙ IDF ⊙ (원래 bag-of-words + augmentation)) · document vector`

`a(q)`는 연속 벡터지만, 0이 아닌 토큰만 추출하면 실제 검색 시에는 확장된 질의 단어 집합이 된다. 또한 `w(q)`를 IDF에 흡수해 새로운 IDF 벡터를 만들 수 있으므로 Pyserini 같은 일반 BM25 구현으로 검색 가능하다.

### 학습

정답 문서와 hard negative·in-batch negative 문서를 사용하는 contrastive loss를 end-to-end로 최적화한다.

문서마다 별도의 신경망 임베딩을 만들지 않는다. 따라서 문서 인덱스를 새로 만들 필요가 없고, 추론 때는 질문에 대해서만 encoder를 실행한다.

### Sparse regularization

질의에 너무 많은 단어가 추가되면 inverted index 검색이 느려지고 후보 문서 수가 폭증한다. 이를 막기 위해 `a(q)`에 L1 정규화를 적용한다.

특히 자주 등장하는 단어에 더 강한 페널티를 준다. 빈번한 단어는 구별력이 낮으므로, 희귀하고 정보량 높은 단어를 남기는 것이 속도와 정확도에 모두 유리하다는 설계다.

## 데이터셋과 비교군

- Natural Questions(NQ)
- EntityQuestions
- MSMARCO passage ranking
- 전이 평가: NQ로 학습 후 TriviaQA와 EntityQuestions에 바로 적용

비교군:

- BM25(Pyserini 기본 토큰화)
- BM25(논문 모델과 동일한 WordPiece 토큰화)
- DPR
- GAR+BM25
- SEAL
- SPLADE

모델은 DistilBERT로 초기화했고, 문서와 질의는 WordPiece 토큰화했다.

## 주요 결과

| 데이터셋 / 방법 | 주 지표 | 보조 지표 | 질의 지연시간 |
|---|---:|---:|---:|
| NQ BM25(Pyserini) | Acc@5 0.436 | Acc@20 0.629 | 0.099초 |
| NQ BM25(ours tokenization) | 0.430 | 0.589 | 0.103초 |
| NQ Ours | **0.557** | **0.694** | 0.146초 |
| EntityQuestions BM25(Pyserini) | 0.616 | 0.720 | 0.060초 |
| EntityQuestions Ours | **0.693** | **0.798** | 0.669초 |
| MSMARCO BM25(Pyserini) | NDCG@10 0.228 | R@100 0.658 | 0.020초 |
| MSMARCO Ours | **0.251** | **0.687** | 0.030초 |

NQ에서 BM25 대비 Acc@5가 0.436에서 0.557로 12.1 percentage points 상승했다. 지연시간은 0.099초에서 0.146초로 약 43ms 증가했다.

EntityQuestions에서는 기존 BM25보다 Acc@5가 0.616에서 0.693으로, Acc@20이 0.720에서 0.798으로 개선됐다. MSMARCO에서는 NDCG@10이 0.228에서 0.251, Recall@100이 0.658에서 0.687로 상승했다.

NQ에서는 DPR·GAR·SEAL보다 절대 검색 성능은 낮지만, 이들 방식의 수십 분 또는 수십 초 수준 비용과 달리 질의당 0.146초로 훨씬 빠르다. MSMARCO에서는 SPLADE보다 낮은 성능이지만 SPLADE의 1.764초보다 훨씬 빠르다.

## 전이 결과

NQ에서 학습한 모델을 추가 fine-tuning 없이 사용했다.

| 데이터셋 | BM25(Pyserini) Acc@5/20 | Ours Acc@5/20 |
|---|---:|---:|
| TriviaQA | 0.677 / 0.773 | 0.662 / 0.755 |
| EntityQuestions | 0.616 / 0.720 | 0.542 / 0.656 |

논문은 OOD 전이에서 “BM25(ours tokenization)”보다 개선된다는 점을 강조하지만, Pyserini 기본 토큰화를 사용한 강한 기준선과 비교하면 성능이 낮아진다. 즉, 전이 가능성은 있지만 토큰화가 결과를 크게 좌우한다.

## Ablation 결과

NQ 기준:

| 설정 | Acc@5 | Acc@20 | 지연시간 | 평균 augmentation 길이 |
|---|---:|---:|---:|---:|
| Full | 0.557 | 0.694 | 0.146초 | 12.334 |
| Weighted L1 제거 | 0.562 | 0.704 | 0.268초 | 15.211 |
| `w(q)` 제거 | 0.545 | 0.683 | 0.269초 | 19.165 |
| BM25 scoring 제거 | 0.487 | 0.635 | 0.225초 | 31.861 |

정확도만 보면 일부 ablation이 약간 높지만, weighted L1은 augmentation 길이와 지연시간을 크게 줄인다. 단순 bag-of-words보다 BM25식 term frequency와 문서 길이 보정이 중요하며, `w(q)`도 선택적으로 단어를 제거해 효율을 높인다.

## 한계

- augmentation 단위가 pretrained tokenizer의 subword라서 자연어 단어 단위 확장이 어렵다.
- WordPiece 토큰화와 기본 BM25 토큰화의 차이가 결과를 크게 바꾼다.
- pretrained encoder가 가진 편향·오류를 검색어에 전파할 수 있다.
- 희귀 단어를 잘못 추가하면 검색 후보가 지나치게 좁아질 수 있다.
- dense retriever나 SPLADE보다 성능이 항상 높은 것은 아니다.

## 최종 평가

이 논문은 “신경 검색을 쓰려면 BM25를 버려야 한다”는 방향과 반대되는 실용적 설계다. 문서 인덱스를 바꾸지 않고 질문만 신경망으로 확장하기 때문에 기존 sparse 검색 인프라에 넣기 쉽다. 검색 품질, latency, 메모리의 균형을 중요하게 보는 first-stage retrieval에서 특히 의미가 있다.

