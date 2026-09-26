# Query Expansion

사용자 질문을 BM25 검색에 유용한 짧은 검색어로 확장하는 **단일 태스크 모델**을 만든다. 우선 실행 환경은 **Runpod의 RTX A5000 단일 GPU**이며, 온프레미스 RTX 3090은 전처리·개발·재현 실험에 활용한다.


```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
qe-train --config configs/smoke_sft.yaml
```

Qwen 연동에는 `.[hf,dev]`가 필요하다. smoke 설정은 다운로드 없는 tiny 모델과 합성 데이터로 실행한다.

## 1. 태스크와 입출력 계약

모델은 질문과 언어 코드만 받아 확장어를 생성한다. 답변 생성, 문서 요약, reranking, 다중 검색 에이전트는 학습 목표에 포함하지 않는다. Question + Answer 데이터의 답변과 정답 문서는 학습 타깃 구축 및 보상 계산에만 사용하며 정책 모델의 입력에 넣지 않는다.

```text
입력
language: ko
query: 전기차 겨울 주행거리가 줄어드는 이유

모델 출력 예시
저온 배터리 성능 난방 에너지 소비 리튬이온 내부 저항

실제 BM25 질의
전기차 겨울 주행거리가 줄어드는 이유 저온 배터리 성능 난방 에너지 소비 리튬이온 내부 저항
```

- 출력은 확장어 한 줄이다. 설명·답변·추론 과정·JSON을 생성하지 않는다.
- 원문 질문은 애플리케이션이 보존하고 뒤에 확장어를 붙인다. 모델이 원문을 재생성하도록 하지 않는다.
- 기본은 입력 언어와 동일한 언어와 영어이다. 고유명사·통용 약어는 허용하며 번역 확장은 별도 실험으로 둔다.
- 최대 생성 길이는 64 model tokens다. 중복 확장어와 원문에 이미 있는 동일 표현은 정규화 후 제거한다. 같은 후처리를 학습 보상·평가·추론에 적용한다.
- 빈 출력·형식 오류·생성 한도까지 EOS 미발생 시 원문 질의로 fallback하고 비율을 기록한다.

전체 흐름은 `질문 → 확장어 생성 → 원문과 결합 → BM25 검색`이다. 학습 시에만 검색 결과와 qrels를 비교해 모델을 업데이트한다.

## 2. Runpod 우선 실행 시나리오

### 시작 구성

| 항목 | Runpod 주 환경 | 온프레미스 보조 환경 |
|---|---|---|
| GPU | RTX A5000 × 1, 24 GB급 VRAM | RTX 3090 × 1, 24 GB급 VRAM |
| 역할 | SFT, 검색 보상 기반 RL, 최종 평가 | 데이터 검사, 인덱스 제작, smoke test, 재현 |
| CPU / RAM | 초기 요청값 8 vCPU 이상 / 64 GB RAM | 동일 데이터·인덱스를 처리할 수 있는 용량 |
| 저장소 | Network Volume 200 GB부터 시작 | 로컬 SSD |
| 실행 방식 | 장시간 학습용 GPU Pod | 동일하게 고정한 실행 환경 |

CPU·RAM·디스크는 프로젝트의 시작 예산이다. 전체 다국어 corpus와 여러 인덱스를 수용한다는 보장은 없으므로 먼저 표본 인덱스 크기를 측정해 조정한다. 학습용 Pod를 먼저 완성하고 Serverless 추론 배포는 후속 작업으로 둔다.

### 저장소와 Pod 생명주기

Network Volume을 Pod 생성 시 연결하고 `/workspace`를 영속 경로로 사용한다. Network Volume은 Pod 삭제와 독립적으로 유지된다. 일반 Volume disk는 Pod 종료(삭제) 시 사라지고 Container disk는 중지 시 사라지므로 체크포인트를 컨테이너 임시 경로에만 저장하지 않는다. 세부 동작은 [Runpod Storage options](https://docs.runpod.io/pods/storage/types)를 따른다.

GPU 재고와 Network Volume을 연결할 수 있는 데이터센터 조합을 먼저 확인한다. 연결 조건은 [Runpod Network volumes](https://docs.runpod.io/storage/network-volumes)를 참조한다. 가격은 실행 시 콘솔에서 확인하고 GPU 시간·저장 용량·전송 비용을 실험별로 기록한다.

```text
/workspace/query-expansion/
├── repo/                  # 소스 및 고정 설정
├── cache/huggingface/     # 기반 모델·데이터 캐시
├── data/{raw,processed}/  # 원본과 정규화 데이터
├── indexes/               # corpus/analyzer별 BM25 인덱스
├── runs/<run_id>/          # manifest, 로그, 평가, 체크포인트
└── exports/<run_id>/       # 배포용 모델 및 tokenizer
```

다음은 Pod에서 디렉터리를 준비하는 명령이다. 학습 프로그램을 설치하거나 실행하는 명령은 아니다.

```bash
export QE_ROOT=/workspace/query-expansion
export HF_HOME="$QE_ROOT/cache/huggingface"
mkdir -p "$QE_ROOT"/{repo,cache/huggingface,data/raw,data/processed,indexes,runs,exports}
nvidia-smi
df -h /workspace
```

### 실행 순서와 통과 조건

1. **환경 고정:** CUDA/PyTorch 포함 컨테이너를 선택하고 image digest, Python, PyTorch, Transformers, PEFT, 검색 엔진, tokenizer 버전을 manifest에 저장한다. Qwen3.5 모델의 load·forward·generate·backward를 확인한 조합으로 lock 파일을 만든다. 버전 미고정 `latest`를 재현 기준으로 삼지 않는다.
2. **소규모 데이터 준비:** 영어 MS MARCO와 한국어 후보 중 문서 연결 검증을 통과한 자료부터 최대 1만 train queries를 준비한다. 공식 평가 split은 보존하고, 없으면 원문 그룹 단위로 분리한다. 연결 문서가 없는 QA는 RL에서 제외한다.
3. **BM25 기준선:** 고정 dev corpus와 기본 analyzer로 원문 질의 성능을 측정한다. 같은 query/corpus/qrels를 이후 실험에도 사용한다.
4. **Smoke test:** 100개 질문으로 SFT 10 optimizer steps 및 RL 2 optimizer steps를 실행한다. 두 optimizer 갱신, frozen parameter 불변, 유한한 loss/reward, 저장·재개를 확인한다.
5. **SFT → RL:** SFT로 출력 규격을 먼저 학습하고 통과한 체크포인트로 RL을 시작한다. 동일 GPU에서 rollout과 gradient update를 순차 실행한다.
6. **평가·내보내기:** dev에서 선택한 모델을 전체 test corpus에 평가한다. 학습된 상위 레이어를 포함한 모델을 내보내고 재로드 결과를 비교한다.
7. **회수:** 체크포인트·manifest·평가표를 영속 경로와 별도 백업에 저장하고 읽기 검증 후 Pod를 종료한다. Network Volume 유지 비용도 관리한다.

처리량 측정 전에는 총 학습시간을 단정하지 않는다. warmup 이후 100 steps의 평균 시간으로 `남은 steps × 초/step`을 추정하고 평가·저장 시간을 더한다. 최초 실행은 1시간 예산으로 환경 검증과 smoke test를 마치는 것을 목표로 한다. 시간 상한에 도달하면 optimizer-step 경계에서 저장 후 정상 종료하도록 구현한다.

## 3. 기반 모델과 파라미터 업데이트

기반 모델은 `Qwen/Qwen3.5-2B-Base`다. 텍스트 decoder 24개 레이어는 linear attention과 full attention이 혼합되어 있고 입력 embedding과 LM head는 가중치를 공유한다. 모델은 vision 구성도 포함하지만 이 프로젝트는 텍스트 경로만 사용한다. 출처: [모델 카드](https://huggingface.co/Qwen/Qwen3.5-2B-Base), [config.json](https://huggingface.co/Qwen/Qwen3.5-2B-Base/blob/main/config.json).

| 영역 | 학습 방식 | Optimizer |
|---|---|---|
| Token embedding / 공유 LM head | Freeze | 없음 |
| 텍스트 decoder 0–11 | 원래 가중치 Freeze + LoRA | AdamW |
| 텍스트 decoder 12–23 | 레이어 내부 전체 파라미터 fine-tuning | SGD |
| decoder 바깥 최종 norm | Freeze, 초기 실험 정책 | 없음 |
| vision 및 사용하지 않는 부가 경로 | Freeze, forward에서 제외 | 없음 |

LoRA는 하위 12개 레이어의 MLP projection을 공통 시작점으로 삼는다. attention projection 추가는 실제 `named_modules()`에서 linear/full attention 각각의 모듈명·타입을 확인한 뒤 명시적으로 구성한다. 모든 레이어가 `q_proj/v_proj`를 제공한다고 가정하거나 전체 모델에 무차별적으로 adapter를 붙이지 않는다. 텍스트 모델 로딩은 고정 버전의 [Transformers Qwen3.5 문서](https://huggingface.co/docs/transformers/model_doc/qwen3_5)를 기준으로 검증한다.

구현 시 다음 조건을 assert한다.

- 텍스트 decoder가 24개인지 확인하고 vision layer index와 구분한다.
- LoRA 삽입 후 trainable flag를 다시 설정한다. PEFT 적용 과정에서 상위 12개 레이어가 의도치 않게 동결되지 않았는지 확인한다.
- 두 optimizer의 parameter ID 집합은 서로 겹치지 않으며 합집합이 전체 trainable parameter와 같다. shared weight는 ID 기준으로 중복 제거한다.
- 하나의 loss로 backward하고 gradient accumulation 완료 시 전체 trainable gradient를 clip한 뒤 두 optimizer를 각각 한 번 step한다. scheduler와 zero_grad도 같은 경계에 맞춘다.
- 하위 레이어의 원래 가중치는 동결해도 forward 전체를 `no_grad()`로 감싸지 않는다. LoRA까지의 gradient 흐름을 유지한다.
- **LoRA adapter만 저장하면 학습 결과가 불완전하다.** 상위 레이어 변경분과 두 optimizer 상태까지 저장해야 한다.

## 4. 데이터 규격과 구성

### 공통 스키마

query, corpus, qrels를 분리한다. 아래는 형식 예시이며 실제 데이터가 아니다.

```json
{
  "query_id": "example:ko:q001",
  "source": "example",
  "language": "ko",
  "split": "train",
  "query": "전기차 겨울 주행거리가 줄어드는 이유",
  "answers": ["저온에 따른 배터리 성능 저하와 난방 소비"],
  "positive_doc_ids": ["example:doc001"],
  "negative_doc_ids": [],
  "expansion_target": null
}
```

- `corpus`: `doc_id`, `language`, `title`, `text`. ID에는 source namespace를 붙인다.
- `qrels`: `query_id`, `doc_id`, `relevance`. 복수 정답과 graded relevance를 보존한다.
- `answers`: SFT 타깃 생성·검수 보조 정보다. 문서 relevance label의 대체물로 쓰지 않는다.
- `expansion_target`: 검증된 확장어 문자열. 없으면 SFT에서 제외하며 qrels가 있으면 RL에는 사용할 수 있다.
- 데이터 manifest에 원본 revision, config/subset, split, 라이선스, adapter 버전, 제외 건수·사유를 기록한다.

### 데이터셋 후보와 도입 순서

기존 후보 목록을 유지한 도입 계획이다. 각 저장소의 존재·접근 권한·라이선스·현재 schema와 corpus 연결 여부는 adapter 구현 전에 확인한다. retrieval/reranking 데이터 사이 query 중복도 검사한다.

| 후보 | 계획된 용도 / 채택 조건 |
|---|---|
| `microsoft/ms_marco` | 영어 MVP. passage와 relevance 연결을 검증한 subset 사용 |
| `BeIR/scifact` | 우선 도메인 전이 평가용으로 보류; 평가 query를 학습에 사용하지 않음 |
| `google-research-datasets/natural_questions` | QA evidence를 안정적인 document ID로 매핑한 뒤 사용 |
| `taeminlee/Ko-StrategyQA` | 한국어 QA 후보. 정답 단독으로 가짜 positive 문서를 만들지 않음 |
| `yjoonjang/markers_bm` | 한국어 후보. query/document/label 구조 확인 후 역할 결정 |
| `mteb/MIRACLRetrieval` | 다국어 retrieval 후보. 언어별 corpus와 qrels 연결 확인 |
| `mteb/MIRACLRetrievalHardNegatives` | MIRACL에 ID로 연결 가능한 hard negatives만 사용 |
| `mteb/MIRACLReranking` | reranking 후보군 활용. 전체 corpus 평가를 대체하지 않음 |
| `xhluca/publichealth-qa` | QA 보조 후보. evidence 문서 확보 시 retrieval 학습에 편입 |
| `facebook/belebele` | 문맥 기반 QA 보조 후보. 선택지 자체를 검색 corpus로 쓰지 않음 |
| `mteb/mrtidy` | 다국어 후보. 언어·split·qrels 확인 후 편입 |
| `mteb/MultiLongDocRetrieval` | 장문 검색 후속 실험. chunk 및 원문 문서 매핑 고정 |
| `mteb/MultiLongDocReranking` | 장문 negative 후보. retrieval 데이터와 중복 제거 |
| `mteb/XPQARetrieval` | 후속 다국어 retrieval 후보. 사용 언어 및 라벨 검사 후 편입 |

### 언어 비율과 누수 방지

최종 학습의 **query sampling 비율** 목표다. corpus 문서 수나 토큰 수 비율이 아니다.

| 언어 | 한국어 | 영어 | 중국어 | 일본어 | 스페인어 | 프랑스어 | 독일어 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 비율 | 25% | 45% | 12% | 8% | 6% | 2% | 2% |

MVP의 한·영 실행에는 아직 이 비율을 강제하지 않는다. 전체 단계에서는 언어 → source → query 순서의 sampler를 구현하고 목표/실현 비율, 고유 query 수, 반복 노출 횟수를 기록한다. 부족한 언어를 무제한 복제하지 않고 목표 미달을 보고한다.

공식 split을 우선한다. 새 split이 필요하면 원문 문서·중복 질문·번역본을 같은 그룹으로 묶어 90/5/5로 분리한다. 공용 retrieval corpus에 평가 문서가 존재하는 것과 평가 query/qrels를 학습에 사용하는 것은 구분한다. test query/qrels는 타깃 생성·negative mining·튜닝에 사용하지 않는다. 장문 chunk는 원문 문서 ID를 보존하고 문서 단위 평가에서 중복 집계하지 않는다.

## 5. 학습 전략

### 단계 A: SFT로 확장어 출력 형식 학습

Base 모델에 바로 검색 보상만 주기 전에 짧은 SFT를 수행한다. Question + Answer + evidence에서 후보 핵심어를 추출하거나 오프라인 teacher로 후보를 생성하고, 원문 대비 train retrieval 성능이 나빠지지 않는 후보를 선별한다. teacher 종류·prompt·revision·생성 비용을 기록한다. 정답 문자열 복사에 편중되지 않도록 표본 검수하고 질문만으로 예측할 수 없는 문서 고유 정보를 강제로 타깃에 넣지 않는다.

정책 입력은 계속 `language + query`다. loss는 확장어와 EOS에만 적용하고 prompt/padding은 mask한다. 기존 tokenizer 토큰을 사용하고 embedding 동결 상태에서 새 special token을 추가하지 않는다. 동일 prompt template을 SFT·RL·추론에 사용한다.

### 단계 B: BM25 검색 보상으로 RL

기존의 BM25 reward 방향을 유지하되 **초기 기본 보상은 BM25 결과의 순위 개선량**으로 구체화한다. raw BM25 점수는 analyzer·corpus에 따라 스케일이 달라 직접 합산하기 어렵다. raw score 보상은 아래 비교 실험으로 남긴다.

질문별 후보 문서 집합 `C(q)`에 알려진 positive를 모두 포함한다. 나머지는 초기 최대 255개로, 원문 BM25의 hard negatives 약 50%와 같은 언어·corpus의 random negatives 약 50%를 섞는다. 알려진 positive·중복 문서를 제외하고 부족하면 가능한 수만 사용해 실제 개수를 기록한다. 미판정 문서는 negative로 가정되므로 false negative를 표본 점검한다.

BM25의 IDF와 평균 문서 길이는 환경별 **고정 corpus 인덱스**에서 얻는다. query별 후보 몇백 개로 인덱스를 다시 만들지 않는다. 후보 내 순위는 학습 비용을 낮추기 위한 근사이며 최종 평가는 전체 corpus로 한다.

환경 `e`, 확장어 `a`, 후처리한 결합 질의 `q′`에 대해:

```text
gain_e = nDCG@10(BM25_e(q′), C(q)) - nDCG@10(BM25_e(q), C(q))
R(q, a) = mean_e(gain_e)
          - 0.02 × min(생성 토큰 수 / 64, 1)
          - 0.10 × invalid_output
```

계수는 초기 제안이다. 유효한 빈 확장은 gain 0의 무확장으로 처리한다. 형식 오류·EOS 미발생은 `invalid_output=1`이며 검색은 원문으로 fallback한다. 길이는 후처리 전 생성 길이로 계산해 반복 출력에도 비용을 부과한다. 동률은 `doc_id` 순으로 정렬하고 양의 qrel이 없는 query는 제외한다. 최종 평가는 페널티를 뺀 실제 retrieval metric으로 보고한다.

raw BM25 비교 실험에서는 환경별 `가장 높은 positive 점수 - 가장 높은 negative 점수`의 원문 대비 개선량을 계산한다. train 데이터에서 고정한 scale로 정규화·clip한 뒤 평균하고 순위 보상과 비교한다. positive 점수만 높이는 방식은 모든 문서 점수를 함께 높여도 좋아 보일 수 있으므로 기준 실험으로만 둔다.

### BM25 환경 다양화

`vocab`은 LLM tokenizer가 아니라 검색 analyzer와 corpus가 만드는 어휘를 의미한다. 처음에는 기본 환경 하나로 검증하고 이후 다음 축으로 2–4개 고정 인덱스를 만든다.

| 축 | 예시 | 고정해야 할 항목 |
|---|---|---|
| 전처리 | stopword 사용/미사용, 정규화 | 언어별 규칙·버전 |
| 토큰화 | 언어별 word/형태소, CJK character n-gram | analyzer, n 범위, query/document 동일 처리 |
| corpus | 도메인·문서 subset | corpus hash, positive 포함 여부 |
| BM25 파라미터 | k1, b | 초기 기준값 k1=1.2, b=0.75 |

동일 질문의 rollout 후보들은 동일한 환경·후보 문서 집합에서 비교한다. corpus subset에도 positive가 있어야 하며 통계는 인덱스 생성 시 고정한다. 환경은 step 사이에서 sampling하고 ID·seed를 기록한다. 학습에 사용하지 않은 analyzer 조합은 일반화 평가용으로 보류한다.

### RL 업데이트와 24 GB 메모리 예산

초기 알고리즘은 critic 없는 **group-relative policy gradient**로 한다. 질문당 `G=4`개 확장을 순차 생성하고 `A_i=(R_i-mean(R))/(std(R)+1e-6)`으로 advantage를 계산한다. reward가 모두 같으면 해당 그룹의 policy update를 건너뛰고 그 비율을 기록한다.

후보를 `no_grad`로 생성한 뒤 token ID를 보관하고 동일 정책의 log-probability를 후보별로 재계산해 gradient를 누적한다. 한 rollout batch당 한 번만 업데이트하는 on-policy REINFORCE 목적 `-mean_i[A_i × sum_t log π(a_it|q,a_i<t)]`를 사용한다. 생성과 재계산 정책을 일치시키기 위해 초기 RL은 temperature 1.0, top-p 1.0, dropout 비활성으로 고정한다. prompt/padding을 제외하고 생성 토큰과 EOS만 목적함수에 포함한다. PPO/GRPO의 clipped multi-epoch update는 후속 구현으로 둔다.

초기에는 별도 GPU reference 모델과 KL 항을 두지 않는다. SFT dev 성능 저하·출력 오류 증가를 early-stop 조건으로 감시한다. KL을 추가한다면 상위 레이어도 변경되므로 **adapter 비활성화만으로 reference 정책이 복원되지 않는다**. 별도 frozen SFT snapshot이 필요하다.

| 설정 | 초기값 |
|---|---|
| 정밀도 | BF16, 비양자화 학습부터 검증 |
| 입력 / 출력 길이 | prompt 포함 256 / 최대 64 tokens |
| SFT micro batch / accumulation | 1 / 16 |
| RL 질문 수 / rollout 수 / accumulation | micro batch당 1 / 질문당 4 / 4 groups |
| LoRA | r=16, alpha=32, dropout=0 |
| SFT learning rate | LoRA AdamW 1e-4 / 상위 레이어 SGD 1e-5 |
| RL learning rate | LoRA AdamW 1e-5 / 상위 레이어 SGD 1e-6 |
| Optimizer 옵션 | AdamW weight decay=0.01, SGD momentum=0, weight decay=0 |
| Scheduler / gradient clip | 각 단계 warmup 3% 후 linear decay / global norm 1.0 |
| 메모리 절감 | gradient checkpointing, 학습 forward의 use_cache=False |
| 평가 / 저장 | 100 optimizer steps마다 dev, 100 steps 또는 10분마다 저장 |

이는 튜닝 출발점이다. 전체 vocab logits, activation, gradient, optimizer state, rollout cache를 포함한 peak VRAM을 실측한다. rollout cache는 재계산 전에 해제하고 BM25는 CPU에서 실행한다. OOM이면 길이·동시 후보 수·micro batch부터 줄이고 activation 설정을 조정한다. 상위 레이어 full tuning을 묵시적으로 LoRA/QLoRA로 바꾸지 않으며 필요하면 별도 실험으로 기록한다.

## 6. 평가와 채택 기준

학습의 샘플 후보 평가와 **전체 corpus 평가**를 별도 표로 보고한다. full-corpus 검색 결과에는 positive를 강제로 삽입하지 않는다.

| 비교군 | 확인할 질문 |
|---|---|
| 원문 BM25 | 확장이 검색 품질을 개선하는가? |
| Base 모델의 고정 prompt 확장 | 학습 자체가 필요한가? |
| SFT만 적용 | 검색 보상 RL의 추가 효과가 있는가? |
| SFT + RL, 단일 BM25 환경 | 기본 정책 학습이 작동하는가? |
| SFT + RL, 복수 BM25 환경 | 보지 못한 analyzer에서도 개선되는가? |

주 지표는 nDCG@10, 보조 지표는 MRR@10·Recall@100이다. 언어별·데이터셋별 수치, 데이터셋 동일 가중 macro 평균, 목표 언어 비율 가중 평균을 함께 보고한다. 생성 길이, fallback 비율, 질문별 성능 악화 비율, 생성/검색 각각의 p50·p95 latency, peak VRAM, queries/sec와 GPU 비용도 기록한다.

첫 채택 기준은 dev 전체 corpus macro nDCG@10이 원문 BM25보다 상대 3% 이상 개선되고 각 언어의 nDCG@10 절대 하락이 0.01 이내이며 fallback이 1% 미만인 것이다. 이는 **프로젝트의 초기 목표이며 검증된 성능이 아니다**. RL 모델은 SFT 기준선도 넘겨야 채택한다. 같은 query에 대한 paired bootstrap 신뢰구간과 가능하면 3개 seed 결과로 안정성을 확인한다. 지연시간 기준은 기준선 실측 후 사용 환경에 맞춰 확정한다.

실험 ID에는 데이터 버전·학습 방식·BM25 환경·seed를 연결한다. test는 모델 선택에 반복 사용하지 않고 dev에서 선택한 최종 모델 검증에만 쓴다.

## 7. 체크포인트, 재개, 배포 산출물

`runs/<run_id>/manifest.json`에는 소스 revision 또는 소스 archive hash, 모델 revision, prompt/설정 hash, 환경 lock, corpus/index hash, 데이터 split, seed, 비용 정보를 저장한다.

재개용 체크포인트에는 모델 변경분 전체 또는 전체 state, LoRA 설정, **AdamW와 SGD 양쪽 state**, 두 scheduler, global optimizer step, RNG(Python/NumPy/CPU/CUDA), sampler 위치, 환경 sampling 상태를 포함한다. optimizer-step 경계에서 임시 경로에 쓴 뒤 완료 marker를 만들고 완료된 체크포인트만 재개한다. 최근 2개와 dev best를 보관하며 다음 저장에 필요한 여유 공간도 검사한다.

Pod를 다시 만들 때 동일 Network Volume·이미지·lock 파일을 사용한다. manifest의 model/data/index hash가 다르면 자동 재개하지 않는다. optimizer 없이 가중치만 읽는 실행은 resume가 아니라 새 fine-tuning run으로 기록한다. smoke test에는 중단 전후 다음 batch·step·loss의 허용 오차 비교를 포함한다.

배포 산출물은 상위 레이어 변경과 하위 LoRA를 반영한 전체 모델, tokenizer, config, prompt, 후처리 규칙, 평가표다. LoRA 병합 지원 여부를 확인하고 병합 전후 logits 및 고정 질문 출력을 비교한다. CPU 또는 한 GPU에 새로 로드해 검색 결과까지 확인한 뒤 export 완료로 표시한다.

## 8. 재구축 상태와 다음 실험

저장소는 프로젝트를 재구축하는 도중이며, 현재 파일 일부만 남아 있다. 아래의 실행 예시와 설계는 완성되거나 현재 동작한다고 주장하는 구현 목록이 아니라 재구축 목표다. 실제 파일과 일치하는지는 구현이 복원된 뒤 확인한다.

### 디렉터리별 담당 구조

모델 훈련 설계에 참여하는 각 전문가는 **자신이 담당하는 모듈 디렉터리 하나만 수정해도 맡은 기능을 구현하고 개선할 수 있도록** 모듈 경계와 인터페이스를 설계한다. 다른 모듈과의 연결은 공개 API와 데이터 형식으로 맞추며, 이 경계가 바뀌면 통합 담당자가 조율한다. 별도 공용 계약 디렉터리를 두지 않고 각 모듈의 공개 API 정의를 해당 모듈 안에 둔다.

```text
train/                              # 실험 통합 담당자: 조립용 entrypoint
  <훈련id>.py
src/query_expansion/
  data_loader/                      # 데이터 전문가
  agents/                           # 모델·정책 전문가
  rewards/                          # 목적함수·보상 전문가
  retriever/                        # 검색 전문가
  utils/                            # 학습 기반·체크포인트 전문가
  evaluation/                       # 평가 전문가
```

| 디렉터리 | 단일 담당 범위 | 공개 책임 |
|---|---|---|
| `src/query_expansion/data_loader/` | 데이터 전문가 | `DatasetStream`, 데이터 adapter, 전략별 전처리, batch 형식 |
| `src/query_expansion/agents/` | 모델·정책 전문가 | `Agents`, 모델 구성·생성·학습 파라미터 선택 |
| `src/query_expansion/rewards/` | 목적함수·보상 전문가 | `Rewards`, 목적함수 registry와 보상 계산 |
| `src/query_expansion/retriever/` | 검색 전문가 | retriever 인터페이스, BM25 인덱스·검색 결과 |
| `src/query_expansion/utils/` | 학습 기반 전문가 | `utils.backward.gradient`, optimizer 갱신·저장·재개 |
| `src/query_expansion/evaluation/` | 평가 전문가 | retrieval metric, 집계와 평가 출력 |
| `train/` | 실험 통합 담당자 | 모듈 초기화·연결, 실험별 학습 절차와 설정 |

학습 entrypoint 구조는 [`CODE_INTERFACE.md`](CODE_INTERFACE.md)를 따른다. `train/<훈련id>.py`는 `DatasetStream`, `Agents`, `Rewards` 등 필요한 구성 요소를 가져와 연결하고, 학습 루프와 forward/loss 흐름을 읽을 수 있게 둔다. 학습 알고리즘의 조립 책임은 entrypoint에 남기며, 공통 backward/update 처리는 `utils.backward.gradient` decorator가 맡는다. gradient accumulation과 LoRA·상위 레이어용 복수 optimizer도 이 기반 모듈이 처리한다.

의존 관계는 `train → 각 모듈`, `rewards → retriever의 공개 인터페이스`로 제한한다. 보상 모듈은 검색 구현 세부사항에 의존하지 않고 entrypoint가 주입한 retriever 인터페이스를 사용한다. 데이터·모델·검색 모듈은 서로의 내부 파일을 직접 수정하거나 import하지 않는다. 각 전문가는 자기 모듈 디렉터리 안에서 구현과 그 모듈의 공개 API 정의를 함께 관리하고, 다른 모듈 변경이 필요하면 통합 담당자에게 인터페이스 변경을 제안한다.

이 구조는 현재 동작하는 구현 목록이 아니라 재구축을 위한 scaffold다. 각 디렉터리의 `README.md`는 담당 범위와 경계만 설명하며, 실제 구현 API와 파일은 해당 모듈 작업에서 추가한다.

첫 GPU milestone은 **A5000 한 장에서 한·영 데이터로 SFT → RL → 전체 corpus 평가 → 저장·재개를 끝까지 통과**하는 것이다.

## 9. 참고 자료

- [BM25 Query Augmentation Learned End-to-End 요약](papers/summary/03-BM25QueryAugmentationLearnedEndtoEnd.md): sparse 확장 길이·검색 비용·토큰화 선택의 영향.
- [Understanding the Behaviors of Environment-aware Information Retrieval 요약](papers/summary/08-UnderstandingtheBehaviorsofEnvironmentawareInformationRetrieval.md): retriever 환경에 따른 확장 전략과 순위 기반 보상.
- [논문 원문 텍스트](papers/parsed/), [논문 PDF](papers/source/).

모델 구조·Runpod 저장소 관련 공식 문서는 2026-09-22에 확인했다. 비용·GPU 재고·라이브러리 호환성은 실제 실행 시 다시 확인한다.
