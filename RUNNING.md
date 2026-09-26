# 구현 실행 및 검증 안내

기존 설계 문서는 변경하지 않았다. 이 파일은 추가된 코드의 사용법과 검증 범위를 설명한다.

## 처음 읽을 코드

1. `train/sft.py`: 구성 요소 선택 → 명시적 반복문 → forward → completion-only loss.
2. `train/rl.py`: query별 후보 생성 → 고정 검색 환경에서 보상 → group advantage → 후보별 log-probability 재계산.
3. `src/query_expansion/utils/backward.py`: backward, gradient clipping, AdamW/SGD와 scheduler의 공통 갱신 경계.
4. `train/common.py`: 모듈 조립, dev 평가, 실행 기록, checkpoint 정책. 모델 호출과 학습 loss는 개별 실험 파일에 있다.

`Agents`는 모델 facade, `Rewards`와 데이터 adapter는 registry, `Retriever`는 주입 가능한 protocol이다. 공개 데이터 형식과 수식은 해당 모듈의 Python docstring에 함께 둔다. 모듈 사이에는 별도의 공용 contracts 디렉터리를 두지 않았다. 신규 데이터 adapter는 `data_loader/adapters.py`에서 등록하고, 새 목적함수는 `rewards/`에서 정의한다. 신규 학습 절차는 `train/<실험명>.py`에서 조립한다. 현재 실행 가능한 전략은 SFT와 RL이며 DPO 등 미구현 전략은 명시적으로 오류를 낸다.

## 다운로드 없는 실행

저장소 루트에서 실행한다. Python 3.11 이상을 사용한다.

```bash
python -m pip install -e '.[dev]'
qe-train --config configs/smoke_sft.yaml
qe-train --config configs/smoke_rl.yaml --init-from runs/smoke_sft/checkpoints/best.json
qe-evaluate --checkpoint runs/smoke_rl/checkpoints/best.json --split test --output runs/smoke_rl/test.json
qe-export --checkpoint runs/smoke_rl/checkpoints/best.json --output exports/smoke_rl
python -m pytest -q
python -m ruff check src train tests
```

모듈 실행도 가능하다: `python -m train.sft --config configs/smoke_sft.yaml`. `qe-*` 실행 파일의 Python 경로가 오래된 가상환경을 가리키면 editable 설치를 다시 수행하거나 `python -m query_expansion.train`을 사용한다.

Smoke는 100개의 합성 train query와 독립된 dev/test query를 사용한다. SFT 10회와 RL 2회의 **실제 optimizer 갱신**을 목표로 한다. 보상이 모두 같은 RL 그룹은 건너뛰므로 설정된 epoch가 먼저 끝나면 목표 step 수에 미달할 수 있다. 반환되는 `global_step`과 `metrics.jsonl`의 `zero_advantage_rate`를 확인한다. tiny 모델의 검색 점수는 모델 품질의 증거가 아니다.

## 재개와 새 단계의 차이

```bash
qe-train --config configs/smoke_rl.yaml --resume runs/smoke_rl/checkpoints/latest.json
```

`--resume`는 두 optimizer, scheduler, RNG, sampler 위치 및 검색 환경 sampling 상태를 복원한다. 모델·데이터·인덱스·설정·소스·실행 버전이 다르면 거부한다. `output_dir`만 재개 identity에서 제외한다. `--init-from`은 전체 모델 가중치만 가져와 새로운 optimizer로 시작하며, 초기 checkpoint의 SHA-256을 기록한다. 이미 manifest가 있는 출력 디렉터리는 새 실행으로 덮어쓰지 않는다.

학습 window에서 오류가 나면 gradient를 비우고 해당 window를 저장하지 않는다. 디스크에 마지막으로 완료된 checkpoint에서 재개한다. 파일의 무결성 검사와 완료 marker를 통과한 checkpoint만 읽는다. PyTorch checkpoint는 신뢰하는 실행에서 생성한 파일만 사용한다.

## 실제 데이터와 GPU 설정

`configs/qwen_sft.yaml`, `configs/qwen_rl.yaml`은 환경변수를 요구하는 실행 템플릿이다. `QE_ROOT`, `QE_MODEL_REVISION`(고정 commit SHA), `QE_IMAGE_DIGEST`, `QE_DATA_REVISION_AND_LICENSE`를 설정한다. `python -m pip install -e '.[hf,dev]'` 후 CUDA가 제공되는 환경에서 실행한다. GPU 패키지 lock은 실제 Runpod에서 검증한 조합으로 별도 고정해야 한다.

JSONL adapter 입력은 README의 정규화된 `queries.jsonl`, `corpus.jsonl`, `qrels.jsonl`이다. queries 파일에 train/dev/test를 모두 넣고 각 행의 `split`로 선택한다. 공식 split은 adapter 이전 단계에서 보존한다. source별 raw dataset 다운로드나 teacher target 생성은 자동 수행하지 않는다. 근거 문서 ID가 없는 QA를 가짜 retrieval label로 변환하지 않는다.

- SFT는 `expansion_target=null`인 행을 제외한다. 빈 문자열 target은 EOS만 학습하는 유효한 무확장이다.
- RL은 양의 qrel이 없는 행을 제외한다. 모든 검색 환경에 positive가 존재해야 한다.
- policy prompt에는 query와 language만 들어간다. 답변과 qrels는 정책 입력에 넣지 않는다.
- prompt 또는 target 길이 초과는 조용히 자르지 않고 오류로 보고한다.
- split 간 정규화된 동일 질문 및 명시적 `group_id` 중복을 검사한다. 번역·원문 문서의 그룹 ID 구성은 데이터 adapter 담당자의 책임이다.

SFT는 target+EOS 토큰 수로 loss를 가중 평균한다. RL은 그룹 수로 정규화하며, 그룹 내부에서는 후보 수로 나눈다. 따라서 부분 accumulation window도 올바른 분모를 사용한다. RL의 동일 보상 그룹은 loss=0으로 취급하고, window 전체가 그러하면 weight decay와 scheduler까지 건너뛴다.

RL 생성은 temperature=1의 전체 vocabulary categorical sampling이며 top-k/top-p 필터를 적용하지 않는다. 재계산에는 생성 token ID를 그대로 사용한다. 모든 dropout을 0으로 설정하고 Qwen 학습 forward에서는 gradient checkpointing을 사용한다.

## 산출물 읽기

- `manifest.json`, `environment.lock`: 설정, 전체 소스 hash, prompt, 데이터·인덱스 hash, 모델 revision, 라이브러리 버전, 초기 checkpoint 및 비용 설정.
- `sampling.json`: 언어별 실제 query 수, query별 노출 수, 제외 수. 부족한 언어를 무제한 복제하지 않는다.
- `metrics.jsonl`: optimizer step별 loss, RL reward, fallback, invalid, zero-advantage와 실제 후보 수·환경 ID.
- `dev-*.json`: **전체 corpus** 검색 평가. nDCG@10/MRR@10/Recall@100, 언어/source별 집계, dataset macro, paired bootstrap 구간, 지연시간, peak VRAM.
- `checkpoints/latest.json`, `best.json`: 최근/최고 dev macro nDCG checkpoint의 포인터. 최근 두 개와 best를 보관한다.
- export: 하위 LoRA를 병합한 전체 모델, tokenizer, prompt/후처리 버전, dev 평가와 manifest. 재로드 logits·greedy 출력·검색 일치 후에만 `COMPLETE.json`을 생성한다.

시간 제한은 optimizer 경계에서 적용한다. 마지막 dev 평가와 저장 시간은 설정된 학습 시간 한도를 초과할 수 있으므로 Pod 회수 여유 시간을 둔다. `early_stop.min_dev_ndcg`, `early_stop.max_invalid_rate`를 설정하면 주기적 dev 평가에서 중단 여부를 판단한다. 실제 RL에는 SFT dev 기준을 측정한 뒤 이 값을 설정한다.

## 검증 범위와 남은 실험

자동 테스트는 동결 가중치, 복수 optimizer, token-weighted accumulation, causal masking, 고정 BM25 통계, 정책 gradient, 완전한 저장·재개, 전체 corpus 평가와 export를 검사한다. Transformers가 설치되어 있으면 무작위 초기화한 작은 24-layer Qwen checkpoint를 만들어 실제 Qwen text loader의 load/forward/backward/generate를 검사한다. 실제 2B 가중치 다운로드나 A5000 성능 검증은 이 테스트에 포함하지 않는다.

현재 BM25는 수식 검증과 소규모 실험을 위한 메모리 기반 구현이다. 대규모 corpus의 처리량, 실제 한국어 형태소 analyzer, raw dataset별 라이선스·schema 검증, teacher target 품질, 3-seed 채택 실험, GPU 비용/VRAM과 Runpod 복구는 별도의 실측 대상이다. 출력 후처리는 단일 행·구조화 형식·중복어 검사를 수행하지만 모든 자연어 답변/설명 문장을 의미적으로 식별하는 분류기는 아니다.
