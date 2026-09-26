# 평가 담당자

## 소유 범위

retrieval metric, 언어·데이터셋 집계, paired bootstrap, 전체 corpus 평가와 보고 형식을 관리한다. 모델 학습, reward penalty, optimizer 갱신, test를 이용한 모델 선택은 담당하지 않는다.

## 공개 API

- `retrieval_metrics(ranking, qrels)`: nDCG@10, MRR@10, Recall@100.
- `aggregate(rows, language_weights=...)`: 언어/source별 성능, dataset macro, 목표 언어 가중값과 fallback/악화 비율.
- `evaluate(records, retriever, expand, split=...)`: 주입된 검색 API와 expansion callback으로 전체 corpus 평가.
- `paired_bootstrap(deltas, ...)`: 동일 query의 성능 차이 신뢰구간.

`expand(query, language)`는 후처리된 출력과 생성 token 수를 반환한다. 평가 모듈은 `Agents`를 직접 구성하거나 import하지 않는다. 모델·후처리 연결은 `train/common.py` 또는 평가 CLI의 통합 코드가 담당한다.

## 판단과 측정의 구분

평가 담당자는 지표를 정확히 측정한다. 어떤 metric으로 best checkpoint를 선택할지, SFT 대비 어느 하락폭에서 멈출지, test를 언제 실행할지는 통합 담당자가 정한다. `utils`에는 보고서와 이미 선택된 score/stop 판단을 전달한다.

후보 문서 기반 학습 reward와 전체 corpus 평가를 구분한다. 전체 corpus 검색에 known positive를 삽입하지 않는다. graded relevance와 doc ID 중복 처리를 유지한다. 새로운 집계 방식은 지표 이름과 평균의 분모를 함께 설명한다.

검증: `tests/test_evaluation.py`, CLI/export 통합 테스트. 작은 합성 corpus의 성능을 실제 모델 품질로 보고하지 않는다.
