# 검색 담당자

## 소유 범위

corpus analyzer, BM25 고정 인덱스 통계, 검색 순위와 `Hit` 형식을 관리한다. 보상 계수, policy loss, 모델 생성과 학습 갱신은 맡지 않는다.

## 공개 API

`Retriever` protocol은 `identity`, `languages`, `search(query, candidates=None, k=100)`를 제공한다. 결과는 `Hit(doc_id, score)` 목록이다. corpus 입력은 `doc_id/language/title/text` 필드가 있는 객체이며 data_loader 내부 클래스를 import하지 않는다.

- `candidates=None`: 전체 corpus 검색. 평가 positive를 강제로 넣지 않는다.
- `candidates` 지정: 같은 고정 corpus 통계로 후보만 scoring. corpus에 없는 ID는 오류.
- score 동률: doc_id 순서로 결정해 재현성을 보장한다.
- `identity`: corpus, analyzer와 BM25 설정 변경을 식별한다.

새 backend는 같은 계약을 만족하도록 이 디렉터리에 구현한다. `train`의 backend 선택 변경은 통합 담당자가 맡는다. 보상 또는 평가 담당자가 검색 인덱스 내부를 직접 수정하게 하지 않는다.

현재 Python BM25는 검증용 reference 구현이다. 대규모 backend를 추가할 때 동일 입력의 순위·후보 제한·동률 및 index identity를 reference와 비교한다. 성능 개선 때문에 relevance 판정이나 후보의 의미를 바꾸지 않는다.

검증: `tests/test_retrieval_rl.py`, `tests/test_evaluation.py`. 특히 검색 전후 corpus IDF가 불변인지 확인한다.
