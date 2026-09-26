# 수동 검토용 제안

README와 CODE_INTERFACE 등 기존 설계 문서는 수정하지 않았다. 아래는 구현 정책을 임의로 바꾸기 위한 승인이 아니라 후속 실험 전에 확정하면 좋은 사항이다.

1. **RL early-stop 임계값**: README가 요구하는 SFT dev 성능 저하와 출력 오류 증가를 판단할 수 있도록 `early_stop.min_dev_ndcg`, `early_stop.max_invalid_rate`를 제공했다. 기준선 대비 허용 하락폭과 연속 위반 횟수를 문서에서 수치로 확정하면 일관된 채택 판단에 도움이 된다. 현재 임계값은 명시적으로 설정해야 한다.
2. **대규모 검색 backend**: 현재 `Retriever` protocol을 구현한 Python BM25는 고정 corpus 통계와 보상 검증에 적합하다. 전체 MS MARCO/MIRACL 이전에 저장형 inverted index backend와 한국어 analyzer를 선택하고 reference BM25와 순위/동률 회귀 검증을 권한다.
3. **출력 형식 판정 범위**: 현재 후처리는 개행, JSON/태그, 일부 답변 접두사, EOS 여부와 중복 표현을 검사한다. 임의의 설명 문장을 모두 판별할 수는 없다. 언어별 유효/무효 예시 집합을 데이터 담당자가 합의하면 학습 보상과 평가에서 같은 판정을 재현할 수 있다.
4. **원문 문서/chunk 및 번역 그룹**: 정규화 스키마의 선택적 `group_id`로 split 누수를 검사한다. 원시 데이터 adapter에서 번역본과 원문 문서 ID를 구성하는 규칙, chunk relevance를 원문 문서로 합치는 규칙을 데이터별로 확정할 필요가 있다. 현재 실행 데이터는 문서 단위 corpus를 전제로 한다.
5. **GPU 검증 후 lock 확정**: 실행마다 버전 lock과 image digest를 기록하지만 이 macOS CPU 검증 결과를 A5000 검증으로 간주할 수 없다. 실제 Qwen 2B의 BF16 load/forward/generate/backward, peak VRAM, 재개 및 export를 GPU에서 통과한 뒤 배포 이미지와 transitive dependency lock을 고정하는 것을 권한다.

-> comment: 이 내용들은 구현하면서 반영하면 될듯.