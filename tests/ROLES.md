# 모듈 검증 및 통합 검증 담당자

각 모듈 담당자는 자신의 공개 동작에 대한 테스트를 유지한다. 모듈 간 연결·재개·CLI·export는 통합 검증 담당자가 확인한다. 현재 테스트는 한 디렉터리에 있지만 소유 책임은 다음과 같이 나뉜다.

| 파일 | 주 담당 |
|---|---|
| `test_data.py` | 데이터 |
| `test_objectives.py`, `test_retrieval_rl.py` | 보상 + 검색 |
| `test_evaluation.py` | 평가 |
| `test_training.py`, `test_qwen_integration.py` | 모델 + 학습 기반 |
| `test_streaming_contract.py` | 학습 기반 + 실험 통합 |
| `test_public_loop.py`, `test_cli.py` | 실험 통합 |

streaming 계약 테스트는 메모리상 saved activation이 다음 후보 계산 전에 해제되는지, 중간 optimizer step이 없는지, partial-window 분모가 맞는지, skipped 그룹이 분모에 남는지, 예외·중복 호출·누락 호출이 갱신을 막는지 검증한다. source 구조 검사만으로 동작을 검증했다고 주장하지 않는다.

`python -m pytest -q`와 `python -m ruff check src train tests`로 실행한다. 실제 Qwen 2B·CUDA·A5000의 메모리와 성능은 별도 환경 검증이며 CPU tiny/축소 Qwen 통과와 구분한다. 다른 모듈의 공개 계약을 바꾸기 위해 실패 테스트를 삭제하거나 기대값만 낮추지 않는다.
