# 데이터 담당자

## 소유 범위

`schema.py`의 Query/Document, `adapters.py`의 데이터 adapter, `stream.py`의 전략별 필터링·batch 형식·sampling 상태를 관리한다. 원본 데이터 revision·라이선스 확인, 안정적인 문서 ID 연결, split 누수와 제외 사유도 데이터 책임이다.

## 공개 API

- `load_dataset(config)` / `register_adapter(name, adapter)`: 정규화된 `(queries, corpus)` 반환. 신규 adapter를 등록하면 train entrypoint의 dataset 분기를 수정할 필요가 없다.
- `DatasetStream(data, strategy, batch_size, ...)`: 한 epoch의 재개 가능한 batch iterator.
- `Batch`: `(inputs, labels, save_flag)`로 unpack할 수 있다. inputs는 query/language만, labels는 SFT target 또는 RL용 Query record다.
- `at_epoch_end`: 현재 epoch의 마지막 batch까지 전달했는지 나타낸다.
- `next_epoch()`, `state_dict()`, `load_state_dict()`: 다음 epoch 및 checkpoint cursor 관리.
- `report()`: 실제 언어별 query 수·반복 노출·제외 수 보고.

`utils.TrainingLoop`는 이 공개 stream 동작으로 epoch와 accumulation 경계를 관리한다. sampler가 optimizer step을 호출하거나 loss를 검사하지 않는다.

## 변경 기준

SFT target이 없는 행과 positive qrel이 없는 RL 행은 각 전략에 맞게 제외한다. QA 답변만으로 positive 문서를 만들어내지 않는다. 공식 split을 보존하고 동일 질문·번역·원문 문서 grouping의 누수를 검사한다. sampling seed와 cursor를 복원하면 다음 query 순서가 같아야 한다.

tokenizer별 prompt encoding은 현재 `Agents.supervised`가 제공한다. 데이터 담당자는 텍스트 target과 전략별 데이터 전처리를 담당하며 모델 내부를 import하지 않는다. 새로운 encoding 책임이 필요하면 입력·출력 계약을 모델 담당자와 먼저 조율한다.

검증: `tests/test_data.py`, `tests/test_streaming_contract.py`, 재개 CLI 테스트. 스키마를 바꾸면 연구 데이터와 checkpoint identity가 어떻게 달라지는지도 확인한다.
