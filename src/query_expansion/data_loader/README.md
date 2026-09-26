# Data loader module

**담당:** 데이터 전문가 한 명 또는 한 역할 담당자. 변경 범위는 이 디렉터리다.

`DatasetStream`, dataset adapter, split 처리, 전략별 전처리와 batch 형식을 관리한다. 선택한 데이터와 학습 전략의 조합을 지원하지 않으면 명확한 오류를 낸다.

모델·보상·검색 구현을 직접 import하거나 수정하지 않는다. `train/<훈련id>.py`가 사용할 공개 입력 인자와 batch 출력 형식은 이 모듈 안에서 문서화한다.
