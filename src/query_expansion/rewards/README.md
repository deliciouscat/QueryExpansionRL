# Rewards module

**담당:** 목적함수·보상 전문가 한 명 또는 한 역할 담당자. 변경 범위는 이 디렉터리다.

`Rewards`, 목적함수 registry, SFT/RL loss 및 reward 계산을 관리한다. 검색 기반 보상은 entrypoint에서 전달받은 retriever 공개 인터페이스를 사용하며, 검색 구현을 직접 import하거나 수정하지 않는다.

목적함수별 입력과 출력, 필요한 retriever 동작은 이 모듈 안에 문서화한다. 새 전략을 지원하지 않는 조합은 조용히 대체하지 말고 명시적으로 오류 처리한다.
