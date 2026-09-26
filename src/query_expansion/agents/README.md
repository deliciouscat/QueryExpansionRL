# Agents module

**담당:** 모델·정책 전문가 한 명 또는 한 역할 담당자. 변경 범위는 이 디렉터리다.

`Agents` 구성, 모델 로딩, 생성 정책, 학습 대상 파라미터 선택을 관리한다. 실험에서 필요한 생성·forward API를 제공하되 data loader, reward, retriever 내부 구현에 의존하지 않는다.

공개 생성 및 forward 입력·출력 형식은 이 모듈 안에서 문서화한다. 실험 entrypoint가 모델 종류와 설정을 선택해 초기화한다.
