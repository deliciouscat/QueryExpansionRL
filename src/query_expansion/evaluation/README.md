# Evaluation module

**담당:** 평가 전문가 한 명 또는 한 역할 담당자. 변경 범위는 이 디렉터리다.

Retrieval metric 계산, 언어·데이터셋별 집계와 평가 출력을 관리한다. 검색은 entrypoint에서 전달받은 retriever 공개 인터페이스로 실행하고, 모델 학습 및 reward 구현에는 의존하지 않는다.

평가 입력, 지표 이름과 출력 형식은 이 모듈 안에서 문서화한다. dev/test 데이터 선택과 평가 실행 시점은 실험 entrypoint가 정한다.
