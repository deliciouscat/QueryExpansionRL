# 진입점 구조
`train/` 디렉토리에 여러 훈련 조합을 별도의 `.py` 파일로 관리한다.
간결하게 유지.
각 모듈()을 별도 디렉토리에서 관리. 각 분야의 전문 연구자가 해당 디렉토리에서만 작업할 수 있도록 한다.
각 모듈은 `train/(훈련id).py`에서 import하여 학습한다. `(훈련id).py`는 각 모듈을 조립하는 역할만 한다.


# Pseudo-code

```train/(훈련id).py
import DatasetStream
import Agents
import Rewards
from utils.backward import gradient

dataset_stream = DatasetStream(
    data: List[str],    #
    strategy: str,      # RLHF, DPO, SFT 등등... 해당 전략에 맞는 전처리가 해당 dataset 전처리 코드에 포함되어 있어야 함. 준비되어 있지 않을 시 error message.
    batch_size: int,
    ...
)

model = Agents(
    model: str,     # 모델 checkpoint
    ...
)

criterion = Rewards(
    strategy: str,  # registry에서 관리되는 reward 중에 골라서 쓸 수 있게
    ...
)


def forward(~):
    """ output 도출 및 loss 계산, reward 계산 등을 포함함. 각 컴포넌트를 조합하는 파트. """
    outputs = model(inputs)        # 2. forward pass
    ...(*"임의의 SFT or RL 프로세스")...
    loss = criterion(outputs, ~)  # 3. loss 계산


for batch in dataset_stream:
    inputs, labels, save_flag = batch
    @gradient
    forward(~)

    if save_flag==True:
        ...(저장)...
    elif *"다른 checkpoint 정책":
        ...(저장)...
```

```utils/backward.py
# trivial한 gradient 작업을 처리하는 decorator
def gradient(forward_block):
    optimizer.zero_grad()          # 1. 이전 gradient 초기화
    forward_block()
    loss.backward()                # 4. backward pass (gradient 계산)
    optimizer.step()               # 5. 파라미터 업데이트
```