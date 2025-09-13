# 부록 A: PyTorch 입문

### 메인 챕터 코드

- [code-part1.ipynb](code-part1.ipynb) 챕터에 나타나는 모든 A.1~A.8 섹션 코드를 포함합니다
- [code-part2.ipynb](code-part2.ipynb) 챕터에 나타나는 모든 A.9 GPU 코드를 포함합니다
- [DDP-script.py](DDP-script.py) 멀티 GPU 사용법을 시연하는 스크립트를 포함합니다 (Jupyter Notebook은 단일 GPU만 지원하므로, 이것은 노트북이 아닌 스크립트입니다). `python DDP-script.py`로 실행할 수 있습니다. 만약 머신에 2개 이상의 GPU가 있다면, `CUDA_VISIBLE_DEVICES=0,1 python DDP-script.py`로 실행하세요.
- [exercise-solutions.ipynb](exercise-solutions.ipynb) 이 챕터의 연습 문제 해답을 포함합니다

### 선택적 코드

- [DDP-script-torchrun.py](DDP-script-torchrun.py) `multiprocessing.spawn`을 통해 직접 여러 프로세스를 생성하고 관리하는 대신 PyTorch `torchrun` 명령을 통해 실행되는 `DDP-script.py` 스크립트의 선택적 버전입니다. `torchrun` 명령은 멀티 노드 협력을 포함하여 분산 초기화를 자동으로 처리하는 장점이 있어, 설정 과정을 약간 단순화합니다. 이 스크립트는 `torchrun --nproc_per_node=2 DDP-script-torchrun.py`로 사용할 수 있습니다