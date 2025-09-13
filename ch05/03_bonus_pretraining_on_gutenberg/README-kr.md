# 프로젝트 구텐베르크 데이터셋으로 GPT 사전학습

이 디렉토리의 코드는 프로젝트 구텐베르크(Project Gutenberg)에서 제공하는 무료 도서로 작은 GPT 모델을 학습시키는 코드를 포함합니다.

프로젝트 구텐베르크 웹사이트에서 명시하듯이, "프로젝트 구텐베르크 전자책의 대부분은 미국에서 퍼블릭 도메인(public domain)입니다."

프로젝트 구텐베르크에서 제공하는 자료 사용에 대한 자세한 정보는 [프로젝트 구텐베르크 허가, 라이선스 및 기타 일반적인 요청](https://www.gutenberg.org/policy/permission.html) 페이지를 참조하시기 바랍니다.

&nbsp;
## 이 코드 사용 방법

&nbsp;

### 1) 데이터셋 다운로드

이 섹션에서는 [`pgcorpus/gutenberg`](https://github.com/pgcorpus/gutenberg) GitHub 저장소의 코드를 사용하여 프로젝트 구텐베르크에서 책을 다운로드합니다.

이 글을 작성하는 시점에서 약 50GB의 디스크 공간이 필요하고 약 10-15시간이 걸리지만, 프로젝트 구텐베르크가 그 이후로 얼마나 성장했는지에 따라 더 오래 걸릴 수 있습니다.

&nbsp;
#### Linux 및 macOS 사용자를 위한 다운로드 지침

Linux 및 macOS 사용자는 다음 단계를 따라 데이터셋을 다운로드할 수 있습니다(Windows 사용자인 경우 아래 주석을 참조하시기 바랍니다):

1. `03_bonus_pretraining_on_gutenberg` 폴더를 작업 디렉토리로 설정하여 이 폴더에 `gutenberg` 저장소를 로컬로 클론합니다(제공된 스크립트 `prepare_dataset.py`와 `pretraining_simple.py`를 실행하는 데 필요합니다). 예를 들어, `LLMs-from-scratch` 저장소 폴더에 있을 때 다음 명령으로 *03_bonus_pretraining_on_gutenberg* 폴더로 이동합니다:
```bash
cd ch05/03_bonus_pretraining_on_gutenberg
```

2. 그곳에서 `gutenberg` 저장소를 클론합니다:
```bash
git clone https://github.com/pgcorpus/gutenberg.git
```

3. 로컬로 클론된 `gutenberg` 저장소 폴더로 이동합니다:
```bash
cd gutenberg
```

4. `gutenberg` 저장소 폴더에서 *requirements.txt*에 정의된 필요한 패키지를 설치합니다:
```bash
pip install -r requirements.txt
```

5. 데이터를 다운로드합니다:
```bash
python get_data.py
```

6. `03_bonus_pretraining_on_gutenberg` 폴더로 돌아갑니다:
```bash
cd ..
```

&nbsp;
#### Windows 사용자를 위한 특별 지침

[`pgcorpus/gutenberg`](https://github.com/pgcorpus/gutenberg) 코드는 Linux와 macOS 모두와 호환됩니다. 하지만 Windows 사용자는 `subprocess` 호출에 `shell=True`를 추가하고 `rsync`를 교체하는 등의 작은 조정이 필요합니다.

또는 Windows에서 이 코드를 실행하는 더 쉬운 방법은 Windows에서 Ubuntu를 사용하여 Linux 환경을 실행할 수 있게 해주는 "Windows Subsystem for Linux" (WSL) 기능을 사용하는 것입니다. 자세한 정보는 [Microsoft의 공식 설치 지침](https://learn.microsoft.com/en-us/windows/wsl/install)과 [튜토리얼](https://learn.microsoft.com/en-us/training/modules/wsl-introduction/)을 참조하시기 바랍니다.

WSL을 사용할 때는 Python 3가 설치되어 있는지 확인하고(`python3 --version`으로 확인하거나, 예를 들어 Python 3.10의 경우 `sudo apt-get install -y python3.10`으로 설치), 다음 패키지들을 설치하시기 바랍니다:

```bash
sudo apt-get update && \
sudo apt-get upgrade -y && \
sudo apt-get install -y python3-pip && \
sudo apt-get install -y python-is-python3 && \
sudo apt-get install -y rsync
```

> **참고:**
> Python 설정 및 패키지 설치 방법에 대한 지침은 [Python 설정 선택 사항](../../setup/01_optional-python-setup-preferences/README.md)과 [Python 라이브러리 설치](../../setup/02_installing-python-libraries/README.md)에서 찾을 수 있습니다.
>
> 선택적으로, 이 저장소와 함께 Ubuntu를 실행하는 Docker 이미지가 제공됩니다. 제공된 Docker 이미지로 컨테이너를 실행하는 방법에 대한 지침은 [선택적 Docker 환경](../../setup/03_optional-docker-environment/README.md)에서 찾을 수 있습니다.

&nbsp;
### 2) 데이터셋 준비

다음으로, `prepare_dataset.py` 스크립트를 실행하여 (이 글을 작성하는 시점에서 60,173개의) 텍스트 파일을 더 큰 파일로 연결합니다. 이렇게 하면 더 효율적으로 전송되고 접근할 수 있습니다:

```bash
python prepare_dataset.py \
  --data_dir gutenberg/data/raw \
  --max_size_mb 500 \
  --output_dir gutenberg_preprocessed
```

```
...
Skipping gutenberg/data/raw/PG29836_raw.txt as it does not contain primarily English text.                                     Skipping gutenberg/data/raw/PG16527_raw.txt as it does not contain primarily English text.                                     100%|██████████████████████████████████████████████████████████| 57250/57250 [25:04<00:00, 38.05it/s]
42 file(s) saved in /Users/sebastian/Developer/LLMs-from-scratch/ch05/03_bonus_pretraining_on_gutenberg/gutenberg_preprocessed
```

> **팁:**
> 생성된 파일은 단순함을 위해 평문 형식으로 저장되며 사전 토큰화되지 않습니다. 하지만 데이터셋을 더 자주 사용하거나 여러 에포크(epoch) 동안 학습시킬 계획이라면 계산 시간을 절약하기 위해 데이터셋을 사전 토큰화된 형태로 저장하도록 코드를 업데이트할 수 있습니다. 자세한 내용은 이 페이지 하단의 *설계 결정 및 개선사항*을 참조하시기 바랍니다.

> **팁:**
> 예를 들어 50MB와 같은 더 작은 파일 크기를 선택할 수 있습니다. 이렇게 하면 더 많은 파일이 생성되지만 테스트 목적으로 작은 수의 파일에서 더 빠른 사전학습 실행에 유용할 수 있습니다.

&nbsp;
### 3) 사전학습 스크립트 실행

다음과 같이 사전학습 스크립트를 실행할 수 있습니다. 추가 명령줄 인수는 설명 목적으로 기본값과 함께 표시됩니다:

```bash
python pretraining_simple.py \
  --data_dir "gutenberg_preprocessed" \
  --n_epochs 1 \
  --batch_size 4 \
  --output_dir model_checkpoints
```

출력은 다음과 같은 형식으로 표시됩니다:

> Total files: 3
> Tokenizing file 1 of 3: data_small/combined_1.txt
> Training ...
> Ep 1 (Step 0): Train loss 9.694, Val loss 9.724
> Ep 1 (Step 100): Train loss 6.672, Val loss 6.683
> Ep 1 (Step 200): Train loss 6.543, Val loss 6.434
> Ep 1 (Step 300): Train loss 5.772, Val loss 6.313
> Ep 1 (Step 400): Train loss 5.547, Val loss 6.249
> Ep 1 (Step 500): Train loss 6.182, Val loss 6.155
> Ep 1 (Step 600): Train loss 5.742, Val loss 6.122
> Ep 1 (Step 700): Train loss 6.309, Val loss 5.984
> Ep 1 (Step 800): Train loss 5.435, Val loss 5.975
> Ep 1 (Step 900): Train loss 5.582, Val loss 5.935
> ...
> Ep 1 (Step 31900): Train loss 3.664, Val loss 3.946
> Ep 1 (Step 32000): Train loss 3.493, Val loss 3.939
> Ep 1 (Step 32100): Train loss 3.940, Val loss 3.961
> Saved model_checkpoints/model_pg_32188.pth
> Book processed 3h 46m 55s
> Total time elapsed 3h 46m 55s
> ETA for remaining books: 7h 33m 50s
> Tokenizing file 2 of 3: data_small/combined_2.txt
> Training ...
> Ep 1 (Step 32200): Train loss 2.982, Val loss 4.094
> Ep 1 (Step 32300): Train loss 3.920, Val loss 4.097
> ...

&nbsp;
> **팁:**
> 실제로 macOS나 Linux를 사용하는 경우, 터미널에 출력하는 동시에 로그 출력을 `log.txt` 파일에 저장하기 위해 `tee` 명령을 사용할 것을 권합니다:

```bash
python -u pretraining_simple.py | tee log.txt
```

&nbsp;
> **경고:**
> `gutenberg_preprocessed` 폴더의 ~500MB 텍스트 파일 중 1개에서 학습하는 것은 V100 GPU에서 약 4시간이 걸립니다.
> 폴더에는 47개의 파일이 포함되어 있으며 완료하는 데 약 200시간(1주일 이상)이 걸립니다. 더 적은 수의 파일에서 실행하는 것을 고려할 수 있습니다.

&nbsp;
## 설계 결정 및 개선사항

이 코드는 교육 목적을 위해 단순함과 최소성을 유지하는 데 중점을 둡니다. 모델링 성능과 학습 효율성을 개선하기 위해 다음과 같은 방법으로 코드를 개선할 수 있습니다:

1. 각 책 파일에서 구텐베르크 표준 텍스트(boilerplate text)를 제거하도록 `prepare_dataset.py` 스크립트를 수정합니다.
2. 사전학습 스크립트를 호출할 때마다 다시 토큰화할 필요가 없도록 데이터셋을 사전 토큰화하고 토큰화된 형태로 저장하도록 데이터 준비 및 로딩 유틸리티를 업데이트합니다.
3. [부록 D: 학습 루프에 유용한 요소 추가](../../appendix-D/01_main-chapter-code/appendix-D.ipynb)에서 도입된 기능들, 즉 코사인 감소(cosine decay), 선형 워밍업(linear warmup), 그래디언트 클리핑(gradient clipping)을 추가하여 `train_model_simple` 스크립트를 업데이트합니다.
4. 옵티마이저 상태를 저장하고(5장의 *5.4 PyTorch에서 가중치 로드 및 저장* 섹션 참조; [ch05.ipynb](../../ch05/01_main-chapter-code/ch05.ipynb)) 기존 모델 및 옵티마이저 체크포인트를 로드하여 학습이 중단된 경우 학습을 계속할 수 있는 옵션을 추가하도록 사전학습 스크립트를 업데이트합니다.
5. 실시간으로 손실 및 검증 곡선을 볼 수 있는 더 고급 로거(예: Weights and Biases)를 추가합니다.
6. 분산 데이터 병렬성(Distributed Data Parallelism, DDP)을 추가하고 여러 GPU에서 모델을 학습시킵니다(부록 A의 *A.9.3 여러 GPU로 학습* 섹션 참조; [DDP-script.py](../../appendix-A/01_main-chapter-code/DDP-script.py)).
7. `previous_chapter.py` 스크립트의 처음부터 구현한 `MultiheadAttention` 클래스를 PyTorch의 `nn.functional.scaled_dot_product_attention` 함수를 통해 Flash Attention을 사용하는 [효율적인 Multi-Head Attention 구현](../../ch03/02_bonus_efficient-multihead-attention/mha-implementations.ipynb) 보너스 섹션에서 구현된 효율적인 `MHAPyTorchScaledDotProduct` 클래스로 교체합니다.
8. [torch.compile](https://pytorch.org/tutorials/intermediate/torch_compile_tutorial.html) (`model = torch.compile`) 또는 [thunder](https://github.com/Lightning-AI/lightning-thunder) (`model = thunder.jit(model)`)를 통해 모델을 최적화하여 학습 속도를 높입니다.
9. 사전학습 과정을 더욱 가속화하기 위해 Gradient Low-Rank Projection (GaLore)을 구현합니다. 이는 `AdamW` 옵티마이저를 [GaLore Python 라이브러리](https://github.com/jiaweizzhao/GaLore)에서 제공하는 `GaLoreAdamW`로 교체하는 것만으로도 달성할 수 있습니다.