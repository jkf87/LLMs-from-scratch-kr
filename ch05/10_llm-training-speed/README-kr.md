# 더 빠른 LLM 학습을 위한 PyTorch 성능 팁

이 책은 교육 목적으로 작성되었으며, 이는 원본 코드가 읽기 쉽도록 의도적으로 단순하게 유지되었음을 의미합니다. 이는 가독성을 돕고 CPU와 GPU를 포함한 다양한 하드웨어에서의 호환성을 보장하기 위함입니다. 하지만 LLM 학습을 더 성능적으로 만들기 위한 더 고급 PyTorch 및 GPU 기능에 대해 궁금할 수 있습니다.

이 폴더는 5장에서 소개된 LLM과 학습 함수의 성능 최적화를 보여주는 3개의 코드 파일을 포함합니다:

1. [`00_orig.py`](00_orig.py): CPU 및 단일 GPU 학습을 위한 원본 5장 코드입니다.  
   ➤ 실행: `python 00_orig.py`

2. [`01_opt_single_gpu.py`](01_opt_single_gpu.py): 단일 GPU 학습을 위한 최적화된 버전입니다.  
   ➤ 실행: `python 01_opt_single_gpu.py`

3. [`02_opt_multi_gpu_ddp.py`](02_opt_multi_gpu_ddp.py): 분산 데이터 병렬성(Distributed Data Parallel, DDP)을 사용한 멀티 GPU 학습을 위한 최적화된 버전입니다.  
   ➤ 실행: `torchrun --nproc_per_node=4 02_opt_multi_gpu_ddp.py`  
   (**참고:** `01_opt_single_gpu.py`와 비교하여 변경사항을 최소한으로 유지하기 위해, 이 스크립트는 위에 표시된 대로 `torchrun`을 통해서만 멀티 프로세싱을 지원합니다. 이는 멀티 GPU 지원이 `python 02_opt_multi_gpu_ddp.py`를 통해서는 **지원되지 않음**을 의미합니다)

**이러한 수정사항들은 학습 속도를 12,525 토큰/초(단일 A100)에서 142,156 토큰/초(단일 A100)로, 그리고 419,259 토큰/초(4x A100s)로 향상시킵니다.**

앞으로 더 자세한 설명을 확장할 계획입니다. 현재로서는 코드에 어떤 개선사항이 추가되었는지 확인하는 가장 쉬운 방법은 Visual Studio Code에서 파일을 열고 "Compare Selected" 기능을 통해 차이점을 보는 것입니다.

![VS compare](https://sebastianraschka.com/images/LLMs-from-scratch-images/bonus/llm-training-speed/vs-code-compare.png)

![PyTorch Tips](https://sebastianraschka.com/images/LLMs-from-scratch-images/bonus/pytorch-tips/pytorch-tips.webp?1)

&nbsp;
## 단일 GPU 속도 비교

위에서 언급했듯이, 앞으로 변경사항에 대해 더 자세히 설명할 계획입니다. 현재 이 섹션은 각 수정사항에 대한 간단한 토큰/초 성능 개요를 포함합니다. 모든 실험은 A100 GPU에서 실행되었습니다.

&nbsp;
### 기준선

`00_orig.py`는 기준선 역할을 하며 다음을 제외하고는 5장의 코드를 그대로 사용하는 중요한 수정사항이 없습니다:

- 4배 더 큰 컨텍스트 길이(`00_orig.py`가 5장에 비해 상대적으로 큰 메모리 사용량을 갖는 이유를 설명);
- 4배 배치 크기 변경(`00_orig.py`의 상대적으로 큰 메모리 사용량에 대한 또 다른 기여 요소);
- 학습 데이터 크기를 증가시키기 위한 더 큰 공개 도메인 책.

하이퍼파라미터는 손실 최소화와 과적합 감소에 대해 매우 최적화되지 않았으며, 마지막에 LLM에서 생성된 텍스트가 매우 정교하지 않을 수 있습니다; 하지만 주요 요점은 여기서 속도 참조로 사용되는 `tok/sec` 메트릭이므로 이는 중요하지 않습니다(높을수록 좋음).

```bash
ubuntu@159-13-52-60:~$ python 00_orig.py
PyTorch version: 2.6.0+cu124
Using cuda
CUDA version: 12.4

Ep 1, Step 000000, Train: 9.535, Val: 9.609, Step tok/sec: 7238, Avg tok/sec: 0
Ep 1, Step 000015, Train: 6.201, Val: 6.152, Step tok/sec: 12545, Avg tok/sec: 12545
Ep 1, Step 000030, Train: 5.663, Val: 5.688, Step tok/sec: 12490, Avg tok/sec: 12517
Ep 1, Step 000045, Train: 5.316, Val: 5.362, Step tok/sec: 12541, Avg tok/sec: 12525
Every effort moves you, and's, and I am not be a

...

Ep 15, Step 000735, Train: 0.227, Val: 6.818, Step tok/sec: 11599, Avg tok/sec: 12248
Ep 15, Step 000750, Train: 0.300, Val: 6.895, Step tok/sec: 12530, Avg tok/sec: 12253
Ep 15, Step 000765, Train: 0.150, Val: 6.914, Step tok/sec: 12532, Avg tok/sec: 12259
Every effort moves you like best to think which he held in the room in him, the interest was the night, the realities of the affairs Bulstrode's duty, now!' the fact is another man, conquests

Allocated memory: 2.5069 GB
Reserved memory: 26.2617 GB
```

`01_opt_single_gpu.py`는 아래에 순차적으로 나열된 모든 수정사항을 포함합니다.

비교는 항상 이전 섹션의 첫 번째 에포크 후 평균 tok/sec와 할당된 메모리를 기반으로 합니다.

&nbsp;
### 1. 동적으로 인과 마스크 생성

- 인과 마스크를 저장하는 대신, 메모리 사용량을 줄이기 위해 인과 마스크를 동적으로 생성합니다(여기서는 최소한의 효과가 있지만, Llama 3.2와 같은 131k 입력 토큰 지원을 가진 긴 컨텍스트 크기 모델에서는 누적될 수 있습니다)

이전:
- `Avg tok/sec: 12525`
- `Reserved memory: 26.2617 GB`

이후:
- `Avg tok/sec: 12526`
- `Reserved memory: 26.2422 GB`

&nbsp;
### 2. 텐서 코어 사용

- 텐서 코어를 사용합니다(A100과 같은 Ampere GPU 이상에서만 작동)

이전:
- `Avg tok/sec: 12526`
- `Reserved memory: 26.2422 GB`

이후:
- `Avg tok/sec: 27648`
- `Reserved memory: 26.2422 GB`

&nbsp;
### 3. 융합 AdamW 옵티마이저

- `fused=True`를 설정하여 `AdamW`에 대한 융합 커널을 사용합니다

이전:
- `Avg tok/sec: 27648`
- `Reserved memory: 26.2422 GB`

이후:
- `Avg tok/sec: 28399`
- `Reserved memory: 26.2422 GB`

&nbsp;
### 4. 데이터 로더에서 고정 메모리

- GPU 메모리를 사전 할당하고 재사용하기 위해 데이터 로더에서 `pin_memory=True`를 사용합니다

이전:
- `Avg tok/sec: 28399`
- `Reserved memory: 26.2422 GB`

이후:
- `Avg tok/sec: 28402`
- `Reserved memory: 26.2422 GB`

&nbsp;
### 5. bfloat16 정밀도 사용

- 32비트 float에서 16비트 brain float (bfloat16) 정밀도로 전환합니다(이 주제에 대해서는 [여기의 제 글](https://magazine.sebastianraschka.com/p/the-missing-bits-llama-2-weights)을 참조하시기 바랍니다)

이전:
- `Avg tok/sec: 28402`
- `Reserved memory: 26.2422 GB`

이후:
- `Avg tok/sec: 45486`
- `Reserved memory: 13.7871 GB`

&nbsp;
### 6. 처음부터 구현한 코드를 PyTorch 클래스로 교체

- LayerNorm과 GeLU의 처음부터 구현을 PyTorch의 네이티브 구현으로 교체합니다

이전:
- `Avg tok/sec: 45486`
- `Reserved memory: 13.7871 GB`

이후:
- `Avg tok/sec: 55256`
- `Reserved memory: 11.5645 GB`

&nbsp;
### 7. FlashAttention 사용

- 처음부터 구현한 멀티헤드 어텐션 구현 대신 FlashAttention을 포함한 PyTorch의 셀프 어텐션 함수를 사용합니다.

이전:
- `Avg tok/sec: 55256`
- `Reserved memory: 11.5645 GB`

이후:
- `Avg tok/sec: 91901`
- `Reserved memory: 5.9004 GB`

&nbsp;
### 8. `pytorch.compile` 사용

- `torch.compile(model)`을 사용합니다. 속도가 향상되기 전에 첫 번째 반복들은 항상 느립니다. `Avg tok/sec` 측정이 평균 계산에서 첫 번째 행만 포함하므로, 이제 에포크 1 끝의 `Step tok/sec`를 사용합니다.

이전:
- `Avg tok/sec: 91901`
- `Reserved memory: 5.9004 GB`

이후:
- `Step tok/sec: 112046`
- `Reserved memory: 6.1875 GB`

&nbsp;
### 9. 어휘 패딩

- 여기서는 어휘 크기를 50,257에서 64의 가장 가까운 배수인 50,304로 약간 증가시킵니다. 이 팁은 제 전 동료인 Carlos Mocholi가 제안했으며, 원래 Andrej Karpathy가 제안했다고 언급했습니다(아마 [이 게시물](https://x.com/karpathy/status/1621578354024677377)에서). Karpathy의 권장사항은 [Bertrand Maher](https://www.linkedin.com/feed/update/urn:li:activity:7309569006057795584?commentUrn=urn%3Ali%3Acomment%3A%28activity%3A7309569006057795584%2C7309754284185669632%29&dashCommentUrn=urn%3Ali%3Afsd_comment%3A%287309754284185669632%2Curn%3Ali%3Aactivity%3A7309569006057795584%29)가 언급한 것처럼 `torch.compile`에 대한 조언을 제공한 PyTorch 팀과의 상호작용에 기반합니다. 이에 대한 좋은 자료는 배치 크기와 선형 레이어 차원이 일반적으로 특정 값의 배수로 선택되는 [NVIDIA의 텐서 모양 가이드라인](https://docs.nvidia.com/deeplearning/performance/mixed-precision-training/index.html#tensor-core-shape)입니다. 또한, 어휘 패딩 트릭은 NVIDIA의 Megatron 팀이 오래전에 설명했습니다(2019년 [Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism](https://arxiv.org/abs/1909.08053) 논문 참조).

이전:
- `Step tok/sec: 112046`
- `Reserved memory: 6.1875 GB`

이후:
- `Step tok/sec: 127345`
- `Reserved memory: 5.8906 GB`

&nbsp;
### 10. 배치 크기 증가

- 마지막으로, GPU에서 지원되는 가장 큰 2의 거듭제곱으로 배치 크기를 증가시킵니다

이전:
- `Step tok/sec: 127345`
- `Reserved memory: 5.8906 GB`

이후:
- `Step tok/sec: 142156`
- `Reserved memory: 22.5078 GB`

&nbsp;
## 멀티 GPU 속도 비교

이제 1개 대신 4개의 GPU를 사용하므로 완전히 공정한 비교는 아닐 수 있지만, 학습이 제한된 GPU 메모리로 인해 병목현상이 발생하지 않는 경우 사용할 수 있는 가장 빠른 멀티 GPU 기술인 분산 데이터 병렬성을 사용하면 당연히 눈에 띄는 속도 향상을 얻을 수 있습니다:

이전(단일 GPU):
- `Step tok/sec: 142156`
- `Reserved memory: 22.5078 GB`

이후(4 GPU):
- `Step tok/sec: 419259`
- `Reserved memory: 22.7969 GB`