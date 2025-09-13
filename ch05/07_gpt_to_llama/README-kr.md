# GPT를 Llama로 변환

이 폴더는 4장과 5장의 GPT 구현을 Meta AI의 Llama 아키텍처로 변환하는 코드를 다음 권장 읽기 순서로 포함합니다:

- [converting-gpt-to-llama2.ipynb](converting-gpt-to-llama2.ipynb): GPT를 Llama 2 7B로 단계별로 변환하고 Meta AI에서 사전학습된 가중치를 로드하는 코드를 포함합니다.
- [converting-llama2-to-llama3.ipynb](converting-llama2-to-llama3.ipynb): Llama 2 모델을 Llama 3, Llama 3.1, Llama 3.2로 변환하는 코드를 포함합니다.
- [standalone-llama32.ipynb](standalone-llama32.ipynb): Llama 3.2를 구현하는 독립적인 노트북입니다.

<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/bonus/gpt-to-llama/gpt-and-all-llamas.webp">

&nbsp;
### `llms-from-scratch` 패키지를 통해 Llama 3.2 사용하기

Llama 3.2 1B 및 3B 모델을 쉽게 사용하는 방법으로, [pkg/llms_from_scratch](../../pkg/llms_from_scratch)의 이 저장소의 소스 코드를 기반으로 한 `llms-from-scratch` PyPI 패키지를 사용할 수도 있습니다.

&nbsp;
#### 1) 설치

```bash
pip install llms_from_scratch blobfile
```

(`blobfile`은 토크나이저를 로드하는 데 필요합니다.)

&nbsp;
#### 2) 모델 및 텍스트 생성 설정

사용할 모델을 지정합니다:

```python
MODEL_FILE = "llama3.2-1B-instruct.pth"
# MODEL_FILE = "llama3.2-1B-base.pth"
# MODEL_FILE = "llama3.2-3B-instruct.pth"
# MODEL_FILE = "llama3.2-3B-base.pth"
```

사용자가 정의할 수 있는 기본 텍스트 생성 설정입니다. 권장되는 8192 토큰 컨텍스트 크기는 텍스트 생성 예제에 대해 약 3GB의 VRAM이 필요합니다.

```python
# 텍스트 생성 설정
if "instruct" in MODEL_FILE:
    PROMPT = "What do llamas eat?"
else:
    PROMPT = "Llamas eat"

MAX_NEW_TOKENS = 150
TEMPERATURE = 0.
TOP_K = 1
```

&nbsp;
#### 3) 가중치 다운로드 및 로딩

이것은 위의 모델 선택에 따라 가중치 파일을 자동으로 다운로드합니다:

```python
import os
import urllib.request

url = f"https://huggingface.co/rasbt/llama-3.2-from-scratch/resolve/main/{MODEL_FILE}"

if not os.path.exists(MODEL_FILE):
    urllib.request.urlretrieve(url, MODEL_FILE)
    print(f"Downloaded to {MODEL_FILE}")
```

모델 가중치는 다음과 같이 로드됩니다:

```python
import torch
from llms_from_scratch.llama3 import Llama3Model

if "1B" in MODEL_FILE:
    from llms_from_scratch.llama3 import LLAMA32_CONFIG_1B as LLAMA32_CONFIG
elif "3B" in MODEL_FILE:
    from llms_from_scratch.llama3 import LLAMA32_CONFIG_3B as LLAMA32_CONFIG
else:
    raise ValueError("Incorrect model file name")

model = Llama3Model(LLAMA32_CONFIG)
model.load_state_dict(torch.load(MODEL_FILE, weights_only=True, map_location="cpu"))

device = (
    torch.device("cuda") if torch.cuda.is_available() else
    torch.device("mps") if torch.backends.mps.is_available() else
    torch.device("cpu")
)
model.to(device)
```

&nbsp;
#### 4) 토크나이저 초기화

다음 코드는 토크나이저를 다운로드하고 초기화합니다:

```python
from llms_from_scratch.llama3 import Llama3Tokenizer, ChatFormat, clean_text

TOKENIZER_FILE = "tokenizer.model"

url = f"https://huggingface.co/rasbt/llama-3.2-from-scratch/resolve/main/{TOKENIZER_FILE}"

if not os.path.exists(TOKENIZER_FILE):
    urllib.request.urlretrieve(url, TOKENIZER_FILE)
    print(f"Downloaded to {TOKENIZER_FILE}")
    
tokenizer = Llama3Tokenizer("tokenizer.model")

if "instruct" in MODEL_FILE:
    tokenizer = ChatFormat(tokenizer)
```

&nbsp;
#### 5) 텍스트 생성

마지막으로 다음 코드를 통해 텍스트를 생성할 수 있습니다:

```python
import time

from llms_from_scratch.ch05 import (
    generate,
    text_to_token_ids,
    token_ids_to_text
)

torch.manual_seed(123)

start = time.time()

token_ids = generate(
    model=model,
    idx=text_to_token_ids(PROMPT, tokenizer).to(device),
    max_new_tokens=MAX_NEW_TOKENS,
    context_size=LLAMA32_CONFIG["context_length"],
    top_k=TOP_K,
    temperature=TEMPERATURE
)

total_time = time.time() - start
print(f"Time: {total_time:.2f} sec")
print(f"{int(len(token_ids[0])/total_time)} tokens/sec")

if torch.cuda.is_available():
    max_mem_bytes = torch.cuda.max_memory_allocated()
    max_mem_gb = max_mem_bytes / (1024 ** 3)
    print(f"Max memory allocated: {max_mem_gb:.2f} GB")

output_text = token_ids_to_text(token_ids, tokenizer)

if "instruct" in MODEL_FILE:
    output_text = clean_text(output_text)

print("\n\nOutput text:\n\n", output_text)
```

Llama 3.2 1B Instruct 모델을 사용할 때, 출력은 아래와 같이 나타날 것입니다:

```
Time: 3.17 sec
50 tokens/sec
Max memory allocated: 2.91 GB


Output text:

 Llamas are herbivores, which means they primarily eat plants. Their diet consists mainly of:

1. Grasses: Llamas love to graze on various types of grasses, including tall grasses and grassy meadows.
2. Hay: Llamas also eat hay, which is a dry, compressed form of grass or other plants.
3. Alfalfa: Alfalfa is a legume that is commonly used as a hay substitute in llama feed.
4. Other plants: Llamas will also eat other plants, such as clover, dandelions, and wild grasses.

It's worth noting that the specific diet of llamas can vary depending on factors such as the breed,
```

&nbsp;
#### 프로 팁 1: FlashAttention으로 추론 속도 향상

`Llama3Model` 대신 `Llama3ModelFast`를 드롭인 교체로 사용할 수 있습니다. 자세한 정보는 [pkg/llms_from_scratch/llama3.py](../../pkg/llms_from_scratch/llama3.py) 코드를 검토하시기 바랍니다.

`Llama3ModelFast`는 `GroupedQueryAttention` 모듈에서 제가 처음부터 구현한 스케일드 닷 프로덕트 코드를 Ampere GPU 이상에서 `FlashAttention`을 사용하는 PyTorch의 `scaled_dot_product` 함수로 교체합니다.

다음 표는 A100에서의 성능 비교를 보여줍니다:

|                 | Tokens/sec | Memory  |
| --------------- | ---------- | ------- |
| Llama3Model     | 42         | 2.91 GB |
| Llama3ModelFast | 54         | 2.91 GB |

&nbsp;
#### 프로 팁 2: 컴파일로 추론 속도 향상

최대 4배 속도 향상을 위해 다음을 교체하세요:

```python
model.to(device)
```

다음으로:

```python
model = torch.compile(model)
model.to(device)
```

참고: 컴파일 시 상당한 몇 분간의 사전 비용이 있으며, 속도 향상은 첫 번째 `generate` 호출 후에 효과가 나타납니다.

다음 표는 연속적인 `generate` 호출에 대한 A100에서의 성능 비교를 보여줍니다:

|                 | Tokens/sec | Memory  |
| --------------- | ---------- | ------- |
| Llama3Model     | 170        | 3.12 GB |
| Llama3ModelFast | 177        | 3.61 GB |

&nbsp;
#### 프로 팁 3: KV 캐시로 추론 속도 향상

CPU에서 모델을 실행할 때 KV 캐시 `Llama3Model` 드롭인 교체를 사용하면 추론 성능을 크게 향상시킬 수 있습니다. (KV 캐시에 대해 더 자세히 알아보려면 제 [LLM에서 KV 캐시 이해하고 처음부터 코딩하기](https://magazine.sebastianraschka.com/p/coding-the-kv-cache-in-llms) 글을 참조하시기 바랍니다.)

```python
from llms_from_scratch.kv_cache.llama3 import Llama3Model
from llms_from_scratch.kv_cache.generate import generate_text_simple

model = Llama3Model(LLAMA32_CONFIG)
# ...
token_ids = generate_text_simple(
    model=model,
    idx=text_to_token_ids(PROMPT, tokenizer).to(device),
    max_new_tokens=MAX_NEW_TOKENS,
    context_size=LLAMA32_CONFIG["context_length"],
)
```

최고 메모리 사용량은 계산하기 더 쉽기 때문에 Nvidia CUDA 디바이스에 대해서만 나열됩니다. 하지만 다른 디바이스의 메모리 사용량은 유사한 정밀도 형식을 사용하므로 비슷할 것이며, KV 캐시 저장으로 인해 생성된 150 토큰 텍스트에 대해 더 낮은 메모리 사용량을 보입니다(하지만 다른 디바이스는 행렬 곱셈을 다르게 구현할 수 있고 다른 최고 메모리 요구 사항을 가질 수 있으며; 더 긴 컨텍스트 길이에서 KV 캐시 메모리가 금지적으로 증가할 수 있습니다).

| Model       | Mode              | Hardware        | Tokens/sec | GPU Memory (VRAM) |
| ----------- | ----------------- | --------------- | ---------- | ----------------- |
| Llama3Model | Regular           | Mac Mini M4 CPU | 1          | -                 |
| Llama3Model | Regular compiled  | Mac Mini M4 CPU | 1          | -                 |
| Llama3Model | KV cache          | Mac Mini M4 CPU | 68         | -                 |
| Llama3Model | KV cache compiled | Mac Mini M4 CPU | 86         | -                 |
|             |                   |                 |            |                   |
| Llama3Model | Regular           | Mac Mini M4 GPU | 15         | -                 |
| Llama3Model | Regular compiled  | Mac Mini M4 GPU | Error      | -                 |
| Llama3Model | KV cache          | Mac Mini M4 GPU | 62         | -                 |
| Llama3Model | KV cache compiled | Mac Mini M4 GPU | Error      | -                 |
|             |                   |                 |            |                   |
| Llama3Model | Regular           | Nvidia A100 GPU | 42         | 2.91 GB           |
| Llama3Model | Regular compiled  | Nvidia A100 GPU | 170        | 3.12 GB           |
| Llama3Model | KV cache          | Nvidia A100 GPU | 58         | 2.87 GB           |
| Llama3Model | KV cache compiled | Nvidia A100 GPU | 161        | 3.61 GB           |

위의 모든 설정은 동일한 텍스트 출력을 생성하는 것으로 테스트되었습니다.