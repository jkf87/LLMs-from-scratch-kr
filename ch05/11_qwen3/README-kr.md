# Qwen3 처음부터 구현

이 폴더의 [standalone-qwen3.ipynb](standalone-qwen3.ipynb) Jupyter 노트북은 Qwen3 0.6B, 1.7B, 4B, 8B, 32B의 처음부터 구현을 포함합니다.

<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/bonus/qwen/qwen-overview.webp">

이 폴더의 [standalone-qwen3-moe.ipynb](standalone-qwen3-moe.ipynb)와 [standalone-qwen3-moe-plus-kvcache.ipynb](standalone-qwen3-moe-plus-kvcache.ipynb) Jupyter 노트북은 Thinking, Instruct, Coder 모델 변형을 포함한 30B-A3B Mixture-of-Experts (MoE)의 처음부터 구현을 포함합니다.

<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/bonus/qwen/qwen3-coder-flash-overview.webp?123" width="430px">

&nbsp;
# `llms-from-scratch` 패키지를 통해 Qwen3 사용하기

Qwen3 처음부터 구현을 쉽게 사용하는 방법으로, [pkg/llms_from_scratch](../../pkg/llms_from_scratch)의 이 저장소의 소스 코드를 기반으로 한 `llms-from-scratch` PyPI 패키지를 사용할 수도 있습니다.

&nbsp;
#### 1) 설치

```bash
pip install llms_from_scratch tokenizers
```

&nbsp;
#### 2) 모델 및 텍스트 생성 설정

사용할 모델을 지정합니다:

```python
USE_REASONING_MODEL = False  # 기본 모델
USE_REASONING_MODEL = True   # "thinking" 모델

# Qwen3 Coder Flash 모델에도
# USE_REASONING_MODEL = True
# 사용
```

사용자가 정의할 수 있는 기본 텍스트 생성 설정입니다. 150 토큰으로 0.6B 모델은 약 1.5GB 메모리가 필요합니다.

```python
MAX_NEW_TOKENS = 150
TEMPERATURE = 0.
TOP_K = 1
```

&nbsp;
#### 3a) 0.6B 모델의 가중치 다운로드 및 로딩

다음은 위의 모델 선택(추론 또는 기본)에 따라 가중치 파일을 자동으로 다운로드합니다. 이 섹션은 0.6B 모델에 중점을 둡니다. 더 큰 모델(1.7B, 4B, 8B, 또는 32B) 중 하나로 작업하려면 이 섹션을 건너뛰고 3b) 섹션을 계속하시기 바랍니다.

```python
from llms_from_scratch.qwen3 import download_from_huggingface

repo_id = "rasbt/qwen3-from-scratch"

if USE_REASONING_MODEL:
    filename = "qwen3-0.6B.pth"
    local_dir = "Qwen3-0.6B"    
else:
    filename = "qwen3-0.6B-base.pth"   
    local_dir = "Qwen3-0.6B-Base"

download_from_huggingface(
    repo_id=repo_id,
    filename=filename,
    local_dir=local_dir
)
```

모델 가중치는 다음과 같이 로드됩니다:

```python
from pathlib import Path
import torch

from llms_from_scratch.qwen3 import Qwen3Model, QWEN_CONFIG_06_B

model_file = Path(local_dir) / filename

model = Qwen3Model(QWEN_CONFIG_06_B)
model.load_state_dict(torch.load(model_file, weights_only=True, map_location="cpu"))

device = (
    torch.device("cuda") if torch.cuda.is_available() else
    torch.device("mps") if torch.backends.mps.is_available() else
    torch.device("cpu")
)
model.to(device);
```

&nbsp;
#### 3b) 더 큰 Qwen 모델의 가중치 다운로드 및 로딩

1.7B, 4B, 8B, 또는 32B와 같은 더 큰 Qwen 모델 중 하나로 작업하는 데 관심이 있다면, 추가 코드 의존성이 필요한 3a) 하의 코드 대신 다음 코드를 사용하시기 바랍니다:

```bash
pip install safetensors huggingface_hub
```

그런 다음 다음 코드를 사용하세요(원하는 모델 크기를 선택하기 위해 `USE_MODEL`을 적절히 변경):

```python
USE_MODEL = "1.7B"

if USE_MODEL == "1.7B":
    from llms_from_scratch.qwen3 import QWEN3_CONFIG_1_7B as QWEN3_CONFIG
elif USE_MODEL == "4B":
    from llms_from_scratch.qwen3 import QWEN3_CONFIG_4B as QWEN3_CONFIG
elif USE_MODEL == "8B":
    from llms_from_scratch.qwen3 import QWEN3_CONFIG_8B as QWEN3_CONFIG
elif USE_MODEL == "14B":
    from llms_from_scratch.qwen3 import QWEN3_CONFIG_14B as QWEN3_CONFIG
elif USE_MODEL == "32B":
    from llms_from_scratch.qwen3 import QWEN3_CONFIG_32B as QWEN3_CONFIG
elif USE_MODEL == "30B-A3B":
    from llms_from_scratch.qwen3 import QWEN3_CONFIG_30B_A3B as QWEN3_CONFIG
else:
    raise ValueError("Invalid USE_MODEL name.")
    
repo_id = f"Qwen/Qwen3-{USE_MODEL}"
local_dir = f"Qwen3-{USE_MODEL}"

if not USE_REASONING_MODEL:
  repo_id = f"{repo_id}-Base"
  local_dir = f"{local_dir}-Base"
```

이제 가중치를 다운로드하고 `model`에 로드합니다:

```python
from llms_from_scratch.qwen3 import (
    Qwen3Model,
    download_from_huggingface_from_snapshots,
    load_weights_into_qwen
)

device = (
    torch.device("cuda") if torch.cuda.is_available() else
    torch.device("mps") if torch.backends.mps.is_available() else
    torch.device("cpu")
)

with device:
    model = Qwen3Model(QWEN3_CONFIG)

weights_dict = download_from_huggingface_from_snapshots(
    repo_id=repo_id,
    local_dir=local_dir
)
load_weights_into_qwen(model, QWEN3_CONFIG, weights_dict)
model.to(device)  # MoE 모델에만 필요
del weights_dict  # 디스크 공간을 확보하기 위해 가중치 딕셔너리 삭제
```

&nbsp;

#### 4) 토크나이저 초기화

다음 코드는 토크나이저를 다운로드하고 초기화합니다:

```python
from llms_from_scratch.qwen3 import Qwen3Tokenizer

if USE_REASONING_MODEL:
    tok_filename = "tokenizer.json"    
else:
    tok_filename = "tokenizer-base.json"   

tokenizer = Qwen3Tokenizer(
    tokenizer_file_path=tok_filename,
    repo_id=repo_id,
    add_generation_prompt=USE_REASONING_MODEL,
    add_thinking=USE_REASONING_MODEL
)
```

&nbsp;

#### 5) 텍스트 생성

마지막으로 다음 코드를 통해 텍스트를 생성할 수 있습니다:

```python
prompt = "Give me a short introduction to large language models."
input_token_ids = tokenizer.encode(prompt)
```

```python
from llms_from_scratch.ch05 import generate
import time

torch.manual_seed(123)

start = time.time()

output_token_ids = generate(
    model=model,
    idx=torch.tensor(input_token_ids, device=device).unsqueeze(0),
    max_new_tokens=150,
    context_size=QWEN_CONFIG_06_B["context_length"],
    top_k=1,
    temperature=0.
)

total_time = time.time() - start
print(f"Time: {total_time:.2f} sec")
print(f"{int(len(output_token_ids[0])/total_time)} tokens/sec")

if torch.cuda.is_available():
    max_mem_bytes = torch.cuda.max_memory_allocated()
    max_mem_gb = max_mem_bytes / (1024 ** 3)
    print(f"Max memory allocated: {max_mem_gb:.2f} GB")

output_text = tokenizer.decode(output_token_ids.squeeze(0).tolist())

print("\n\nOutput text:\n\n", output_text + "...")
```

Qwen3 0.6B 추론 모델을 사용할 때, 출력은 아래와 같이 나타날 것입니다(A100에서 실행):

```
Time: 6.35 sec
25 tokens/sec
Max memory allocated: 1.49 GB


Output text:

 <|im_start|>user
Give me a short introduction to large language models.<|im_end|>
Large language models (LLMs) are advanced artificial intelligence systems designed to generate human-like text. They are trained on vast amounts of text data, allowing them to understand and generate coherent, contextually relevant responses. LLMs are used in a variety of applications, including chatbots, virtual assistants, content generation, and more. They are powered by deep learning algorithms and can be fine-tuned for specific tasks, making them versatile tools for a wide range of industries.<|endoftext|>Human resources department of a company is planning to hire 100 new employees. The company has a budget of $100,000 for the recruitment process. The company has a minimum wage of $10 per hour. The company has a total of...
```

더 큰 모델의 경우, 각 토큰이 생성되는 즉시 출력하는 스트리밍 변형을 선호할 수 있습니다:

```python
from llms_from_scratch.generate import generate_text_simple_stream

input_token_ids_tensor = torch.tensor(input_token_ids, device=device).unsqueeze(0)

for token in generate_text_simple_stream(
    model=model,
    token_ids=input_token_ids_tensor,
    max_new_tokens=150,
    eos_token_id=tokenizer.eos_token_id
):
    token_id = token.squeeze(0).tolist()
    print(
        tokenizer.decode(token_id),
        end="",
        flush=True
    )
```

```
 <|im_start|>user
Give me a short introduction to large language models.<|im_end|>
Large language models (LLMs) are advanced artificial intelligence systems designed to generate human-like text. They are trained on vast amounts of text data, allowing them to understand and generate coherent, contextually relevant responses. LLMs are used in a variety of applications, including chatbots, virtual assistants, content generation, and more. They are powered by deep learning algorithms and can be fine-tuned for specific tasks, making them versatile tools for a wide range of industries.<|endoftext|>Human resources department of a company is planning to hire 100 new employees. The company has a budget of $100,000 for the recruitment process. The company has a minimum wage of $10 per hour. The company has a total of...
```

&nbsp;

#### 프로 팁 1: 컴파일로 추론 속도 향상

최대 4배 속도 향상을 위해 다음을 교체하세요:

```python
model.to(device)
```

다음으로:

```python
model.to(device)
model = torch.compile(model)
```

참고: 컴파일 시 상당한 몇 분간의 사전 비용이 있으며, 속도 향상은 첫 번째 `generate` 호출 후에 효과가 나타납니다.

다음 표는 연속적인 `generate` 호출에 대한 A100에서의 성능 비교를 보여줍니다:

|                          | Hardware        | Tokens/sec | Memory   |
| ------------------------ | ----------------|----------- | -------- |
| Qwen3Model 0.6B          | Nvidia A100 GPU | 25         | 1.49 GB  |
| Qwen3Model 0.6B compiled | Nvidia A100 GPU | 107        | 1.99 GB  |

&nbsp;
#### 프로 팁 2: KV 캐시로 추론 속도 향상

CPU에서 모델을 실행할 때 KV 캐시 `Qwen3Model` 드롭인 교체를 사용하면 추론 성능을 크게 향상시킬 수 있습니다. (KV 캐시에 대해 더 자세히 알아보려면 제 [LLM에서 KV 캐시 이해하고 처음부터 코딩하기](https://magazine.sebastianraschka.com/p/coding-the-kv-cache-in-llms) 글을 참조하시기 바랍니다.)

```python
from llms_from_scratch.kv_cache.qwen3 import Qwen3Model
from llms_from_scratch.kv_cache.generate import generate_text_simple

model = Qwen3Model(QWEN_CONFIG_06_B)
# ...
token_ids = generate_text_simple(
    model=model,
    idx=text_to_token_ids(PROMPT, tokenizer).to(device),
    max_new_tokens=MAX_NEW_TOKENS,
    context_size=QWEN_CONFIG_06_B["context_length"],
)
```

최고 메모리 사용량은 계산하기 더 쉽기 때문에 Nvidia CUDA 디바이스에 대해서만 나열됩니다. 하지만 다른 디바이스의 메모리 사용량은 유사한 정밀도 형식을 사용하므로 비슷할 것이며, KV 캐시 저장으로 인해 생성된 150 토큰 텍스트에 대해 더 낮은 메모리 사용량을 보입니다(하지만 다른 디바이스는 행렬 곱셈을 다르게 구현할 수 있고 다른 최고 메모리 요구 사항을 가질 수 있으며; 더 긴 컨텍스트 길이에서 KV 캐시 메모리가 금지적으로 증가할 수 있습니다).

| Model           | Mode              | Hardware        | Tokens/sec | GPU Memory (VRAM) |
| --------------- | ----------------- | --------------- | ---------- | ----------------- |
| Qwen3Model 0.6B | Regular           | Mac Mini M4 CPU | 1          | -                 |
| Qwen3Model 0.6B | Regular compiled  | Mac Mini M4 CPU | 1          | -                 |
| Qwen3Model 0.6B | KV cache          | Mac Mini M4 CPU | 80         | -                 |
| Qwen3Model 0.6B | KV cache compiled | Mac Mini M4 CPU | 137        | -                 |
|                 |                   |                 |            |                   |
| Qwen3Model 0.6B | Regular           | Mac Mini M4 GPU | 21         | -                 |
| Qwen3Model 0.6B | Regular compiled  | Mac Mini M4 GPU | Error      | -                 |
| Qwen3Model 0.6B | KV cache          | Mac Mini M4 GPU | 28         | -                 |
| Qwen3Model 0.6B | KV cache compiled | Mac Mini M4 GPU | Error      | -                 |
|                 |                   |                 |            |                   |
| Qwen3Model 0.6B | Regular           | Nvidia A100 GPU | 26         | 1.49 GB           |
| Qwen3Model 0.6B | Regular compiled  | Nvidia A100 GPU | 107        | 1.99 GB           |
| Qwen3Model 0.6B | KV cache          | Nvidia A100 GPU | 25         | 1.47 GB           |
| Qwen3Model 0.6B | KV cache compiled | Nvidia A100 GPU | 90         | 1.48 GB           |

위의 모든 설정은 동일한 텍스트 출력을 생성하는 것으로 테스트되었습니다.

&nbsp;

#### 프로 팁 3: 배치 추론

배치 추론을 통해 처리량을 더욱 증가시킬 수 있습니다. 이제 더 많은 수의 입력 시퀀스로 추론을 실행하므로 일대일 비교는 아니지만, 메모리 사용량 증가와 교환하여 초당 토큰 처리량을 증가시킵니다.

이는 프롬프트 준비에 대한 작은 코드 수정만 필요합니다. 예를 들어, 아래의 배치 프롬프트를 고려해보세요:

```python
from llms_from_scratch.ch04 import generate_text_simple
from llms_from_scratch.qwen3 import Qwen3Model, QWEN_CONFIG_06_B
# ...

prompts = [
    "Give me a short introduction to neural networks.",
    "Give me a short introduction to machine learning.",
    "Give me a short introduction to deep learning models.",
    "Give me a short introduction to natural language processing.",
    "Give me a short introduction to generative AI systems.",
    "Give me a short introduction to transformer architectures.",
    "Give me a short introduction to supervised learning methods.",
    "Give me a short introduction to unsupervised learning.",
]

tokenized_prompts = [tokenizer.encode(p) for p in prompts]
max_len = max(len(t) for t in tokenized_prompts)
padded_token_ids = [
    t + [tokenizer.pad_token_id] * (max_len - len(t)) for t in tokenized_prompts
]
input_tensor = torch.tensor(padded_token_ids).to(device)

output_token_ids = generate_text_simple(
    model=model,
    idx=input_tensor,
    max_new_tokens=150,
    context_size=QWEN_CONFIG_06_B["context_length"],
)
```

KV 캐시 버전의 코드는 비슷하지만, 다음 드롭인 교체를 사용해야 합니다:

```python
from llms_from_scratch.kv_cache_batched.generate import generate_text_simple
from llms_from_scratch.kv_cache_batched.qwen3 import Qwen3Model
```

아래 실험은 배치 크기 8로 실행됩니다.

| Model            | Mode              | Hardware        | Batch size | Tokens/sec | GPU Memory (VRAM) |
| ---------------- | ----------------- | --------------- | ---------- | ---------- | ----------------- |
| Qwen3Model  0.6B | Regular           | Mac Mini M4 CPU | 8          | 2          | -                 |
| Qwen3Model 0.6B  | Regular compiled  | Mac Mini M4 CPU | 8          | -          | -                 |
| Qwen3Model 0.6B  | KV cache          | Mac Mini M4 CPU | 8          | 92         | -                 |
| Qwen3Model 0.6B  | KV cache compiled | Mac Mini M4 CPU | 8          | 128        | -                 |
|                  |                   |                 |            |            |                   |
| Qwen3Model 0.6B  | Regular           | Mac Mini M4 GPU | 8          | 36         | -                 |
| Qwen3Model 0.6B  | Regular compiled  | Mac Mini M4 GPU | 8          | -          | -                 |
| Qwen3Model 0.6B  | KV cache          | Mac Mini M4 GPU | 8          | 61         | -                 |
| Qwen3Model 0.6B  | KV cache compiled | Mac Mini M4 GPU | 8          | -          | -                 |
|                  |                   |                 |            |            |                   |
| Qwen3Model 0.6B  | Regular           | Nvidia A100 GPU | 8          | 184        | 2.19 GB           |
| Qwen3Model 0.6B  | Regular compiled  | Nvidia A100 GPU | 8          | 351        | 2.19 GB           |
| Qwen3Model 0.6B  | KV cache          | Nvidia A100 GPU | 8          | 140        | 3.13 GB           |
| Qwen3Model 0.6B  | KV cache compiled | Nvidia A100 GPU | 8          | 280        | 1.75 GB           |