# 보너스 자료: KV 캐시

**이 폴더는 GPT 모델에 KV 캐시를 추가하는 방법을 구현합니다.**

&nbsp;
## 개요

요약하면, KV 캐시는 추론 중 재사용을 위해 중간 키(K) 및 값(V) 계산을 저장하여, 응답을 생성할 때 상당한 속도 향상을 제공합니다. 단점은 코드에 복잡성을 추가하고, 메모리 사용량을 증가시키며, 훈련 중에는 사용할 수 없다는 것입니다. 하지만 LLM을 배포할 때는 코드 복잡성과 메모리 면에서의 절충점을 고려해도 추론 속도 향상이 충분히 가치 있는 경우가 많습니다.

&nbsp;
## 작동 방식

LLM이 어떤 텍스트를 생성한다고 상상해보세요. 구체적으로 LLM이 다음과 같은 프롬프트를 받았다고 가정해보겠습니다: "Time flies".

아래 그림은 키(key)와 값(value) 벡터가 강조된 3장의 수정된 그래픽을 사용하여 기본적인 어텐션 점수 계산의 일부를 보여줍니다:

<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/bonus/kv-cache/kv-cache-attn-1.png?3" width=800>

이제, 2장과 4장에서 배운 바와 같이, LLM은 한 번에 하나의 단어(또는 토큰)을 생성합니다. LLM이 "fast"라는 단어를 생성하여 다음 라운드의 프롬프트가 "Time flies fast"가 되었다고 가정해보겠습니다. 이는 다음 그림에서 설명됩니다:

<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/bonus/kv-cache/kv-cache-attn-2.png?3" width=800>

이전 두 그림을 비교해보면, 처음 두 토큰에 대한 키와 값 벡터는 정확히 동일하며, 각 다음 토큰 텍스트 생성 라운드에서 이들을 다시 계산하는 것은 낭비적입니다.

따라서 KV 캐시의 아이디어는 이전에 생성된 키와 값 벡터를 재사용을 위해 저장하는 캐싱 메커니즘을 구현하여 불필요한 재계산을 피하는 것입니다.

&nbsp;

## KV 캐시 구현

KV 캐시를 구현하는 방법은 여러 가지가 있으며, 주요 아이디어는 각 생성 단계에서 새로 생성된 토큰에 대해서만 키와 값 텐서를 계산한다는 것입니다.

저는 코드 가독성을 강조하는 간단한 방법을 선택했습니다. 구현 방법을 보려면 코드 변경 사항을 훑어보는 것이 가장 쉽다고 생각합니다.

이 폴더에는 두 개의 파일이 있습니다:

1. [`gpt_ch04.py`](gpt_ch04.py): 3장과 4장에서 가져온 독립적인 코드로 LLM을 구현하고 간단한 텍스트 생성 함수를 실행합니다
2. [`gpt_with_kv_cache.py`](gpt_with_kv_cache.py): 위와 동일하지만 KV 캐시를 구현하기 위한 필요한 변경사항이 적용되었습니다.

다음 중 하나를 선택할 수 있습니다:

a. [`gpt_with_kv_cache.py`](gpt_with_kv_cache.py) 파일을 열고 새로운 변경사항을 표시하는 `# NEW` 섹션을 찾아보세요:

<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/bonus/kv-cache/new-sections.png?3" width=800>

b. 원하는 파일 비교 도구를 통해 두 코드 파일의 변경사항을 확인하세요:

<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/bonus/kv-cache/file-diff.png?3" width=800>

구현 세부사항을 요약하면, 다음과 같은 간단한 설명입니다.

&nbsp;

### 1. 캐시 버퍼 등록

`MultiHeadAttention` 생성자 내부에서 단계별로 연결된 키와 값을 보관할 두 개의 비영구적 버퍼인 `cache_k`와 `cache_v`를 추가합니다:

```python
self.register_buffer("cache_k", None, persistent=False)
self.register_buffer("cache_v", None, persistent=False)
```

&nbsp;

### 2. `use_cache` 플래그가 있는 순전파

다음으로, `MultiHeadAttention` 클래스의 `forward` 메서드를 `use_cache` 인수를 받도록 확장합니다. 새로운 토큰 덩어리를 `keys_new`, `values_new`, `queries`로 프로젝션한 후, kv 캐시를 초기화하거나 캐시에 추가합니다:

```python
def forward(self, x, use_cache=False):
    b, num_tokens, d_in = x.shape

    keys_new = self.W_key(x)  # Shape: (b, num_tokens, d_out)
    values_new = self.W_value(x)
    queries = self.W_query(x)
    #...

    if use_cache:
        if self.cache_k is None:
            self.cache_k, self.cache_v = keys_new, values_new
        else:
            self.cache_k = torch.cat([self.cache_k, keys_new], dim=1)
            self.cache_v = torch.cat([self.cache_v, values_new], dim=1)
        keys, values = self.cache_k, self.cache_v
    else:
        keys, values = keys_new, values_new
        
    # ...
    
    num_tokens_Q = queries.shape[-2]
    num_tokens_K = keys.shape[-2]
    if use_cache:
        mask_bool = self.mask.bool()[
            self.ptr_current_pos:self.ptr_current_pos + num_tokens_Q, :num_tokens_K
        ]
        self.ptr_current_pos += num_tokens_Q
    else:
        mask_bool = self.mask.bool()[:num_tokens_Q, :num_tokens_K]
```

&nbsp;

### 3. 캐시 지우기

텍스트를 생성할 때, 독립적인 시퀀스들 사이에 (예를 들어 텍스트 생성 호출 간에) 두 버퍼를 모두 재설정해야 하므로, `MultiHeadAttention` 클래스에 캐시 재설정 메서드도 추가합니다:

```python
def reset_cache(self):
    self.cache_k, self.cache_v = None, None
    self.ptr_current_pos = 0
```

&nbsp;

### 4. 전체 모델에서 `use_cache` 전파

`MultiHeadAttention` 클래스의 변경사항이 완료되면, 이제 `GPTModel` 클래스를 수정합니다. 먼저 생성자에 토큰 인덱스를 위한 위치 추적을 추가합니다:

```python
self.current_pos = 0
```

그런 다음, 한 줄짜리 블록 호출을 명시적인 루프로 교체하여 각 트랜스포머 블록을 통해 `use_cache`를 전달합니다:

```python
def forward(self, in_idx, use_cache=False):
    # ...
 
    if use_cache:
        pos_ids = torch.arange(
            self.current_pos, self.current_pos + seq_len,            
            device=in_idx.device, dtype=torch.long
        )
        self.current_pos += seq_len
    else:
        pos_ids = torch.arange(
            0, seq_len, device=in_idx.device, dtype=torch.long
        )
    
    pos_embeds = self.pos_emb(pos_ids).unsqueeze(0)
    x = tok_embeds + pos_embeds
    # ...
    for blk in self.trf_blocks:
        x = blk(x, use_cache=use_cache)
```

위의 변경사항은 `TransformerBlock` 클래스가 `use_cache` 인수를 받도록 하는 작은 수정도 필요합니다:
```python
    def forward(self, x, use_cache=False):
        # ...
        self.att(x, use_cache=use_cache)
```

마지막으로, 편의를 위해 모든 블록 캐시를 한 번에 지우는 모델 수준의 재설정을 `GPTModel`에 추가합니다:

```python
def reset_kv_cache(self):
    for blk in self.trf_blocks:
        blk.att.reset_cache()
    self.current_pos = 0
```

&nbsp;

### 5. 생성에서 캐시 사용

`GPTModel`, `TransformerBlock`, `MultiHeadAttention`에 대한 변경사항이 완료되면, 마지막으로 간단한 텍스트 생성 함수에서 KV 캐시를 사용하는 방법은 다음과 같습니다:

```python
def generate_text_simple_cached(model, idx, max_new_tokens, 
                                context_size=None, use_cache=True):
    model.eval()
    ctx_len = context_size or model.pos_emb.num_embeddings

    with torch.no_grad():
        if use_cache:
            # 전체 프롬프트로 캐시 초기화
            model.reset_kv_cache()
            logits = model(idx[:, -ctx_len:], use_cache=True)

            for _ in range(max_new_tokens):
                # a) 가장 높은 로그 확률을 가진 토큰 선택 (탐욕적 샘플링)
                next_idx = logits[:, -1].argmax(dim=-1, keepdim=True)
                # b) 실행 중인 시퀀스에 추가
                idx = torch.cat([idx, next_idx], dim=1)
                # c) 모델에 새 토큰만 공급
                logits = model(next_idx, use_cache=True)
        else:
            for _ in range(max_new_tokens):
                logits = model(idx[:, -ctx_len:], use_cache=False)
                next_idx = logits[:, -1].argmax(dim=-1, keepdim=True)
                idx = torch.cat([idx, next_idx], dim=1)

    return idx
```

c)에서 `logits = model(next_idx, use_cache=True)`를 통해 모델에 새 토큰만 공급한다는 점을 주목하세요. 캐싱 없이는 저장된 키와 값을 재사용할 수 없으므로 `logits = model(idx[:, -ctx_len:], use_cache=False)`와 같이 모델에 전체 입력을 공급합니다.

&nbsp;

## 간단한 성능 비교

개념적 수준에서 KV 캐시를 다룬 후, 중요한 질문은 실제로 작은 예제에서 얼마나 잘 작동하는지입니다. 구현을 시험해보기 위해 앞서 언급한 두 코드 파일을 Python 스크립트로 실행할 수 있습니다. 이 스크립트들은 작은 124M 파라미터 LLM을 실행하여 200개의 새로운 토큰을 생성합니다 (시작할 때 4토큰 프롬프트 "Hello, I am"이 주어집니다):

```bash
pip install -r https://raw.githubusercontent.com/rasbt/LLMs-from-scratch/refs/heads/main/requirements.txt

python gpt_ch04.py

python gpt_with_kv_cache.py
```

M4 칩이 탑재된 Mac Mini(CPU)에서의 결과는 다음과 같습니다:

|                        | Tokens/sec |
| ---------------------- | ---------- |
| `gpt_ch04.py`          | 27         |
| `gpt_with_kv_cache.py` | 144        |

보시다시피, 작은 124M 파라미터 모델과 짧은 200토큰 시퀀스 길이로도 이미 약 5배의 속도 향상을 얻을 수 있습니다. (이 구현은 코드 가독성을 위해 최적화되었으며 CUDA나 MPS 런타임 속도를 위해 최적화되지 않았습니다. 이를 위해서는 텐서를 재생성하고 연결하는 대신 미리 할당하는 것이 필요합니다.)

**참고:** 두 경우 모두 모델은 다음과 같은 "의미 없는" 텍스트를 생성합니다:

> Output text: Hello, I am Featureiman Byeswickattribute argue logger Normandy Compton analogous bore ITVEGIN ministriesysics Kle functional recountrictionchangingVirgin embarrassedgl ...

이는 모델을 아직 훈련하지 않았기 때문입니다. 다음 장에서는 모델을 훈련하고, 훈련된 모델에서 KV 캐시를 사용하여 (하지만 KV 캐시는 추론 중에만 사용됩니다) 일관된 텍스트를 생성할 수 있습니다. 여기서는 코드를 더 간단하게 유지하기 위해 훈련되지 않은 모델을 사용합니다.

하지만 더 중요한 것은 `gpt_ch04.py`와 `gpt_with_kv_cache.py` 구현이 정확히 동일한 텍스트를 생성한다는 것입니다. 이는 KV 캐시가 올바르게 구현되었음을 알려줍니다 -- 다른 결과를 초래할 수 있는 인덱싱 실수를 하기 쉽습니다.

&nbsp;

## KV 캐시의 장단점

시퀀스 길이가 증가함에 따라 KV 캐시의 장점과 단점이 다음과 같은 방식으로 더욱 두드러집니다:

- [좋음] **계산 효율성 증가**: 캐싱 없이는 단계 *t*에서의 어텐션이 새로운 쿼리를 *t*개의 이전 키와 비교해야 하므로 누적 작업이 제곱적으로, O(n²)로 확장됩니다. 캐시를 사용하면 각 키와 값이 한 번 계산되고 재사용되어 단계당 총 복잡도가 선형인 O(n)으로 줄어듭니다.

- [나쁨] **메모리 사용량이 선형적으로 증가**: 각 새로운 토큰이 KV 캐시에 추가됩니다. 긴 시퀀스와 큰 LLM의 경우 누적 KV 캐시가 커져서 상당한 양의 (GPU) 메모리를 소모하거나 심지어 금지적일 수도 있습니다. 해결책으로 KV 캐시를 잘라낼 수 있지만, 이는 더 많은 복잡성을 추가합니다 (하지만 다시, LLM을 배포할 때는 그만한 가치가 있을 수 있습니다.)

&nbsp;
## KV 캐시 구현 최적화

위의 KV 캐시에 대한 개념적 구현은 명확성에 도움이 되며 주로 코드 가독성과 교육 목적에 맞춰져 있지만, 실제 시나리오(특히 더 큰 모델과 더 긴 시퀀스 길이)에서 배포하려면 더 세심한 최적화가 필요합니다.

&nbsp;
### 캐시 확장 시 일반적인 함정

- **메모리 단편화와 반복적인 할당**: 앞서 보여준 것처럼 `torch.cat`을 통해 텐서를 계속 연결하면 빈번한 메모리 할당과 재할당으로 인해 성능 병목현상이 발생합니다.

- **메모리 사용량의 선형 증가**: 적절한 처리 없이는 KV 캐시 크기가 매우 긴 시퀀스에 대해 비실용적이 됩니다.

&nbsp;
#### 팁 1: 메모리 미리 할당

텐서를 반복적으로 연결하는 대신, 예상되는 최대 시퀀스 길이에 기반하여 충분히 큰 텐서를 미리 할당할 수 있습니다. 이렇게 하면 일관된 메모리 사용을 보장하고 오버헤드를 줄입니다. 의사 코드로는 다음과 같습니다:

```python
# 키와 값에 대한 미리 할당 예시
max_seq_len = 1024  # 예상되는 최대 시퀀스 길이
cache_k = torch.zeros((batch_size, num_heads, max_seq_len, head_dim), device=device)
cache_v = torch.zeros((batch_size, num_heads, max_seq_len, head_dim), device=device)
```

추론 중에는 이러한 미리 할당된 텐서의 슬라이스에 간단히 쓸 수 있습니다.

&nbsp;
#### 팁 2: 슬라이딩 윈도우를 통한 캐시 절단

GPU 메모리 폭발을 방지하기 위해 동적 절단과 함께 슬라이딩 윈도우 접근법을 구현할 수 있습니다. 슬라이딩 윈도우를 통해 캐시에서 마지막 `window_size` 토큰만 유지합니다:

```python
# 슬라이딩 윈도우 캐시 구현
window_size = 512
cache_k = cache_k[:, :, -window_size:, :]
cache_v = cache_v[:, :, -window_size:, :]
```

&nbsp;
#### 실제 최적화

이러한 최적화는 [`gpt_with_kv_cache_optimized.py`](gpt_with_kv_cache_optimized.py) 파일에서 찾을 수 있습니다.

M4 칩이 탑재된 Mac Mini(CPU)에서 200토큰 생성과 컨텍스트 길이와 같은 윈도우 크기(동일한 결과를 보장하기 위해)로 아래 코드 실행 시간을 비교하면 다음과 같습니다:

|                                  | Tokens/sec |
| -------------------------------- | ---------- |
| `gpt_ch04.py`                    | 27         |
| `gpt_with_kv_cache.py`           | 144        |
| `gpt_with_kv_cache_optimized.py` | 166        |

안타깝게도 이것이 작은 모델이므로 CUDA 기기에서는 속도 이점이 사라지며, 기기 전송과 통신이 이 작은 모델에 대한 KV 캐시의 이점을 능가합니다.

&nbsp;
## 추가 자료

1. [Qwen3 from-scratch KV 캐시 벤치마크](../../ch05/11_qwen3#pro-tip-2-speed-up-inference-with-compilation)
2. [Llama 3 from-scratch KV 캐시 벤치마크](../../ch05/07_gpt_to_llama/README.md#pro-tip-3-speed-up-inference-with-compilation)
3. [Understanding and Coding the KV Cache in LLMs from Scratch](https://magazine.sebastianraschka.com/p/coding-the-kv-cache-in-llms) -- 이 README의 더 자세한 설명