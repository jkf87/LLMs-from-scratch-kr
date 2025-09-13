# 더 효율적인 멀티헤드 어텐션 구현

- [mha-implementations.ipynb](mha-implementations.ipynb)는 멀티헤드 어텐션의 다양한 구현을 포함하고 비교합니다

### 요약

아래 그림들은 성능 벤치마크 결과를 요약합니다 (낮을수록 좋음).

&nbsp;
#### 순전파만

<a href="mha-implementations.ipynb"><img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/bonus/mha-benchmark/1_forward-only.webp?1" width="500px"></a>

&nbsp;
#### 순전파와 역전파

<a href="mha-implementations.ipynb"><img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/bonus/mha-benchmark/2_forward-and-backward.webp?1" width="500px"></a>

&nbsp;
#### 컴파일 후 순전파와 역전파

<a href="mha-implementations.ipynb"><img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/bonus/mha-benchmark/3_forward-and-backward-compiled.webp?1" width="500px"></a>