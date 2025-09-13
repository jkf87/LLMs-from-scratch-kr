# 처음부터 구축하는 대형 언어 모델(Build a Large Language Model From Scratch)
[English](README.md) | [한국어](README-kr.md)

이 저장소는 GPT와 유사한 LLM을 개발, 사전 학습, 미세 조정하는 코드를 포함하며, [처음부터 구축하는 대형 언어 모델(Build a Large Language Model From Scratch)](https://amzn.to/4fqvn0D) 책의 공식 코드 저장소입니다.

<br>
<br>

<a href="https://amzn.to/4fqvn0D"><img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/cover.jpg?123" width="250px"></a>

<br>

[*처음부터 구축하는 대형 언어 모델(Build a Large Language Model From Scratch)*](http://mng.bz/orYv)에서는 대형 언어 모델(LLM)이 내부적으로 어떻게 작동하는지 배우고 이해할 수 있습니다. 단계별로 코드를 처음부터 작성하여 학습합니다. 이 책에서는 명확한 텍스트, 다이어그램, 예제를 통해 각 단계를 설명하면서 자신만의 LLM을 만드는 방법을 안내합니다.

이 책에서 설명하는 교육 목적의 소규모이지만 기능적인 모델을 훈련하고 개발하는 방법은 ChatGPT와 같은 대규모 기반 모델을 만드는 데 사용되는 접근 방식과 유사합니다. 또한 이 책에는 미세 조정을 위해 더 큰 사전 훈련된 모델의 가중치를 로드하는 코드도 포함되어 있습니다.

- 공식 [소스 코드 저장소](https://github.com/rasbt/LLMs-from-scratch) 링크
- [Manning(출판사 웹사이트)의 책 링크](http://mng.bz/orYv)
- [Amazon.com의 책 페이지 링크](https://www.amazon.com/gp/product/1633437167)
- ISBN 9781633437166

<a href="http://mng.bz/orYv#reviews"><img src="https://sebastianraschka.com//images/LLMs-from-scratch-images/other/reviews.png" width="220px"></a>


<br>
<br>

이 저장소의 사본을 다운로드하려면 [Download ZIP](https://github.com/rasbt/LLMs-from-scratch/archive/refs/heads/main.zip) 버튼을 클릭하거나 터미널에서 다음 명령을 실행하세요:

```bash
git clone --depth 1 https://github.com/rasbt/LLMs-from-scratch.git
```

<br>

(Manning 웹사이트에서 코드 번들을 다운로드한 경우, 최신 업데이트를 위해 GitHub의 공식 코드 저장소 [https://github.com/rasbt/LLMs-from-scratch](https://github.com/rasbt/LLMs-from-scratch)를 방문해 주시기 바랍니다.)

<br>
<br>


# 목차

이 `README.md` 파일은 마크다운(`.md`) 파일입니다. Manning 웹사이트에서 이 코드 번들을 다운로드하여 로컬 컴퓨터에서 보고 있다면, 적절한 보기를 위해 마크다운 편집기나 뷰어를 사용하는 것을 권장합니다. 아직 마크다운 편집기를 설치하지 않았다면 [Ghostwriter](https://ghostwriter.kde.org)가 좋은 무료 옵션입니다.

또는 브라우저에서 GitHub의 [https://github.com/rasbt/LLMs-from-scratch](https://github.com/rasbt/LLMs-from-scratch)에서 이 파일과 다른 파일들을 볼 수 있습니다. GitHub는 마크다운을 자동으로 렌더링합니다.

<br>
<br>


> **팁:**
> Python과 Python 패키지 설치 및 코드 환경 설정에 대한 안내가 필요하다면, [setup](setup) 디렉토리에 있는 [README.md](setup/README.md) 파일을 읽어보시기 바랍니다.

<br>
<br>

[![Code tests Linux](https://github.com/rasbt/LLMs-from-scratch/actions/workflows/basic-tests-linux-uv.yml/badge.svg)](https://github.com/rasbt/LLMs-from-scratch/actions/workflows/basic-tests-linux-uv.yml)
[![Code tests Windows](https://github.com/rasbt/LLMs-from-scratch/actions/workflows/basic-tests-windows-uv-pip.yml/badge.svg)](https://github.com/rasbt/LLMs-from-scratch/actions/workflows/basic-tests-windows-uv-pip.yml)
[![Code tests macOS](https://github.com/rasbt/LLMs-from-scratch/actions/workflows/basic-tests-macos-uv.yml/badge.svg)](https://github.com/rasbt/LLMs-from-scratch/actions/workflows/basic-tests-macos-uv.yml)




<br>

| 장 제목                                                     | 메인 코드(빠른 접근)                                                                                                    | 모든 코드 + 보충 자료         |
|------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------|-------------------------------|
| [설정 권장사항](setup)                                       | -                                                                                                                               | -                             |
| 1장: 대형 언어 모델 이해하기                                  | 코드 없음                                                                                                                         | -                             |
| 2장: 텍스트 데이터 다루기                                     | - [ch02.ipynb](ch02/01_main-chapter-code/ch02.ipynb)<br/>- [dataloader.ipynb](ch02/01_main-chapter-code/dataloader.ipynb) (요약)<br/>- [exercise-solutions.ipynb](ch02/01_main-chapter-code/exercise-solutions.ipynb)               | [./ch02](./ch02)            |
| 3장: 어텐션 메커니즘 코딩                                     | - [ch03.ipynb](ch03/01_main-chapter-code/ch03.ipynb)<br/>- [multihead-attention.ipynb](ch03/01_main-chapter-code/multihead-attention.ipynb) (요약) <br/>- [exercise-solutions.ipynb](ch03/01_main-chapter-code/exercise-solutions.ipynb)| [./ch03](./ch03)             |
| 4장: GPT 모델을 처음부터 구현                                | - [ch04.ipynb](ch04/01_main-chapter-code/ch04.ipynb)<br/>- [gpt.py](ch04/01_main-chapter-code/gpt.py) (요약)<br/>- [exercise-solutions.ipynb](ch04/01_main-chapter-code/exercise-solutions.ipynb) | [./ch04](./ch04)           |
| 5장: 비라벨 데이터로 사전학습                                 | - [ch05.ipynb](ch05/01_main-chapter-code/ch05.ipynb)<br/>- [gpt_train.py](ch05/01_main-chapter-code/gpt_train.py) (요약) <br/>- [gpt_generate.py](ch05/01_main-chapter-code/gpt_generate.py) (요약) <br/>- [exercise-solutions.ipynb](ch05/01_main-chapter-code/exercise-solutions.ipynb) | [./ch05](./ch05)              |
| 6장: 텍스트 분류를 위한 미세조정                              | - [ch06.ipynb](ch06/01_main-chapter-code/ch06.ipynb)  <br/>- [gpt_class_finetune.py](ch06/01_main-chapter-code/gpt_class_finetune.py)  <br/>- [exercise-solutions.ipynb](ch06/01_main-chapter-code/exercise-solutions.ipynb) | [./ch06](./ch06)              |
| 7장: 지시 따르기 미세조정                                    | - [ch07.ipynb](ch07/01_main-chapter-code/ch07.ipynb)<br/>- [gpt_instruction_finetuning.py](ch07/01_main-chapter-code/gpt_instruction_finetuning.py) (요약)<br/>- [ollama_evaluate.py](ch07/01_main-chapter-code/ollama_evaluate.py) (요약)<br/>- [exercise-solutions.ipynb](ch07/01_main-chapter-code/exercise-solutions.ipynb) | [./ch07](./ch07)  |
| 부록 A: PyTorch 입문                                        | - [code-part1.ipynb](appendix-A/01_main-chapter-code/code-part1.ipynb)<br/>- [code-part2.ipynb](appendix-A/01_main-chapter-code/code-part2.ipynb)<br/>- [DDP-script.py](appendix-A/01_main-chapter-code/DDP-script.py)<br/>- [exercise-solutions.ipynb](appendix-A/01_main-chapter-code/exercise-solutions.ipynb) | [./appendix-A](./appendix-A) |
| 부록 B: 참고문헌과 추가 읽을거리                             | 코드 없음                                                                                                                         | -                             |
| 부록 C: 연습 문제 해답                                       | 코드 없음                                                                                                                         | -                             |
| 부록 D: 학습 루프에 유용한 요소 추가                         | - [appendix-D.ipynb](appendix-D/01_main-chapter-code/appendix-D.ipynb)                                                          | [./appendix-D](./appendix-D)  |
| 부록 E: LoRA로 파라미터 효율적 미세조정                      | - [appendix-E.ipynb](appendix-E/01_main-chapter-code/appendix-E.ipynb)                                                          | [./appendix-E](./appendix-E) |

<br>
&nbsp;

아래의 멘탈 모델은 이 책에서 다루는 내용을 요약합니다.

<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/mental-model.jpg" width="650px">


<br>
&nbsp;

## 전제 조건

가장 중요한 전제 조건은 Python 프로그래밍에 대한 탄탄한 기초입니다.
이 지식을 바탕으로 LLM의 매혹적인 세계를 탐험하고
이 책에서 제시되는 개념과 코드 예제를 이해할 수 있을 것입니다.

깊은 신경망(deep neural networks)에 대한 경험이 있다면 LLM이 이러한 아키텍처를 기반으로 구축되기 때문에 특정 개념들이 더 친숙하게 느껴질 수 있습니다.

이 책은 외부 LLM 라이브러리를 사용하지 않고 처음부터 코드를 구현하기 위해 PyTorch를 사용합니다. PyTorch에 대한 숙련도가 전제 조건은 아니지만, PyTorch 기초에 대한 친숙함은 확실히 유용합니다. PyTorch를 처음 접한다면, 부록 A에서 PyTorch에 대한 간결한 소개를 제공합니다. 또는 제가 쓴 책 [PyTorch in One Hour: From Tensors to Training Neural Networks on Multiple GPUs](https://sebastianraschka.com/teaching/pytorch-1h/)가 기본 사항을 학습하는 데 도움이 될 수 있습니다.



<br>
&nbsp;

## 하드웨어 요구사항

이 책의 주요 장들에 있는 코드는 합리적인 시간 내에 일반적인 노트북에서 실행되도록 설계되었으며 특수한 하드웨어를 필요로 하지 않습니다. 이러한 접근 방식은 광범위한 독자들이 자료에 참여할 수 있도록 보장합니다. 또한 GPU가 사용 가능한 경우 코드가 자동으로 GPU를 활용합니다. (추가 권장사항은 [setup](https://github.com/rasbt/LLMs-from-scratch/blob/main/setup/README.md) 문서를 참조해 주세요.)


&nbsp;
## 비디오 강의

책의 각 장을 코딩하는 [17시간 15분의 동반 비디오 강의](https://www.manning.com/livevideo/master-and-build-large-language-models)가 있습니다. 강의는 책의 구조를 반영하는 장과 섹션으로 구성되어 있어 책의 독립적인 대안이나 보완적인 코드 실습 자료로 사용할 수 있습니다.

<a href="https://www.manning.com/livevideo/master-and-build-large-language-models"><img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/video-screenshot.webp?123" width="350px"></a>



&nbsp;
## 연습 문제

책의 각 장에는 여러 연습 문제가 포함되어 있습니다. 해답은 부록 C에 요약되어 있으며, 해당 코드 노트북은 이 저장소의 메인 장 폴더에서 사용할 수 있습니다(예: [./ch02/01_main-chapter-code/exercise-solutions.ipynb](./ch02/01_main-chapter-code/exercise-solutions.ipynb)).

코드 연습 문제 외에도, Manning 웹사이트에서 [처음부터 구축하는 대형 언어 모델 자가 테스트(Test Yourself On Build a Large Language Model From Scratch)](https://www.manning.com/books/test-yourself-on-build-a-large-language-model-from-scratch)라는 제목의 무료 170페이지 PDF를 다운로드할 수 있습니다. 이 PDF에는 이해도를 테스트하는 데 도움이 되는 장당 약 30개의 퀴즈 문제와 해답이 포함되어 있습니다.

<a href="https://www.manning.com/books/test-yourself-on-build-a-large-language-model-from-scratch"><img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/test-yourself-cover.jpg?123" width="150px"></a>



&nbsp;
## 보너스 자료

여러 폴더에는 관심 있는 독자들을 위한 선택적 자료들이 보너스로 포함되어 있습니다:

- **설정**
  - [Python 설정 팁](setup/01_optional-python-setup-preferences)
  - [이 책에서 사용되는 Python 패키지와 라이브러리 설치](setup/02_installing-python-libraries)
  - [Docker 환경 설정 가이드](setup/03_optional-docker-environment)
- **2장: 텍스트 데이터 다루기**
  - [바이트 페어 인코딩(BPE) 토크나이저 처음부터 구현](ch02/05_bpe-from-scratch/bpe-from-scratch.ipynb)
  - [다양한 바이트 페어 인코딩(BPE) 구현 비교](ch02/02_bonus_bytepair-encoder)
  - [임베딩 레이어와 선형 레이어 간의 차이점 이해](ch02/03_bonus_embedding-vs-matmul)
  - [간단한 숫자로 데이터로더 직관 이해](ch02/04_bonus_dataloader-intuition)
- **3장: 어텐션 메커니즘 코딩**
  - [효율적인 멀티헤드 어텐션 구현 비교](ch03/02_bonus_efficient-multihead-attention/mha-implementations.ipynb)
  - [PyTorch 버퍼 이해](ch03/03_understanding-buffers/understanding-buffers.ipynb)
- **4장: GPT 모델을 처음부터 구현**
  - [FLOPS 분석](ch04/02_performance-analysis/flops-analysis.ipynb)
  - [KV 캐시](ch04/03_kv-cache)
- **5장: 비라벨 데이터로 사전학습:**
  - [대체 가중치 로딩 방법](ch05/02_alternative_weight_loading/)
  - [프로젝트 구텐베르크 데이터셋으로 GPT 사전학습](ch05/03_bonus_pretraining_on_gutenberg)
  - [학습 루프에 유용한 요소들 추가](ch05/04_learning_rate_schedulers)
  - [사전학습을 위한 하이퍼파라미터 최적화](ch05/05_bonus_hparam_tuning)
  - [사전학습된 LLM과 상호작용하기 위한 사용자 인터페이스 구축](ch05/06_user_interface)
  - [GPT를 Llama로 변환](ch05/07_gpt_to_llama)
  - [Llama 3.2 처음부터 구현](ch05/07_gpt_to_llama/standalone-llama32.ipynb)
  - [Qwen3 Dense와 Mixture-of-Experts (MoE) 처음부터 구현](ch05/11_qwen3/)
  - [메모리 효율적인 모델 가중치 로딩](ch05/08_memory_efficient_weight_loading/memory-efficient-state-dict.ipynb)
  - [새 토큰으로 Tiktoken BPE 토크나이저 확장](ch05/09_extending-tokenizers/extend-tiktoken.ipynb)
  - [더 빠른 LLM 훈련을 위한 PyTorch 성능 팁](ch05/10_llm-training-speed)
- **6장: 분류를 위한 미세조정**
  - [다양한 레이어 미세조정과 더 큰 모델 사용 추가 실험](ch06/02_bonus_additional-experiments)
  - [50k IMDB 영화 리뷰 데이터셋으로 다양한 모델 미세조정](ch06/03_bonus_imdb-classification)
  - [GPT 기반 스팸 분류기와 상호작용하기 위한 사용자 인터페이스 구축](ch06/04_user_interface)
- **7장: 지시 따르기 미세조정**
  - [거의 중복된 항목 찾기와 수동태 항목 생성을 위한 데이터셋 유틸리티](ch07/02_dataset-utilities)
  - [OpenAI API와 Ollama를 사용한 지시 응답 평가](ch07/03_model-evaluation)
  - [지시 미세조정을 위한 데이터셋 생성](ch07/05_dataset-generation/llama3-ollama.ipynb)
  - [지시 미세조정을 위한 데이터셋 개선](ch07/05_dataset-generation/reflection-gpt4.ipynb)
  - [Llama 3.1 70B와 Ollama로 선호도 데이터셋 생성](ch07/04_preference-tuning-with-dpo/create-preference-data-ollama.ipynb)
  - [LLM 정렬을 위한 직접 선호도 최적화(DPO)](ch07/04_preference-tuning-with-dpo/dpo-from-scratch.ipynb)
  - [지시 미세조정된 GPT 모델과 상호작용하기 위한 사용자 인터페이스 구축](ch07/06_user_interface)

<br>
&nbsp;

## 질문, 피드백, 그리고 이 저장소에 기여하기


[Manning 포럼](https://livebook.manning.com/forum?product=raschka&page=1)이나 [GitHub Discussions](https://github.com/rasbt/LLMs-from-scratch/discussions)를 통해 공유되는 모든 종류의 피드백을 환영합니다. 마찬가지로 질문이 있거나 다른 사람들과 아이디어를 논의하고 싶다면 주저하지 말고 포럼에 게시해 주세요.

이 저장소는 인쇄된 책에 해당하는 코드를 포함하고 있기 때문에, 현재 메인 장 코드의 내용을 확장하는 기여는 받을 수 없습니다. 물리적 책과의 차이를 야기하기 때문입니다. 일관성을 유지하는 것은 모든 사람에게 원활한 경험을 보장하는 데 도움이 됩니다.


&nbsp;
## 인용

이 책이나 코드가 연구에 유용하다면 인용을 고려해 주세요.

시카고 스타일 인용:

> Raschka, Sebastian. *Build A Large Language Model (From Scratch)*. Manning, 2024. ISBN: 978-1633437166.

BibTeX 항목:

```
@book{build-llms-from-scratch-book,
  author       = {Sebastian Raschka},
  title        = {Build A Large Language Model (From Scratch)},
  publisher    = {Manning},
  year         = {2024},
  isbn         = {978-1633437166},
  url          = {https://www.manning.com/books/build-a-large-language-model-from-scratch},
  github       = {https://github.com/rasbt/LLMs-from-scratch}
}
```