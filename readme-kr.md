# 대형 언어 모델 만들기 (From Scratch)

이 저장소는 GPT와 같은 LLM을 설계, 사전학습(pretraining), 미세조정(finetuning)하는 코드를 포함하며, 도서 [Build a Large Language Model (From Scratch)](https://amzn.to/4fqvn0D)의 공식 코드 저장소입니다.

<br>
<br>

<a href="https://amzn.to/4fqvn0D"><img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/cover.jpg?123" width="250px"></a>

<br>

[*Build a Large Language Model (From Scratch)*](http://mng.bz/orYv)에서는, LLM이 내부적으로 어떻게 작동하는지 단계별로 직접 코드를 작성하며 이해하게 됩니다. 책 전반에 걸쳐, 직접 LLM을 만드는 전 과정을 명확한 설명, 도식, 예제와 함께 안내합니다.

이 책에서 다루는 교육용 소형 모델의 학습·개발 방식은 ChatGPT와 같은 대규모 기저 모델을 만들 때 사용되는 방법과 유사합니다. 또한 더 큰 사전학습 모델의 가중치를 불러와 미세조정하는 코드 역시 포함되어 있습니다.

- 공식 [소스 코드 저장소](https://github.com/rasbt/LLMs-from-scratch)
- [출판사(매닝) 도서 페이지](http://mng.bz/orYv)
- [아마존 도서 페이지](https://www.amazon.com/gp/product/1633437167)
- ISBN 9781633437166

<a href="http://mng.bz/orYv#reviews"><img src="https://sebastianraschka.com//images/LLMs-from-scratch-images/other/reviews.png" width="220px"></a>

<br>
<br>

이 저장소를 다운로드하려면 GitHub의 [Download ZIP](https://github.com/rasbt/LLMs-from-scratch/archive/refs/heads/main.zip) 버튼을 클릭하거나 다음 명령을 실행하세요:

```bash
git clone --depth 1 https://github.com/rasbt/LLMs-from-scratch.git
```

<br>

(매닝 웹사이트에서 코드 번들을 내려받았다면, 최신 업데이트를 위해 공식 GitHub 저장소 [https://github.com/rasbt/LLMs-from-scratch](https://github.com/rasbt/LLMs-from-scratch)를 방문하는 것을 권장합니다.)

<br>
<br>

# 목차 (Table of Contents)

이 `README.md`는 마크다운(`.md`) 파일입니다. 매닝 웹사이트에서 코드 번들을 내려받아 로컬에서 이 파일을 보고 있다면, 마크다운 미리보기 기능을 지원하는 에디터 사용을 권장합니다. 아직 설치하지 않았다면 [Ghostwriter](https://ghostwriter.kde.org)가 좋은 무료 옵션입니다.

또는 GitHub에서 [https://github.com/rasbt/LLMs-from-scratch](https://github.com/rasbt/LLMs-from-scratch)를 열어 브라우저에서 바로 렌더링된 문서를 볼 수도 있습니다.

<br>
<br>

> **팁:**
> 파이썬과 패키지 설치, 코드 실행 환경 설정에 관한 가이드는 [setup](setup) 디렉터리의 [README.md](setup/README.md)를 참고하세요.

<br>
<br>

[![Code tests Linux](https://github.com/rasbt/LLMs-from-scratch/actions/workflows/basic-tests-linux-uv.yml/badge.svg)](https://github.com/rasbt/LLMs-from-scratch/actions/workflows/basic-tests-linux-uv.yml)
[![Code tests Windows](https://github.com/rasbt/LLMs-from-scratch/actions/workflows/basic-tests-windows-uv-pip.yml/badge.svg)](https://github.com/rasbt/LLMs-from-scratch/actions/workflows/basic-tests-windows-uv-pip.yml)
[![Code tests macOS](https://github.com/rasbt/LLMs-from-scratch/actions/workflows/basic-tests-macos-uv.yml/badge.svg)](https://github.com/rasbt/LLMs-from-scratch/actions/workflows/basic-tests-macos-uv.yml)

<br>

| 장(Chapter) 제목                                            | 주요 코드(빠른 링크)                                                                                                        | 전체 코드 + 보충 자료         |
|------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------|-------------------------------|
| [설치/환경 설정](setup)                                     | -                                                                                                                           | -                             |
| 1장: 대형 언어 모델 이해하기                               | 코드 없음                                                                                                                   | -                             |
| 2장: 텍스트 데이터 다루기                                  | - [ch02.ipynb](ch02/01_main-chapter-code/ch02.ipynb)<br/>- [dataloader.ipynb](ch02/01_main-chapter-code/dataloader.ipynb) (요약)<br/>- [exercise-solutions.ipynb](ch02/01_main-chapter-code/exercise-solutions.ipynb) | [./ch02](./ch02)             |
| 3장: 어텐션 메커니즘 코딩                                  | - [ch03.ipynb](ch03/01_main-chapter-code/ch03.ipynb)<br/>- [multihead-attention.ipynb](ch03/01_main-chapter-code/multihead-attention.ipynb) (요약) <br/>- [exercise-solutions.ipynb](ch03/01_main-chapter-code/exercise-solutions.ipynb) | [./ch03](./ch03)             |
| 4장: GPT 모델을 처음부터 구현                               | - [ch04.ipynb](ch04/01_main-chapter-code/ch04.ipynb)<br/>- [gpt.py](ch04/01_main-chapter-code/gpt.py) (요약)<br/>- [exercise-solutions.ipynb](ch04/01_main-chapter-code/exercise-solutions.ipynb) | [./ch04](./ch04)             |
| 5장: 비라벨 데이터로 사전학습                              | - [ch05.ipynb](ch05/01_main-chapter-code/ch05.ipynb)<br/>- [gpt_train.py](ch05/01_main-chapter-code/gpt_train.py) (요약) <br/>- [gpt_generate.py](ch05/01_main-chapter-code/gpt_generate.py) (요약) <br/>- [exercise-solutions.ipynb](ch05/01_main-chapter-code/exercise-solutions.ipynb) | [./ch05](./ch05)             |
| 6장: 텍스트 분류를 위한 미세조정                           | - [ch06.ipynb](ch06/01_main-chapter-code/ch06.ipynb)  <br/>- [gpt_class_finetune.py](ch06/01_main-chapter-code/gpt_class_finetune.py)  <br/>- [exercise-solutions.ipynb](ch06/01_main-chapter-code/exercise-solutions.ipynb) | [./ch06](./ch06)             |
| 7장: 지시 따르기(following instructions)를 위한 미세조정    | - [ch07.ipynb](ch07/01_main-chapter-code/ch07.ipynb)<br/>- [gpt_instruction_finetuning.py](ch07/01_main-chapter-code/gpt_instruction_finetuning.py) (요약)<br/>- [ollama_evaluate.py](ch07/01_main-chapter-code/ollama_evaluate.py) (요약)<br/>- [exercise-solutions.ipynb](ch07/01_main-chapter-code/exercise-solutions.ipynb) | [./ch07](./ch07)             |
| 부록 A: PyTorch 입문                                       | - [code-part1.ipynb](appendix-A/01_main-chapter-code/code-part1.ipynb)<br/>- [code-part2.ipynb](appendix-A/01_main-chapter-code/code-part2.ipynb)<br/>- [DDP-script.py](appendix-A/01_main-chapter-code/DDP-script.py)<br/>- [exercise-solutions.ipynb](appendix-A/01_main-chapter-code/exercise-solutions.ipynb) | [./appendix-A](./appendix-A) |
| 부록 B: 참고문헌과 추가 읽을거리                           | 코드 없음                                                                                                                   | -                             |
| 부록 C: 연습 문제 해설 요약                                 | 코드 없음                                                                                                                   | -                             |
| 부록 D: 학습 루프에 유용한 요소 추가                        | - [appendix-D.ipynb](appendix-D/01_main-chapter-code/appendix-D.ipynb)                                                      | [./appendix-D](./appendix-D) |
| 부록 E: LoRA로 파라미터 효율적 미세조정                     | - [appendix-E.ipynb](appendix-E/01_main-chapter-code/appendix-E.ipynb)                                                      | [./appendix-E](./appendix-E) |

<br>
&nbsp;

아래의 개념도는 이 책에서 다루는 내용을 요약한 것입니다.

<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/mental-model.jpg" width="650px">

<br>
&nbsp;

## 사전 지식

가장 중요한 전제는 파이썬 프로그래밍에 대한 탄탄한 기초입니다. 이를 바탕으로 LLM의 흥미로운 세계를 탐구하고, 본서의 개념과 예제를 이해하는 데 충분합니다.

심층 신경망에 대한 경험이 있다면 더 익숙하게 느낄 수 있습니다. LLM은 이러한 아키텍처 위에 구축되기 때문입니다.

이 책은 외부 LLM 라이브러리에 의존하지 않고, PyTorch만으로 처음부터 구현합니다. PyTorch가 필수는 아니지만, 기본에 익숙하다면 도움이 됩니다. PyTorch가 처음이라면 부록 A에서 간단히 소개하고 있으며, 또는 [PyTorch in One Hour: From Tensors to Training Neural Networks on Multiple GPUs](https://sebastianraschka.com/teaching/pytorch-1h/)도 참고할 수 있습니다.

<br>
&nbsp;

## 하드웨어 요구 사항

본서의 주요 장(chapter)에 포함된 코드는 일반적인 노트북에서도 합리적인 시간 내 실행되도록 설계되었고, 특수 하드웨어가 필수는 아닙니다. 또한 가능할 경우 GPU를 자동으로 활용합니다. (추가 권장 사항은 [setup](https://github.com/rasbt/LLMs-from-scratch/blob/main/setup/README.md) 문서를 참조하세요.)

&nbsp;

## 비디오 코스

[총 17시간 15분 분량의 동반 비디오 코스](https://www.manning.com/livevideo/master-and-build-large-language-models)는 책의 구조와 동일한 챕터·섹션으로 이루어져 있어, 책의 대체재로도, 코드-얼롱용 보조 자료로도 활용할 수 있습니다.

<a href="https://www.manning.com/livevideo/master-and-build-large-language-models"><img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/video-screenshot.webp?123" width="350px"></a>

&nbsp;

## 연습 문제

각 장에는 여러 연습 문제가 포함되어 있습니다. 해설은 부록 C에 요약되어 있으며, 해당 코드 노트북은 각 장의 폴더에 포함되어 있습니다(예: [./ch02/01_main-chapter-code/exercise-solutions.ipynb](./ch02/01_main-chapter-code/exercise-solutions.ipynb)).

또한 매닝 웹사이트에서 무료 170페이지 분량의 PDF인 [Test Yourself On Build a Large Language Model (From Scratch)](https://www.manning.com/books/test-yourself-on-build-a-large-language-model-from-scratch)을 내려받을 수 있습니다. 장별로 약 30개의 퀴즈와 해설이 수록되어 있어 이해도를 점검하는 데 도움이 됩니다.

<a href="https://www.manning.com/books/test-yourself-on-build-a-large-language-model-from-scratch"><img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/test-yourself-cover.jpg?123" width="150px"></a>

&nbsp;

## 보너스 자료

관심 있는 독자를 위한 선택적 보너스 자료는 다음과 같습니다:

- **설치/환경 설정**
  - [파이썬 설정 팁](setup/01_optional-python-setup-preferences)
  - [본서에서 사용하는 파이썬 패키지 설치](setup/02_installing-python-libraries)
  - [도커 환경 구성 가이드](setup/03_optional-docker-environment)
- **2장: 텍스트 데이터 다루기**
  - [Byte Pair Encoding(BPE) 토크나이저를 처음부터 구현](ch02/05_bpe-from-scratch/bpe-from-scratch.ipynb)
  - [여러 BPE 구현 비교](ch02/02_bonus_bytepair-encoder)
  - [임베딩 레이어와 선형 레이어의 차이 이해](ch02/03_bonus_embedding-vs-matmul)
  - [간단한 숫자로 직관적인 데이터로더 이해](ch02/04_bonus_dataloader-intuition)
- **3장: 어텐션 메커니즘 코딩**
  - [효율적인 멀티헤드 어텐션 구현 비교](ch03/02_bonus_efficient-multihead-attention/mha-implementations.ipynb)
  - [PyTorch 버퍼 이해하기](ch03/03_understanding-buffers/understanding-buffers.ipynb)
- **4장: GPT 모델을 처음부터 구현**
  - [FLOPS 분석](ch04/02_performance-analysis/flops-analysis.ipynb)
  - [KV 캐시](ch04/03_kv-cache)
- **5장: 비라벨 데이터로 사전학습**
  - [대체 가중치 로딩 방법](ch05/02_alternative_weight_loading/)
  - [Project Gutenberg 데이터셋으로 GPT 사전학습](ch05/03_bonus_pretraining_on_gutenberg)
  - [학습 루프에 유용한 기능 추가](ch05/04_learning_rate_schedulers)
  - [사전학습 하이퍼파라미터 최적화](ch05/05_bonus_hparam_tuning)
  - [사전학습 LLM과 상호작용하는 사용자 인터페이스 구축](ch05/06_user_interface)
  - [GPT를 Llama로 변환](ch05/07_gpt_to_llama)
  - [Llama 3.2를 처음부터 구현](ch05/07_gpt_to_llama/standalone-llama32.ipynb)
  - [Qwen3 Dense 및 MoE를 처음부터 구현](ch05/11_qwen3/)
  - [메모리 효율적 모델 가중치 로딩](ch05/08_memory_efficient_weight_loading/memory-efficient-state-dict.ipynb)
  - [Tiktoken BPE 토크나이저에 새 토큰 확장](ch05/09_extending-tokenizers/extend-tiktoken.ipynb)
  - [LLM 학습 속도 향상을 위한 PyTorch 팁](ch05/10_llm-training-speed)
- **6장: 분류용 미세조정**
  - [다양한 레이어/큰 모델 미세조정 추가 실험](ch06/02_bonus_additional-experiments)
  - [IMDB 5만 리뷰 데이터셋으로 다양한 모델 미세조정](ch06/03_bonus_imdb-classification)
  - [GPT 기반 스팸 분류기와 상호작용하는 UI 구축](ch06/04_user_interface)
- **7장: 지시 따르기 미세조정**
  - [중복 문서 탐색 및 수동태 문장 데이터셋 유틸리티](ch07/02_dataset-utilities)
  - [OpenAI API와 Ollama를 활용한 응답 평가](ch07/03_model-evaluation)
  - [지시 미세조정용 데이터셋 생성](ch07/05_dataset-generation/llama3-ollama.ipynb)
  - [지시 미세조정용 데이터셋 개선](ch07/05_dataset-generation/reflection-gpt4.ipynb)
  - [Llama 3.1 70B와 Ollama로 선호 데이터 생성](ch07/04_preference-tuning-with-dpo/create-preference-data-ollama.ipynb)
  - [LLM 정렬을 위한 DPO(Direct Preference Optimization)](ch07/04_preference-tuning-with-dpo/dpo-from-scratch.ipynb)
  - [지시 미세조정된 GPT 모델과 상호작용하는 UI 구축](ch07/06_user_interface)

<br>
&nbsp;

## 질문, 피드백, 그리고 기여 방법

다양한 피드백을 환영합니다. [Manning 포럼](https://livebook.manning.com/forum?product=raschka&page=1) 또는 [GitHub Discussions](https://github.com/rasbt/LLMs-from-scratch/discussions)을 통해 공유해 주세요. 질문이나 아이디어가 있다면 포럼에 자유롭게 남겨 주세요.

이 저장소는 종이책과 1:1로 대응되는 코드이므로, 현재로서는 본문 메인 코드의 내용을 확장하는 형태의 기여는 받기 어렵습니다. 책과의 일치성을 유지하는 것이 독자 경험을 위해 중요하기 때문입니다.

&nbsp;

## 인용(Citation)

이 책 또는 코드를 연구에 활용했다면, 아래와 같이 인용을 부탁드립니다.

시카고 스타일:

> Raschka, Sebastian. *Build A Large Language Model (From Scratch)*. Manning, 2024. ISBN: 978-1633437166.

BibTeX:

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