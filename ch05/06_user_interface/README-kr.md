# 사전학습된 LLM과 상호작용하기 위한 사용자 인터페이스 구축

이 보너스 폴더는 아래 그림과 같이 5장의 사전학습된 LLM과 상호작용하기 위한 ChatGPT와 유사한 사용자 인터페이스를 실행하는 코드를 포함합니다.

![Chainlit UI example](https://sebastianraschka.com/images/LLMs-from-scratch-images/bonus/chainlit/chainlit-orig.webp)

이 사용자 인터페이스를 구현하기 위해 오픈소스 [Chainlit Python 패키지](https://github.com/Chainlit/chainlit)를 사용합니다.

&nbsp;
## 단계 1: 의존성 설치

먼저 다음 명령으로 `chainlit` 패키지를 설치합니다:

```bash
pip install chainlit
```

(또는 `pip install -r requirements-extra.txt`를 실행합니다.)

&nbsp;
## 단계 2: `app` 코드 실행

이 폴더는 2개의 파일을 포함합니다:

1. [`app_orig.py`](app_orig.py): 이 파일은 OpenAI의 원본 GPT-2 가중치를 로드하고 사용합니다.
2. [`app_own.py`](app_own.py): 이 파일은 5장에서 생성한 GPT-2 가중치를 로드하고 사용합니다. 이를 위해서는 먼저 [`../01_main-chapter-code/ch05.ipynb`](../01_main-chapter-code/ch05.ipynb) 파일을 실행해야 합니다.

(이 파일들을 열고 검토하여 더 자세히 알아보시기 바랍니다.)

터미널에서 다음 명령 중 하나를 실행하여 UI 서버를 시작합니다:

```bash
chainlit run app_orig.py
```

또는

```bash
chainlit run app_own.py
```

위 명령 중 하나를 실행하면 모델과 상호작용할 수 있는 새 브라우저 탭이 열립니다. 브라우저 탭이 자동으로 열리지 않으면 터미널 명령을 확인하고 로컬 주소를 브라우저 주소 표시줄에 복사하시기 바랍니다(보통 주소는 `http://localhost:8000`입니다).