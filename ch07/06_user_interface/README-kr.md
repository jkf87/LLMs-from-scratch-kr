# 지시사항 미세조정된 GPT 모델과 상호작용하는 사용자 인터페이스 구축 (Building a User Interface to Interact With the Instruction Finetuned GPT Model)



이 보너스 폴더에는 아래와 같이 챕터 7의 지시사항 미세조정된 GPT와 상호작용하는 ChatGPT와 같은 사용자 인터페이스를 실행하는 코드가 포함되어 있습니다.



![Chainlit UI example](https://sebastianraschka.com/images/LLMs-from-scratch-images/bonus/chainlit/chainlit-sft.webp?2)



이 사용자 인터페이스를 구현하기 위해 오픈소스 [Chainlit Python 패키지](https://github.com/Chainlit/chainlit)를 사용합니다.

&nbsp;
## 1단계: 의존성 설치

먼저 다음을 통해 `chainlit` 패키지를 설치합니다:

```bash
pip install chainlit
```

(또는 `pip install -r requirements-extra.txt`를 실행하세요.)

&nbsp;
## 2단계: `app` 코드 실행

[`app.py`](app.py) 파일에는 UI 코드가 포함되어 있습니다. 더 자세히 알아보려면 이 파일들을 열어서 살펴보세요.

이 파일은 챕터 7에서 생성한 GPT-2 가중치를 로드하고 사용합니다. 이를 위해서는 먼저 [`../01_main-chapter-code/ch07.ipynb`](../01_main-chapter-code/ch07.ipynb) 파일을 실행해야 합니다.

터미널에서 다음 명령을 실행하여 UI 서버를 시작합니다:

```bash
chainlit run app.py
```

위 명령을 실행하면 모델과 상호작용할 수 있는 새 브라우저 탭이 열릴 것입니다. 브라우저 탭이 자동으로 열리지 않으면 터미널 명령을 확인하고 로컬 주소를 브라우저 주소 표시줄에 복사하세요(보통 주소는 `http://localhost:8000`입니다).