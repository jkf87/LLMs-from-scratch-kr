# 이 책에서 사용되는 Python 패키지와 라이브러리 설치

이 문서는 설치된 Python 버전과 패키지를 다시 확인하는 방법에 대한 추가 정보를 제공합니다. (Python 및 Python 패키지 설치에 대한 자세한 정보는 [../01_optional-python-setup-preferences](../01_optional-python-setup-preferences) 폴더를 참조하세요.)

이 책에 대해 [여기](https://github.com/rasbt/LLMs-from-scratch/blob/main/requirements.txt)에 나열된 다음 라이브러리들을 사용했습니다. 이러한 라이브러리의 새 버전들도 호환될 가능성이 높습니다. 하지만 코드에서 문제가 발생하면, 대안으로 이러한 라이브러리 버전을 시도해볼 수 있습니다.



> **참고:**
> [옵션 1: uv 사용](../01_optional-python-setup-preferences/README.md)에서 설명한 대로 `uv`를 사용하는 경우, 아래 명령에서 `pip`를 `uv pip`로 바꿀 수 있습니다. 예를 들어, `pip install -r requirements.txt`는 `uv pip install -r requirements.txt`가 됩니다.



이러한 요구사항을 가장 편리하게 설치하려면, 이 코드 저장소의 루트 디렉토리에 있는 `requirements.txt` 파일을 사용하고 다음 명령을 실행할 수 있습니다:

```bash
pip install -r requirements.txt
```

또는 다음과 같이 GitHub URL을 통해 설치할 수 있습니다:

```bash
pip install -r https://raw.githubusercontent.com/rasbt/LLMs-from-scratch/main/requirements.txt
```


그런 다음, 설치가 완료된 후 다음을 사용하여 모든 패키지가 설치되고 최신 상태인지 확인하세요

```bash
python python_environment_check.py
```

<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/setup/02_installing-python-libraries/check_1.jpg" width="600px">

이상적으로는 위와 같은 결과를 얻을 수 있는 이 디렉토리의 `python_environment_check.ipynb`를 실행하여 JupyterLab에서 버전을 확인하는 것도 권장됩니다.

<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/setup/02_installing-python-libraries/check_2.jpg" width="500px">

다음과 같은 문제가 발생하면, JupyterLab 인스턴스가 잘못된 conda 환경에 연결되어 있을 가능성이 높습니다:

<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/setup/02_installing-python-libraries/jupyter-issues.jpg" width="450px">

이 경우, `--conda` 플래그와 함께 `watermark`를 사용하여 올바른 conda 환경에서 JupyterLab 인스턴스를 열었는지 확인할 수 있습니다:

<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/setup/02_installing-python-libraries/watermark.jpg" width="350px">


&nbsp;
## PyTorch 설치

PyTorch는 pip를 사용하여 다른 Python 라이브러리나 패키지처럼 설치할 수 있습니다. 예를 들어:

```bash
pip install torch
```

하지만 PyTorch는 CPU 및 GPU 호환 코드를 특징으로 하는 포괄적인 라이브러리이므로, 설치에는 추가 설정과 설명이 필요할 수 있습니다(자세한 내용은 책의 *A.1.3 PyTorch 설치* 참조).

또한 [https://pytorch.org](https://pytorch.org)의 공식 PyTorch 웹사이트의 설치 가이드 메뉴를 참조하는 것을 강력히 권장합니다.

<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/setup/02_installing-python-libraries/pytorch-installer.jpg" width="600px">

<br>

---




질문이 있으시면 [토론 포럼](https://github.com/rasbt/LLMs-from-scratch/discussions)에서 언제든지 연락하세요.