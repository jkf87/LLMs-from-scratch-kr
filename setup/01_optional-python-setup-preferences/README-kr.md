# Python 설정 팁



Python을 설치하고 컴퓨팅 환경을 설정하는 방법은 여러 가지가 있습니다. 여기서는 제 개인적인 선호사항을 공유합니다.

<br>

> **참고:** 
> Google Colab에서 노트북을 실행하고 의존성을 설치하려면, 노트북 상단의 새 셀에서 다음 코드를 실행하고 이 튜토리얼의 나머지 부분은 건너뛰세요:
> `pip install uv && uv pip install --system -r https://raw.githubusercontent.com/rasbt/LLMs-from-scratch/refs/heads/main/requirements.txt`

아래 섹션들은 로컬 머신에서 Python 환경과 패키지를 관리하는 방법을 설명합니다.

저는 오랫동안 [Conda](https://anaconda.org/anaconda/conda)와 [pip](https://pypi.org/project/pip/)를 사용해왔지만, 최근에는 [uv](https://github.com/astral-sh/uv) 패키지가 패키지 설치와 의존성 해결을 더 빠르고 효율적으로 제공하여 상당한 주목을 받고 있습니다.

2025년 현재 더 현대적인 접근 방식인 *옵션 1: uv 사용*부터 시작하는 것을 권장합니다. *옵션 1*에서 문제가 발생하면 *옵션 2: Conda 사용*을 고려해보세요.

이 튜토리얼에서는 macOS를 실행하는 컴퓨터를 사용하고 있지만, 이 워크플로우는 Linux 머신에서도 유사하며 다른 운영체제에서도 작동할 수 있습니다.


&nbsp;
# 옵션 1: uv 사용

이 섹션은 `uv pip` 인터페이스를 통해 `uv`를 사용하여 Python 설정 및 패키지 설치 절차를 안내합니다. `uv pip` 인터페이스는 이전에 pip를 사용한 대부분의 Python 사용자에게 네이티브 `uv` 명령보다 더 친숙하게 느껴질 수 있습니다.

&nbsp;
> **참고:**
> Python을 설치하고 `uv`를 사용하는 대안적인 방법들이 있습니다. 예를 들어, `uv`를 통해 직접 Python을 설치하고 더 빠른 패키지 관리를 위해 `uv pip install` 대신 `uv add`를 사용할 수 있습니다.
>
> macOS나 Linux 사용자이고 네이티브 `uv` 명령을 선호한다면, [./native-uv.md 튜토리얼](./native-uv.md)을 참조하세요. 공식 [`uv` 문서](https://docs.astral.sh/uv/)도 확인하는 것을 권장합니다.
>
> `uv add` 구문은 Windows 사용자에게도 적용됩니다. 하지만 `pyproject.toml`의 일부 의존성이 Windows에서 문제를 일으킨다는 것을 발견했습니다. 따라서 Windows 사용자에게는 `uv add`와 유사한 `pixi add` 워크플로우를 가진 `pix`를 대신 권장합니다. 더 자세한 정보는 [./native-pixi.md 튜토리얼](./native-pixi.md)을 참조하세요.
>
> `uv add`와 `pixi add`는 추가적인 속도 장점을 제공하지만, `uv pip`가 약간 더 사용자 친화적이어서 초보자에게는 좋은 출발점이라고 생각합니다. 하지만 Python 패키지 관리에 처음이라면, 네이티브 `uv` 인터페이스도 처음부터 배울 수 있는 훌륭한 기회입니다. 지금은 이것이 제가 `uv`를 사용하는 방식이기도 하지만, `pip`와 `conda`에서 온다면 진입 장벽이 약간 더 높다는 것을 인식하고 있습니다.




&nbsp;
## 1. Python 설치 (설치되지 않은 경우)

이전에 시스템에 Python을 수동으로 설치한 적이 없다면, 그렇게 하는 것을 강력히 권장합니다. 이는 운영 체제의 내장 Python 설치와의 잠재적 충돌을 방지하는 데 도움이 되며, 이로 인해 문제가 발생할 수 있습니다.

하지만 이전에 시스템에 Python을 설치한 적이 있더라도, 최신 버전의 Python이 설치되어 있는지 확인하세요(3.10 이상 권장). 터미널에서 다음 코드를 실행하여 확인할 수 있습니다:

```bash
python --version
```
3.10 이상을 반환하면 추가 조치가 필요하지 않습니다.

&nbsp;
> **참고:**
> `python --version`이 Python 버전이 설치되지 않았음을 나타내면, 시스템이 `python3` 명령을 대신 사용하도록 구성되어 있을 수 있으므로 `python3 --version`도 확인해보세요.

&nbsp;
> **참고:**
> PyTorch 호환성을 보장하기 위해 최신 릴리스보다 최소 2버전 이전의 Python 버전을 설치하는 것을 권장합니다. 예를 들어, 최신 버전이 Python 3.13이라면, 버전 3.10 또는 3.11을 설치하는 것을 권장합니다.

그렇지 않고 Python이 설치되지 않았거나 구버전이라면, 아래에 설명된 대로 운영 체제에 맞게 설치할 수 있습니다.

<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/setup/uv-setup/python-not-found.png" width="500" height="auto" alt="No Python Found">

<br>

**Linux (Ubuntu/Debian)**

```bash
sudo apt update
sudo apt install python3.10 python3.10-venv python3.10-dev
```

<br>

**macOS**

Homebrew를 사용한다면 다음으로 Python을 설치하세요:

```bash
brew install python@3.10
```

또는 공식 웹사이트에서 설치 프로그램을 다운로드하여 실행하세요: [https://www.python.org/downloads/](https://www.python.org/downloads/).


<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/setup/uv-setup/python-version.png" width="700" height="auto" alt="Python version">

<br>

**Windows**

공식 웹사이트에서 설치 프로그램을 다운로드하여 실행하세요: [https://www.python.org/downloads/](https://www.python.org/downloads/).


&nbsp;

## 2. 가상 환경 생성

OS가 의존할 수 있는 시스템 전체 패키지 수정을 피하기 위해 별도의 가상 환경에 Python 패키지를 설치하는 것을 강력히 권장합니다. 현재 폴더에 가상 환경을 생성하려면 아래 세 단계를 따르세요.

<br>

**1. uv 설치**

```bash
pip install uv
```

<br>

**2. 가상 환경 생성**

```bash
uv venv --python=python3.10
```

<br>

**3. 가상 환경 활성화**

```bash
source .venv/bin/activate
```

&nbsp;
> **참고:**
> Windows를 사용하는 경우, 위 명령을 `source .venv/Scripts/activate` 또는 `.venv/Scripts/activate`로 바꿔야 할 수 있습니다.



새 터미널 세션을 시작할 때마다 가상 환경을 활성화해야 합니다. 예를 들어, 터미널이나 컴퓨터를 다시 시작하고 다음 날 프로젝트를 계속 작업하려면, 프로젝트 폴더에서 `source .venv/bin/activate`를 실행하여 가상 환경을 다시 활성화하면 됩니다.

<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/setup/uv-setup/venv-activate-1.png" width="600" height="auto" alt="Venv activated">

선택적으로 `deactivate` 명령을 실행하여 환경을 비활성화할 수 있습니다.

<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/setup/uv-setup/venv-activate-2.png" width="800" height="auto" alt="Venv deactivated">

&nbsp;
## 3. 패키지 설치

가상 환경을 활성화한 후, `uv`를 사용하여 Python 패키지를 설치할 수 있습니다. 예를 들어:

```bash
uv pip install packaging
```

`requirements.txt` 파일(이 GitHub 저장소의 최상위 레벨에 있는 파일과 같은)에서 모든 필요한 패키지를 설치하려면, 파일이 터미널 세션과 같은 디렉토리에 있다고 가정하고 다음 명령을 실행하세요:

```bash
uv pip install -r requirements.txt
```

또는 저장소에서 직접 최신 의존성을 설치하세요:

```bash
uv pip install -r https://raw.githubusercontent.com/rasbt/LLMs-from-scratch/refs/heads/main/requirements.txt
```


<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/setup/uv-setup/uv-install.png" width="700" height="auto" alt="Uv install">

&nbsp;

> **참고:**
> 특정 의존성으로 인해 위 명령에서 문제가 발생하는 경우(예: Windows를 사용하는 경우), 항상 일반 pip 사용으로 되돌릴 수 있습니다:
> `pip install -r requirements.txt`
> 또는
> `pip install -U -r https://raw.githubusercontent.com/rasbt/LLMs-from-scratch/refs/heads/main/requirements.txt`

<br>

**설정 마무리**

끝입니다! 이제 환경이 저장소의 코드를 실행할 준비가 되었습니다.

선택적으로, 이 저장소의 `python_environment_check.py` 스크립트를 실행하여 환경 확인을 실행할 수 있습니다:

```bash
python setup/02_installing-python-libraries/python_environment_check.py
```

<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/setup/uv-setup/env-check.png" width="700" height="auto" alt="Environment check">

특정 패키지에서 문제가 발생하면 다음을 사용하여 재설치해보세요:

```bash
uv pip install packagename
```

(여기서 `packagename`은 문제가 있는 패키지 이름으로 교체해야 하는 플레이스홀더 이름입니다.)

문제가 지속되면 GitHub에서 [토론 열기](https://github.com/rasbt/LLMs-from-scratch/discussions)를 고려하거나 아래의 *옵션 2: Conda 사용* 섹션을 진행하세요.

<br>

**코드 작업 시작**

모든 것이 설정되면 코드 파일로 작업을 시작할 수 있습니다. 예를 들어, 다음을 실행하여 [JupyterLab](https://jupyterlab.readthedocs.io/en/latest/)을 시작하세요:

```bash
jupyter lab
```

&nbsp;
> **참고:**
> jupyter lab 명령에서 문제가 발생하면, 가상 환경 내부의 전체 경로를 사용하여 시작할 수도 있습니다. 예를 들어, Linux/macOS에서는 `.venv/bin/jupyter lab`을, Windows에서는 `.venv\Scripts\jupyter-lab`을 사용하세요.

&nbsp;

<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/setup/uv-setup/jupyter.png" width="900" height="auto" alt="Uv install">

&nbsp;
<br>
<br>
&nbsp;

# 옵션 2: Conda 사용



이 섹션은 [miniforge](https://github.com/conda-forge/miniforge)를 통해 [`conda`](https://www.google.com/search?client=safari&rls=en&q=conda&ie=UTF-8&oe=UTF-8)를 사용하여 Python 설정 및 패키지 설치 절차를 안내합니다.

이 튜토리얼에서는 macOS를 실행하는 컴퓨터를 사용하고 있지만, 이 워크플로우는 Linux 머신에서도 유사하며 다른 운영체제에서도 작동할 수 있습니다.


&nbsp;
## 1. Miniforge 다운로드 및 설치

GitHub 저장소 [여기](https://github.com/conda-forge/miniforge)에서 miniforge를 다운로드하세요.

<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/setup/01_optional-python-setup-preferences/download.png" alt="download" width="600px">

운영 체제에 따라 `.sh` (macOS, Linux) 또는 `.exe` 파일 (Windows)이 다운로드됩니다.

`.sh` 파일의 경우, 명령줄 터미널을 열고 다음 명령을 실행하세요

```bash
sh ~/Desktop/Miniforge3-MacOSX-arm64.sh
```

여기서 `Desktop/`은 Miniforge 설치 프로그램이 다운로드된 폴더입니다. 컴퓨터에서는 `Downloads/`로 바꿔야 할 수 있습니다.

<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/setup/01_optional-python-setup-preferences/miniforge-install.png" alt="miniforge-install" width="600px">

다음으로, "Enter"로 확인하면서 다운로드 지침을 단계별로 진행하세요.


&nbsp;
## 2. 새 가상 환경 생성

설치가 성공적으로 완료된 후, `LLMs`라는 새 가상 환경을 생성하는 것을 권장합니다. 다음을 실행하여 생성할 수 있습니다

```bash
conda create -n LLMs python=3.10
```

<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/setup/01_optional-python-setup-preferences/new-env.png" alt="new-env" width="600px">

> 많은 과학 컴퓨팅 라이브러리들이 최신 버전의 Python을 즉시 지원하지 않습니다. 따라서 PyTorch를 설치할 때는 한두 릴리스 이전 버전의 Python을 사용하는 것이 좋습니다. 예를 들어, Python의 최신 버전이 3.13이라면, Python 3.10 또는 3.11을 사용하는 것을 권장합니다.

다음으로, 새 가상 환경을 활성화하세요(새 터미널 창이나 탭을 열 때마다 해야 합니다):

```bash
conda activate LLMs
```

<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/setup/01_optional-python-setup-preferences/activate-env.png" alt="activate-env" width="600px">


&nbsp;
## 선택사항: 터미널 스타일링

활성화된 가상 환경을 볼 수 있도록 제 것과 유사하게 터미널을 스타일링하려면, [Oh My Zsh](https://github.com/ohmyzsh/ohmyzsh) 프로젝트를 확인해보세요.

&nbsp;
## 3. 새 Python 라이브러리 설치



이제 `conda` 패키지 설치 프로그램을 사용하여 새 Python 라이브러리를 설치할 수 있습니다. 예를 들어, [JupyterLab](https://jupyter.org/install)와 [watermark](https://github.com/rasbt/watermark)를 다음과 같이 설치할 수 있습니다:

```bash
conda install jupyterlab watermark
```

<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/setup/01_optional-python-setup-preferences/conda-install.png" alt="conda-install" width="600px">



`pip`를 사용하여 라이브러리를 설치할 수도 있습니다. 기본적으로 `pip`는 새로운 `LLms` conda 환경에 연결되어야 합니다:

<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/setup/01_optional-python-setup-preferences/check-pip.png" alt="check-pip" width="600px">

&nbsp;
## 4. PyTorch 설치

PyTorch는 pip를 사용하여 다른 Python 라이브러리나 패키지처럼 설치할 수 있습니다. 예를 들어:

```bash
pip install torch
```

하지만 PyTorch는 CPU 및 GPU 호환 코드를 특징으로 하는 포괄적인 라이브러리이므로, 설치에는 추가 설정과 설명이 필요할 수 있습니다(자세한 내용은 책의 *A.1.3 PyTorch 설치* 참조).

또한 [https://pytorch.org](https://pytorch.org)의 공식 PyTorch 웹사이트의 설치 가이드 메뉴를 참조하는 것을 강력히 권장합니다.

<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/setup/01_optional-python-setup-preferences/pytorch-installer.jpg" width="600px">

&nbsp;
## 5. 이 책에서 사용되는 Python 패키지와 라이브러리 설치

필요한 라이브러리 설치 방법에 대한 지침은 [이 책에서 사용되는 Python 패키지와 라이브러리 설치](../02_installing-python-libraries/README.md) 문서를 참조하세요.

<br>

---




질문이 있으시면 [토론 포럼](https://github.com/rasbt/LLMs-from-scratch/discussions)에서 언제든지 연락하세요.