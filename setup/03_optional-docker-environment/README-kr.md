# Docker 환경 설정 가이드

프로젝트의 의존성과 구성을 격리하는 개발 설정을 선호한다면, Docker를 사용하는 것이 매우 효과적인 솔루션입니다. 이 접근 방식은 소프트웨어 패키지와 라이브러리를 수동으로 설치할 필요를 없애고 일관된 개발 환경을 보장합니다.

이 가이드는 [../01_optional-python-setup-preferences](../01_optional-python-setup-preferences)와 [../02_installing-python-libraries](../02_installing-python-libraries)에서 설명한 conda 접근법보다 이것을 선호하는 경우, 이 책을 위한 선택적 docker 환경 설정 과정을 안내합니다.

<br>

## Docker 다운로드 및 설치

Docker를 시작하는 가장 쉬운 방법은 해당 플랫폼용 [Docker Desktop](https://docs.docker.com/desktop/)을 설치하는 것입니다.

Linux (Ubuntu) 사용자는 대신 [Docker Engine](https://docs.docker.com/engine/install/ubuntu/)을 설치하고 [설치 후](https://docs.docker.com/engine/install/linux-postinstall/) 단계를 따르는 것을 선호할 수 있습니다.

<br>

## Visual Studio Code에서 Docker DevContainer 사용

Docker DevContainer 또는 Development Container는 개발자가 Docker 컨테이너를 완전한 개발 환경으로 사용할 수 있게 해주는 도구입니다. 이 접근 방식은 사용자가 로컬 머신 설정에 관계없이 일관된 개발 환경으로 빠르게 시작하고 실행할 수 있도록 보장합니다.

DevContainer는 다른 IDE와도 작동하지만, DevContainer로 작업하기 위해 일반적으로 사용되는 IDE/편집기는 Visual Studio Code (VS Code)입니다. 아래 가이드는 VS Code 컨텍스트에서 이 책의 DevContainer를 사용하는 방법을 설명하지만, 유사한 과정이 PyCharm에도 적용되어야 합니다. 없고 사용하려면 [설치](https://code.visualstudio.com/download)하세요.

1. 이 GitHub 저장소를 복제하고 프로젝트 루트 디렉토리로 `cd`합니다.

```bash
git clone https://github.com/rasbt/LLMs-from-scratch.git
cd LLMs-from-scratch
```

2. `setup/03_optional-docker-environment/`에서 `.devcontainer` 폴더를 현재 디렉토리(프로젝트 루트)로 이동합니다.

```bash
mv setup/03_optional-docker-environment/.devcontainer ./
```

3. Docker Desktop에서, **_desktop-linux_ 빌더**가 실행 중이고 Docker 컨테이너를 빌드하는 데 사용될 것인지 확인하세요(_Docker Desktop_ -> _Change settings_ -> _Builders_ -> _desktop-linux_ -> _..._ -> _Use_ 참조)

4. [CUDA 지원 GPU](https://developer.nvidia.com/cuda-gpus)가 있다면, 학습과 추론을 가속화할 수 있습니다:

    4.1 [여기](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html#installing-with-apt)에 설명된 대로 **NVIDIA Container Toolkit**을 설치하세요. NVIDIA Container Toolkit은 [여기](https://docs.nvidia.com/cuda/wsl-user-guide/index.html#nvidia-compute-software-support-on-wsl-2)에 작성된 대로 지원됩니다.

    4.2 Docker Engine 데몬 구성에서 _nvidia_를 런타임으로 추가하세요(_Docker Desktop_ -> _Change settings_ -> _Docker Engine_ 참조). 구성에 다음 라인을 추가하세요:

    ```json
    "runtimes": {
        "nvidia": {
        "path": "nvidia-container-runtime",
        "runtimeArgs": []
    ```

    예를 들어, 전체 Docker Engine 데몬 구성 json 코드는 다음과 같아야 합니다:

    ```json
    {
      "builder": {
        "gc": {
          "defaultKeepStorage": "20GB",
          "enabled": true
        }
      },
      "experimental": false,
      "runtimes": {
        "nvidia": {
          "path": "nvidia-container-runtime",
          "runtimeArgs": []
        }
      }
    }
    ```

    그리고 Docker Desktop을 다시 시작하세요.

5. 터미널에서 `code .`를 입력하여 VS Code에서 프로젝트를 여세요. 또는 VS Code를 시작하고 UI에서 열 프로젝트를 선택할 수 있습니다.

6. 왼쪽의 VS Code _Extensions_ 메뉴에서 **Remote Development** 확장을 설치하세요.

7. DevContainer를 여세요.

`.devcontainer` 폴더가 메인 `LLMs-from-scratch` 디렉토리에 있으므로(`.`로 시작하는 폴더는 설정에 따라 OS에서 보이지 않을 수 있음), VS Code는 자동으로 이를 감지하고 프로젝트를 devcontainer에서 열지 묻습니다. 그렇지 않으면, `Ctrl + Shift + P`를 눌러 명령 팔레트를 열고 `dev containers`를 입력하여 모든 DevContainer 특정 옵션 목록을 보세요.

8. **Reopen in Container**를 선택하세요.

Docker는 이제 이전에 빌드되지 않은 경우 `.devcontainer` 구성에 지정된 Docker 이미지를 빌드하거나, 레지스트리에서 사용 가능한 경우 이미지를 가져오는 과정을 시작합니다.

전체 과정은 자동화되며 시스템과 인터넷 속도에 따라 몇 분이 걸릴 수 있습니다. 선택적으로 VS Code의 오른쪽 하단 모서리에 있는 "Starting Dev Container (show log)"를 클릭하여 현재 빌드 진행 상황을 볼 수 있습니다.

완료되면, VS Code는 자동으로 컨테이너에 연결하고 새로 생성된 Docker 개발 환경 내에서 프로젝트를 다시 엽니다. 로컬 머신에서 실행되는 것처럼 코드를 작성, 실행, 디버그할 수 있지만 Docker의 격리와 일관성의 추가 이점이 있습니다.

> **경고:**
> 빌드 과정에서 오류가 발생하는 경우, 머신에 호환 가능한 GPU가 없어서 NVIDIA 컨테이너 툴킷을 지원하지 않기 때문일 가능성이 높습니다. 이 경우, `devcontainer.json` 파일을 편집하여 `"runArgs": ["--runtime=nvidia", "--gpus=all"],` 라인을 제거하고 "Reopen Dev Container" 절차를 다시 실행하세요.

9. 완료.

이미지가 가져와지고 빌드되면, 모든 패키지가 설치된 상태로 컨테이너 내부에 프로젝트가 마운트되어 개발할 준비가 되어 있어야 합니다.

<br>

## Docker 이미지 제거

더 이상 사용할 계획이 없다면 Docker 컨테이너와 이미지를 제거하는 지침은 다음과 같습니다. 이 과정은 시스템에서 Docker 자체를 제거하지는 않고 프로젝트별 Docker 아티팩트를 정리합니다.

1. DevContainer와 연결된 이미지를 찾기 위해 모든 Docker 이미지를 나열하세요:

```bash
docker image ls
```

2. 이미지 ID 또는 이름을 사용하여 Docker 이미지를 제거하세요:

```bash
docker image rm [IMAGE_ID_OR_NAME]
```

<br>

## Docker 제거

Docker가 자신에게 맞지 않는다고 판단하고 제거하려면, 특정 운영 체제에 대한 단계를 설명하는 공식 문서를 [여기](https://docs.docker.com/desktop/uninstall/)에서 참조하세요.