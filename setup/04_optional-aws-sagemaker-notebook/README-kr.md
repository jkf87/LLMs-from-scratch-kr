# AWS CloudFormation 템플릿: LLMs-from-scratch 저장소를 포함한 Jupyter Notebook

이 CloudFormation 템플릿은 실행 역할과 LLMs-from-scratch GitHub 저장소를 포함한 Amazon SageMaker에서 GPU 지원 Jupyter 노트북을 생성합니다.

## 기능:

1. SageMaker 노트북 인스턴스에 필요한 권한을 가진 IAM 역할을 생성합니다.
2. 노트북 인스턴스 암호화를 위한 KMS 키와 별칭을 생성합니다.
3. 다음을 수행하는 노트북 인스턴스 라이프사이클 구성 스크립트를 구성합니다:
   - 사용자의 홈 디렉토리에 별도의 Miniconda 설치를 설치합니다.
   - CUDA 지원을 포함한 TensorFlow 2.15.0과 PyTorch 2.1.0이 포함된 사용자 정의 Python 환경을 생성합니다.
   - Jupyter Lab, Matplotlib 및 기타 유용한 라이브러리와 같은 추가 패키지를 설치합니다.
   - 사용자 정의 환경을 Jupyter 커널로 등록합니다.
4. GPU 지원 인스턴스 유형, 실행 역할 및 기본 코드 저장소를 포함한 지정된 구성으로 SageMaker 노트북 인스턴스를 생성합니다.

## 사용 방법:

1. CloudFormation 템플릿 파일(`cloudformation-template.yml`)을 다운로드합니다.
2. AWS Management Console에서 CloudFormation 서비스로 이동합니다.
3. 새 스택을 생성하고 템플릿 파일을 업로드합니다.
4. 노트북 인스턴스의 이름을 제공합니다(예: "LLMsFromScratchNotebook") (기본값은 LLMs-from-scratch GitHub 저장소).
5. 템플릿의 매개변수를 검토하고 수락한 후, 스택을 생성합니다.
6. 스택 생성이 완료되면, SageMaker 콘솔에서 SageMaker 노트북 인스턴스를 사용할 수 있습니다.
7. 노트북 인스턴스를 열고 미리 구성된 환경을 사용하여 LLMs-from-scratch 프로젝트 작업을 시작합니다.

## 주요 사항:

- 템플릿은 50GB 스토리지를 가진 GPU 지원(`ml.g4dn.xlarge`) 노트북 인스턴스를 생성합니다.
- CUDA 지원을 포함한 TensorFlow 2.15.0과 PyTorch 2.1.0이 포함된 사용자 정의 Miniconda 환경을 설정합니다.
- 사용자 정의 환경은 Jupyter 커널로 등록되어 노트북에서 사용할 수 있습니다.
- 템플릿은 또한 노트북 인스턴스 암호화를 위한 KMS 키와 필요한 권한을 가진 IAM 역할을 생성합니다.