# 사전학습된 가중치를 로드하는 대안적 접근법

이 폴더는 OpenAI에서 가중치를 사용할 수 없는 경우를 대비한 대안적 가중치 로딩 전략을 포함합니다.

- [weight-loading-pytorch.ipynb](weight-loading-pytorch.ipynb): (권장) 원본 TensorFlow 가중치를 변환하여 생성한 PyTorch state dict에서 가중치를 로드하는 코드를 포함합니다.

- [weight-loading-hf-transformers.ipynb](weight-loading-hf-transformers.ipynb): `transformers` 라이브러리를 통해 Hugging Face Model Hub에서 가중치를 로드하는 코드를 포함합니다.

- [weight-loading-hf-safetensors.ipynb](weight-loading-hf-safetensors.ipynb): `safetensors` 라이브러리를 통해 직접 Hugging Face Model Hub에서 가중치를 로드하는 코드를 포함합니다(Hugging Face transformer 모델 인스턴스화 과정을 생략).