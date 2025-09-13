# 5장: 비라벨 데이터로 사전학습

### 메인 장 코드

- [ch05.ipynb](ch05.ipynb)는 장에 나타나는 모든 코드를 포함합니다.
- [previous_chapters.py](previous_chapters.py)는 이전 장의 `MultiHeadAttention` 모듈과 `GPTModel` 클래스를 포함하는 Python 모듈로, GPT 모델을 사전학습하기 위해 [ch05.ipynb](ch05.ipynb)에서 가져옵니다.
- [gpt_download.py](gpt_download.py)는 사전학습된 GPT 모델 가중치를 다운로드하기 위한 유틸리티 함수를 포함합니다.
- [exercise-solutions.ipynb](exercise-solutions.ipynb)는 이 장의 연습 문제 해답을 포함합니다.

### 선택적 코드

- [gpt_train.py](gpt_train.py)는 GPT 모델을 학습시키기 위해 [ch05.ipynb](ch05.ipynb)에서 구현한 코드가 포함된 독립적인 Python 스크립트 파일입니다(이 장을 요약하는 코드 파일로 생각할 수 있습니다).
- [gpt_generate.py](gpt_generate.py)는 OpenAI에서 사전학습된 모델 가중치를 로드하고 사용하기 위해 [ch05.ipynb](ch05.ipynb)에서 구현한 코드가 포함된 독립적인 Python 스크립트 파일입니다.
