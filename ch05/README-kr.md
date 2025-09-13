# 5장: 비라벨 데이터로 사전학습

&nbsp;
## 메인 장 코드

- [01_main-chapter-code](01_main-chapter-code)는 메인 장 코드를 포함합니다.

&nbsp;
## 보너스 자료

- [02_alternative_weight_loading](02_alternative_weight_loading)는 OpenAI에서 모델 가중치를 사용할 수 없는 경우를 대비해 대체 위치에서 GPT 모델 가중치를 로드하는 코드를 포함합니다.
- [03_bonus_pretraining_on_gutenberg](03_bonus_pretraining_on_gutenberg)는 프로젝트 구텐베르크의 전체 도서 말뭉치에서 LLM을 더 오래 사전학습시키는 코드를 포함합니다.
- [04_learning_rate_schedulers](04_learning_rate_schedulers)는 학습률 스케줄러(learning rate schedulers)와 그래디언트 클리핑(gradient clipping)을 포함한 더 정교한 학습 함수를 구현하는 코드를 포함합니다.
- [05_bonus_hparam_tuning](05_bonus_hparam_tuning)는 선택적 하이퍼파라미터 튜닝(hyperparameter tuning) 스크립트를 포함합니다.
- [06_user_interface](06_user_interface)는 사전학습된 LLM과 상호작용하기 위한 대화형 사용자 인터페이스를 구현합니다.
- [07_gpt_to_llama](07_gpt_to_llama)는 GPT 아키텍처 구현을 Llama 3.2로 변환하고 Meta AI에서 사전학습된 가중치를 로드하는 단계별 가이드를 포함합니다.
- [08_memory_efficient_weight_loading](08_memory_efficient_weight_loading)는 PyTorch의 `load_state_dict` 메서드를 통해 모델 가중치를 더 효율적으로 로드하는 방법을 보여주는 보너스 노트북을 포함합니다.
- [09_extending-tokenizers](09_extending-tokenizers)는 GPT-2 BPE 토크나이저(tokenizer)의 처음부터(from-scratch) 구현을 포함합니다.
- [10_llm-training-speed](10_llm-training-speed)는 LLM 학습 속도를 개선하기 위한 PyTorch 성능 팁을 보여줍니다.
- [11_qwen3](11_qwen3)는 기본(base), 추론(reasoning), 코딩(coding) 모델 변형의 사전학습된 가중치를 로드하는 코드를 포함하여 Qwen3 0.6B와 Qwen3 30B-A3B (Mixture-of-Experts)의 처음부터 구현을 제공합니다.



<br>
<br>

[![Link to the video](https://img.youtube.com/vi/Zar2TJv-sE0/0.jpg)](https://www.youtube.com/watch?v=Zar2TJv-sE0)