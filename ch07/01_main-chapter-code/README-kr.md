# Chapter 7: 지시사항을 따르기 위한 미세조정 (Finetuning to Follow Instructions)

### 메인 챕터 코드

- [ch07.ipynb](ch07.ipynb) 챕터에 나타나는 모든 코드를 포함합니다
- [previous_chapters.py](previous_chapters.py) 이전 챕터에서 코딩하고 훈련한 GPT 모델과 많은 유틸리티 함수가 포함된 Python 모듈로, 이 챕터에서 재사용합니다
- [gpt_download.py](gpt_download.py) 사전훈련된 GPT 모델 가중치를 다운로드하기 위한 유틸리티 함수를 포함합니다
- [exercise-solutions.ipynb](exercise-solutions.ipynb) 이 챕터의 연습 문제 해답을 포함합니다


### 선택적 코드

- [load-finetuned-model.ipynb](load-finetuned-model.ipynb) 이 챕터에서 생성한 지시사항 미세조정된 모델을 로드하는 독립적인 Jupyter 노트북입니다

- [gpt_instruction_finetuning.py](gpt_instruction_finetuning.py) 메인 챕터에서 설명한 대로 모델을 지시사항 미세조정하는 독립적인 Python 스크립트입니다 (미세조정 부분에 집중한 챕터 요약이라고 생각하면 됩니다)

사용법:

```bash
python gpt_instruction_finetuning.py
```

```
matplotlib version: 3.9.0
tiktoken version: 0.7.0
torch version: 2.3.1
tqdm version: 4.66.4
tensorflow version: 2.16.1
--------------------------------------------------
Training set length: 935
Validation set length: 55
Test set length: 110
--------------------------------------------------
Device: cpu
--------------------------------------------------
File already exists and is up-to-date: gpt2/355M/checkpoint
File already exists and is up-to-date: gpt2/355M/encoder.json
File already exists and is up-to-date: gpt2/355M/hparams.json
File already exists and is up-to-date: gpt2/355M/model.ckpt.data-00000-of-00001
File already exists and is up-to-date: gpt2/355M/model.ckpt.index
File already exists and is up-to-date: gpt2/355M/model.ckpt.meta
File already exists and is up-to-date: gpt2/355M/vocab.bpe
Loaded model: gpt2-medium (355M)
--------------------------------------------------
Initial losses
   Training loss: 3.839039182662964
   Validation loss: 3.7619192123413088
Ep 1 (Step 000000): Train loss 2.611, Val loss 2.668
Ep 1 (Step 000005): Train loss 1.161, Val loss 1.131
Ep 1 (Step 000010): Train loss 0.939, Val loss 0.973
...
Training completed in 15.66 minutes.
Plot saved as loss-plot-standalone.pdf
--------------------------------------------------
Generating responses
100%|█████████████████████████████████████████████████████████| 110/110 [06:57<00:00,  3.80s/it]
Responses saved as instruction-data-with-response-standalone.json
Model saved as gpt2-medium355M-sft-standalone.pth
```

- [ollama_evaluate.py](ollama_evaluate.py) 메인 챕터에서 설명한 대로 미세조정된 모델의 응답을 평가하는 독립적인 Python 스크립트입니다 (평가 부분에 집중한 챕터 요약이라고 생각하면 됩니다)

사용법:

```bash
python ollama_evaluate.py --file_path instruction-data-with-response-standalone.json
```

```
Ollama running: True
Scoring entries: 100%|███████████████████████████████████████| 110/110 [01:08<00:00,  1.62it/s]
Number of scores: 110 of 110
Average score: 51.75
```

- [exercise_experiments.py](exercise_experiments.py) 연습 문제 해답을 구현하는 선택적 스크립트입니다; 자세한 내용은 [exercise-solutions.ipynb](exercise-solutions.ipynb)를 참조하세요