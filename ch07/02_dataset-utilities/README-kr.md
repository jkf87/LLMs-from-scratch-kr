# Chapter 7: 지시사항을 따르기 위한 미세조정 (Finetuning to Follow Instructions)

이 폴더에는 지시사항 데이터셋을 준비하는 데 사용할 수 있는 유틸리티 코드가 포함되어 있습니다.

다음을 통해 추가 패키지 요구사항을 설치합니다:

```bash
pip install -r requirements-extra.txt
```




### 근사 중복 찾기 (Finding Near Duplicates)

`find-near-duplicates.py` 함수는 지시사항 데이터셋에서 중복 및 근사 중복을 식별하는 데 사용할 수 있습니다. 예를 들어,



```bash
python find-near-duplicates.py --json_file instruction-examples.json
```

```
scikit-learn version: 1.3.1


==================================================
Searching 'instruction' for duplicates ...
==================================================
Duplicate pair found with similarity 0.94:
1. Edit the following sentence to make it more formal.
2. Edit the sentence to make it more formal.

Duplicate pair found with similarity 1.00:
1. Name a dwarf planet in our solar system.
2. Name a dwarf planet in our solar system.

Duplicate pair found with similarity 0.91:
1. Change the sentences from active voice to passive voice.
2. Change the sentence from passive to active voice.



==================================================
Searching 'input' for duplicates ...
==================================================
No duplicates found


==================================================
Searching 'output' for duplicates ...
==================================================
Duplicate pair found with similarity 1.00:
1. One dwarf planet in our solar system is Pluto.
2. One dwarf planet in our solar system is Pluto.


```

&nbsp;
민감도를 줄이거나 높이기 위해 0과 1 사이의 값으로 `--threshold` 설정을 사용할 수 있습니다.
기본 임계값은 0.9입니다.



&nbsp;
## 수동태 항목 생성 (Creating Passive Voice Entries)

- [create-passive-voice-entries.ipynb](create-passive-voice-entries.ipynb) 노트북은 OpenAI의 GPT-4를 사용하여 지시사항 데이터셋에 대한 "수동태" 항목을 생성합니다. 아래 예시와 같습니다:

```python
{  
   'instruction': 'Identify the verb in the following sentence',
   'input': 'The cat sleeps on the couch.',
   'output': 'The verb in the sentence is "sleeps."',
   'output_2': 'The sentence is "sleeps."'   #  <---- 새로 생성된 항목
}  
```