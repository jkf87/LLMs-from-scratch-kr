# Chapter 7: 지시사항을 따르기 위한 미세조정 (Finetuning to Follow Instructions)

이 폴더에는 모델 평가에 사용할 수 있는 유틸리티 코드가 포함되어 있습니다.



&nbsp;
## OpenAI API를 사용한 지시사항 응답 평가 (Evaluating Instruction Responses Using the OpenAI API)


- [llm-instruction-eval-openai.ipynb](llm-instruction-eval-openai.ipynb) 노트북은 OpenAI의 GPT-4를 사용하여 지시사항 미세조정된 모델이 생성한 응답을 평가합니다. 다음 형식의 JSON 파일과 함께 작동합니다:

```python
{
    "instruction": "What is the atomic number of helium?",
    "input": "",
    "output": "The atomic number of helium is 2.",               # <-- 테스트 세트에서 제공되는 목표값
    "model 1 response": "\nThe atomic number of helium is 2.0.", # <-- LLM의 응답
    "model 2 response": "\nThe atomic number of helium is 3."    # <-- 두 번째 LLM의 응답
},
```

&nbsp;
## Ollama를 사용한 로컬 지시사항 응답 평가 (Evaluating Instruction Responses Locally Using Ollama)

- [llm-instruction-eval-ollama.ipynb](llm-instruction-eval-ollama.ipynb) 노트북은 위의 노트북에 대한 대안을 제공하며, Ollama를 통해 로컬로 다운로드된 Llama 3 모델을 활용합니다.