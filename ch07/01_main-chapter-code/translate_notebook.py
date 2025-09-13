#!/usr/bin/env python3
"""
Jupyter Notebook 번역 스크립트
ch07.ipynb를 한국어로 번역하여 ch07-kr.ipynb로 저장
"""

import json
import sys
import os

def translate_text(text):
    """영어 텍스트를 한국어로 번역하는 함수"""
    
    # 기본 번역 매핑
    translations = {
        # Chapter 제목
        "Chapter 7: Following Instructions to Finetune a LLM": "7장: 지시사항을 따라 LLM 미세조정하기",
        "Following Instructions to Finetune a LLM": "지시사항을 따라 LLM 미세조정하기",
        
        # 섹션 제목
        "7.1 Preparing a dataset for supervised instruction finetuning": "7.1 지도 지시사항 미세조정을 위한 데이터셋 준비",
        "7.2 Batching inputs of variable lengths": "7.2 가변 길이 입력의 배치 처리",
        "7.3 Creating data loaders for instruction finetuning": "7.3 지시사항 미세조정용 데이터 로더 생성",
        "7.4 Loading a pretrained model for finetuning": "7.4 미세조정용 사전훈련된 모델 로딩",
        "7.5 Instruction finetuning the LLM": "7.5 LLM 지시사항 미세조정",
        "7.6 Evaluating instruction responses": "7.6 지시사항 응답 평가",
        "7.7 Summary": "7.7 요약",
        
        # 일반적인 용어들
        "instruction finetuning": "지시사항 미세조정(instruction finetuning)",
        "supervised finetuning": "지도 미세조정(supervised finetuning)",
        "pretrained": "사전훈련된(pretrained)",
        "dataset": "데이터셋(dataset)",
        "batch": "배치(batch)",
        "data loader": "데이터 로더(data loader)",
        "model": "모델(model)",
        "evaluation": "평가(evaluation)",
        "response": "응답(response)",
        "prompt": "프롬프트(prompt)",
        "tokenizer": "토크나이저(tokenizer)",
        "fine-tuning": "미세조정(fine-tuning)",
        "finetuning": "미세조정(finetuning)",
        "training": "훈련(training)",
        "validation": "검증(validation)",
        "loss": "손실(loss)",
        "accuracy": "정확도(accuracy)",
        "performance": "성능(performance)",
        
        # 길고 복잡한 텍스트들
        "In this chapter, we will instruction-finetune the 124 million parameter GPT model from Chapter 5 to follow basic instructions such as answering questions about a piece of text.": "이 장에서는 5장의 1억 2천 4백만 개 매개변수 GPT 모델을 지시사항 미세조정하여 텍스트 조각에 대한 질문 답변과 같은 기본 지시사항을 따르도록 만들겠습니다.",
        
        "Instruction finetuning, also known as supervised finetuning, trains a pretrained LLM on instruction-response pairs.": "지시사항 미세조정(instruction finetuning), 지도 미세조정(supervised finetuning)이라고도 알려진 이 방법은 사전훈련된 LLM을 지시-응답 쌍에서 훈련시킵니다.",
        
        "The goal is to improve the model's ability to understand and follow human instructions across various tasks.": "목표는 다양한 작업에서 인간의 지시사항을 이해하고 따르는 모델의 능력을 향상시키는 것입니다.",
        
        "This chapter covers instruction finetuning specifically.": "이 장은 구체적으로 지시사항 미세조정을 다룹니다.",
    }
    
    # 번역된 텍스트가 있으면 반환
    for eng, kor in translations.items():
        if eng.lower() in text.lower():
            text = text.replace(eng, kor)
    
    # 기타 일반적인 패턴 번역
    if text.startswith("Below is an instruction"):
        return "다음은 작업을 설명하는 지시사항입니다. 요청을 적절히 완성하는 응답을 작성하세요."
    
    return text

def process_notebook(input_path, output_path):
    """노트북 파일을 읽어서 번역한 후 새 파일로 저장"""
    
    try:
        with open(input_path, 'r', encoding='utf-8') as f:
            notebook = json.load(f)
        
        # 셀들을 순회하며 마크다운 셀만 번역
        for cell in notebook.get('cells', []):
            if cell.get('cell_type') == 'markdown':
                if 'source' in cell and isinstance(cell['source'], list):
                    # 리스트 형태의 source 처리
                    translated_source = []
                    for line in cell['source']:
                        translated_line = translate_text(line)
                        translated_source.append(translated_line)
                    cell['source'] = translated_source
                elif 'source' in cell and isinstance(cell['source'], str):
                    # 문자열 형태의 source 처리  
                    cell['source'] = translate_text(cell['source'])
        
        # 번역된 노트북을 새 파일로 저장
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(notebook, f, ensure_ascii=False, indent=1)
        
        print(f"번역 완료: {output_path}")
        return True
        
    except Exception as e:
        print(f"오류 발생: {e}")
        return False

if __name__ == "__main__":
    input_file = "/Users/conanssam-m4/LLMs-from-scratch-kr/ch07/01_main-chapter-code/ch07.ipynb"
    output_file = "/Users/conanssam-m4/LLMs-from-scratch-kr/ch07/01_main-chapter-code/ch07-kr.ipynb"
    
    if process_notebook(input_file, output_file):
        print("ch07.ipynb 번역이 완료되었습니다.")
    else:
        print("번역 중 오류가 발생했습니다.")