#!/usr/bin/env python3
"""
코드 주석 번역 스크립트
Jupyter Notebook의 코드 셀 내 주석들을 한국어로 번역
"""

import json
import re
import sys

def translate_comment(comment):
    """영어 주석을 한국어로 번역"""
    
    # 주석 번역 매핑
    translations = {
        # 일반적인 주석들
        "# Pre-tokenize texts": "# 텍스트를 사전 토큰화합니다",
        "# Add an <|endoftext|> token": "# <|endoftext|> 토큰을 추가합니다",
        "# Pad sequences to max_length": "# 시퀀스를 max_length로 패딩합니다",
        "# Truncate the last token for inputs": "# 입력용으로 마지막 토큰을 자릅니다",
        "# Shift +1 to the right for targets": "# 대상용으로 +1 오른쪽으로 시프트합니다",
        "# Find the longest sequence in the batch": "# 배치에서 가장 긴 시퀀스를 찾습니다",
        "# Pad and prepare inputs and targets": "# 입력과 대상을 패딩하고 준비합니다",
        "# Convert list of inputs and targets to tensors": "# 입력과 대상 리스트를 텐서로 변환합니다",
        "# Transfer to target device": "# 대상 장치로 전송합니다",
        "# Replace all but the first padding tokens": "# 첫 번째를 제외한 모든 패딩 토큰을 교체합니다",
        "# Tokenizer": "# 토크나이저",
        "# Deep learning library": "# 딥러닝 라이브러리",
        "# Vocabulary size": "# 어휘 크기",
        "# Context length": "# 컨텍스트 길이", 
        "# Dropout rate": "# 드롭아웃 비율",
        "# Query-key-value bias": "# Query-key-value 편향",
        "# Separate list for instruction lengths": "# 지시사항 길이를 위한 별도 리스트",
        "# collect instruction lengths": "# 지시사항 길이를 수집합니다",
        "# return both instruction lengths and texts separately": "# 지시사항 길이와 텍스트를 별도로 반환합니다",
        "# batch is now a tuple": "# batch가 이제 튜플입니다",
        "# Mask all input and instruction tokens in the targets": "# 대상에서 모든 입력 및 지시사항 토큰을 마스킹합니다",
        "# Optionally truncate to maximum sequence length": "# 선택적으로 최대 시퀀스 길이로 자릅니다",
        
        # 새로운 내용 관련
        "# NEW: Use `format_input_phi` and adjust the response text template": "# 새로운 내용: `format_input_phi`를 사용하고 응답 텍스트 템플릿을 조정합니다",
        "# New: Adjust ###Response -> <|assistant|>": "# 새로운 내용: ###Response -> <|assistant|>로 조정합니다",
        "# New: return both instruction lengths and texts separately": "# 새로운 내용: 지시사항 길이와 텍스트를 별도로 반환합니다",
        "# New: batch is now a tuple": "# 새로운 내용: batch가 이제 튜플입니다",
        "# New: Mask all input and instruction tokens in the targets": "# 새로운 내용: 대상에서 모든 입력 및 지시사항 토큰을 마스킹합니다",
        "# New: Separate list for instruction lengths": "# 새로운 내용: 지시사항 길이를 위한 별도 리스트",
        "# New: collect instruction lengths": "# 새로운 내용: 지시사항 길이를 수집합니다"
    }
    
    # 정확한 매치 찾기
    for eng, kor in translations.items():
        if comment.strip() == eng:
            return kor
    
    # 부분 매치 시도
    for eng, kor in translations.items():
        if eng.lower() in comment.lower():
            return comment.replace(eng, kor)
    
    return comment

def translate_code_comments(notebook_path):
    """노트북의 코드 셀에서 주석들을 번역"""
    
    try:
        with open(notebook_path, 'r', encoding='utf-8') as f:
            notebook = json.load(f)
        
        modified = False
        
        # 코드 셀들을 순회하며 주석 번역
        for cell in notebook.get('cells', []):
            if cell.get('cell_type') == 'code':
                if 'source' in cell and isinstance(cell['source'], list):
                    # 리스트 형태의 source 처리
                    new_source = []
                    for line in cell['source']:
                        # 주석이 포함된 라인 찾기
                        if '#' in line:
                            # 주석 부분만 번역
                            parts = line.split('#', 1)  # 첫 번째 #에서만 분할
                            if len(parts) == 2:
                                code_part = parts[0]
                                comment_part = '#' + parts[1]
                                translated_comment = translate_comment(comment_part)
                                if translated_comment != comment_part:
                                    line = code_part + translated_comment
                                    modified = True
                        new_source.append(line)
                    
                    cell['source'] = new_source
        
        if modified:
            # 수정된 노트북을 다시 저장
            with open(notebook_path, 'w', encoding='utf-8') as f:
                json.dump(notebook, f, ensure_ascii=False, indent=1)
            print(f"주석 번역 완료: {notebook_path}")
        else:
            print(f"번역할 주석을 찾을 수 없음: {notebook_path}")
        
        return True
        
    except Exception as e:
        print(f"오류 발생: {e}")
        return False

def main():
    """메인 함수"""
    
    files_to_translate = [
        "/Users/conanssam-m4/LLMs-from-scratch-kr/ch07/01_main-chapter-code/ch07-kr.ipynb",
        "/Users/conanssam-m4/LLMs-from-scratch-kr/ch07/01_main-chapter-code/exercise-solutions-kr.ipynb",
        "/Users/conanssam-m4/LLMs-from-scratch-kr/ch07/01_main-chapter-code/load-finetuned-model-kr.ipynb"
    ]
    
    for file_path in files_to_translate:
        translate_code_comments(file_path)

if __name__ == "__main__":
    main()