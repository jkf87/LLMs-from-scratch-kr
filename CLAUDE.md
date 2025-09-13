# LLMs from Scratch Korean Translation 프로젝트

## 개요
이 프로젝트는 Sebastian Raschka의 "Build a Large Language Model (From Scratch)" 책의 한국어 번역과 예제 코드를 포함합니다.

## 장별 빠른 링크

### 설정/환경 설정
- 전체 가이드: `setup/README.md`

### 1장: 대형 언어 모델 이해하기
- 코드 없음

### 2장: 텍스트 데이터 다루기
- 메인 노트북: `ch02/01_main-chapter-code/ch02.ipynb`
- 요약(데이터로더): `ch02/01_main-chapter-code/dataloader.ipynb`
- 연습 해설: `ch02/01_main-chapter-code/exercise-solutions.ipynb`

### 3장: 어텐션 메커니즘 코딩
- 메인 노트북: `ch03/01_main-chapter-code/ch03.ipynb`
- 요약(MHA): `ch03/01_main-chapter-code/multihead-attention.ipynb`
- 연습 해설: `ch03/01_main-chapter-code/exercise-solutions.ipynb`

### 4장: GPT 모델을 처음부터 구현
- 메인 노트북: `ch04/01_main-chapter-code/ch04.ipynb`
- 핵심 코드(요약): `ch04/01_main-chapter-code/gpt.py`
- 연습 해설: `ch04/01_main-chapter-code/exercise-solutions.ipynb`

### 5장: 비라벨 데이터로 사전학습
- 메인 노트북: `ch05/01_main-chapter-code/ch05.ipynb`
- 학습 스크립트(요약): `ch05/01_main-chapter-code/gpt_train.py`
- 생성 스크립트(요약): `ch05/01_main-chapter-code/gpt_generate.py`
- 연습 해설: `ch05/01_main-chapter-code/exercise-solutions.ipynb`

### 6장: 텍스트 분류를 위한 미세조정
- 메인 노트북: `ch06/01_main-chapter-code/ch06.ipynb`
- 분류 미세조정 스크립트: `ch06/01_main-chapter-code/gpt_class_finetune.py`
- 연습 해설: `ch06/01_main-chapter-code/exercise-solutions.ipynb`

### 7장: 지시 따르기 미세조정
- 메인 노트북: `ch07/01_main-chapter-code/ch07.ipynb`
- 지시 미세조정 스크립트(요약): `ch07/01_main-chapter-code/gpt_instruction_finetuning.py`
- Ollama 평가(요약): `ch07/01_main-chapter-code/ollama_evaluate.py`
- 연습 해설: `ch07/01_main-chapter-code/exercise-solutions.ipynb`

### 부록 A: PyTorch 입문
- Part 1: `appendix-A/01_main-chapter-code/code-part1.ipynb`
- Part 2: `appendix-A/01_main-chapter-code/code-part2.ipynb`
- DDP 예제 스크립트: `appendix-A/01_main-chapter-code/DDP-script.py`
- 연습 해설: `appendix-A/01_main-chapter-code/exercise-solutions.ipynb`

### 부록 B: 참고문헌과 추가 읽을거리
- 코드 없음

### 부록 C: 연습 문제 해설 요약
- 코드 없음

### 부록 D: 학습 루프에 유용한 요소 추가
- 메인 노트북: `appendix-D/01_main-chapter-code/appendix-D.ipynb`

### 부록 E: LoRA로 파라미터 효율적 미세조정
- 메인 노트북: `appendix-E/01_main-chapter-code/appendix-E.ipynb`

## Claude Code 명령어
- 테스트 실행: 각 디렉토리의 README를 확인하여 해당 테스트 명령어 확인
- 환경 설정: `setup/README.md` 참고
- 의존성 설치: 각 chapter의 requirements.txt 확인

## 번역 방법론

### Jupyter 노트북 번역 지침
1. **파일명 규칙**: 원본 파일명 뒤에 `-kr` 접미사 추가 (예: `ch02.ipynb` → `ch02-kr.ipynb`)

2. **번역 범위**:
   - 마크다운 셀: 모든 텍스트 번역
   - 코드 셀: 코드는 유지, 주석만 번역
   - 이미지/링크: 원본 그대로 유지

3. **번역 원칙**:
   - 기술 용어는 한국어 번역 후 영어 원문을 괄호로 병기 (예: "어텐션 메커니즘(attention mechanism)")
   - 섹션 제목과 헤더는 완전 번역
   - 설명 텍스트는 자연스러운 한국어로 번역
   - 변수명, 함수명, 클래스명은 번역하지 않고 원문 유지

4. **코드 주석 번역**:
   - 인라인 주석: `# 각 항목에서 공백을 제거한 후 빈 문자열을 필터링합니다.`
   - 블록 주석: 전체 의미를 살려 번역
   - 기술적 설명: 정확한 번역 우선

5. **품질 관리**:
   - 전문 용어 일관성 유지
   - 문맥에 맞는 자연스러운 번역
   - 원문의 구조와 형식 유지

### 번역 완료 파일
- ✅ `ch02/01_main-chapter-code/ch02-kr.ipynb` (2장 메인 노트북)
- ✅ `ch02/01_main-chapter-code/dataloader-kr.ipynb` (2장 데이터로더)
- ✅ `ch02/01_main-chapter-code/exercise-solutions-kr.ipynb` (2장 연습 문제 해답)

## 프로젝트 구조
- 각 장은 `ch##/` 형태의 디렉토리로 구성
- 주요 코드는 `01_main-chapter-code/` 하위에 위치
- 연습 문제와 해설은 각 장의 노트북에 포함
- 한국어 번역 파일은 원본 파일명에 `-kr` 접미사 추가