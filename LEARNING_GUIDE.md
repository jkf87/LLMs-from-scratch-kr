# LLMs from Scratch 한국어 학습 가이드

## 📚 소개

이 학습 가이드는 Sebastian Raschka의 "Build a Large Language Model (From Scratch)" 책의 한국어 번역본을 효과적으로 학습하기 위한 로드맵을 제공합니다.

## 🎯 학습 목표

이 과정을 통해 다음을 배울 수 있습니다:
- 대형 언어 모델(LLM)의 기본 원리와 작동 방식
- PyTorch를 사용한 GPT 모델 구현
- 텍스트 데이터 처리 및 토큰화
- 어텐션 메커니즘의 이해와 구현
- 모델 사전학습 및 미세조정 기법
- 실제 응용을 위한 모델 최적화

## 🛠️ 시작하기 전에

### 환경 설정
1. **전체 설정 가이드**: `setup/README.md` 참고
2. **의존성 설치**: 각 장의 `requirements.txt` 확인 및 설치
3. **Python 환경**: Python 3.8 이상 권장
4. **PyTorch**: GPU 사용 가능 환경 권장 (선택사항)

### 선수 지식
- Python 프로그래밍 기초
- 기본적인 머신러닝 개념
- (선택) PyTorch 기초 - 부록 A에서 학습 가능

## 📖 장별 학습 로드맵

### 1장: 대형 언어 모델 이해하기
**학습 목표**: LLM의 개념과 역사, 활용 사례 이해

**학습 자료**:
- 이론 중심 (코드 없음)

**학습 팁**:
- LLM의 전체적인 그림을 파악하는 단계
- 다음 장들에서 구현할 내용의 개요 이해

---

### 2장: 텍스트 데이터 다루기
**학습 목표**:
- 텍스트 토큰화 이해
- 데이터로더 구현
- 임베딩 레이어 작성

**학습 자료**:
- 📓 메인 노트북: `ch02/01_main-chapter-code/ch02.ipynb` (또는 `-kr.ipynb`)
- 📓 요약 노트북: `ch02/01_main-chapter-code/dataloader.ipynb` (또는 `-kr.ipynb`)
- 📝 연습 문제: `ch02/01_main-chapter-code/exercise-solutions.ipynb` (또는 `-kr.ipynb`)

**학습 순서**:
1. 메인 노트북으로 전체 개념 학습
2. 데이터로더 요약 노트북으로 핵심 복습
3. 연습 문제로 이해도 점검

**핵심 개념**:
- Byte Pair Encoding (BPE)
- 토큰화와 어휘 구축
- 슬라이딩 윈도우 데이터로더

---

### 3장: 어텐션 메커니즘 코딩
**학습 목표**:
- Self-attention 이해 및 구현
- Multi-head attention 구현
- 인과적 어텐션(causal attention) 이해

**학습 자료**:
- 📓 메인 노트북: `ch03/01_main-chapter-code/ch03.ipynb` (또는 `-kr.ipynb`)
- 📓 요약 노트북: `ch03/01_main-chapter-code/multihead-attention.ipynb`
- 📝 연습 문제: `ch03/01_main-chapter-code/exercise-solutions.ipynb`

**학습 순서**:
1. Self-attention의 수학적 원리 이해
2. 단계별 구현 따라하기
3. Multi-head attention으로 확장
4. 마스킹을 통한 인과적 어텐션 구현

**핵심 개념**:
- Query, Key, Value 행렬
- Scaled dot-product attention
- Multi-head attention
- Causal masking

---

### 4장: GPT 모델을 처음부터 구현
**학습 목표**:
- 완전한 GPT 아키텍처 구현
- Transformer 블록 이해
- Layer normalization과 feed-forward 네트워크

**학습 자료**:
- 📓 메인 노트북: `ch04/01_main-chapter-code/ch04.ipynb`
- 🐍 핵심 코드: `ch04/01_main-chapter-code/gpt.py`
- 📝 연습 문제: `ch04/01_main-chapter-code/exercise-solutions.ipynb`

**학습 순서**:
1. Transformer 블록 구조 이해
2. 레이어별 구현 (layer norm, feed-forward)
3. 전체 GPT 모델 조립
4. `gpt.py`로 최종 구현 확인

**핵심 개념**:
- Transformer 블록
- Residual connections
- Layer normalization
- Position-wise feed-forward networks

---

### 5장: 비라벨 데이터로 사전학습
**학습 목표**:
- 언어 모델 사전학습 과정 이해
- 학습 루프 구현
- 모델 평가 및 저장

**학습 자료**:
- 📓 메인 노트북: `ch05/01_main-chapter-code/ch05.ipynb`
- 🐍 학습 스크립트: `ch05/01_main-chapter-code/gpt_train.py`
- 🐍 생성 스크립트: `ch05/01_main-chapter-code/gpt_generate.py`
- 📝 연습 문제: `ch05/01_main-chapter-code/exercise-solutions.ipynb`

**학습 순서**:
1. 사전학습 데이터 준비
2. 학습 루프 구현
3. 모델 학습 실행
4. 텍스트 생성으로 결과 확인

**핵심 개념**:
- Cross-entropy loss
- AdamW optimizer
- Learning rate scheduling
- Text generation strategies

---

### 6장: 텍스트 분류를 위한 미세조정
**학습 목표**:
- 사전학습된 모델 미세조정
- 분류 작업을 위한 모델 수정
- 평가 지표 이해

**학습 자료**:
- 📓 메인 노트북: `ch06/01_main-chapter-code/ch06.ipynb`
- 🐍 미세조정 스크립트: `ch06/01_main-chapter-code/gpt_class_finetune.py`
- 📝 연습 문제: `ch06/01_main-chapter-code/exercise-solutions.ipynb`

**학습 순서**:
1. 분류를 위한 모델 헤드 추가
2. 미세조정 데이터셋 준비
3. 미세조정 실행
4. 성능 평가

**핵심 개념**:
- Transfer learning
- Classification head
- Fine-tuning strategies
- Evaluation metrics

---

### 7장: 지시 따르기 미세조정
**학습 목표**:
- 지시사항 기반 미세조정
- 프롬프트 엔지니어링
- 모델 평가 방법

**학습 자료**:
- 📓 메인 노트북: `ch07/01_main-chapter-code/ch07.ipynb`
- 🐍 지시 미세조정 스크립트: `ch07/01_main-chapter-code/gpt_instruction_finetuning.py`
- 🐍 평가 스크립트: `ch07/01_main-chapter-code/ollama_evaluate.py`
- 📝 연습 문제: `ch07/01_main-chapter-code/exercise-solutions.ipynb`

**학습 순서**:
1. 지시사항 데이터셋 형식 이해
2. 미세조정 실행
3. Ollama를 사용한 평가
4. 프롬프트 최적화

**핵심 개념**:
- Instruction tuning
- Prompt engineering
- Model evaluation
- Response quality assessment

---

## 📑 부록 활용 가이드

### 부록 A: PyTorch 입문
**대상**: PyTorch 초보자

**학습 자료**:
- 📓 Part 1: `appendix-A/01_main-chapter-code/code-part1.ipynb`
- 📓 Part 2: `appendix-A/01_main-chapter-code/code-part2.ipynb`
- 🐍 DDP 예제: `appendix-A/01_main-chapter-code/DDP-script.py`
- 📝 연습 문제: `appendix-A/01_main-chapter-code/exercise-solutions.ipynb`

**권장 학습 시점**: 2장 시작 전

---

### 부록 D: 학습 루프에 유용한 요소 추가
**대상**: 학습 과정 최적화에 관심 있는 학습자

**학습 자료**:
- 📓 메인 노트북: `appendix-D/01_main-chapter-code/appendix-D.ipynb`

**권장 학습 시점**: 5장 완료 후

**핵심 개념**:
- Gradient clipping
- Learning rate warmup
- Model checkpointing
- Early stopping

---

### 부록 E: LoRA로 파라미터 효율적 미세조정
**대상**: 효율적인 미세조정 기법에 관심 있는 학습자

**학습 자료**:
- 📓 메인 노트북: `appendix-E/01_main-chapter-code/appendix-E.ipynb`

**권장 학습 시점**: 6장 또는 7장 완료 후

**핵심 개념**:
- Low-Rank Adaptation (LoRA)
- Parameter-efficient fine-tuning
- Memory optimization

---

## 🎓 학습 전략

### 추천 학습 순서

#### 🟢 기초 과정 (필수)
1. **1장** → LLM 개념 이해
2. **부록 A** (필요시) → PyTorch 기초
3. **2장** → 텍스트 데이터 처리
4. **3장** → 어텐션 메커니즘
5. **4장** → GPT 구현

#### 🟡 중급 과정 (권장)
6. **5장** → 사전학습
7. **부록 D** → 학습 최적화
8. **6장** → 분류 미세조정
9. **7장** → 지시 미세조정

#### 🔴 고급 과정 (선택)
10. **부록 E** → LoRA 미세조정

### 효과적인 학습 방법

1. **코드 실행 우선**
   - 노트북을 처음부터 끝까지 실행하며 결과 확인
   - 각 셀의 출력값을 이해하려 노력

2. **주석 읽기**
   - 한국어 번역본(`-kr.ipynb`)의 주석을 꼼꼼히 읽기
   - 이해가 안 되는 부분은 원문 참조

3. **실습 반복**
   - 동일한 노트북을 여러 번 실행
   - 파라미터를 변경하며 결과 관찰

4. **연습 문제 풀이**
   - 각 장의 연습 문제를 반드시 풀어보기
   - 막히면 해설 참고

5. **요약 노트북 활용**
   - 메인 노트북 학습 후 요약 노트북으로 복습
   - 핵심 코드만 모아둔 `.py` 파일 참고

## 🔧 실습 팁

### 노트북 실행
```bash
# Jupyter Notebook 실행
jupyter notebook

# 또는 JupyterLab
jupyter lab
```

### 의존성 설치
```bash
# 특정 장의 의존성 설치
cd ch02/01_main-chapter-code
pip install -r requirements.txt
```

### 테스트 실행
각 디렉토리의 README를 확인하여 해당 테스트 명령어 실행

## 📝 번역 파일 활용

### 한국어 번역 완료 파일
- ✅ `ch02/01_main-chapter-code/ch02-kr.ipynb` (2장 메인)
- ✅ `ch02/01_main-chapter-code/dataloader-kr.ipynb` (2장 데이터로더)
- ✅ `ch02/01_main-chapter-code/exercise-solutions-kr.ipynb` (2장 연습문제)

### 파일명 규칙
- 원본 파일: `ch02.ipynb`
- 한국어 번역: `ch02-kr.ipynb`

## 🤝 학습 커뮤니티

### 질문하기
- 코드 이슈: GitHub Issues 활용
- 번역 개선: Pull Request 환영

### 기여하기
- 번역 오류 수정
- 추가 예제 작성
- 학습 가이드 개선

## 📌 참고 자료

### 공식 자료
- 원서: "Build a Large Language Model (From Scratch)" by Sebastian Raschka
- 원본 코드: [GitHub Repository]

### 추가 학습 자료
- **부록 B**: 참고문헌과 추가 읽을거리
- **부록 C**: 연습 문제 해설 요약

## ⚡ 빠른 참조

### 장별 핵심 파일

| 장 | 메인 노트북 | 요약/스크립트 | 연습문제 |
|---|-----------|------------|---------|
| 2 | `ch02.ipynb` | `dataloader.ipynb` | `exercise-solutions.ipynb` |
| 3 | `ch03.ipynb` | `multihead-attention.ipynb` | `exercise-solutions.ipynb` |
| 4 | `ch04.ipynb` | `gpt.py` | `exercise-solutions.ipynb` |
| 5 | `ch05.ipynb` | `gpt_train.py`, `gpt_generate.py` | `exercise-solutions.ipynb` |
| 6 | `ch06.ipynb` | `gpt_class_finetune.py` | `exercise-solutions.ipynb` |
| 7 | `ch07.ipynb` | `gpt_instruction_finetuning.py` | `exercise-solutions.ipynb` |

---

**Happy Learning! 즐거운 학습 되세요!** 🚀
