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

## 🧠 핵심 개념 오버뷰

### LLM 구축의 전체 여정

이 책은 **7단계**로 나누어 GPT 스타일의 대형 언어 모델을 처음부터 구축합니다:

```
┌─────────────────────────────────────────────────────────────┐
│  1장: LLM이란 무엇인가?                                        │
│  개념: Transformer, GPT, 사전학습, 미세조정                     │
└─────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────┐
│  2장: 텍스트 → 숫자 변환                                       │
│  개념: 토큰화(Tokenization) → 어휘(Vocabulary) → 임베딩         │
│       → 데이터로더(DataLoader)                                │
└─────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────┐
│  3장: 어텐션 메커니즘 (핵심!)                                   │
│  개념: Self-Attention → Multi-Head Attention                │
│       → Causal Masking (미래 정보 차단)                       │
└─────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────┐
│  4장: GPT 아키텍처 조립                                        │
│  개념: Transformer Block → Layer Norm → Residual Connection │
│       → Feed-Forward Network → 전체 GPT 모델                 │
└─────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────┐
│  5장: 사전학습 (Pre-training)                                 │
│  개념: 다음 단어 예측 → Loss 계산 → Optimizer → 학습 루프     │
│       → 텍스트 생성                                           │
└─────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────┐
│  6장: 분류 미세조정 (Classification Fine-tuning)              │
│  개념: Transfer Learning → Classification Head               │
│       → Task-Specific Fine-tuning                           │
└─────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────┐
│  7장: 지시 따르기 미세조정 (Instruction Fine-tuning)           │
│  개념: Instruction Dataset → Prompt Engineering              │
│       → Conversational AI                                   │
└─────────────────────────────────────────────────────────────┘
```

### 핵심 개념 맵

#### 📊 레이어 1: 데이터 처리 (2장)
- **토큰화(Tokenization)**: 텍스트를 작은 단위로 분리
  - BPE (Byte Pair Encoding): 효율적인 단어 분리 알고리즘
- **어휘(Vocabulary)**: 모든 가능한 토큰의 사전
- **임베딩(Embedding)**: 토큰을 숫자 벡터로 변환
- **위치 인코딩(Positional Encoding)**: 단어 순서 정보 추가

#### 🎯 레이어 2: 어텐션 메커니즘 (3장)
- **Self-Attention**: 문장 내 단어들 간의 관계 파악
  - Query, Key, Value 행렬 이해
  - Scaled Dot-Product Attention
- **Multi-Head Attention**: 여러 관점에서 동시에 어텐션 수행
- **Causal Masking**: 미래 단어를 보지 못하도록 차단 (GPT의 핵심!)

#### 🏗️ 레이어 3: 모델 아키텍처 (4장)
- **Transformer Block**: GPT의 기본 빌딩 블록
  - Multi-Head Attention 레이어
  - Feed-Forward Network
  - Layer Normalization
  - Residual Connection (Skip Connection)
- **GPT 모델**: Transformer Block을 여러 개 쌓아서 완성

#### 🔥 레이어 4: 학습 (5장)
- **사전학습(Pre-training)**: 대량의 텍스트로 언어 패턴 학습
  - 다음 단어 예측 (Next Token Prediction)
  - Cross-Entropy Loss
  - AdamW Optimizer
  - Learning Rate Scheduling
- **텍스트 생성**: 학습된 모델로 새로운 텍스트 생성
  - Greedy Decoding
  - Temperature Sampling
  - Top-k / Top-p Sampling

#### 🎨 레이어 5: 응용 (6-7장)
- **미세조정(Fine-tuning)**: 특정 작업에 맞게 모델 조정
  - 분류(Classification): 감정 분석, 스팸 탐지 등
  - 지시 따르기(Instruction Following): ChatGPT 스타일
- **Transfer Learning**: 사전학습 지식을 새로운 작업에 활용

### 개념의 계층 구조

```
최종 목표: 대화형 AI (ChatGPT 스타일)
    ↑
7장: 지시 따르기 미세조정
    ↑
6장: 분류 미세조정
    ↑
5장: 사전학습된 언어 모델
    ↑
4장: GPT 아키텍처 (Transformer Blocks × N)
    ↑
3장: Self-Attention 메커니즘
    ↑
2장: 텍스트 → 숫자 임베딩
    ↑
1장: 이론적 기반 (LLM이란?)
```

### 각 단계에서 구현하는 것

| 장 | 입력 | 출력 | 핵심 질문 |
|---|------|------|----------|
| 2 | 텍스트 문자열 | 숫자 벡터 | 어떻게 컴퓨터가 텍스트를 이해할까? |
| 3 | 단어 벡터들 | 문맥을 고려한 벡터 | 단어들이 어떻게 서로 영향을 주나? |
| 4 | 문장 | 다음 단어 확률 분포 | GPT의 전체 구조는? |
| 5 | 대량의 텍스트 | 학습된 GPT 모델 | 어떻게 언어 패턴을 학습하나? |
| 6 | 라벨된 데이터 | 분류 모델 | 특정 작업을 어떻게 수행하나? |
| 7 | 지시사항 데이터 | 대화형 AI | 어떻게 지시를 따르게 만드나? |

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
