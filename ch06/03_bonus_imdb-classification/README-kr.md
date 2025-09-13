# 50k IMDB 영화 리뷰 감정 분류 추가 실험 (Additional Experiments Classifying the Sentiment of 50k IMDB Movie Reviews)

## 개요

이 폴더에는 챕터 6의 (디코더 스타일) GPT-2 (2018) 모델을 [BERT (2018)](https://arxiv.org/abs/1810.04805), [RoBERTa (2019)](https://arxiv.org/abs/1907.11692), [ModernBERT (2024)](https://arxiv.org/abs/2412.13663)와 같은 인코더 스타일 LLM과 비교하는 추가 실험이 포함되어 있습니다. 챕터 6의 작은 SPAM 데이터셋을 사용하는 대신, 리뷰어가 영화를 좋아했는지 아닌지를 예측하는 이진 분류 목표로 IMDb의 50k 영화 리뷰 데이터셋([데이터셋 소스](https://ai.stanford.edu/~amaas/data/sentiment/))을 사용합니다. 이는 균형 잡힌 데이터셋이므로 랜덤 예측은 50% 정확도를 가져야 합니다.




|       | 모델                        | 테스트 정확도 |
| ----- | ---------------------------- | ------------- |
| **1** | 124M GPT-2 베이스라인          | 91.88%        |
| **2** | 340M BERT                    | 90.89%        |
| **3** | 66M DistilBERT               | 91.40%        |
| **4** | 355M RoBERTa                 | 92.95%        |
| **5** | 304M DeBERTa-v3              | 94.69%        |
| **6** | 149M ModernBERT Base         | 93.79%        |
| **7** | 395M ModernBERT Large        | 95.07%        |
| **8** | 로지스틱 회귀 베이스라인 | 88.85%        |






&nbsp;
## 1단계: 의존성 설치

다음을 통해 추가 의존성을 설치합니다:

```bash
pip install -r requirements-extra.txt
```

&nbsp;
## 2단계: 데이터셋 다운로드

코드는 영화 리뷰가 긍정적인지 부정적인지를 예측하기 위해 IMDb의 50k 영화 리뷰([데이터셋 소스](https://ai.stanford.edu/~amaas/data/sentiment/))를 사용합니다.

다음 코드를 실행하여 `train.csv`, `validation.csv`, `test.csv` 데이터셋을 생성합니다:

```bash
python download_prepare_dataset.py
```


&nbsp;
## 3단계: 모델 실행

&nbsp;
### 1) 124M GPT-2 베이스라인

챕터 6에서 사용한 124M GPT-2 모델로, 사전훈련된 가중치로 시작하여 모든 가중치를 미세조정합니다:

```bash
python train_gpt.py --trainable_layers "all" --num_epochs 1
```

```
Ep 1 (Step 000000): Train loss 3.706, Val loss 3.853
Ep 1 (Step 000050): Train loss 0.682, Val loss 0.706
...
Ep 1 (Step 004300): Train loss 0.199, Val loss 0.285
Ep 1 (Step 004350): Train loss 0.188, Val loss 0.208
Training accuracy: 95.62% | Validation accuracy: 95.00%
Training completed in 9.48 minutes.

Evaluating on the full datasets ...

Training accuracy: 95.64%
Validation accuracy: 92.32%
Test accuracy: 91.88%
```


<br>

---

<br>

&nbsp;
### 2) 340M BERT


340M 파라미터 인코더 스타일 [BERT](https://arxiv.org/abs/1810.04805) 모델:

```bash
python train_bert_hf.py --trainable_layers "all" --num_epochs 1 --model "bert"
```

```
Ep 1 (Step 000000): Train loss 0.848, Val loss 0.775
Ep 1 (Step 000050): Train loss 0.655, Val loss 0.682
...
Ep 1 (Step 004300): Train loss 0.146, Val loss 0.318
Ep 1 (Step 004350): Train loss 0.204, Val loss 0.217
Training accuracy: 92.50% | Validation accuracy: 88.75%
Training completed in 7.65 minutes.

Evaluating on the full datasets ...

Training accuracy: 94.35%
Validation accuracy: 90.74%
Test accuracy: 90.89%
```

<br>

---

<br>

&nbsp;
### 3) 66M DistilBERT

66M 파라미터 인코더 스타일 [DistilBERT](https://arxiv.org/abs/1910.01108) 모델(340M 파라미터 BERT 모델에서 증류됨), 사전훈련된 가중치로 시작하여 마지막 트랜스포머 블록과 출력 레이어만 훈련:



```bash
python train_bert_hf.py --trainable_layers "all" --num_epochs 1 --model "distilbert"
```

```
Ep 1 (Step 000000): Train loss 0.693, Val loss 0.688
Ep 1 (Step 000050): Train loss 0.452, Val loss 0.460
...
Ep 1 (Step 004300): Train loss 0.179, Val loss 0.272
Ep 1 (Step 004350): Train loss 0.199, Val loss 0.182
Training accuracy: 95.62% | Validation accuracy: 91.25%
Training completed in 4.26 minutes.

Evaluating on the full datasets ...

Training accuracy: 95.30%
Validation accuracy: 91.12%
Test accuracy: 91.40%
```
<br>

---

<br>

&nbsp;
### 4) 355M RoBERTa

355M 파라미터 인코더 스타일 [RoBERTa](https://arxiv.org/abs/1907.11692) 모델, 사전훈련된 가중치로 시작하여 마지막 트랜스포머 블록과 출력 레이어만 훈련:


```bash
python train_bert_hf.py --trainable_layers "last_block" --num_epochs 1 --model "roberta" 
```

```
Ep 1 (Step 000000): Train loss 0.695, Val loss 0.698
Ep 1 (Step 000050): Train loss 0.670, Val loss 0.690
...
Ep 1 (Step 004300): Train loss 0.083, Val loss 0.098
Ep 1 (Step 004350): Train loss 0.170, Val loss 0.086
Training accuracy: 98.12% | Validation accuracy: 96.88%
Training completed in 11.22 minutes.

Evaluating on the full datasets ...

Training accuracy: 96.23%
Validation accuracy: 94.52%
Test accuracy: 94.69%
```

<br>

---

<br>

&nbsp;
### 5) 304M DeBERTa-v3

304M 파라미터 인코더 스타일 [DeBERTa-v3](https://arxiv.org/abs/2111.09543) 모델. DeBERTa-v3는 분리된 어텐션과 개선된 위치 인코딩으로 이전 버전을 개선합니다.


```bash
python train_bert_hf.py --trainable_layers "all" --num_epochs 1 --model "deberta-v3-base"
```

```
Ep 1 (Step 000000): Train loss 0.689, Val loss 0.694
Ep 1 (Step 000050): Train loss 0.673, Val loss 0.683
...
Ep 1 (Step 004300): Train loss 0.126, Val loss 0.149
Ep 1 (Step 004350): Train loss 0.211, Val loss 0.138
Training accuracy: 92.50% | Validation accuracy: 94.38%
Training completed in 7.20 minutes.

Evaluating on the full datasets ...

Training accuracy: 93.44%
Validation accuracy: 93.02%
Test accuracy: 92.95%
```

<br>

---

<br>



&nbsp;
### 6) 149M ModernBERT Base

[ModernBERT (2024)](https://arxiv.org/abs/2412.13663)는 효율성과 성능을 향상시키기 위해 병렬 잔여 연결(parallel residual connections)과 게이트 선형 유닛(gated linear units, GLUs)과 같은 아키텍처 개선을 통합한 BERT의 최적화된 재구현입니다. BERT의 원래 사전훈련 목표를 유지하면서 현대 하드웨어에서 더 빠른 추론과 더 나은 확장성을 달성합니다.

```bash
python train_bert_hf.py --trainable_layers "all" --num_epochs 1 --model "modernbert-base"
```



```
Ep 1 (Step 000000): Train loss 0.699, Val loss 0.698
Ep 1 (Step 000050): Train loss 0.564, Val loss 0.606
...
Ep 1 (Step 004300): Train loss 0.086, Val loss 0.168
Ep 1 (Step 004350): Train loss 0.160, Val loss 0.131
Training accuracy: 95.62% | Validation accuracy: 93.75%
Training completed in 10.27 minutes.

Evaluating on the full datasets ...

Training accuracy: 95.72%
Validation accuracy: 94.00%
Test accuracy: 93.79%
```

<br>

---

<br>


&nbsp;
### 7) 395M ModernBERT Large

위와 동일하지만 더 큰 ModernBERT 변형을 사용합니다.

```bash
python train_bert_hf.py --trainable_layers "all" --num_epochs 1 --model "modernbert-large"
```



```
Ep 1 (Step 000000): Train loss 0.666, Val loss 0.662
Ep 1 (Step 000050): Train loss 0.548, Val loss 0.556
...
Ep 1 (Step 004300): Train loss 0.083, Val loss 0.115
Ep 1 (Step 004350): Train loss 0.154, Val loss 0.116
Training accuracy: 96.88% | Validation accuracy: 95.62%
Training completed in 27.69 minutes.

Evaluating on the full datasets ...

Training accuracy: 97.04%
Validation accuracy: 95.30%
Test accuracy: 95.07%
```




<br>

---

<br>

&nbsp;
### 8) 로지스틱 회귀 베이스라인

베이스라인으로 scikit-learn [로지스틱 회귀](https://sebastianraschka.com/blog/2022/losses-learned-part1.html) 분류기:


```bash
python train_sklearn_logreg.py
```

```
Dummy classifier:
Training Accuracy: 50.01%
Validation Accuracy: 50.14%
Test Accuracy: 49.91%


Logistic regression classifier:
Training Accuracy: 99.80%
Validation Accuracy: 88.62%
Test Accuracy: 88.85%
```