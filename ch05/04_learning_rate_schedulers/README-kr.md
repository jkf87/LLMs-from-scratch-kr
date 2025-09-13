# 학습 루프에 유용한 요소 추가

메인 장에서는 5장을 페이지 제한 내에서 맞추고 코드를 읽기 쉽게 유지하기 위해 상대적으로 단순한 학습 함수를 사용했습니다. 선택적으로 학습 안정성과 수렴성을 개선하기 위해 선형 워밍업(linear warm-up), 코사인 감소 스케줄(cosine decay schedule), 그래디언트 클리핑(gradient clipping)을 추가할 수 있습니다.

이러한 더 정교한 학습 함수의 코드는 [부록 D: 학습 루프에 유용한 요소 추가](../../appendix-D/01_main-chapter-code/appendix-D.ipynb)에서 찾을 수 있습니다.