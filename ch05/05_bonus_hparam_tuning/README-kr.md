# 사전학습을 위한 하이퍼파라미터 최적화

[부록 D: 학습 루프에 유용한 요소 추가](../../appendix-D/01_main-chapter-code/appendix-D.ipynb)의 확장된 학습 함수를 기반으로 한 [hparam_search.py](hparam_search.py) 스크립트는 그리드 서치(grid search)를 통해 최적의 하이퍼파라미터를 찾도록 설계되었습니다.

>[!NOTE]
이 스크립트는 실행하는 데 오랜 시간이 걸립니다. 상단의 `HPARAM_GRID` 딕셔너리에서 탐색할 하이퍼파라미터 구성의 수를 줄이는 것을 고려할 수 있습니다.