# GOAD
This repository contains a PyTorch implementation of the method presented in ["Classification-Based Anomaly Detection for General Data"](https://openreview.net/pdf?id=H1lK_lBtvS) by Liron Bergman and Yedid Hoshen, ICLR 2020.

## GOAD 코드 완벽 정리 

train_ad_image.py로 CIFAR10 작업 

앞으로 구축한 일반 이미지 데이터셋에 대해서도 적용 가능하게 만들기 

### CIFAR10 결과 

```
CiFAR10

AUC - vaild, test
airplane : 71.90%, 72.49%
automobile : 94.00%, 91.92%
bird : 71.69%, 69.83% 
cat : 61.73%, 60.88%
deer : 75.12%, 74.07%
dog : 79.38%, 78.80%
frog : 69.22%, 68.93%
horse : 93.08%, 93.32%
ship : 92.38%, 92.65%
truck : 89.35%, 87.67%

```

## Citation
If you find this useful, please cite our paper:
```
@inproceedings{bergman2020goad,
  author    = {Liron Bergman and Yedid Hoshen},
  title     = {Classification-Based Anomaly Detection for General Data},
  booktitle = {International Conference on Learning Representations (ICLR)},
  year      = {2020}
}
```
