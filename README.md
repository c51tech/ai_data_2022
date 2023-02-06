# ai_data_2022
2022년 인공지능 학습용 데이터 구축 사업 2-060-2 금속 3D프린팅 스파크 이미지 데이터 분석 및 예측 모델 개발


## 1. 환경 설정

python3 + pandas + matplotlib/seaborn/plotly + scikit-learn/pytorch

```bash
$ conda create -n mmdl -c anaconda -c conda-forge jupyter jupytext matplotlib seaborn tqdm scipy pandas munch tensorboard scikit-learn pyarrow openpyxl
$ conda activate mmdl
(mmdl)$ pip install opencv-python
(mmdl)$ conda install -c plotly plotly

(mmdl)$ conda install pytorch torchvision pytorch-cuda=11.6 -c pytorch -c nvidia
(mmdl)$ conda install -c conda-forge pytorch-lightning 
```

참고: cuda 11.3 으로 설치 시
```bash
(mmdl)$ conda install pytorch torchvision cudatoolkit=11.3 -c pytorch 
```
