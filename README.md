# ai_data_2022
2022년 인공지능 학습용 데이터 구축 사업 2-060-2 금속 3D프린팅 스파크 이미지 데이터 분석 및 예측 모델 개발

## 1. 과업 개요
- 금속 3D 프린팅(PBF방식) 제작물의 기계적 물성치 예측
  - 항복강도, 인장강도, 연신율, 밀도
- 레이저 출력, 스캔 스피드 등의 공정 조건과 제조 중에 촬영한 스파크 이미지 동영상을 입력 받아 예측함
- 다양한 타입의 데이터를 학습/처리하는 Multimodal Deep Learning 기법 적용

## 2. 모델 정보
- Multimodal Deep Learning 기법 중, VSCNN(Visual Social Network) 모델을 변형해 적용
- 인공지능 학습용 데이터 구축 사업 유효성 평가에는 ver13-8 버전 모델을 제출하였음
  - Model 명칭: VSCNN
  - Model Task: Multimodal Deep Learning 기반 기계적 특성치 추정
  - Model Version: 13-8

## 3. 환경 설정

python3 + pandas + matplotlib/seaborn/plotly + scikit-learn/pytorch + opencv

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

## 4. License
MIT License
