# +
import os
import sys
import json
import random
import pandas as pd
import numpy as np
import scipy.stats as stats
from datetime import datetime
from PIL import Image
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import ToTensor, Normalize, Lambda, Compose

np.set_printoptions(precision=3)
pd.set_option('display.max_rows', None)

import seaborn as sns 
import matplotlib.pyplot as plt
from matplotlib import rc
# %matplotlib inline
rc('font', family='NanumGothic')
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.figsize'] = [20, 16]

import plotly.graph_objects as go
import plotly.io as pio   
import plotly.express as px
from plotly.subplots import make_subplots

from IPython.display import display, HTML
display(HTML("<style>.container { width:80% !important; }</style>"))
display(HTML("<style>div.output_scroll { height: 100em; }</style>"))
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_colwidth', 200)

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=DeprecationWarning)

import sklearn.neighbors._base
sys.modules['sklearn.neighbors.base'] = sklearn.neighbors._base

import re
re_num = re.compile(r'([0-9\.]+)(.*)')

import train_vscnn as tv

# +
start_time = datetime.now()

print("=" * 50)
print("Start Model Evaluation : ", start_time)
print("=" * 50)
# -

# # Load Model

log_base_dir = 'logs/'
exp_title = 'ver13'

model_dir = tv.search_model_ckpt(log_base_dir, exp_title)
model_ckpt = torch.load(model_dir)

params = model_ckpt['hyper_parameters']['params']
data_params = params['data_params']

tv.seed_rngs(data_params.random_seed)

# +
params.is_train = False
params.num_workers = 0

model = tv._build_model(params)

tv._init_model(params, model)

model
# -

# # Load Data

# +
# base_dir = "/data"

# base_dir = "/home/kikim/data/ai_data_230102/final/191_ 금속 스파크 이미지 데이터"
# base_dir = '/home/kikim/data/ai_data_230102/processed/'
base_dir = '/home/kikim/data/ai_data_230102/final/'
# image_sub_dir = os.path.join('원천데이터')
# label_sub_dir = os.path.join('라벨링데이터')

image_sub_dir = os.path.join('원천데이터', '기계적 특성치 데이터')
label_sub_dir = os.path.join('라벨링데이터', '기계적 특성치 데이터')

image_dir = os.path.join(base_dir, image_sub_dir)
label_dir = os.path.join(base_dir, label_sub_dir)
# -

data_params.image_dir = image_dir
data_params.label_dir = label_dir

# +
variable_names = ['material.name', 'material.size', 
                  'equipment.name', 'equipment.chamber.oxygen.density', 'equipment.chamber.temp',
                  'equipment.chamber.oxygen.density', 'equipment.base.hd',
                  'condition.laser.power', 'condition.scan.speed', 'condition.lamination.direction',
                  'specimen.position', 
                  'yielding.stress', 'tensile.stress', 'elongation', 'density', #'distortion.x'
                 ]

on_axis_meta_dict = tv.read_json_files(label_dir, 'On_Axis_Images/JSON', variable_names)
off_axis_meta_dict = tv.read_json_files(label_dir, 'Off_Axis_Images/JSON', variable_names)

on_axis_meta_dict = { k:tv.convert_value_to_float(v) for k, v in on_axis_meta_dict.items() }
off_axis_meta_dict = { k:tv.convert_value_to_float(v) for k, v in off_axis_meta_dict.items() }
# -

df_on_axis = pd.DataFrame.from_dict(on_axis_meta_dict, orient='index')
df_on_axis['laser_density'] = df_on_axis['condition.laser.power'] / df_on_axis['condition.scan.speed']

df_off_axis = pd.DataFrame.from_dict(off_axis_meta_dict, orient='index')
df_off_axis['laser_density'] = df_off_axis['condition.laser.power'] / df_off_axis['condition.scan.speed']

# +
df_meta = df_on_axis.copy()

df_meta = df_meta[['material.name', 'material.size', 
                'equipment.name', 'equipment.chamber.oxygen.density', 'equipment.chamber.temp', 'equipment.base.hd', 
                'condition.laser.power', 'condition.scan.speed', 'condition.lamination.direction', 'laser_density', 
                ] + ['yielding.stress', 'tensile.stress', 'elongation', 'density']]
df_meta = tv.replace_col_1hot(df_meta, ['material.name', 'equipment.name', 'condition.lamination.direction'])

## 1-hot encoding 변수명 업데이트
data_params.vars_x = df_meta.columns[:-4].tolist()
data_params.vars_y = df_meta.columns[-4:].tolist()

## condition.scan.speed == 0 인 경우 제외
df_meta.drop(index=df_meta.index[df_meta['condition.scan.speed'] == 0], inplace=True)
df_meta.loc[df_meta['condition.scan.speed'] == 0, :]

# +
specimen_id_list = df_meta.index.tolist()

train_specimen_id_list, valid_specimen_id_list, test_specimen_id_list = tv.split_data(specimen_id_list)

assert set(train_specimen_id_list).intersection(set(valid_specimen_id_list)) == set()
assert set(train_specimen_id_list).intersection(set(test_specimen_id_list)) == set()
assert set(valid_specimen_id_list).intersection(set(test_specimen_id_list)) == set()

print(len(specimen_id_list), len(train_specimen_id_list), len(valid_specimen_id_list), len(test_specimen_id_list))
# -

train_dataset = tv.make_dataset(data_params, df_meta, train_specimen_id_list)
valid_dataset = tv.make_dataset(data_params, df_meta, valid_specimen_id_list)
test_dataset = tv.make_dataset(data_params, df_meta, test_specimen_id_list)

data_module = tv.DataModule(params, train_dataset, test_dataset, valid_dataset)

# +

import pytorch_lightning as pl

trainer = pl.Trainer(accelerator='gpu', auto_select_gpus=True)

# +
from munch import Munch
from scipy import stats

def summarize(model, trainer, params, data_module, plot_chart=True): 
    #mse = trainer.test(model)
    dm_loaders = Munch( 
#         train = data_module.train_dataloader(shuffle=False), 
#         val = data_module.val_dataloader(shuffle=False), 
        test = data_module.test_dataloader(shuffle=False))

    print('experiment {}'.format(params.exp_title))
    print('-'*20)
    
    exp_results = {}
#     tag_mode = ['train', 'val', 'test']
    tag_mode = ['test']
    for mode in tag_mode:  
        print('Evaluate :', tag_mode)
        
        y_pred, y_real, x_real = tv.evaluate(model, dm_loaders[mode], gpu=params.gpu)
        y_pred = tv._inv_norm_y(y_pred, dm_loaders[mode])
        y_real = tv._inv_norm_y(y_real, dm_loaders[mode])
        
        title = '[{}] {}\n'.format(params.exp_title, mode)
            
        print('Compute Metrics')
        
        if plot_chart:
            fig = plt.figure(figsize=(24, 6))
            axs = fig.subplots(1, 4)
            fig.suptitle(mode)
        
        metrics = []
        vars_y = dm_loaders[mode].dataset.vars_y
        for i, _var_y in enumerate(vars_y):
            _pred, _real = y_pred[:, i], y_real[:, i]
            mape = tv.cal_mape(_pred, _real)
            rmse = tv.cal_rmse(_pred, _real)
            r2 = tv.cal_r2(_pred, _real)
            spearmanr = tv.cal_spearmanr(_pred, _real)
            
            print()
            print(f'[{vars_y}]')
            print('mape      ({} {}): {:.6f}'.format(mode, _var_y, mape))
            print('rmse      ({} {}): {:.6f}'.format(mode, _var_y, rmse))
            print('r2        ({} {}): {:.6f}'.format(mode, _var_y, r2))
            print('spearmanr ({} {}): {:.6f}'.format(mode, _var_y, spearmanr))
            
            metrics.append({
                'mode': mode,
                'mape': mape,
                'rmse': rmse,
                'r2': r2, 
                'spearmanr': spearmanr,    
            })
            
            if plot_chart:
                tv.plot_scatter(_pred, _real, title=_var_y, ax=axs[i])
        
        df_results = pd.DataFrame.from_dict(metrics)
        df_results.index = vars_y
        exp_results[mode] = df_results
            
        
    #plt.legend()
    #plt.show()
    print('experiment {}'.format(params.exp_title))
    
    return exp_results

# +
exp_results = summarize(model, trainer, params, data_module, plot_chart=False)

for mode, df_result in exp_results.items():
    display(mode)
    display(df_result)

# +
mean_r2 = exp_results['test']['r2'].mean()
mean_spearmanr = exp_results['test']['spearmanr'].mean()

print()
print('*' * 25)
print('Evaluation Results [Test]')
print(f'R2               = {mean_r2:.4}', )
print(f"Spearman's rho   = {mean_spearmanr:.4}")
print('*' * 25)
# -

end_time = datetime.now()
elapsed_time = end_time - start_time
print("=" * 70)
print("Model Evaluation Finished : ", end_time, f'({round(elapsed_time.total_seconds(), 1)} sec)')
print("=" * 70)

# +
# # %load_ext tensorboard
# -

# %tensorboard --logdir 'logs/ver13' --host=0.0.0.0


