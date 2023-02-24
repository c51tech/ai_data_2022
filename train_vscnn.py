# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.14.4
#   kernelspec:
#     display_name: Python [conda env:mmdl]
#     language: python
#     name: conda-env-mmdl-py
# ---

# # VSCNN 학습

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

from tqdm import tqdm
from munch import Munch
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

#import env
os.getcwd()
# -

# # 데이터 읽기 / 전처리

# ## 메타데이터

# +
if __name__=="__main__":
    
#     base_dir = "/data"
    
#     base_dir = "/home/kikim/data/ai_data_230102/final/191_ 금속 스파크 이미지 데이터"
    base_dir = '/home/kikim/data/ai_data_230102/processed/'
    image_sub_dir = os.path.join('원천데이터')
    label_sub_dir = os.path.join('라벨링데이터')

#     image_sub_dir = os.path.join('원천데이터', '기계적 특성치 데이터')
#     label_sub_dir = os.path.join('라벨링데이터', '기계적 특성치 데이터')
    
    image_dir = os.path.join(base_dir, image_sub_dir)
    label_dir = os.path.join(base_dir, label_sub_dir)


# +
def sort_elements_by_specimen_id(elements, delim='_', id_idx=0):
    ids = [d.split(delim)[id_idx] for d in elements]
    ids = [int(i) for i in ids if i.isdigit()]
    
    return [d for i, d in sorted(zip(ids, elements))]


def sort_dirs_by_specimen_id(root_dir):
    return sort_elements_by_specimen_id(os.listdir(root_dir))


def select_dict_by_keys(_dict, keys=None):
    if keys is None:
        keys = _dict.keys()
        
    return {k:v for k, v in _dict.items() if k in keys}


def read_json_files(label_dir, suffix_path, keys=None):
    meta_dict = {}
    for d in sort_dirs_by_specimen_id(label_dir):
        specimen_id = d.split('_')[0]

        json_dir = os.path.join(label_dir, d, suffix_path)
        if not os.path.exists(json_dir):
            continue

        _json_files = [f for f in os.listdir(json_dir) if f.endswith('.json')]
        if len(_json_files) == 0:
            continue

        try:
            _1st_json_path = os.path.join(json_dir, _json_files[0])
            with open(_1st_json_path) as f:
                _json_dict = json.load(f)
        except json.decoder.JSONDecodeError as ex:
            with open(_1st_json_path, encoding='utf-8-sig') as f:
                print(ex, _1st_json_path)
                _json_dict = json.load(f)

        meta_dict[int(specimen_id)] = select_dict_by_keys(_json_dict, keys)
        
    return meta_dict


def read_json_files__melt_temp(label_dir, suffix_path, keys=None, melt_temp_key='max_temperature'):
    meta_dict = {}
    for d in sort_dirs_by_specimen_id(label_dir):
        specimen_id = d.split('_')[0]

        json_dir = os.path.join(label_dir, d, suffix_path)
        if not os.path.exists(json_dir):
            continue

        _json_files = [f for f in os.listdir(json_dir) if f.endswith('.json')]
        if len(_json_files) == 0:
            continue

        with open(os.path.join(json_dir, _json_files[0])) as f:
            _json_dict = json.load(f)

        meta_dict[int(specimen_id)] = select_dict_by_keys(_json_dict, keys)
        
        _max_temp_list = []
        for f_name in sort_elements_by_specimen_id(os.listdir(json_dir)):
            _json_path = os.path.join(json_dir, f_name)
            if not _json_path.endswith('.json') or not os.path.exists(_json_path):
                continue

            with open(_json_path) as f:
                _json_dict = json.load(f)

            _max_temp_list.append(_json_dict[melt_temp_key])

        if len(_max_temp_list) > 0:
            _json_dict = select_dict_by_keys(_json_dict, keys)
            _json_dict[melt_temp_key] = [float(t) for t in _max_temp_list]
            meta_dict[int(specimen_id)] = _json_dict
        
    return meta_dict

def convert_value_to_float(json_dict):
    regex = re.compile('(\d+\.?\d*)([^-]*)')
    
    _converted_json_dict = {}
    for k, v in json_dict.items():
        if isinstance(v, str):
            m = regex.match(v.strip())
            if m is not None:
                v = m.groups()[0]
                _converted_json_dict[k] = float(v)
        
        if k not in _converted_json_dict:
            _converted_json_dict[k] = v
            
    return _converted_json_dict


# -

if __name__=="__main__":

    variable_names = ['material.name', 'material.size', 
                      'equipment.name', 'equipment.chamber.oxygen.density', 'equipment.chamber.temp',
                      'equipment.chamber.oxygen.density', 'equipment.base.hd',
                      'condition.laser.power', 'condition.scan.speed', 'condition.lamination.direction',
                      'specimen.position', 
                      'yielding.stress', 'tensile.stress', 'elongation', 'density', #'distortion.x'
                     ]

    on_axis_meta_dict = read_json_files(label_dir, 'On_Axis_Images/JSON', variable_names)
    off_axis_meta_dict = read_json_files(label_dir, 'Off_Axis_Images/JSON', variable_names)

    on_axis_meta_dict = { k:convert_value_to_float(v) for k, v in on_axis_meta_dict.items() }
    off_axis_meta_dict = { k:convert_value_to_float(v) for k, v in off_axis_meta_dict.items() }

# +
# if __name__=="__main__":
#     melt_temp_meta_dict = read_json_files__melt_temp(label_dir, 'melt_temperature', variable_names)
#     melt_temp_meta_dict = { k:convert_value_to_float(v) for k, v in melt_temp_meta_dict.items() }
# -

if __name__=="__main__":
    df_on_axis = pd.DataFrame.from_dict(on_axis_meta_dict, orient='index')
    df_on_axis['laser_density'] = df_on_axis['condition.laser.power'] / df_on_axis['condition.scan.speed']
    
    display(df_on_axis.iloc[1680:1696, :])

if __name__=="__main__":
    df_off_axis = pd.DataFrame.from_dict(off_axis_meta_dict, orient='index')
    df_off_axis['laser_density'] = df_off_axis['condition.laser.power'] / df_off_axis['condition.scan.speed']

# +
# if __name__=="__main__":

#     for k, _dict in melt_temp_meta_dict.items():
#         _melt_temps = np.array(_dict['max_temperature'])
#         _dict['max_temperature__mean'] = _melt_temps.mean()
#         _dict['max_temperature__max'] = _melt_temps.max()
#         _dict['max_temperature__min'] = _melt_temps.min()
#         _dict['max_temperature__median'] = _melt_temps.min()
#         _ = _dict.pop('max_temperature')
    
#     df_melt_temp = pd.DataFrame.from_dict(melt_temp_meta_dict, orient='index')
# -

# ## Video Data

if __name__=="__main__":

    specimen_dirs = [d for d in os.listdir(image_dir) if os.path.isdir(os.path.join(image_dir, d))]
    specimen_id_to_img_dirs = {int(d.split('_')[0]):d for d in os.listdir(image_dir) if os.path.isdir(os.path.join(image_dir, d))}

# # Prepare Data

if __name__=="__main__":
    data_params = Munch(
        random_seed=1,
        label_dir=label_dir,
        image_dir=image_dir,
        vars_x=['material.name', 'material.size', 
                'equipment.name', 'equipment.chamber.oxygen.density', 'equipment.chamber.temp', 'equipment.base.hd', 
                'condition.laser.power', 'condition.scan.speed', 'condition.lamination.direction', 'laser_density', 
                ],
        vars_y=['yielding.stress', 'tensile.stress', 'elongation', 'density'],

    )


# +
def split_data(specimen_id_list, ratio=(.8, .1, .1)):
    num_specimen = len(specimen_id_list)

    num_valid = int(round(num_specimen * ratio[1]))
    num_test = int(round(num_specimen * ratio[2]))
    num_train = num_specimen - num_valid - num_test
    
    _shuffled = np.random.permutation(specimen_id_list)
    
    return sorted(_shuffled[:num_train]), \
            sorted(_shuffled[num_train:-num_test]), \
            sorted(_shuffled[-num_test:])


def replace_col_1hot(df, cols_1hot):
    col_names = df.columns
    
    _df = pd.DataFrame(index=df.index)
    for c in col_names:
        if c in cols_1hot:
            df_1hot = pd.get_dummies(df[c])
            for c_1hot in df_1hot.columns:
                _df[f'{c}:{c_1hot}'] = df_1hot[c_1hot]
        else:
            _df[c] = df[c]
    
    return _df


def seed_rngs(seed: int, pytorch: bool = True) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    if pytorch:
        torch.manual_seed(seed)
        
if __name__=="__main__":
    seed_rngs(data_params.random_seed)
# -

if __name__=="__main__":
    df_meta = df_on_axis.copy()

    df_meta = df_meta[data_params.vars_x + data_params.vars_y]
    df_meta = replace_col_1hot(df_meta, ['material.name', 'equipment.name', 'condition.lamination.direction'])
    
    ## 1-hot encoding 변수명 업데이트
    data_params.vars_x = df_meta.columns[:-4].tolist()
    data_params.vars_y = df_meta.columns[-4:].tolist()
    
    ## condition.scan.speed == 0 인 경우 제외
    df_meta.drop(index=df_meta.index[df_meta['condition.scan.speed'] == 0], inplace=True)
    df_meta.loc[df_meta['condition.scan.speed'] == 0, :]

if __name__=="__main__":

    specimen_id_list = df_meta.index.tolist()

    train_specimen_id_list, valid_specimen_id_list, test_specimen_id_list = split_data(specimen_id_list)

    assert set(train_specimen_id_list).intersection(set(valid_specimen_id_list)) == set()
    assert set(train_specimen_id_list).intersection(set(test_specimen_id_list)) == set()
    assert set(valid_specimen_id_list).intersection(set(test_specimen_id_list)) == set()

    print(len(specimen_id_list), len(train_specimen_id_list), len(valid_specimen_id_list), len(test_specimen_id_list))


def _get_a_file_path(root_dir, prefix, suffix='.avi'):
    files = [f for f in os.listdir(root_dir) if f.startswith(prefix) and f.endswith(suffix)]
    
    if len(files) == 0:
        print('No .avi File', root_dir, f'{prefix}_Axis_Images.avi')
        return
    
    if not files[0].endswith('_Axis_Images.avi'):
        print('Warning: Bad .avi File Name ', root_dir, files[0])
        
    return os.path.join(root_dir, files[0])


# +
class SpecimenDataset(Dataset):
    
    def __init__(self, df_meta, img_dir, 
                 transform_x=None, 
                 transform_x_on_img=None,
                 transform_x_off_img=None,
                 transform_y=None,
#                  melt_temp_dict=None,
#                  transform_x_melt_temp=None,
                 vars_x=None,
                 vars_y=None
                ):
        
        if vars_x is None:
            vars_x = self.df_meta.columns[:-4].tolist()
        
        if vars_y is None:
            vars_y = self.df_meta.columns[-4:].tolist()
        
        self.df_meta = df_meta
        self.specimen_id_list = df_meta.index.tolist()
        
        self.img_dir = img_dir
        specimen_dirs = [d for d in os.listdir(img_dir) if os.path.isdir(os.path.join(img_dir, d))]
        specimen_id_to_img_dirs = {int(d.split('_')[0]):d for d in os.listdir(img_dir) if os.path.isdir(os.path.join(img_dir, d))}
        
        self.specimen_id_to_on_img_paths = {s:_get_a_file_path(os.path.join(img_dir, d, 'video'), 'On') 
                                            for s, d in specimen_id_to_img_dirs.items() if s in self.specimen_id_list}
        self.specimen_id_to_off_img_paths = {s:_get_a_file_path(os.path.join(img_dir, d, 'video'), 'Off') 
                                             for s, d in specimen_id_to_img_dirs.items() if s in self.specimen_id_list}
        
        self.transform_x = transform_x
        self.transform_x_on_img = transform_x_on_img
        self.transform_x_off_img = transform_x_off_img
        self.transform_y = transform_y
        
        self.vars_x = vars_x
        self.vars_y = vars_y
        
    def __getitem__(self, idx):
        specimen_id = self.specimen_id_list[idx]
        x = self.df_meta.loc[specimen_id, self.vars_x].values
        y = self.df_meta.loc[specimen_id, self.vars_y].values
        
        try:
        
            if self.transform_x is not None:
                x = self.transform_x(x)

            if self.transform_y is not None:
                y = self.transform_y(y)

            on_imgs = self._read_video_frames(self.specimen_id_to_on_img_paths[specimen_id])
            off_imgs = self._read_video_frames(self.specimen_id_to_off_img_paths[specimen_id])

            if self.transform_x_on_img is not None:
                on_imgs = self.transform_x_on_img(on_imgs)

            if self.transform_x_off_img is not None:
                off_imgs = self.transform_x_off_img(off_imgs)
        except Exception as ex:
            print(ex, idx, specimen_id)
            raise ex
            
        return x, y, on_imgs, off_imgs, specimen_id, idx
        
    def __len__(self):
        return len(self.specimen_id_list)
    
    def _read_video_frames(self, video_path):
        vidcap = cv2.VideoCapture(video_path)
        img_list = []
        
        success,image = vidcap.read()
        while success:
            img_list.append(image[:, :, 0])
            success,image = vidcap.read()

        return np.array(img_list)
    

class Normalizer:
    """
    최소/최대 [-1, 1] 범위로 정규화
    """ 
    def __init__(self, mins=np.zeros(0), maxs=np.zeros(0), diffs=np.ones(0)):
        self.mins = np.array(mins)
        self.maxs = np.array(maxs)
        self.diffs = np.array(diffs)
        
    def fit(self, array):
        self.mins = array.min(axis=0)
        self.maxs = array.max(axis=0)
        self.diffs = (self.maxs - self.mins) * 0.5
        self.diffs[self.diffs == 0] = 1
        
        return self
        
    def __call__(self, sample):
        return (sample - self.mins) / self.diffs - 1
    
    def inverse_tranform(self, normalized):
        return (normalized + 1) * self.diffs[None, :] + self.mins[None, :]
    
    def to_dict(self):
        return dict(mins=self.mins, maxs=self.maxs, diffs=self.diffs)


class Standardizer:
    """
    정규화
    """ 
    def __init__(self, means=np.zeros(0), stds=np.ones(0)):
        self.means = means
        self.stds = stds
        
    def fit(self, array):
        self.means = array.mean(axis=0)
        self.stds = array.std(axis=0)
        
        self.stds[self.stds == 0] = 1
        
        return self
        
    def __call__(self, sample):
        return (sample - self.means) / self.stds
    
    def inverse_tranform(self, normalized):
        return normalized * self.stds[None, :] + self.means[None, :]

    def to_dict(self):
        return dict(means=self.means, stds=self.stds)

    
class Subsampler:
    """
    
    """ 
    def __init__(self, length=128):
        self.length = length
        
    def __call__(self, array):
        _len = array.shape[0]
        
        idxes = np.arange(_len)
        
        if  self.length < _len:
            _sampled = np.random.permutation(idxes)[:self.length]
        elif self.length == _len:
            _sampled = idxes
        elif self.length / 2 < _len:
            _sampled = idxes.tolist() + np.random.choice(idxes, self.length - _len).tolist()
        else:
            _sampled = np.random.choice(idxes, self.length).tolist()
            
        _sampled = np.sort(_sampled)
        
        return array[_sampled]
        
    def to_dict(self):
        return dict(length=self.length)

        
class ToTensor:
    """
    
    """ 
        
    def __call__(self, array):
        return torch.Tensor(array)#.unsqueeze(1)


# -



# +
from torchvision.transforms import Resize, Lambda, Compose

if __name__=="__main__":
    vars_x = data_params.vars_x
    vars_y = data_params.vars_y

    transform_img_temp = Compose([Subsampler(10), ToTensor(), 
                                  Resize((300, 300))
                                 ])

    temp_dataset = SpecimenDataset(df_meta.loc[:, :], image_dir, 
                                   vars_x=vars_x, vars_y=vars_y,
                                   transform_x_on_img=transform_img_temp,
                                   transform_x_off_img=transform_img_temp,
                                  )

    x, y, on_imgs, off_imgs, specimen_id, idx = next(iter(DataLoader(temp_dataset, batch_size=16, shuffle=True)))

    min_pixel = on_imgs.min().item()
    max_pixel = on_imgs.max().item()

# -

if __name__=="__main__":

    normalizer_x = Normalizer().fit(df_meta.loc[train_specimen_id_list, vars_x].values)
    normalizer_y = Normalizer().fit(df_meta.loc[train_specimen_id_list, vars_y].values)
    normalizer_img = Lambda(lambda x: (x - min_pixel) / (max_pixel - min_pixel))


if __name__=="__main__":
    
    data_params['norm_x'] = {k:v.tolist() for k, v in normalizer_x.to_dict().items()}
    data_params['norm_y'] = {k:v.tolist() for k, v in normalizer_y.to_dict().items()}
    data_params['norm_x_img'] = dict(min_pixel=min_pixel, max_pixel=max_pixel)

    data_params['num_imgs'] = 128
    data_params['min_pixel'] = min_pixel
    data_params['max_pixel'] = max_pixel

    data_params


def make_dataset(data_params, df_meta, specimen_id_list):
    
    lamdba_float = Lambda(lambda x: torch.Tensor(x).float())
    norm_img = Lambda(lambda x: (x - data_params.min_pixel) / (data_params.max_pixel - data_params.min_pixel))
    
    transform_x = Compose([Normalizer(**data_params.norm_x), lamdba_float])
    transform_y = Compose([Normalizer(**data_params.norm_y), lamdba_float])
    transform_img = Compose([Subsampler(data_params.num_imgs), ToTensor(), 
                             Resize((300, 300)), norm_img
                            ])

    dataset = SpecimenDataset(df_meta.loc[specimen_id_list, :], data_params.image_dir, 
                              vars_x=data_params.vars_x, vars_y=data_params.vars_y,
                              transform_x=transform_x,
                              transform_y=transform_y,
                              transform_x_on_img=transform_img,
                              transform_x_off_img=transform_img,
                             )
    return dataset


if __name__=="__main__":

    train_dataset = make_dataset(data_params, df_meta, train_specimen_id_list)
    valid_dataset = make_dataset(data_params, df_meta, valid_specimen_id_list)
    test_dataset = make_dataset(data_params, df_meta, test_specimen_id_list)




# # Learn

# +
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning import LightningModule
from resnet import ResNet, BasicBlock
from munch import Munch

#torch.backends.cudnn.enabled=False


class DataModule(pl.LightningDataModule):
    def __init__(self, params, train_dataset, test_dataset, val_dataset=None):
        super().__init__()

        self.train = train_dataset
        self.val = val_dataset
        self.test = test_dataset
        self.params = params
        
    def train_dataloader(self, shuffle=True, drop_last=True):
        train_loader = DataLoader(dataset=self.train, 
                                  batch_size=self.params.batch_size, 
                                  num_workers=self.params.num_workers,
                                  shuffle=shuffle,
                                  drop_last=drop_last)
        return train_loader
            
    def val_dataloader(self, shuffle=False, drop_last=True):
        if self.val:
            val_loader = DataLoader(dataset=self.val, 
                                    batch_size=self.params.batch_size, 
                                    num_workers=self.params.num_workers,
                                    shuffle=shuffle,
                                    drop_last=drop_last)
            return val_loader
        else:
            return None
            
    def test_dataloader(self, shuffle=False, drop_last=True):
        test_loader = DataLoader(dataset=self.test, 
                        batch_size=self.params.batch_size, #self.test.__len__(), 
                        num_workers=self.params.num_workers,
                        shuffle=shuffle,
                        drop_last=drop_last)
        return test_loader


class RegressionModule(LightningModule):
    def __init__(self, params):
        super().__init__()
        self.learning_rate = params.learning_rate
        self.weight_decay = params.weight_decay
        self.output_size = params.output_size
        self.input_size = params.input_size
        self.hidden_size = params.hidden_size
        self.num_layers = params.num_layers
        self.num_imgs = params.num_imgs
        
        self.inp_enc_out = params.inp_enc_out
        self.on_img_enc_out = params.on_img_enc_out
        self.off_img_enc_out = params.off_img_enc_out
        
        self.inp_encoder = nn.Sequential(
            nn.Linear(self.input_size, self.inp_enc_out), 
            nn.ReLU(), 
            nn.Dropout(params.dropout), 
        )
        
#         #self.img_encoder = create_model()
        self.on_img_encoder = ResNet(BasicBlock, [1,1,1,1], num_classes=self.on_img_enc_out, num_input=self.num_imgs)
        self.off_img_encoder = ResNet(BasicBlock, [1,1,1,1], num_classes=self.off_img_enc_out, num_input=self.num_imgs)
#         #self.model = _resnet("resnet", BasicBlock, [1,1,1,1], pretrained=False, progress=False, num_classes=1)
        
        self.layers = nn.Sequential(
            nn.Linear(self.inp_enc_out + self.on_img_enc_out + self.off_img_enc_out, 128), nn.ReLU(), nn.Dropout(params.dropout), 
            nn.Linear(128, 32), nn.ReLU(), nn.Dropout(params.dropout), 
            nn.Linear(32, self.output_size),
        )
        
        self.save_hyperparameters()
    
    def forward(self, x_inp, x_on_img, x_off_img, debug=False): #[B,L,I]
        
        if debug: print(x_inp.shape, x_on_img.shape, x_off_img.shape)
            
        out_inp = self.inp_encoder(x_inp)
        out_on_img = self.on_img_encoder(x_on_img)
        out_off_img = self.off_img_encoder(x_off_img)

        if debug: print(out_inp.shape, out_on_img.shape, out_off_img.shape)
            
        out = torch.cat((out_inp.view(-1, self.inp_enc_out), 
                         out_on_img.view(-1, self.on_img_enc_out), 
                         out_off_img.view(-1, self.off_img_enc_out), ), dim=-1)
        
        if debug: print(out.shape)
        
        return self.layers(out)
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(),
                                     lr=self.learning_rate, 
                                     weight_decay=self.weight_decay) 
        return optimizer
    
    def _step(self, batch, mode='train'):
        x_inp, y, x_on_imgs, x_off_imgs, sample_num, idx = batch
        
        if torch.cuda.is_available():
            x_inp, y = x_inp.cuda(),  y.cuda()
            x_on_imgs, x_off_imgs = x_on_imgs.cuda(), x_off_imgs.cuda()
        y_pred = self.forward(x_inp, x_on_imgs, x_off_imgs)
        loss = F.mse_loss(y_pred, y)
        self.log('{}_loss'.format(mode), loss)
        return loss
    
    def training_step(self, batch, batch_idx):
        return self._step(batch, mode='train')
    
    def validation_step(self, batch, batch_idx):
        return self._step(batch, mode='val')
    
    def test_step(self, batch, batch_idx):
        return self._step(batch, mode='test')

    
def he_init(module):
    if isinstance(module, nn.Conv2d):
        nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)
    if isinstance(module, nn.Linear):
        nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)

def to_gpu(module):
    device = torch.cuda.current_device()
    module.to(device)


# -

log_base_dir = 'logs/'

if __name__=="__main__":
    params = Munch(
        window_size=1, 
        batch_size=8, #8, 
        num_workers=8,
        input_size=len(vars_x), 
        output_size=len(vars_y),
        hidden_size=32, 
        num_layers=4,
        learning_rate=5e-5,
        max_epochs=100,
        weight_decay=0.0,
        dropout=0.0,
        gpu=torch.cuda.is_available(),
        exp_title='ver05',
        is_train=True, 
        model_path=None,
        num_imgs=data_params.num_imgs,

        inp_enc_out=32,
        on_img_enc_out=16,
        off_img_enc_out=16,
        )

    params.exp_title='ver14'
    params.weight_decay=0.0
    params.dropout=0.0
    
    display(params)


if __name__=="__main__":
    params.data_params=data_params
    params.specimen_id_list = dict(
        train=train_specimen_id_list,
        valid=valid_specimen_id_list,
        test=test_specimen_id_list
    )

if __name__=="__main__":
    data_module = DataModule(params, train_dataset, test_dataset, valid_dataset)


# +
def _build_model(params):
    model = RegressionModule(params)
    return model

def _init_model(params, model):
    if params.is_train:
        model.apply(he_init)       
        if params.gpu:
            model.apply(to_gpu)

    else:
        model_dir = search_model_ckpt(log_base_dir, params.exp_title)
        #model_dir = os.path.join(os.path.join(log_base_dir, params.exp_title),'epoch={}.ckpt'.format(params.max_epochs))
        model_ckpt = torch.load(model_dir)
        model.load_state_dict(model_ckpt['state_dict'])
        if params.gpu:
            model.apply(to_gpu)
            
            
if __name__=="__main__":
    model = _build_model(params)
    display(model)

# +
# if __name__=="__main__":
#     x, y, on_imgs, off_imgs, _, _ = next(iter(data_module.test_dataloader()))

#     model(x, on_imgs, off_imgs, True)

# +
from datetime import datetime

if __name__=="__main__":
    
    if params.is_train:
        start_time = datetime.now()
        print(start_time, 'Start Training')
        _init_model(params, model)
        trainer = pl.Trainer(
            gpus=1, auto_select_gpus=True, 
            max_epochs=params.max_epochs, 
            default_root_dir=os.path.join(log_base_dir, params.exp_title),
            logger=pl.loggers.TensorBoardLogger(log_base_dir, name=params.exp_title))
        trainer.fit(model, data_module)
        trainer.save_checkpoint(os.path.join(os.path.join(log_base_dir, params.exp_title),'epoch={}.ckpt'.format(params.max_epochs)))
        print(datetime.now(), 'Training Complete')
        print('Elapsed Time:', round((datetime.now() - start_time).total_seconds() / 3600, 1), 'Hours')
    else:
        _init_model(params, model)
        trainer = pl.Trainer(accelerator='gpu', auto_select_gpus=True)
# -

# # Evaluate

# +
#from sklearn.metrics import mean_absolute_percentage_error 
from scipy import stats
from sklearn.metrics import r2_score 
import seaborn as sns
import warnings 
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots
pio.renderers.default = "notebook"

warnings.simplefilter(action='ignore', category=FutureWarning)

def evaluate(model, test_loader, inv_norm=True, gpu=True):
    y_real = list()
    y_pred = list()
    x_real = list()
    with torch.no_grad():
        if gpu:
            model = model.cuda(0)
        for batch in tqdm(test_loader):
            x_inp, y, x_on_imgs, x_off_imgs, sample_num, idx = batch
            if gpu:
                x_inp, y, = x_inp.cuda(), y.cuda()
                x_on_imgs, x_off_imgs = x_on_imgs.cuda(), x_off_imgs.cuda()

            y_real.extend(y.data.cpu().numpy())
            x_real.append(x_inp.data.cpu().numpy())
            _y_pred = model.forward(x_inp, x_on_imgs, x_off_imgs)
            y_pred.extend(_y_pred.data.cpu().numpy())

    y_pred = np.array(y_pred)
    y_real = np.array(y_real)
    x_real = np.array(x_real)
    x_real.reshape((-1, x_real.shape[-1]))
    
    return y_pred, y_real, x_real

def _inv_norm_x_inp(x_inp: np.ndarray, data_loader: DataLoader):
    norm_x_inp = data_loader.dataset.transform_x.transforms[0]
    return norm_x_inp.inverse_tranform(x_inp)

def _inv_norm_y(y: np.ndarray, data_loader: DataLoader):
    norm_y = data_loader.dataset.transform_y.transforms[0]
    return norm_y.inverse_tranform(y)

def cal_mape(y_pred, y_real):
    #return mean_absolute_percentage_error(y_real, y_pred) * 100
    #non_zero = np.where(y_real != 0.0)
    non_zero = np.where(np.abs(y_real) > 1e-6)
    out = np.abs(np.divide((y_pred[non_zero] - y_real[non_zero]), y_real[non_zero]))
    return np.nansum(out) / (y_real.shape[0]) * 100

def cal_rmse(y_pred, y_real):
    non_zero = np.where(np.abs(y_real) > 1e-6)
    out = np.sqrt( np.sum(np.square((y_pred[non_zero] - y_real[non_zero])))/ (y_real.shape[0]))
    return out

def cal_r2(y_pred, y_real):
    return r2_score(y_real, y_pred)
#     ss_res = np.sum( np.square(y_real - y_pred) ) 
#     ss_tot = np.sum( np.square(y_real - np.mean(y_real)))
#     return 1 - ss_res / ss_tot

def cal_spearmanr(y_pred, y_real):
    return stats.spearmanr(y_pred, y_real, nan_policy='omit').correlation

def plotly_compact(data, title=None, static=True, xdata=None):
    fig = make_subplots()
    for key, val in data.items():
        if xdata is None:
            fig.add_trace(go.Scatter(y=val, name=key))
        else:
            fig.add_trace(go.Scatter(x=xdata, y=val, name=key, mode='markers'))
            
    _layout_dict = dict()
    if title:
        _layout_dict['title'] = title
    #_layout_dict['yaxis'] = dict(range=[750, 1050 ],)
    #_layout_dict['xaxis'] = dict(range=[17350, 17850],)

    if static:
        _layout_dict['width'] = 1600
        fig.update_layout(_layout_dict)
        fig.show(renderer="png")  
    else:
        fig.update_layout(_layout_dict)
        fig.show()

def plot_scatter(predicted, actual, title='', ax=None):
    if ax is None:
        fig, ax = plt.subplots()
        fig.title(title)
    else:
        ax.set_title(title)
    ax.scatter(predicted, actual)
    ax.set_box_aspect(1)
#     ax.set_xlim(-1.3, 1.3)
#     ax.set_ylim(-1.3, 1.3)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')

def summarize(model, trainer, params, data_module, plot_chart=True): 
    #mse = trainer.test(model)
    dm_loaders = Munch( train = data_module.train_dataloader(shuffle=False), 
                        val = data_module.val_dataloader(shuffle=False), 
                        test = data_module.test_dataloader(shuffle=False))

    print('experiment {}'.format(params.exp_title))
    print('-'*20)
    
    exp_results = {}
    tag_mode = ['train', 'val', 'test'] #'val'
    for mode in tag_mode:  
        y_pred, y_real, x_real = evaluate(model, dm_loaders[mode], gpu=params.gpu)
        y_pred = _inv_norm_y(y_pred, dm_loaders[mode])
        y_real = _inv_norm_y(y_real, dm_loaders[mode])
        
        title = '[{}] {}\n'.format(params.exp_title, mode)
            
        
        if plot_chart:
            fig, axs = plt.subplots(1, 4)
            fig.suptitle(mode)
        
        metrics = []
        vars_y = dm_loaders[mode].dataset.vars_y
        for i, _var_y in enumerate(vars_y):
            _pred, _real = y_pred[:, i], y_real[:, i]
            mape = cal_mape(_pred, _real)
            rmse = cal_rmse(_pred, _real)
            r2 = cal_r2(_pred, _real)
            spearmanr = cal_spearmanr(_pred, _real)
            
            print()
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
                plot_scatter(_pred, _real, title=_var_y, ax=axs[i])
        
        df_results = pd.DataFrame.from_dict(metrics)
        df_results.index = vars_y
        exp_results[mode] = df_results
            
    if plot_chart:
        plt.legend()
        plt.show()
    
    print('experiment {}'.format(params.exp_title))

    return exp_results

    
def search_model_ckpt(base_dir, exp_title):
    base_dir = os.path.join(os.getcwd(), base_dir)
    model_dir = os.path.join(base_dir, exp_title)
    print(sorted(os.listdir(model_dir)))
    model_dir = os.path.join(model_dir, sorted(os.listdir(model_dir))[-1])
    model_dir = os.path.join(model_dir, 'checkpoints')
    model_dir = os.path.join(model_dir, os.listdir(model_dir)[0])
    print(model_dir)
    return model_dir


# -

if __name__=="__main__":

    exp_results = summarize(model, trainer, params, data_module, plot_chart=False)

    for mode, df_result in exp_results.items():
        display(mode)
        display(df_result)



# %load_ext tensorboard

# %tensorboard --logdir 'logs/ver14' --host=0.0.0.0


