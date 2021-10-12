from typing import Union, List, Dict
import warnings
import requests
import gzip
from pathlib import Path
from datetime import datetime, timezone, timedelta
from logging import Logger
import yaml
import subprocess
import numpy as np
import pandas as pd
from attrdict import AttrDict
from collections import namedtuple
from glob import glob
from tqdm import tqdm
from enum import Enum
import torch

from .logger import get_logger

warnings.filterwarnings('ignore')

NVIDIA_SMI_DEFAULT_ATTRIBUTES = (
    'index',
    'uuid',
    'name',
    'timestamp',
    'memory.total',
    'memory.free',
    'memory.used',
    'utilization.gpu',
    'utilization.memory'
)

class Phase(Enum):
    DEV = 1
    TRAIN = 2
    VALID = 3
    TEST = 4
    SUBMISSION = 5

def now():
    JST = timezone(timedelta(hours=9))
    return datetime.now(JST)

def describe_gpu(nvidia_smi_path='nvidia-smi', keys=NVIDIA_SMI_DEFAULT_ATTRIBUTES, no_units=True, logger:Logger=None):
    if logger is None:
        logger = get_logger('gpu_info.log')
    nu_opt = '' if not no_units else ',nounits'
    cmd = f'{nvidia_smi_path} --query-gpu={",".join(keys)} --format=csv,noheader{nu_opt}'
    output = subprocess.check_output(cmd, shell=True)
    lines = output.decode().split('\n')
    lines = [line.strip() for line in lines if line.strip() != '']
    lines = [{ k: v for k, v in zip(keys, line.split(', '))} for line in lines ]
    
    logger.info('====== show GPU information =========')
    for line in lines:
        for k, v in line.items():
            logger.info(f'{k:25s}: {v}')
    logger.info('=====================================')


def load_config(config_path:str, logger:Logger=None, no_log:bool=False):
    if logger is None and no_log == False:
        logger = get_logger('load_config.log')
    config = yaml.safe_load(open(config_path))
    config['device'] = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if no_log == False:
        logger.info('====== show config =========')
        for key, value in config.items():
            logger.info(f'config: {key:20s}: {value}')
        logger.info('============================')

        if torch.cuda.is_available():
            describe_gpu(logger=logger)

    return AttrDict(config)

def describe_model(model:torch.nn.Module, logger:Logger):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    total_params = 0
    logger.info('====== describe model ======')
    logger.info('{:20s}: {}'.format('device', device))
    for name, param in model.named_parameters():
        logger.info(f'{name:20s}: {param.shape}')
        total_params += param.numel() 
    logger.info('----------------------------')
    logger.info('{:20s}: {}'.format('total_params', total_params))
    logger.info('============================')

def load_mnist(path, kind:Phase=Phase.TRAIN):
    '''Load MNIST data from `path`'''

    kind_name_map = {Phase.TRAIN: 'train', Phase.TEST: 't10k'}
    kind_name = kind_name_map[kind]

    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    labels_path = path / f'{kind_name}-labels-idx1-ubyte.gz'
    images_path = path / f'{kind_name}-images-idx3-ubyte.gz'

    if not labels_path.exists() or not images_path.exists():
        urls = [
            'http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-images-idx3-ubyte.gz',
            'http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-labels-idx1-ubyte.gz',
            'http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-images-idx3-ubyte.gz',
            'http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-labels-idx1-ubyte.gz',
        ]
        for url in tqdm(urls, desc='loading mnist data...'):
            urlData = requests.get(url).content
            with open(path / url.split('/')[-1], 'wb') as f:
                f.write(urlData)

    with gzip.open(str(labels_path), 'rb') as lbpath:
        labels = np.frombuffer(lbpath.read(), dtype=np.uint8, offset=8)
    with gzip.open(str(images_path), 'rb') as imgpath:
        images = np.frombuffer(imgpath.read(), dtype=np.uint8, offset=16).reshape(len(labels), 784)
    return images, labels
