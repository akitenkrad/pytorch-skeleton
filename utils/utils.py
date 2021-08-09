import os
import requests
import gzip
from pathlib import Path
from datetime import datetime, timezone, timedelta
from logging import Logger
import yaml
import numpy as np
from attrdict import AttrDict
from glob import glob
from tqdm import tqdm
import torch
from .logger import get_logger

def now():
    JST = timezone(timedelta(hours=9))
    return datetime.now(JST)

def load_mnist(path, kind='train'):
    '''Load MNIST data from `path`'''

    assert kind in ['train', 'test']
    if kind == 'test':
        kind = 't10k'

    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    labels_path = path / f'{kind}-labels-idx1-ubyte.gz'
    images_path = path / f'{kind}-images-idx3-ubyte.gz'

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

def load_config(config_path:str, logger:Logger=None):
    if logger is None:
        logger = get_logger('load_config.log')
    config = yaml.safe_load(open(config_path))
    config['device'] = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    logger.info('====== show config =========')
    for key, value in config.items():
        logger.info(f'config: {key:20s}: {value}')
    logger.info('============================')

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
