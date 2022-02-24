from typing import Union
import os, sys
from os import PathLike
import string
import random
import requests
import cpuinfo
import gzip
from pathlib import Path
from datetime import datetime, timezone, timedelta
from logging import Logger
import yaml
import shutil
import subprocess
import numpy as np
import pandas as pd
from attrdict import AttrDict
from collections import namedtuple
from glob import glob
from enum import Enum
import torch
from torchinfo import summary

from utils.logger import get_logger, kill_logger

def is_colab():
    return 'google.colab' in sys.modules

if is_colab():
    from tqdm.notebook import tqdm
    print('running on google colab -> use tqdm.notebook')
else:
    from tqdm import tqdm

class Phase(Enum):
    DEV = 1
    TRAIN = 2
    VALID = 3
    TEST = 4
    SUBMISSION = 5

class Config(AttrDict):

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

    def __init__(self, config_path:PathLike):
        self['config_path'] = Path(config_path)
        self.timestamp:datetime = self.now()
        self.__load_config(config_path)

    @classmethod
    def get_hash(cls, size:int=12) -> str:
        chars = string.ascii_lowercase + string.digits
        return ''.join(random.SystemRandom().choice(chars) for _ in range(size))

    @classmethod
    def now(cls) -> datetime:
        JST = timezone(timedelta(hours=9))
        return datetime.now(JST)

    def __load_config(self, config_path:str):
        config = yaml.safe_load(open(config_path))
        for key, value in config.items():
            self[key] = value
        self['device'] = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self['log_dir'] = str(Path(self['log_dir']) / self.timestamp.strftime('%Y%m%d%H%M%S'))
        self['log_file'] = str(Path(self['log_dir']) / self['log_filename'])
        self['weights_dir'] = str(Path(self['log_dir']) / 'weights')
        self['backup']['backup_dir'] = str(Path(self['backup']['backup_dir']) / Path(self['log_dir']).name)
        self['loggers'] = AttrDict()

        if hasattr(self, '__logger') and isinstance(self.__logger, Logger):
            kill_logger(self.__logger)

        self['loggers']['logger']= get_logger(name='config', logfile=self['log_file'])
        self['logger'] = self['loggers']['logger']
        config['hash'] = Config.get_hash(16)

        self.logger.info('====== show config =========')
        attrdict_attrs = list(dir(AttrDict()))
        for key, value in self.items():
            if key not in attrdict_attrs:
                self.logger.info(f'config: {key:20s}: {value}')
        self.logger.info('============================')

        # CPU info
        self.describe_cpu()

        # GPU info
        if torch.cuda.is_available():
            self.describe_gpu()

        return AttrDict(config)

    def describe_cpu(self):
        self.logger.info('====== cpu info ============')
        for key, value in cpuinfo.get_cpu_info().items():
            self.logger.info(f'CPU INFO: {key:20s}: {value}')
        self.logger.info('============================')

    def describe_gpu(self, nvidia_smi_path='nvidia-smi', no_units=True):

        try:
            keys = self.NVIDIA_SMI_DEFAULT_ATTRIBUTES
            nu_opt = '' if not no_units else ',nounits'
            cmd = f'{nvidia_smi_path} --query-gpu={",".join(keys)} --format=csv,noheader{nu_opt}'
            output = subprocess.check_output(cmd, shell=True)
            lines = output.decode().split('\n')
            lines = [line.strip() for line in lines if line.strip() != '']
            lines = [{ k: v for k, v in zip(keys, line.split(', '))} for line in lines ]

            self.logger.info('====== show GPU information =========')
            for line in lines:
                for k, v in line.items():
                    self.logger.info(f'{k:25s}: {v}')
            self.logger.info('=====================================')
        except:
            self.logger.info('====== show GPU information =========')
            self.logger.info('  No GPU was found.')
            self.logger.info('=====================================')

    def describe_model(self, model:torch.nn.Module, input_size:tuple=None, input_data=None):
        if input_data is None:
            summary_str = summary(model,
                input_size=input_size,
                col_names=['input_size', 'output_size', 'num_params', 'kernel_size', 'mult_adds'],
                col_width=18,
                row_settings=['var_names'],
                verbose=0)
        else:
            summary_str = summary(model,
                input_data=input_data,
                col_names=['input_size', 'output_size', 'num_params', 'kernel_size', 'mult_adds'],
                col_width=18,
                row_settings=['var_names'],
                verbose=0)

        for line in summary_str.__str__().split('\n'):
            self.logger.info(line)

    def backup_logs(self):
        '''copy log directory to config.backup'''
        backup_dir = Path(self.backup.backup_dir)
        if backup_dir.exists():
            shutil.rmtree(str(backup_dir))
        backup_dir.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(self.log_dir, self.backup.backup_dir)

    def add_logger(self, name:str):
        self['loggers'][name]= get_logger(name=name, logfile=self['log_file'])
        self[name] = self['loggers'][name]
