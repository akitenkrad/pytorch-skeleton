import click

import torch
from torch.utils.data import DataLoader
import torch.optim as optim

from utils.logger import get_logger
from utils.utils import load_config, describe_model, Phase, Path
from utils.lr_finder import lr_finder, save_figure
from datasets.base import MnistDataset
from models.dla import dla34
from train import train as train_func
from predict import predict as pred_func
from validate import validate as valid_func
from explore import calculate_ipca

@click.group()
def cli():
    pass

@cli.command()
@click.option('--config', type=click.Path(exists=True), help='path to config.yml')
def find_lr(config):
    # load config
    config = load_config(config)
    logger = get_logger('lr_finder', logfile=config.log_file, silent=True)

    # prepare dataset
    dataset = MnistDataset(config.dataset_path, config.test_size, Phase.TRAIN, logger)
    dl = DataLoader(dataset, batch_size=config.batch_size)

    # load model
    model = dla34(num_classes=10 ,pool_size=1)
    model = model.train().to(config.device)
    describe_model(model, logger)

    # optimizer, loss
    optimizer = optim.Adam(model.parameters())
    loss_func = torch.nn.CrossEntropyLoss()

    # run lr_finder
    log_lrs, losses = lr_finder(config, model, dl, optimizer, loss_func, config.lr_finder.initial_value, config.lr_finder.final_value, config.lr_finder.beta, logger)
    save_figure(log_lrs, losses)

@cli.command()
@click.option('--config', type=click.Path(exists=True), help='path to config.yml')
def train(config):
    # load config   
    config = load_config(config)
    logger = get_logger('train', logfile=config.log_file, silent=True)

    # prepare dataset
    dataset = MnistDataset(config.dataset_path, config.test_size, Phase.TRAIN, logger)
    dataset.train()

    # prepare model
    model = dla34(num_classes=10, pool_size=1)
    describe_model(model, logger)

    # optimizer, loss
    optimizer = optim.Adam(model.parameters(), lr=config.lr)
    lr_scheduler = optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=config.lr_decay)
    loss_func = torch.nn.CrossEntropyLoss()

    # run train
    train_func(config, dataset, model, optimizer, lr_scheduler, loss_func, logger)

@cli.command()
@click.option('--config', type=click.Path(exists=True), help='path to config.yml')
def predict(config):
    # load config   
    config = load_config(config)
    logger = get_logger('predict', logfile=config.log_file, silent=True)

    # prepare dataset
    dataset = MnistDataset(config.dataset_path, config.test_size, Phase.TEST, logger)
    dataset.submission()

    # prepare model
    model = dla34(num_classes=10, pool_size=1)
    describe_model(model, logger)

    # run predict
    pred_func(config, dataset, model, logger)

@cli.command()
@click.option('--config', type=click.Path(exists=True), help='path to config.yml')
def validate(config):
    # load config
    config = load_config(config)
    logger = get_logger('validate', logfile=config.log_file, silent=True)

    # prepare dataset
    dataset = MnistDataset(config.dataset_path, config.test_size, Phase.TEST, logger)
    dataset.test()

    # prepare model
    model = dla34(num_classes=10, pool_size=1)
    describe_model(model, logger)

    # run validate
    valid_func(config, dataset, model, logger)

@cli.command()
@click.option('--config', type=click.Path(exists=True), help='path to config.yml')
@click.option('--feat-dir', type=click.Path(exists=True), help='path to feature dir')
@click.option('--n-components', type=int, default=10, help='number of components of PCA')
@click.option('--batch-size', type=int, default=64, help='batch size of incremental PCA')
def generate_features(config, feat_dir, n_components, batch_size):
    # load config
    config = load_config(config)
    logger = get_logger('generate_features', logfile=config.log_file, silent=True)

    # extract features and exec pca
    calculate_ipca(Path(config.dataset_path), Path(feat_dir), n_components, batch_size)

if __name__ == '__main__':
    cli()
