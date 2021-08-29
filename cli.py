import click

import torch
from torch.utils.data import DataLoader
import torch.optim as optim

from utils.logger import get_logger
from utils.utils import load_config, describe_model, Phase
from utils.lr_finder import lr_finder, save_figure
from train import train as train_func, predict as pred_func
from datasets import MnistDataset
from models.dla import dla34

@click.group()
def cli():
    pass

@cli.command()
@click.option('--config', type=click.Path(exists=True), help='path to config.yml')
def train(config):
    # load config   
    logger = get_logger('train.log', silent=True)
    config = load_config(config, logger)

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
    logger = get_logger('predict.log', silent=True)
    config = load_config(config, logger)

    # prepare dataset
    dataset = MnistDataset(config.dataset_path, config.test_size, Phase.TEST, logger)
    dataset.test()

    # prepare model
    model = dla34(num_classes=10, pool_size=1)
    describe_model(model, logger)

    # run predict
    pred_func(config, dataset, model, logger)

@cli.command()
@click.option('--config', type=click.Path(exists=True), help='path to config.yml')
def find_lr(config):
    # load config
    logger = get_logger('lr_finder.log', silent=True)
    config = load_config(config, logger)

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

if __name__ == '__main__':
    cli()
