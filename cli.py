import click
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
from utils.logger import get_logger
from utils.utils import load_config, describe_model
from utils.lr_finder import lr_finder, save_figure
from train import train as train_func
from datasets import MnistDataset
from models import dla34

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
    dataset = MnistDataset(config.dataset_path, config.test_size, 'train', logger)

    # prepare model
    model = dla34(num_classes=10, pool_size=1)
    describe_model(model, logger)

    # run train
    train_func(config, dataset, model, logger)

@cli.command()
@click.option('--config', type=click.Path(exists=True), help='path to config.yml')
def find_lr(config):
    # load config
    logger = get_logger('lr_finder.log', silent=True)
    config = load_config(config, logger)

    # prepare dataset
    dataset = MnistDataset(config.dataset_path, config.test_size, 'train', logger)
    dl = DataLoader(dataset, batch_size=config.batch_size)

    # load model
    model = dla34(num_classes=10 ,pool_size=1)
    model = model.train().to(config.device)
    describe_model(model, logger)

    # optimizer, loss
    optimizer = optim.Adam(model.parameters())
    loss_func = torch.nn.CrossEntropyLoss()

    # run lr_finder
    log_lrs, losses = lr_finder(config, model, dl, optimizer, loss_func, config.lr_finder.initial_value, config.lr_finder.final_value, config.lr_finder.beta)
    save_figure(log_lrs, losses)

if __name__ == '__main__':
    cli()
