import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path
from attrdict import AttrDict
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from utils.logger import get_logger, Logger
from utils.watchers import SimpleWatcher, AucWatcher
from utils.utils import tqdm, load_config, describe_model, now 
from datasets import BaseDataset

def validate(batch_size, epoch, ds:BaseDataset, model, loss_func, threshold):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dl = DataLoader(ds, batch_size=batch_size)

    loss_watcher = SimpleWatcher('loss', default_value=sys.maxsize)
    auc_watcher = AucWatcher('auc', threshold=threshold)

    with tqdm(dl, total=len(dl), desc=f'[Epoch {epoch:4d} - Validate:{ds.phase}]', leave=False) as valid_it:
        for x, y in valid_it:
            x, y = x.type(torch.float32).to(device), y.type(torch.float32).to(device)
            with torch.no_grad():
                out = model(x)
                out, y = out.squeeze(), y.squeeze()
                loss = loss_func(out, y)

                loss_watcher.put(loss.item())
                auc_watcher.put(out.cpu().numpy(), y.cpu().numpy())
    ds.train()
    return auc_watcher.auc, loss_watcher.mean()

def save_model(config, model, name):
    save_dir = Path(config.weights_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), str(save_dir / name))

def train(config:AttrDict, dataset:BaseDataset, model:nn.Module, logger:Logger):

    # load model
    model = model.train().to(config.device)

    # optimizer, loss
    optimizer = optim.Adam(model.parameters(), lr=config.lr)
    optim_scheduler = optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=config.lr_decay)
    loss_func = torch.nn.BCELoss()

    log_dir = Path(config.log_dir) / 'exp_{}'.format(datetime.now(timezone(timedelta(hours=9), 'JST')).strftime('%Y%m%d-%H%M%S'))
    with SummaryWriter(str(log_dir)) as writer:

        # k-fold cross validation
        kfold = KFold(n_splits=config.k_folds, shuffle=True)
        with tqdm(enumerate(kfold.split(dataset)), total=config.k_folds, desc='[Fold   0]') as fold_it:
            for fold, (train_indices, valid_indices) in fold_it:
                fold_it.set_description(f'[Fold {fold:2d}]')

                # prepare dataloader
                dataset.train_indices = train_indices
                dataset.valid_indices = valid_indices
                dataset.train()
                dl = DataLoader(dataset, batch_size=config.batch_size)

                # Epoch roop
                with tqdm(range(config.epochs), total=config.epochs, desc=f'[Fold {fold:2d} / Epoch   0]', leave=False) as epoch_it:
                    auc_watcher = SimpleWatcher('auc', default_value=-1, patience=config.early_stop_patience)

                    for epoch in epoch_it:
                        loss_watcher = SimpleWatcher('loss', default_value=sys.maxsize)

                        # Batch roop
                        with tqdm(enumerate(dl), total=len(dl), desc=f'[Epoch {epoch:3d} / Batch {0:3d}]', leave=False) as batch_it:
                            for batch, (x, y) in batch_it:

                                x = x.type(torch.float32).to(config.device)
                                y = y.type(torch.float32).to(config.device)

                                # model output
                                out = model(x)

                                # calculate loss
                                loss = loss_func(out.squeeze(), y)

                                # update parameters
                                optimizer.zero_grad()
                                loss.backward()
                                optimizer.step()

                                # put logs
                                loss_watcher.put(loss.item())
                                batch_it.set_description(f'[Epoch {epoch:3d} / Batch {batch:3d}] Loss: {loss.item():.5f}')

                        # evaluation
                        dataset.valid()
                        val_auc, val_loss = validate(config.batch_size, epoch, dataset, model, loss_func, config.threshold)

                        # step learning rate
                        optim_scheduler.step()

                        # update iteration description
                        desc = f'[Fold {fold:2d} / Epoch {epoch:3d}] Train Loss: {loss_watcher.mean():.5f} | Valid Loss:{val_loss:.5f} | Valid AUC: {val_auc:.5f}'
                        epoch_it.set_description(desc)

                        # logging
                        last_lr = optim_scheduler.get_last_lr()[0]
                        logger.info(f'[Fold {fold:2d} / Epoch {epoch:3d}] Train Loss: {loss_watcher.mean():.5f} | Valid Loss:{val_loss:.5f} | Valid AUC: {val_auc:.5f} | LR: {last_lr:.7f}')
                        writer.add_text('train_log', desc, epoch)
                        writer.add_scalar('train loss', loss_watcher.mean(), epoch)
                        writer.add_scalar('valid loss', val_loss, epoch)
                        writer.add_scalar('valid auc', val_auc, epoch)

                        # save best model
                        if auc_watcher.is_best:
                            save_model(config, model, f'{config.model}_best.pt')
                        auc_watcher.put(val_auc)

                        # save model regularly
                        if epoch % 5 == 0:
                            save_model(config, model, f'{config.model}_last.pt')

                        # early stopping
                        if auc_watcher.early_stop:
                            logger.info(f'====== Early Stopping @epoch: {epoch} @AUC: {auc_watcher.best_score:5.10f} ======')
                            save_model(config, model, f'{config.model}_last.pt')
                            break

                    save_model(config, model, f'{config.model}_last.pt')

def predict(config):
    logger = get_logger('predict.log')
    config = load_config(config, logger)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # load dataset
    image_size = (config.image_size.height, config.image_size.width, config.image_size.length)
    dataset = BaseDataset(config.dataset_path, image_size=image_size, test_size=config.test_size, phase='test', logger=logger)
    dl = DataLoader(dataset, batch_size=config.batch_size)

    # load model
    blocks_args, global_params = get_model_params('efficientnet-b0', {'num_classes': 1, 'image_size': image_size})
    model = EfficientNet(blocks_args, global_params)
    model = model.train().to(device)
    describe_model(model, logger)

    # load weights
    weights_path = Path(config.weights_dir) / f'{config.model}_best.pt'
    if torch.cuda.is_available():
        model.load_state_dict(torch.load(str(weights_path)))
    else:
        model.load_state_dict(torch.load(str(weights_path), map_location=device))

    with tqdm(dl, total=len(dl), desc=f'predicting...', leave=False) as batch_it:
        outputs = []
        for x, case in batch_it:
            
            x = x.type(torch.float32).to(device)
            with torch.no_grad():
                out = model(x)
                out = out.squeeze().numpy()
                for o, c in zip(out, case):
                    outputs.append({'BraTS21ID': case, 'MGMT_value': o})
        
        out_path = Path(config.out_dir) / f'{config.model}_{now().strftime("%Y%m%d%H%M%S")}' / 'submission.csv'
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame(outputs)
        df.to_csv(str(out_path), index=False, header=True, float_format='%.15f')
        logger.info(f'saved -> {str(out_path)}')
