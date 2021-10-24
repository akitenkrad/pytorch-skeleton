from datetime import datetime, timezone, timedelta
from pathlib import Path
from attrdict import AttrDict
from sklearn.model_selection import KFold

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader

from utils.logger import Logger
from utils.watchers import LossWatcher
from utils.utils import is_colab, backup
from utils.step import step
from datasets import BaseDataset

if is_colab():
    from tqdm.notebook import tqdm
else:
    from tqdm import tqdm

def validate(epoch, valid_dl:DataLoader, model, loss_func, threshold):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    loss_watcher = LossWatcher('loss')

    with tqdm(valid_dl, total=len(valid_dl), desc=f'[Epoch {epoch:4d} - Validate]', leave=False) as valid_it:
        for x, y in valid_it:
            with torch.no_grad():
                loss, out = step(model, device, x, y, loss_func)
                loss_watcher.put(loss.item())

    return loss_watcher.mean

def save_model(config, model, name):
    save_dir = Path(config.weights_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), str(save_dir / name))

def train(config:AttrDict, dataset:BaseDataset, model:nn.Module, optimizer:optim.Optimizer, lr_scheduler, loss_func, logger:Logger):

    # load model
    model = model.train().to(config.device)

    log_dir = Path(config.log_dir) / 'tensorboard' / 'exp_{}'.format(datetime.now(timezone(timedelta(hours=9), 'JST')).strftime('%Y%m%d-%H%M%S'))
    global_step = 0
    with SummaryWriter(str(log_dir)) as writer:

        # k-fold cross validation
        kfold = KFold(n_splits=config.k_folds, shuffle=True)
        with tqdm(enumerate(kfold.split(dataset)), total=config.k_folds, desc='[Fold   0]') as fold_it:
            for fold, (train_indices, valid_indices) in fold_it:
                fold_it.set_description(f'[Fold {fold:2d}]')

                # prepare dataloader
                train_subsampler = torch.utils.data.SubsetRandomSampler(train_indices)
                valid_subsampler = torch.utils.data.SubsetRandomSampler(valid_indices)
                train_dl = DataLoader(dataset, batch_size=config.batch_size, sampler=train_subsampler)
                valid_dl = DataLoader(dataset, batch_size=config.batch_size, sampler=valid_subsampler)

                # initialize learning rate
                optimizer.param_groups[0]['lr'] = config.lr

                # Epoch roop
                with tqdm(range(config.epochs), total=config.epochs, desc=f'[Fold {fold:2d} | Epoch   0]', leave=False) as epoch_it:
                    valid_loss_watcher = LossWatcher('valid_loss', patience=config.early_stop_patience)

                    for epoch in epoch_it:
                        loss_watcher = LossWatcher('loss')

                        # Batch roop
                        with tqdm(enumerate(train_dl), total=len(train_dl), desc=f'[Epoch {epoch:3d} | Batch {0:3d}]', leave=False) as batch_it:
                            for batch, (x, y) in batch_it:

                                # process model and calculate loss
                                loss, out = step(model, config.device, x, y, loss_func)

                                # update parameters
                                optimizer.zero_grad()
                                loss.backward()
                                optimizer.step()

                                # put logs
                                writer.add_scalar('train loss', loss.item(), global_step)
                                loss_watcher.put(loss.item())
                                batch_it.set_description(f'[Epoch {epoch:3d} | Batch {batch:3d}] Loss: {loss.item():.3f}')

                                # global step for summary writer
                                global_step += 1

                                if batch > 0 and batch % config.logging_per_batch == 0:
                                    logger.info(f'[Fold {fold:02d} | Epoch {epoch:03d} | Batch {batch:05d}/{len(train_dl):05d} ({(batch/len(train_dl)) * 100.0:.2f}%)] Loss:{loss.item():.3f}')

                        # evaluation
                        val_loss = validate(epoch, valid_dl, model, loss_func, config.threshold)

                        # step scheduler
                        lr_scheduler.step()

                        # update iteration description
                        desc = f'[Fold {fold:2d} | Epoch {epoch:3d}] Train Loss: {loss_watcher.mean:.5f} | Valid Loss:{val_loss:.5f}'
                        epoch_it.set_description(desc)

                        # logging
                        last_lr = lr_scheduler.get_last_lr()[0]
                        logger.info(f'[Fold {fold:2d} / Epoch {epoch:3d}] Train Loss: {loss_watcher.mean:.5f} | Valid Loss:{val_loss:.5f} | LR: {last_lr:.7f}')
                        writer.add_text('train_log', desc, global_step)
                        writer.add_scalar('valid loss', val_loss, global_step)

                        # save best model
                        if valid_loss_watcher.is_best:
                            save_model(config, model, f'{config.model.name}_best.pt')
                        valid_loss_watcher.put(val_loss)

                        # save model regularly
                        if epoch % 5 == 0:
                            save_model(config, model, f'{config.model.name}_f{fold}e{epoch}.pt')

                        # early stopping
                        if valid_loss_watcher.early_stop:
                            logger.info(f'====== Early Stopping @epoch: {epoch} @Loss: {valid_loss_watcher.best_score:5.10f} ======')
                            break

                        # backup files
                        if config.backup.backup:
                            backup(config)
                            logger.info(f'backup logs -> {str(config.backup.backup_dir.resolve().absolute())}')

                    save_model(config, model, f'{config.model.name}_last_f{fold}.pt')
