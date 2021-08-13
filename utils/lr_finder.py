from attrdict import AttrDict
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from .step import step

def lr_finder(config:AttrDict, model:nn.Module, dataloader:DataLoader, optimizer:optim.Optimizer, loss_func, init_value=1e-8, final_value=10.0, beta=0.98):
    num = len(dataloader) - 1
    mult = (final_value / init_value) ** (1 / num)
    lr = init_value
    optimizer.param_groups[0]['lr'] = lr
    avg_loss = 0.0
    best_loss = 0.0
    losses = []
    log_lrs = []
    
    with tqdm(enumerate(dataloader), total=len(dataloader), desc='[B:{:05d}] lr:{:.8f} best_loss:{:.3f}'.format(0, lr, -1)) as it:
        for idx, (x, y) in it:

            # process model and calculate loss
            loss = step(model, config.device, x, y, loss_func)

            # compute the smoothed loss
            avg_loss = beta * avg_loss + (1-beta) * loss.item()
            smoothed_loss = avg_loss / (1 - beta**(idx+1))

            # stop if the loss is exploding
            if idx > 0 and smoothed_loss > 4 * best_loss:
                return log_lrs, losses

            # record the best loss
            if smoothed_loss < best_loss or idx == 0:
                best_loss = smoothed_loss

            # store the values
            losses.append(smoothed_loss)
            log_lrs.append(lr)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # update progress bar
            it.set_description('[B:{:05d}] lr:{:.8f} best_loss:{:.3f}'.format(idx+1, lr, best_loss))
            
            # update learning rate
            lr *= mult
            optimizer.param_groups[0]['lr'] = lr
        
        return log_lrs, losses

def save_figure(log_lrs, losses, save_file='lr_loss_curve.png'):
        # save figure    
        plt.plot(log_lrs[10:-5], losses[10:-5])
        plt.xscale('log')
        plt.xlabel('Learning Rate')
        plt.ylabel('Loss')
        plt.savefig(save_file)
        print('saved ->', save_file)
