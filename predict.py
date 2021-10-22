from attrdict import AttrDict
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from utils.utils import Path, now, is_colab
from utils.logger import Logger
from utils.step import step_without_loss
from datasets import BaseDataset

if is_colab():
    from tqdm.notebook import tqdm
else:
    from tqdm import tqdm

def predict(config:AttrDict, dataset:BaseDataset, model:nn.Module, logger:Logger):

    # prepare dataloader
    dl = DataLoader(dataset, batch_size=config.batch_size)

    # load model
    model = model.eval().to(config.device)

    # load weights
    weights_path = Path(config.weights_dir) / f'{config.model.name}_best.pt'
    if torch.cuda.is_available():
        model.load_state_dict(torch.load(str(weights_path)))
    else:
        model.load_state_dict(torch.load(str(weights_path), map_location=config.device))
    logger.info(f'Loaded model weights: {str(weights_path.resolve().absolute())}')

    # output path
    out_path = Path(config.out_dir) / f'{config.model.name}_{now().strftime("%Y%m%d%H%M%S")}' / 'submission.csv'
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with tqdm(enumerate(dl), total=len(dl), desc=f'[Records {0:06d}]') as batch_it:

        outputs = []
        for batch, items in batch_it:
            out:torch.Tensor = step_without_loss(model, config.device, items)
            out = torch.sigmoid(out)
            out = out.squeeze().cpu().detach().numpy()
            ids = items.id
            out = [{'id': i, 'target': o} for i, o in zip(ids, out)]
            outputs += out

            batch_it.set_description(f'[Records {len(outputs):06d}]')

    df = pd.DataFrame(outputs)
    df.to_csv(str(out_path), index=False, header=True, float_format='%.2f')
    logger.info(f'saved -> {str(out_path.resolve().absolute())}')