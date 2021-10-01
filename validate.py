import pickle
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from utils.utils import Path, AttrDict, tqdm, now, pd, np
from utils.logger import Logger
from utils.step import step_without_loss
from datasets import BaseDataset


def validate(config:AttrDict, dataset:BaseDataset, model:nn.Module, logger:Logger):

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
    out_path = Path(config.out_dir) / f'{config.model.name}_{now().strftime("%Y%m%d%H%M%S")}' / 'features'
    out_path.mkdir(parents=True, exist_ok=True)

    with tqdm(enumerate(dl), total=len(dl), desc=f'[Batch {0:05d} / {len(dl)}]') as batch_it:

        for batch, items in batch_it:
            out, feat = step_without_loss(model, config.device, items)
            out = torch.sigmoid(out)

            out = out.squeeze().cpu().detach().numpy()
            feat = feat.squeeze().cpu().detach().numpy()
            ids = items.id

            outputs = [{'item': dataset.id2data[i], 'out': o, 'feat': f.mean(axis=-1).reshape(1, -1)} for i, o, f in zip(ids, out, feat)]

            for output in outputs:
                outfile = out_path / (str(output['item'].id) + '.npy')
                np.save(str(outfile), output)

            batch_it.set_description(f'[Batch {batch:05d} / {len(dl)}]')