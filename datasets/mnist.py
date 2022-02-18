from pathlib import Path
import gzip
import requests
from tqdm import tqdm
import numpy as np
from torchvision.transforms import transforms

from utils.logger import Logger
from utils.utils import namedtuple, Phase, Config
from datasets.base import BaseDataset


MnistItem = namedtuple('MnistItem', ('image', 'label'))

class MnistDataset(BaseDataset):
    def __init__(self, config:Config, dataset_path:str, test_size:float=0.2, phase:Phase=Phase.TRAIN, logger:Logger=None):
        super().__init__(dataset_path, test_size, phase, logger)
        self.config = Config
    
    def __initialize__(self):
        self.train_data, self.test_data = self.__load_data__(self.dataset_path)
        self.transform = transforms.Compose([
            transforms.ToTensor(),
        ])

    def __load_data__(self, dataset_path:Path):
        train_images, train_labels = self.load_mnist(dataset_path, Phase.TRAIN)
        test_images, test_labels = self.load_mnist(dataset_path, Phase.TEST)
        
        # list -> dict
        train_data = {idx: MnistItem(data.reshape(28, 28), label) for idx, (data, label) in enumerate(zip(train_images, train_labels))}
        test_data = {idx: MnistItem(data.reshape(28, 28), label) for idx, (data, label) in enumerate(zip(test_images, test_labels))}

        return train_data, test_data

    def __getitem__(self, index):

        if self.phase == Phase.TRAIN or self.phase == Phase.VALID:
            data:MnistItem = self.train_data[index]
            label = data.label

            data = self.transform(np.array(data.image))
            return data, label

        elif self.phase == Phase.DEV:
            data:MnistItem = self.dev_data[index]
            label = data.label

            data = self.transform(np.array(data.image))
            return data, label

        elif self.phase == Phase.TEST:
            data:MnistItem = self.test_data[index]
            data = self.transform(np.array(data.image))
            return data

        raise RuntimeError(f'Unknown phase: {self.pahse}')

    def load_mnist(self, path, kind:Phase=Phase.TRAIN):
        '''Load MNIST data from `path`'''

        kind_name_map = {Phase.TRAIN: 'train', Phase.TEST: 't10k'}
        kind_name = kind_name_map[kind]

        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        labels_path = path / f'{kind_name}-labels-idx1-ubyte.gz'
        images_path = path / f'{kind_name}-images-idx3-ubyte.gz'

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