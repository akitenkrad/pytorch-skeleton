from pathlib import Path
import numpy as np

from torchvision.transforms import transforms

from utils.logger import Logger
from utils.utils import namedtuple, load_mnist, Phase, Config
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
        train_images, train_labels = load_mnist(dataset_path, Phase.TRAIN)
        test_images, test_labels = load_mnist(dataset_path, Phase.TEST)
        
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
