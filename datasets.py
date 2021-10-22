from pathlib import Path
import numpy as np

from torchvision.transforms import transforms
from torch.utils.data import Dataset

from utils.logger import get_logger, Logger
from utils.utils import namedtuple, load_mnist, Phase

class BaseDataset(Dataset):
    def __init__(self, dataset_path:str, test_size:float=0.2, phase:Phase=Phase.TRAIN, logger:Logger=None):
        super().__init__()
        self.dataset_path = Path(dataset_path)
        self.test_size = test_size
        self.phase = phase
        self.logger = logger if logger is not None else get_logger('Dataset')
        self.__initialize__()
        self.dev_data = self.__dev_data()

    def __initialize__(self):
        # self.train_data, self.test_data = self.__load_data__(self.dataset_path)
        self.train_data, self.test_data = None, None
        raise NotImplementedError()

    def __dev_data(self):
        return {idx: self.train_data[idx] for idx in range(1000)}

    def __load_data__(self, dataset_path:Path):
        '''return (train_data, test_data, label_data)'''
        # return train_data, test_data, label_data
        raise NotImplementedError()

    def __len__(self):
        if self.phase == Phase.TRAIN:
            return len(self.train_data)
        elif self.phase == Phase.DEV:
            return len(self.dev_data)
        elif self.phase == Phase.TEST:
            return len(self.test_data)
        raise RuntimeError(f'Unknown phase: {self.pahse}')

    def __getitem__(self, index):
        if self.phase == Phase.TRAIN:
            target_index = self.train_indices[index]
            data = self.train_data[target_index]
            raise NotImplementedError()

        elif self.phase == Phase.TEST:
            target_index = self.test_indices[index]
            data = self.test_data[target_index]
            raise NotImplementedError()

        raise RuntimeError(f'Unknown phase: {self.pahse}')

    # phase change functions
    def train(self):
        self.phase = Phase.TRAIN
    def test(self):
        self.phase = Phase.TEST

MnistItem = namedtuple('MnistItem', ('image', 'label'))

class MnistDataset(BaseDataset):
    def __init__(self, dataset_path:str, test_size:float=0.2, phase:Phase=Phase.TRAIN, logger:Logger=None):
        super().__init__(dataset_path, test_size, phase, logger)
    
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
