from pathlib import Path
import numpy as np
from torchvision.transforms import transforms
from torch.utils.data import Dataset
from utils.logger import get_logger, Logger
from utils.utils import glob, tqdm, load_mnist

class BaseDataset(Dataset):
    def __init__(self, dataset_path:str, test_size:float=0.2, phase:str='train', logger:Logger=None):
        super().__init__()
        self.dataset_path = Path(dataset_path)
        self.test_size = test_size
        self.phase = phase
        self.logger = logger if logger is not None else get_logger('Dataset')
        self.__initialize__()

    def __initialize__(self):
        # self.train_data, self.test_data, self.label_data = self.__load_data__(self.dataset_path)
        raise NotImplementedError()

    def __load_data__(self, dataset_path:Path):
        '''return (train_data, test_data, label_data)'''
        # return train_data, test_data, label_data
        raise NotImplementedError()

    def __len__(self):
        if self.phase == 'train':
            return len(self.train_data)
        elif self.phase == 'test':
            return len(self.test_data)
        raise RuntimeError(f'Unknown phase: {self.pahse}')

    def __getitem__(self, index):
        if self.phase == 'train':
            target_index = self.train_indices[index]
            data = self.train_data[target_index]
            raise NotImplementedError()

        elif self.phase == 'test':
            target_index = self.test_indices[index]
            data = self.test_data[target_index]
            raise NotImplementedError()

        raise RuntimeError(f'Unknown phase: {self.pahse}')

    # phase change functions
    def train(self):
        self.phase = 'train'
    def test(self):
        self.phase = 'test'

class MnistDataset(BaseDataset):
    def __init__(self, dataset_path:str, test_size:float=0.2, phase:str='train', logger:Logger=None):
        super().__init__(dataset_path, test_size, phase, logger)
    
    def __initialize__(self):
        self.train_data, self.train_labels, self.test_data, self.test_labels = self.__load_data__(self.dataset_path)
        self.transform = transforms.Compose([
            transforms.ToTensor(),
        ])

    def __load_data__(self, dataset_path:Path):
        train_images, train_labels = load_mnist(dataset_path, 'train')
        test_images, test_labels = load_mnist(dataset_path, 'test')
        
        # list -> dict
        train_images = {idx:data.reshape(28, 28) for idx, data in enumerate(train_images)}
        train_labels = {idx:data for idx, data in enumerate(train_images)}
        test_images = {idx:data.reshape(28, 28) for idx, data in enumerate(test_images)}
        test_labels = {idx:data for idx, data in enumerate(test_labels)}

        return train_images, train_labels, test_images, test_labels

    def __getitem__(self, index):

        if self.phase == 'train':
            data = self.train_data[index]
            label = self.train_labels[index]

            data = self.transform(np.array(data))
            return data, label

        elif self.phase == 'test':
            data = self.test_data[index]
            data = self.transform(data)
            return data

        raise RuntimeError(f'Unknown phase: {self.pahse}')
