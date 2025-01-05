import torch
from torchvision import datasets, transforms

from transforms import Transforms

from torch.utils.data import DataLoader

class DatasetLoader:
    def __init__(self, batch_size=128, num_workers=4):
        self.kwargs = {
            'batch_size': batch_size,
            'shuffle': True,
            'num_workers': num_workers,
        }
        self.transforms = Transforms()

    def train_loader(self):
        train_dataset = datasets.CIFAR10(
            root='./data',
            train=True,
            download=True,
            transform=self.transforms.train_transforms()
        )
        return DataLoader(train_dataset, **self.kwargs)

        

    def test_loader(self):
        """
        Test loader for the dataset
        """

        test_dataset = datasets.CIFAR10('./data', train=False, download=True, transform=self.transforms.test_transforms())
        
        return DataLoader(test_dataset, **self.kwargs)