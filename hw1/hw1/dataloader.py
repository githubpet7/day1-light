import os 
import torch
import numpy as np
import torchvision.transforms as transforms
from PIL import Image
from torchvision.transforms.transforms import Compose
from torchvision.transforms import functional as F
from torch.utils.data import Dataset, DataLoader

def train_dataloader(path, batch_size=64, num_workers=0, use_transform=True):
    data_dir = os.path.join(path,'train')
    T = None
    if use_transform:
        T = Compose([
            transforms.RandomCrop(256),
            transforms.ToTensor()
        ])
    dataloader = DataLoader(
        RainDataset(data_dir,transform=T),
        batch_size= batch_size,
        shuffle = True,
        num_workers = num_workers,
        pin_memory = True
    )
    return dataloader

def test_dataloader(path, batch_size=1, num_workers=0):
    data_dir = os.path.join(path,'test')
    T = Compose([
        transforms.CenterCrop(256),
        transforms.ToTensor()
    ])
    dataloader = DataLoader(
        RainDataset(data_dir,transform=T, is_test= True),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    return dataloader

class RainDataset(Dataset):
    def __init__(self, data_dir, transform=None, is_test=False):
        self.data_dir = data_dir
        self.transform = transform
        self.is_test = is_test
        
        if self.is_test:
            self.image_list = os.listdir(os.path.join(data_dir))
            
        else:
            heavy_folder = os.listdir(os.path.join(data_dir, 'Heavy'))
            light_folder = os.listdir(os.path.join(data_dir, 'Light'))
            self.image_list = ['Heavy/'+str(i) for i in heavy_folder] + ['Light/'+str(i) for i in light_folder]
        
        self._check_image(self.image_list)
        self.image_list.sort()
    
    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        image = Image.open(os.path.join(self.data_dir, self.image_list[idx]))
        if self.transform:
            image = self.transform(image)
        else:
            image = F.to_tensor(image)
        
        if self.is_test:
            name = self.image_list[idx]
            return image, name

        label = 1 if self.image_list[idx][:1] == "H" else 0
        return image, label

    @staticmethod
    def _check_image(lst):
        for x in lst:
            splits = x.split('.')
            if splits[-1] not in ['png', 'jpg', 'jpeg']:
                raise ValueError