import pandas as pd
import numpy as np
from PIL import Image
from PIL import ImageOps
import PIL
import torch, torchvision
from torchvision import transforms, datasets
from torch.utils.data import Dataset, DataLoader

class convResDataset(Dataset):
    def __init__(self, arr = None, img_path='/home/rliu/TDD-Net/data/', pad_size=50, transforms=None):
        """
        Args:
            df: dataframes of training data
            transform: pytorch transforms for transforms and tensor conversion
        """
        self.data = arr
        self.img_path = img_path
        self.transforms = transforms
        self.pad_size = pad_size

    def __getitem__(self, index):
        img = Image.open(self.img_path + '%06.0f.jpg' % self.data[index]).convert('L')
        img_resized = torchvision.transforms.functional.resize(img, (4064,4064), interpolation=2)
        img_resized = ImageOps.expand(img_resized, border=112, fill=0)
        if self.transforms is not None:
            img_resized = self.transforms(img_resized)
        toTensor = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[0.3019], std=[0.1909])]) # toTensor
        img_tensor = toTensor(img_resized)
        # Return image and the label
        return img_tensor

    def __len__(self):
        return len(self.data)