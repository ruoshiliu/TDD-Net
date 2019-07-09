import pandas as pd
import numpy as np
from PIL import Image
from PIL import ImageOps
import PIL
import torch, torchvision
from torchvision import transforms, datasets
from torch.utils.data import Dataset, DataLoader
from util import create_circular_mask, sample_point_circular, split_and_sample, sample_rec


class defectDataset_csv(Dataset):
    def __init__(self, csv_path='/home/rliu/yolo2/v2_pytorch_yolo2/data/an_data/VOCdevkit/VOC2007/csv_labels/train.csv', img_path='/home/rliu/TDD-Net/data/', window_size=50, pad_size=50, mask = create_circular_mask(200,200), transforms=None):
        """
        Args:
            csv_path (string): path to csv file
            transform: pytorch transforms for transforms and tensor conversion
        """
        self.data = pd.read_csv(csv_path, sep=" ")
        self.img_path = img_path
        self.transforms = transforms
        self.window_size = window_size
        self.pad_size = pad_size
        self.mask = mask

    def __getitem__(self, index):
        labels = self.data.loc[index]
        single_image_label = int(labels['class']) # float
        x = labels['x']
        y = 1 - labels['y'] # origin of PIL image is top-left
        img_index = labels['image_index']
        img = Image.open(self.img_path + '%06.0f.jpg' % img_index)
        img = img.convert('L')
        img = torchvision.transforms.functional.resize(img, (300,300), interpolation=2)
        width, height = img.size
        img = ImageOps.expand(img, border=self.pad_size, fill=0)
        xmin = width * x - self.window_size/2 + self.pad_size
        ymin = height * y - self.window_size/2 + self.pad_size
        xmax = width * x + self.window_size/2 + self.pad_size
        ymax = height * y + self.window_size/2 + self.pad_size
        img_resized = img.crop((xmin, ymin, xmax, ymax))
        img_resized = torchvision.transforms.functional.resize(img_resized, (200,200), interpolation=2)
        img_masked = img_resized * self.mask
        img_masked = Image.fromarray(img_masked.astype('uint8'), 'L')
        if self.transforms is not None:
            img_masked = self.transforms(img_masked)
        # Return image and the label
        return (img_masked, single_image_label)

    def __len__(self):
        return len(self.data.index)

class defectDataset_df(Dataset):
    def __init__(self, df = pd.read_csv('/home/rliu/yolo2/v2_pytorch_yolo2/data/an_data/VOCdevkit/VOC2007/csv_labels/train.csv', sep=" "), img_path='/home/rliu/TDD-Net/data/', window_size=50, pad_size=50, mask = create_circular_mask(200,200), transforms=None):
        """
        Args:
            df: dataframes of training data
            transform: pytorch transforms for transforms and tensor conversion
        """
        self.data = df
        self.img_path = img_path
        self.transforms = transforms
        self.window_size = window_size
        self.pad_size = pad_size
        self.mask = mask

    def __getitem__(self, index):
        labels = self.data.loc[index]
        single_image_label = int(labels['class']) # float
        x = labels['x']
        y = 1 - labels['y'] # origin of PIL image is top-left
        img_index = labels['image_index']
        img = Image.open(self.img_path + '%06.0f.jpg' % img_index).convert('L')
#         img = torchvision.transforms.functional.resize(img, (300,300), interpolation=2)
        width, height = img.size
        img = ImageOps.expand(img, border=self.pad_size, fill=0)
        xmin = width * x - self.window_size/2 + self.pad_size
        ymin = height * y - self.window_size/2 + self.pad_size
        xmax = width * x + self.window_size/2 + self.pad_size
        ymax = height * y + self.window_size/2 + self.pad_size
        img_resized = img.crop((xmin, ymin, xmax, ymax))
        img_resized = torchvision.transforms.functional.resize(img_resized, (200,200), interpolation=2)
        img_masked = img_resized * self.mask
        img_masked = Image.fromarray(img_masked.astype('uint8'), 'L')
        if self.transforms is not None:
            img_masked = self.transforms(img_masked)
        # Return image and the label
        return (img_masked, single_image_label)

    def __len__(self):
        return len(self.data.index)
        

class defectDataset_convolution(Dataset):
    def __init__(self, image_index = 6501, img_path='/home/rliu/TDD-Net/data/',window_size=45, mask = create_circular_mask(200,200), stride = 2, transforms=None):
        """
        Args:
            image_index: index of image being processed
            window_size: size of sliding window
            transform: pytorch transforms for transforms and tensor conversion
        """
        self.image = Image.open(img_path + '%06.0f.jpg' % image_index).convert('L')
        coord_list = np.empty([0,2],dtype=int)
        for i in np.arange(0,self.image.size[0],stride):
            for j in np.arange(0,self.image.size[1],stride):
                coord_list = np.append(coord_list,[[i,j]],axis = 0);
        self.coords = coord_list
        self.mask = mask
        self.window_size = window_size
        self.transforms = transforms

    def __getitem__(self, index):
        x,y = self.coords[index]
        img_resized = self.image.crop(box=(x - self.window_size/2,y - self.window_size/2, x + self.window_size/2, y + self.window_size/2))
        img_resized = torchvision.transforms.functional.resize(img_resized, (200,200), interpolation=2)
        img_masked = img_resized * self.mask
        img_masked = Image.fromarray(img_masked.astype('uint8'), 'L')
        # Transform image to tensor
        if self.transforms is not None:
            img_masked = self.transforms(img_masked)
        # Return image and the label
        return (img_masked)

    def __len__(self):
        return self.coords.shape[0]