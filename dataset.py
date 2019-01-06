import pandas as pd
import numpy as np
from PIL import Image
from PIL import ImageOps
import PIL
import torch, torchvision
from torchvision import transforms, datasets
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

window_size = 50
pad_size= window_size

def create_circular_mask(h, w, center=None, radius=None):

    if center is None: # use the middle of the image
        center = [int(w/2), int(h/2)]
    if radius is None: # use the smallest distance between the center and image walls
        radius = min(center[0], center[1], w-center[0], h-center[1])

    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0])**2 + (Y-center[1])**2)

    mask = dist_from_center <= radius
    mask = mask.astype(float)
    return mask
mask = create_circular_mask(window_size, window_size)

class defectDataset(Dataset):
    def __init__(self, csv_path='/home/rliu/yolo2/v2_pytorch_yolo2/data/an_data/VOCdevkit/VOC2007/csv_labels/train.csv', img_path='/home/rliu/yolo2/v2_pytorch_yolo2/data/an_data/VOCdevkit/VOC2007/JPEGImages/', window_size=50, pad_size=50, mask = mask, transforms=None):
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
#         img_resized = img_resized * mask
#         img_resized = img_resized*mask
#         if self.mask is not None:
#             img_resized[~mask] = 0
        # Transform image to tensor
        if self.transforms is not None:
            img_resized = self.transforms(img_resized)
        # Return image and the label
        return (img_resized, single_image_label)

    def __len__(self):
        return len(self.data.index)
        

if __name__ == "__main__":
    transformations = transforms.Compose([transforms.ToTensor()])
#     defect_from_csv = \
#         defectDataset('../data/mnist_in_csv.csv', 28, 28, transformations)