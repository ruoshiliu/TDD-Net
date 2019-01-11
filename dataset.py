import pandas as pd
import numpy as np
from PIL import Image
from PIL import ImageOps
import PIL
import torch, torchvision
from torchvision import transforms, datasets
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import math
import random

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
    mask = mask.astype(int)
    return mask

mask = create_circular_mask(200,200)


class defectDataset_csv(Dataset):
    def __init__(self, csv_path='/home/rliu/yolo2/v2_pytorch_yolo2/data/an_data/VOCdevkit/VOC2007/csv_labels/train.csv', img_path='/home/rliu/yolo2/v2_pytorch_yolo2/data/an_data/VOCdevkit/VOC2007/JPEGImages/', window_size=50, pad_size=50, mask = create_circular_mask(200,200), transforms=None):
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
        img_masked = img_resized * mask
        img_masked = Image.fromarray(img_masked.astype('uint8'), 'L')
#         img_resized = img_resized * mask
#         img_resized = img_resized*mask
#         if self.mask is not None:
#             img_resized[~mask] = 0
        # Transform image to tensor
        if self.transforms is not None:
            img_masked = self.transforms(img_masked)
        # Return image and the label
        return (img_masked, single_image_label)

    def __len__(self):
        return len(self.data.index)
        

if __name__ == "__main__":
    transformations = transforms.Compose([transforms.ToTensor()])
#     defect_from_csv = \
#         defectDataset('../data/mnist_in_csv.csv', 28, 28, transformations)

class defectDataset_df(Dataset):
    def __init__(self, df = pd.read_csv('/home/rliu/yolo2/v2_pytorch_yolo2/data/an_data/VOCdevkit/VOC2007/csv_labels/train.csv', sep=" "), img_path='/home/rliu/yolo2/v2_pytorch_yolo2/data/an_data/VOCdevkit/VOC2007/JPEGImages/', window_size=50, pad_size=50, mask = create_circular_mask(200,200), transforms=None):
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
        img_masked = img_resized * mask
        img_masked = Image.fromarray(img_masked.astype('uint8'), 'L')
#         img_resized = img_resized * mask
#         img_resized = img_resized*mask
#         if self.mask is not None:
#             img_resized[~mask] = 0
        # Transform image to tensor
        if self.transforms is not None:
            img_masked = self.transforms(img_masked)
        # Return image and the label
        return (img_masked, single_image_label)

    def __len__(self):
        return len(self.data.index)
        

if __name__ == "__main__":
    transformations = transforms.Compose([transforms.ToTensor()])
#     defect_from_csv = \
#         defectDataset('../data/mnist_in_csv.csv', 28, 28, transformations)


def sample_point_circular(circle_min = 0.02, circle_max = 0.07):
    # random angle
    alpha = 2 * math.pi * random.random()
    # random radius
    r = (circle_max - circle_min) * random.random() + circle_min
    # calculating coordinates
    x = r * math.cos(alpha)
    y = r * math.sin(alpha)
    return x,y

def split_and_sample(
    df_labels = pd.read_csv('/home/rliu/yolo2/v2_pytorch_yolo2/data/an_data/VOCdevkit/VOC2007/csv_labels/train.csv', sep=" "),
    df_yolo = pd.read_csv('/home/rliu/github/defect_classifier/yolo2_dm/results/train_yolo.csv', sep=' ')
    , n_samples = 1000, non_pos_ratio = 1, non_inner_circle = 0.02, non_outer_circle = 0.07, method = 'uniform'):
    
    df_pos = df_labels[df_labels['class'] == 0]
    df_neg = df_labels[df_labels['class'] == 1]
    df_pos_o = df_labels[df_labels['class'] == 2]
    df_nuc = df_labels[df_labels['class'] == 3]
    frames = [df_pos.sample(n=n_samples), df_neg.sample(n=n_samples), df_pos_o.sample(n=n_samples), df_nuc.sample(n=n_samples)]
    df_labels_samples = pd.concat(frames)
    # print(df_labels_non.head(10))
    if method=='hard':
        df_labels_non = df_labels_samples.sample(n=n_samples*non_pos_ratio)
        for index, row in df_labels_non.iterrows():
            min_dis = 0  # make sure go in for loop
            while min_dis < non_inner_circle:
                dx,dy = sample_point_circular(non_inner_circle, non_outer_circle)
                new_point = [row.x+dx, row.y+dy]
                df_image = df_labels[df_labels['image_index'] == row['image_index']] # retrive all points in the image
                min_dis = 1
                for index_im, row_im in df_image.iterrows(): # check distance from new_point to each point, if smaller than threshold than thow away
                    dis = math.sqrt(math.pow(new_point[0]-row.x,2)+math.pow(new_point[1]-row.y,2))
                    if dis < min_dis:
                        min_dis = dis
            df_labels_non.at[index,'x'] = new_point[0]
            df_labels_non.at[index,'y'] = new_point[1]
            df_labels_non.at[index,'class'] = 4
    elif method=='uniform':
        df_labels_non = df_labels_samples.sample(n=n_samples*non_pos_ratio)
        for index, row in df_labels_non.iterrows():
            min_dis = 0
            while min_dis < non_inner_circle:
                new_point = [random.random(),random.random()]
                df_image = df_labels[df_labels['image_index'] == row['image_index']]
                min_dis = 1
                for index_im, row_im in df_image.iterrows():
                    dis = math.sqrt(math.pow(new_point[0]-row.x,2)+math.pow(new_point[1]-row.y,2))
                    if dis < min_dis:
                        min_dis = dis
            df_labels_non.at[index,'x'] = new_point[0]
            df_labels_non.at[index,'y'] = new_point[1]
            df_labels_non.at[index,'class'] = 4

    elif method=='yolo':
        columns = ['image_index','class', 'x', 'y']
        df_labels_non = pd.DataFrame(index=np.arange(2*n_samples*non_pos_ratio), columns=columns)
        df_labels_non['class'] = 4 # assign class number
        df_yolo_samples = df_yolo.sample(n=2*n_samples*non_pos_ratio)
        df_yolo_samples = df_yolo_samples.reset_index() #reset index
        df_yolo_samples = df_yolo_samples.drop(columns='index')
        for index, row in df_yolo_samples.iterrows():
            pts = sample_rec(row) # sample one point from a bounding box
            df_labels_non.at[index,'x'] = pts[0]
            df_labels_non.at[index,'y'] = pts[1]
            df_labels_non.at[index,'image_index'] = row['image_index']
        df_labels_non = df_labels_non.reset_index()
        df_labels_non = df_labels_non.drop(columns='index')
        for index_non, row_non in df_labels_non.iterrows(): # check if points are too close to a defect
            df_image = df_labels[df_labels['image_index'] == row_non['image_index']] # extract all defect labels
            min_dis = 1
            for index_im, row_im in df_image.iterrows():
                dis = math.sqrt(math.pow(row_non.x-row_im.x,2)+math.pow(row_non.x-row_im.y,2))
                if dis < min_dis:
                    min_dis = dis
    #         print(min_dis)
            if min_dis < non_inner_circle:
                df_labels_non = df_labels_non.drop(index_non)
        df_labels_non = df_labels_non.sample(n=n_samples*non_pos_ratio)
    df_labels_samples = df_labels_samples.append(df_labels_non)
    df_labels_samples = df_labels_samples.reset_index()
    df_labels_samples = df_labels_samples.drop(['index'],axis = 1)
    return df_labels_samples

def sample_rec(row):
    width = row.x2 - row.x1
    height = row.y2 - row.y1
    pts = [random.random(),random.random()]
    pts[0] = pts[0] * width + row.x1 # sample points from bounding boxes
    pts[1] = pts[1] * height + row.y1
    return pts