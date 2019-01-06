{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 166,
   "metadata": {},
   "outputs": [],
   "source": [
    "import pandas as pd\n",
    "import numpy as np\n",
    "from PIL import Image\n",
    "from PIL import ImageOps\n",
    "import PIL\n",
    "import torch, torchvision\n",
    "from torchvision import transforms, datasets\n",
    "from torch.utils.data import Dataset, DataLoader\n",
    "import matplotlib.pyplot as plt"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 167,
   "metadata": {},
   "outputs": [],
   "source": [
    "img_path = '/home/rliu/yolo2/v2_pytorch_yolo2/data/an_data/VOCdevkit/VOC2007/JPEGImages/'\n",
    "csv_path = '/home/rliu/yolo2/v2_pytorch_yolo2/data/an_data/VOCdevkit/VOC2007/csv_labels/train.csv'"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 168,
   "metadata": {},
   "outputs": [],
   "source": [
    "df_defect = pd.read_csv('/home/rliu/yolo2/v2_pytorch_yolo2/data/an_data/VOCdevkit/VOC2007/csv_labels/labels.csv', sep=\" \")\n",
    "df_train = pd.read_csv('/home/rliu/yolo2/v2_pytorch_yolo2/data/an_data/VOCdevkit/VOC2007/csv_labels/train.csv', sep=\" \")\n",
    "df_test = pd.read_csv('/home/rliu/yolo2/v2_pytorch_yolo2/data/an_data/VOCdevkit/VOC2007/csv_labels/test.csv', sep=\" \")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 169,
   "metadata": {},
   "outputs": [],
   "source": [
    "window_size = 50\n",
    "pad_size= window_size"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 170,
   "metadata": {},
   "outputs": [],
   "source": [
    "def create_circular_mask(h, w, center=None, radius=None):\n",
    "\n",
    "    if center is None: # use the middle of the image\n",
    "        center = [int(w/2), int(h/2)]\n",
    "    if radius is None: # use the smallest distance between the center and image walls\n",
    "        radius = min(center[0], center[1], w-center[0], h-center[1])\n",
    "\n",
    "    Y, X = np.ogrid[:h, :w]\n",
    "    dist_from_center = np.sqrt((X - center[0])**2 + (Y-center[1])**2)\n",
    "\n",
    "    mask = dist_from_center <= radius\n",
    "    mask = mask.astype(float)\n",
    "    return mask\n",
    "mask = create_circular_mask(window_size, window_size)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 171,
   "metadata": {},
   "outputs": [],
   "source": [
    "class defectDataset(Dataset):\n",
    "    def __init__(self, csv_path='/home/rliu/yolo2/v2_pytorch_yolo2/data/an_data/VOCdevkit/VOC2007/csv_labels/train.csv', img_path='/home/rliu/yolo2/v2_pytorch_yolo2/data/an_data/VOCdevkit/VOC2007/JPEGImages/', window_size=50, pad_size=50, mask = mask, transforms=None):\n",
    "        \"\"\"\n",
    "        Args:\n",
    "            csv_path (string): path to csv file\n",
    "            transform: pytorch transforms for transforms and tensor conversion\n",
    "        \"\"\"\n",
    "        self.data = pd.read_csv(csv_path, sep=\" \")\n",
    "        self.img_path = img_path\n",
    "        self.transforms = transforms\n",
    "        self.window_size = window_size\n",
    "        self.pad_size = pad_size\n",
    "        self.mask = mask\n",
    "\n",
    "    def __getitem__(self, index):\n",
    "        labels = self.data.loc[index]\n",
    "        single_image_label = int(labels['class']) # float\n",
    "        x = labels['x']\n",
    "        y = 1 - labels['y'] # origin of PIL image is top-left\n",
    "        img_index = labels['image_index']\n",
    "        img = Image.open(self.img_path + '%06.0f.jpg' % img_index)\n",
    "        img = img.convert('L')\n",
    "        img = torchvision.transforms.functional.resize(img, (300,300), interpolation=2)\n",
    "        width, height = img.size\n",
    "        img = ImageOps.expand(img, border=self.pad_size, fill=0)\n",
    "        xmin = width * x - self.window_size/2 + self.pad_size\n",
    "        ymin = height * y - self.window_size/2 + self.pad_size\n",
    "        xmax = width * x + self.window_size/2 + self.pad_size\n",
    "        ymax = height * y + self.window_size/2 + self.pad_size\n",
    "        img_resized = img.crop((xmin, ymin, xmax, ymax))\n",
    "#         img_resized = img_resized * mask\n",
    "#         img_resized = img_resized*mask\n",
    "#         if self.mask is not None:\n",
    "#             img_resized[~mask] = 0\n",
    "        # Transform image to tensor\n",
    "        if self.transforms is not None:\n",
    "            img_resized = self.transforms(img_resized)\n",
    "        # Return image and the label\n",
    "        return (img_resized, single_image_label)\n",
    "\n",
    "    def __len__(self):\n",
    "        return len(self.data.index)\n",
    "        \n",
    "\n",
    "if __name__ == \"__main__\":\n",
    "    transformations = transforms.Compose([transforms.ToTensor()])\n",
    "#     defect_from_csv = \\\n",
    "#         defectDataset('../data/mnist_in_csv.csv', 28, 28, transformations)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 172,
   "metadata": {},
   "outputs": [],
   "source": [
    "data_transform = transforms.Compose([\n",
    "        transforms.RandomResizedCrop(200, scale=(1, 1), ratio=(1, 1)),\n",
    "#         transforms.RandomHorizontalFlip(),\n",
    "#         transforms.RandomRotation((-90,90)),\n",
    "        transforms.ToTensor(),\n",
    "        transforms.Normalize(mean=[0.0040],\n",
    "                             std=[0.9908])\n",
    "    ])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 173,
   "metadata": {},
   "outputs": [],
   "source": [
    "defect_dataset = defectDataset(window_size = window_size, transforms=data_transform)\n",
    "dataset_loader = torch.utils.data.DataLoader(defect_dataset,\n",
    "                                             batch_size=4, shuffle=True,\n",
    "                                             num_workers=4)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 174,
   "metadata": {},
   "outputs": [],
   "source": [
    "def imshow(img):\n",
    "    img = img * 0.9908 + 0.0040     # unnormalize\n",
    "    npimg = img.numpy()\n",
    "    plt.imshow(np.transpose(npimg, (1, 2, 0)))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "GroundTruth:  pos_o   pos   nuc pos_o\n",
      "tensor([2, 0, 3, 2])\n"
     ]
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "dataiter = iter(dataset_loader)\n",
    "images, labels = dataiter.next()\n",
    "imshow(torchvision.utils.make_grid(images))\n",
    "classes = [\"pos\",\"neg\",\"pos_o\",\"nuc\",\"non\"]\n",
    "print('GroundTruth: ', ' '.join('%5s' % classes[labels[j]] for j in range(4)))\n",
    "print(labels)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 20,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "512.0\n",
      "1024.0\n",
      "1536.0\n",
      "2048.0\n",
      "2560.0\n",
      "3072.0\n",
      "3584.0\n",
      "4096.0\n",
      "4608.0\n",
      "5120.0\n",
      "5632.0\n",
      "6144.0\n",
      "6656.0\n",
      "7168.0\n",
      "7680.0\n",
      "8192.0\n",
      "8704.0\n",
      "9216.0\n",
      "9728.0\n",
      "10240.0\n",
      "10752.0\n",
      "11264.0\n",
      "11776.0\n",
      "12288.0\n",
      "12800.0\n",
      "13312.0\n",
      "13824.0\n",
      "14336.0\n",
      "14848.0\n",
      "15360.0\n",
      "15872.0\n",
      "16384.0\n",
      "16896.0\n",
      "17408.0\n",
      "17920.0\n",
      "18432.0\n",
      "18944.0\n",
      "19456.0\n",
      "19968.0\n",
      "20480.0\n",
      "20992.0\n",
      "21504.0\n",
      "22016.0\n",
      "22528.0\n",
      "23040.0\n",
      "23552.0\n",
      "24064.0\n",
      "24576.0\n",
      "25088.0\n",
      "25600.0\n",
      "26112.0\n",
      "26624.0\n",
      "27136.0\n",
      "27648.0\n",
      "28160.0\n",
      "28672.0\n",
      "29184.0\n",
      "29696.0\n",
      "30208.0\n",
      "30430.0\n"
     ]
    }
   ],
   "source": [
    "defect_dataset = defectDataset(window_size = 50, transforms=data_transform)\n",
    "dataset_loader = torch.utils.data.DataLoader(defect_dataset,\n",
    "                                             batch_size=512, shuffle=False,\n",
    "                                             num_workers=4)\n",
    "\n",
    "\n",
    "mean = 0.\n",
    "std = 0.\n",
    "nb_samples = 0.\n",
    "for data, labels in dataset_loader:\n",
    "    batch_samples = data.size(0)\n",
    "    data = data.view(batch_samples, data.size(1), -1)\n",
    "    mean += data.mean(2).sum(0)\n",
    "    std += data.std(2).sum(0)\n",
    "    nb_samples += batch_samples\n",
    "    print(nb_samples)\n",
    "mean /= nb_samples\n",
    "std /= nb_samples"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "mean, std"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 32,
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": 38,
   "metadata": {},
   "outputs": [
    {
     "ename": "NameError",
     "evalue": "name 'im_crop' is not defined",
     "output_type": "error",
     "traceback": [
      "\u001b[0;31m---------------------------------------------------------------------------\u001b[0m",
      "\u001b[0;31mNameError\u001b[0m                                 Traceback (most recent call last)",
      "\u001b[0;32m<ipython-input-38-2365a98a46bf>\u001b[0m in \u001b[0;36m<module>\u001b[0;34m\u001b[0m\n\u001b[0;32m----> 1\u001b[0;31m \u001b[0mim_crop\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m",
      "\u001b[0;31mNameError\u001b[0m: name 'im_crop' is not defined"
     ]
    }
   ],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": 18,
   "metadata": {},
   "outputs": [],
   "source": [
    "A = create_circular_mask(200,200)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 20,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([[False, False, False, ..., False, False, False],\n",
       "       [False, False, False, ..., False, False, False],\n",
       "       [False, False, False, ..., False, False, False],\n",
       "       ...,\n",
       "       [False, False, False, ..., False, False, False],\n",
       "       [False, False, False, ..., False, False, False],\n",
       "       [False, False, False, ..., False, False, False]])"
      ]
     },
     "execution_count": 20,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "mask"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.6.5"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}
