import pandas as pd
import numpy as np
from PIL import Image
from PIL import ImageOps
import PIL
import torch, torchvision
from torchvision import transforms, datasets
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from dataset import *
import random
import math
import seaborn
from  matplotlib import pyplot
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable
import time
from torch.optim import lr_scheduler

input_size = 10       # The image size = 28 x 28 = 784
hidden_size = 100      # The number of nodes at the hidden layer
num_classes = 5       # The number of output classes. In this case, from 0 to 9
num_epochs = 40         # The number of times entire dataset is trained
batch_size = 256       # The size of input data took for one iteration
learning_rate = 0.005  # The speed of convergence
non_pos_ratio = 10
weight_decay=5e-4

train_transform = transforms.Compose([
        transforms.RandomResizedCrop(200, scale=(1, 1), ratio=(1, 1)),
        transforms.RandomRotation((-90,90)),
        torchvision.transforms.RandomVerticalFlip(p=0.5),
        torchvision.transforms.RandomHorizontalFlip(p=0.5),
#         torchvision.transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0, hue=0),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.3019],
                             std=[0.1909])
    ])
test_transform = transforms.Compose([
        transforms.RandomResizedCrop(200, scale=(1, 1), ratio=(1, 1)),
#         transforms.RandomRotation((-90,90)),
#         torchvision.transforms.RandomVerticalFlip(p=0.5),
#         torchvision.transforms.RandomHorizontalFlip(p=0.5),
#         torchvision.transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0, hue=0),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.3019],
                             std=[0.1909])
    ])


class Net(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(Net, self).__init__()                    # Inherited from the parent class nn.Module
        self.fc1 = nn.Linear(input_size, hidden_size)  # 1st Full-Connected Layer: 10 (input data) -> 500 (hidden node)
        self.relu = nn.ReLU()                          # Non-Linear ReLU Layer: max(0,x)
        self.fc2 = nn.Linear(hidden_size, num_classes) # 2nd Full-Connected Layer: 500 (hidden node) -> 5 (output class)
#         self.relu = nn.ReLU()                          # Non-Linear ReLU Layer: max(0,x)
#         self.fc3 = nn.Linear(hidden_size, num_classes) # 3rd Full-Connected Layer: 500 (hidden node) -> 5 (output class)
    
    def forward(self, x):                              # Forward pass: stacking each layer together
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
#         out = self.relu(out)
#         out = self.fc3(out)
        return out


net = Net(input_size, hidden_size, num_classes)


use_gpu = torch.cuda.is_available()

if use_gpu:
    print("GPU in use")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

classes = ["pos","neg","pos_o","nuc","non"]
num_of_classes = len(classes)

model_uniform = torch.load('/home/rliu/defect_classifier/models/python/res34_600epo_uniform_01-07-18.model')
model_uniform.eval()
model_hard = torch.load('/home/rliu/defect_classifier/models/python/res34_600epo_hard_01-07-18.model')
model_hard.eval()


if use_gpu:
#     model_uniform = torch.nn.DataParallel(model_uniform)
    model_uniform.to(device)
#     model_hard = torch.nn.DataParallel(model_hard)
    model_hard.to(device)
    net.to(device)
    
weights = [1.0, 1.0, 1.0, 1.0, 1.0/non_pos_ratio]
class_weights = torch.FloatTensor(weights).to(device)
criterion = nn.CrossEntropyLoss(weight = class_weights)
optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate, weight_decay=weight_decay)
scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.2)

since = time.time()
best_model_wts = net.state_dict()
best_acc = 0.0
for epoch in range(num_epochs):
    trainset = defectDataset_df(df = split_and_sample(method = 'yolo',n_samples = 1995, non_pos_ratio=non_pos_ratio), window_size = window_size,
                                             transforms=train_transform)
    trainloader = torch.utils.data.DataLoader(trainset,
                                                 batch_size=batch_size, shuffle=True,
                                                 num_workers=8, drop_last=True)
    print("trainloader ready!")

    testset = defectDataset_df(df = split_and_sample(df_labels = pd.read_csv('/home/rliu/yolo2/v2_pytorch_yolo2/data/an_data/VOCdevkit/VOC2007/csv_labels/test.csv', sep=" "),
                                            method = 'yolo',n_samples = 800), window_size = window_size, transforms=test_transform)
    testloader = torch.utils.data.DataLoader(testset,
                                                 batch_size=batch_size, shuffle=True,
                                                 num_workers=8)
    print("testloader ready!")
    print('Epoch {}/{}'.format(epoch, num_epochs - 1))
    print('-' * 10)
    scheduler.step()
    model_uniform.train(False)
    model_hard.train(False)
    net.train(True)
    running_loss = 0.0
    running_corrects = 0
    for data in trainloader:
        
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        with torch.no_grad():
            outputs_uniform = model_uniform(inputs)
            outputs_hard = model_hard(inputs)
        outputs_in = torch.cat((outputs_uniform, outputs_hard), dim=1)
        outputs_out = net(outputs_in)
        _, preds = torch.max(outputs_out.data, 1)
        loss = criterion(outputs_out, labels)

        optimizer.zero_grad()                             # Intialize the hidden weight to all zeros
        loss.backward()                                   # Backward pass: compute the weight
        optimizer.step()                                  # Optimizer: update the weights of hidden nodes

#         if (i+1) % 100 == 0:                              # Logging
#             print('Epoch [%d/%d], Step [%d/%d], Loss: %.4f'
#                  %(epoch+1, num_epochs, i+1, len(train_dataset)//batch_size, loss.item()))

        
        # statistics
        iter_loss = loss.item()
        correct = torch.sum(preds == labels.data).item()
        batch_accuracy = correct / batch_size
        running_loss += loss.item()
        running_corrects_tensor = torch.sum(preds == labels.data)
        running_corrects += running_corrects_tensor.item()        
        epoch_loss = running_loss / len(trainset)
        epoch_acc = running_corrects / len(trainset)

        print('{} Loss: {:.4f} Acc: {:.4f} batch_loss: {:.4f} correct: {:d} batch_accuracy: {:.4f}'.format(
            "train", epoch_loss, epoch_acc, iter_loss, correct, batch_accuracy))
        
    correct = 0
    total = 0
    class_correct = list(0. for i in range(5))
    class_total = list(0. for i in range(5))    
    with torch.no_grad():
        for data in testloader:
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            outputs_uniform = model_uniform(inputs)
            outputs_hard = model_hard(inputs)
            outputs_in = torch.cat((outputs_uniform, outputs_hard), dim=1)
            outputs_out = net(outputs_in)
            _, predicted = torch.max(outputs_out.data, 1)
            c = (predicted == labels).squeeze()
            for i in range(batch_size):
                if len(labels) == batch_size:
                    label = labels[i]
                    class_correct[label] += c[i].item()
                    class_total[label] += 1
                    correct += c[i].item()
                    total += 1
#             if len(labels) == batch_size:
#                 total += labels.size(0)
#                 correct += (predicted == labels).sum().item()
    #         print(predicted)
    #         print(labels)
    #       print('processed: %d' % total)
    #       print('correct: %d' % correct)
        print('Accuracy of the network on the test images: %.5f %%' % (100 * correct / total))
        for i in range(5):
            print('Accuracy of %5s : %2d %%' % (classes[i], 100 * class_correct[i] / class_total[i]))
        print('class total: ',class_total)
        print('class correct: ',class_correct)
        print('total: ', total)
        print('correct: ', correct)
time_elapsed = time.time() - since
print('Training complete in {:.0f}m {:.0f}s'.format(
    time_elapsed // 60, time_elapsed % 60))
print('Best val Acc: {:4f}'.format(best_acc))

# load best model weights
net.load_state_dict(best_model_wts)

output_path = '/home/rliu/defect_classifier/models/python/FNN/2x100_28epo_yolo_01-13-18.model'
torch.save(net, output_path)

