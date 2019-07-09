import pandas as pd
import numpy as np
import torch, torchvision
from torch.utils.data import Dataset, DataLoader
from dataset import defectDataset_df, create_circular_mask, split_and_sample
import torch.nn as nn
from torch.autograd import Variable
from torchvision import datasets, models, transforms
import time
import os

def train_model(model, criterion, optimizer, scheduler, transform, train_num, test_num, non_pos_ratio, window_size, batch_size, device, classes, num_epochs=500, method = 'uniform', use_gpu = True):
    since = time.time()

    best_model_wts = model.state_dict()
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training phase
        scheduler.step()
        model.train(True)  # Set model to training mode
        running_loss = 0.0
        running_corrects = 0

        trainset = defectDataset_df(df = split_and_sample(df_labels = pd.read_csv('/home/rliu/yolo2/v2_pytorch_yolo2/data/an_data/VOCdevkit/VOC2007/csv_labels/train.csv', sep=" "),
                                                          method = method, n_samples = train_num, non_pos_ratio = non_pos_ratio), window_size = window_size, transforms=transform)
        trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=16, drop_last=True)
        print("trainloader ready!")

        testset = defectDataset_df(df = split_and_sample(df_labels = pd.read_csv('/home/rliu/yolo2/v2_pytorch_yolo2/data/an_data/VOCdevkit/VOC2007/csv_labels/test.csv', sep=" "),
                                                              method = method, n_samples = test_num), window_size = window_size, transforms=transform)
        testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=16)
        print("testloader ready!")
        # Iterate over data.
        for data in trainloader:
            # get the inputs
            inputs, labels = data
            # wrap them in Variable
            if use_gpu:
                inputs, labels = inputs.to(device), labels.to(device)
            else:
                inputs, labels = Variable(inputs), Variable(labels)

            # zero the parameter gradients
            optimizer.zero_grad()
            m = nn.LogSoftmax(dim=0)
            outputs = model(inputs)
            _, preds = torch.max(outputs.data, 1)
            loss = criterion(m(outputs), labels)

            loss.backward()
            optimizer.step()

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
            model.train(False)
            for data in testloader:
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                c = (predicted == labels).squeeze()
                for i in range(batch_size):
                    if len(labels) == batch_size:
                        label = labels[i]
                        class_correct[label] += c[i].item()
                        class_total[label] += 1
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            print('Accuracy of the network on the test images: %.5f %%' % (100 * correct / total))
            for i in range(5):
                print('Accuracy of %5s : %2d %%' % (classes[i], 100 * class_correct[i] / class_total[i]))

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model