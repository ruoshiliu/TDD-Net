from dataset import convResDataset
import torch, torchvision
import matplotlib.pyplot as plt
import numpy as np
import os
import torch.nn.functional as F

os.environ["CUDA_VISIBLE_DEVICES"]="1,0"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model_uniform = torch.load('/home/rliu/TDD-Net/models/python/res34-150epo_uniform_07-12-19-hpcc/res34-150epo_uniform_07-12-19_hpcc.model')
model_uniform.eval();
model_uniform = torch.nn.DataParallel(model_uniform)
model_uniform.to(device)

train_set = convResDataset(arr = np.arange(1,11001))
train_loader = torch.utils.data.DataLoader(train_set, batch_size=6, shuffle=False, num_workers=6)

index = 1

with torch.no_grad():
    for image in train_loader:
        model_uniform.train(False)
        image = image.to(device)
        output = model_uniform(image)
        for i in range(output.shape[0]):
            conf_map = output[i]
            conf_map = conf_map.reshape(3,128,128)
            confidences = F.softmax(conf_map, dim=0)
            confidences = confidences.cpu().detach().numpy()
            np.save('/home/rliu/TDD-Net/convResNet/input/uniform/%0.6d.npy' % index, confidences)
            print('%0.6d processed' % index)
            index += 1