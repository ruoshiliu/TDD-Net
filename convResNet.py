from dataset_crn import convResDataset
import torch, torchvision
import matplotlib.pyplot as plt
import numpy as np

os.environ["CUDA_VISIBLE_DEVICES"]="2"

model_uniform = torch.load('/home/rliu/TDD-Net/models/python/res34-150epo_uniform_07-12-19-hpcc/res34-150epo_uniform_07-12-19_hpcc.model')
model_uniform.eval();
model_uniform.cuda();

train_set = convResDataset(arr = np.arange(1,11001))
train_loader = torch.utils.data.DataLoader(train_set, batch_size=4, shuffle=False, num_workers=1)

index = 1

for image in train_loader:
    image = image.cuda()
    output = model_uniform(image)
    conf_map = output.reshape(3,128,128)
    confidences = F.softmax(conf_map, dim=0)
    confidences = confidences.cpu().detach().numpy()
    np.save('/home/rliu/TDD-Net/convResNet/uniform/%0.6d.npy' % index, confidences)
    index += 1
    print(index + ' processed')