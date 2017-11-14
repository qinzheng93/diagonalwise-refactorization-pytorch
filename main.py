from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from time import clock

from MobileNet import MobileNet

model = MobileNet()
model.cuda()

print('Model created.')

transform = transforms.Compose([
    transforms.RandomSizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

traindir = '/home/zheng/Datasets/ILSVRC/ILSVRC2012_images_train'
train = datasets.ImageFolder(traindir, transform)

print('Dataset created.')

train_loader = torch.utils.data.DataLoader(train, batch_size=64, shuffle=True, num_workers=4)

criterion = nn.CrossEntropyLoss().cuda()
learning_rate = 0.01
optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), learning_rate, 0.9)

print('Begin Training...')

num_epoch = 20
max_batches = 50
tot_time = 0

# preload data to eliminate data loading time
data = 0
for batch in train_loader:
    data = batch
    break
inputs, labels = data
inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())

# time testing
for _ in range(max_batches + 1):
    t0 = clock()

    optimizer.zero_grad()
    outputs = model(inputs)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()

    t1 = clock()
    if _ > 0:
        tot_time += t1 - t0

    print('{}: {} seconds'.format(_, t1 - t0))

print(tot_time / float(max_batches))
