---
title: Notebook Code Overview 
author: Adina Tung, James Froelich
date: 2023-06-03
category: Jekyll
layout: post
cover: https://birdsbeetsbattlestargalactica.github.io/assets/birds_better.gif
---

The following code was adapted from [Joseph Redmon's tutorial][1]. 

Libraries used:
```python
import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
import itertools
import pandas

from pandas import DataFrame
from torchvision.models import resnet18, ResNet18_Weights
```

We used custom datasets in Kaggle to import checkpoints from previous runs.  
```python
### set up directories
prev_cpts = '/kaggle/input/bbbg_cpts/'
checkpoints = '/kaggle/working/
```

### Load source images for training and testing  
Base definition of our data loaders, see experimentation for changes made.  
```python
def get_bird_data(augmentation=0, input_size=128):
    transform_train = transforms.Compose([
        transforms.Resize(input_size),
        transforms.RandomCrop(input_size, padding=8, padding_mode='edge'), # Take 128x128 crops from padded images
        transforms.RandomHorizontalFlip(),    # 50% of time flip image along y-axis
        transforms.ToTensor()
    ])
    
    transform_test = transforms.Compose([
        transforms.Resize(input_size),
        transforms.ToTensor(),
    ])

    data_path = '/kaggle/input/birds23sp/birds/'

    trainset = torchvision.datasets.ImageFolder(root=data_path + 'train', transform=transform_train)
    testset = torchvision.datasets.ImageFolder(root=data_path + 'test', transform=transform_test)
    
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)
    testloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False, num_workers=2)
    
    classes = open(data_path + "names.txt").read().strip().split("\n")
    class_to_idx = trainset.class_to_idx
    idx_to_class = {int(v): int(k) for k, v in class_to_idx.items()}
    idx_to_name = {k: classes[v] for k,v in idx_to_class.items()}
    return {'train': trainloader, 'test': testloader, 'to_class': idx_to_class, 'to_name':idx_to_name}
```

### Training function  
```python
def train(net, dataloader, epochs=1, start_epoch=0, lr=0.01, momentum=0.9, decay=0.0005, 
          verbose=1, print_every=10, state=None, schedule={}, checkpoint_path=None):
    net.to(device)
    net.train()
    losses = []
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=lr, momentum=momentum, weight_decay=decay)

    # Load previous training state
    if state:
        net.load_state_dict(state['net'])
        optimizer.load_state_dict(state['optimizer'])
        start_epoch = state['epoch']
        losses = state['losses']

    # Fast forward lr schedule through already trained epochs
    for epoch in range(start_epoch):
        if epoch in schedule:
            print ("Learning rate: %f"% schedule[epoch])
            for g in optimizer.param_groups:
                g['lr'] = schedule[epoch]

    for epoch in range(start_epoch, epochs):
        sum_loss = 0.0

        # Update learning rate when scheduled
        if epoch in schedule:
            print ("Learning rate: %f"% schedule[epoch])
            for g in optimizer.param_groups:
                g['lr'] = schedule[epoch]

        for i, batch in enumerate(dataloader, 0):
            inputs, labels = batch[0].to(device), batch[1].to(device)

            optimizer.zero_grad()

            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()  # autograd magic, computes all the partial derivatives
            optimizer.step() # takes a step in gradient direction

            losses.append(loss.item())
            sum_loss += loss.item()

            if i % print_every == print_every-1:    # print every 10 mini-batches
                if verbose:
                  print('[%d, %5d] loss: %.3f' % (epoch, i + 1, sum_loss / print_every))
                sum_loss = 0.0
        if checkpoint_path:
            state = {'epoch': epoch+1, 'net': net.state_dict(), 'optimizer': optimizer.state_dict(), 'losses': losses}
            torch.save(state, checkpoint_path + 'checkpoint-%d.pkl'%(epoch+1))
            plt.plot(smooth(state['losses'], 50))
            plt.savefig('checkpoint-%d.png'%(epoch+1))
    return losses
```

### Smooth loss data  
Helper function for smoothing loss data (for generating graphs)  
```python
def smooth(x, size):
  return np.convolve(x, np.ones(size)/size, mode='valid')
```

### Main code  
Train datasets from scratch or starting with previously saved checkpoints.  
```python
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
data = get_bird_data(input_size=256)
resnet = torch.hub.load('pytorch/vision:v0.6.0', 'resnet18', weights=ResNet18_Weights.IMAGENET1K_V1)
resnet.fc = nn.Linear(512, 555)

# selectable between starting at a checkpoint or from scratch.
if (1):
    state = torch.load(prev_cpts + 'checkpoint-15.pkl')
    resnet.load_state_dict(state['net'])
    losses = train(resnet, data['train'], epochs=20, lr=.0001, print_every=10, checkpoint_path=checkpoints, state=state)
else: 
    losses = train(resnet, data['train'], epochs=10, schedule={0:.01, 8:0.001}, lr=.01, print_every=10, checkpoint_path=checkpoints)

```

### Prediction code  
Generates a `.csv` file of our model's predictions on the test dataset.
```python
def predict(net, dataloader, ofname):
    out = open(ofname, 'w')
    out.write("path,class\n")
    net.to(device)
    net.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for i, (images, labels) in enumerate(dataloader, 0):
            if i%100 == 0:
                print(i)
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            fname, _ = dataloader.dataset.samples[i]
            out.write("test/{},{}\n".format(fname.split('/')[-1], data['to_class'][predicted.item()]))
    out.close()


state = torch.load(checkpoints + 'checkpoint-20.pkl')
resnet.load_state_dict(state['net'])
predict(resnet, data['test'], checkpoints + "preds.csv")
```

### Calculate accuracy  
Compares the labelling from our model to the actual labels.
```python
birds_folder = '/kaggle/input/birds23sp/birds/'

def calc_accuracy(file_path):
    df = pandas.read_csv(file_path, header=0)
    df.columns = df.columns.str.removeprefix("text/")
    predictions = [tuple(x) for x in df.itertuples(index=False)]
    
    df = pandas.read_csv(birds_folder + 'labels.csv', header=0)
    actuals = df.set_index('path')['class'].to_dict()
    
    total = 0
    correct = 0
    for i in predictions:
        if (int(actuals.get(i[0], -1)) == i[1]):
            correct += 1
        total += 1
    return correct / total
    
    
acc = calc_accuracy('preds-train2.csv')
print("accuracy = {}".format(acc))
```

[1]: https://colab.research.google.com/drive/1kHo8VT-onDxbtS3FM77VImG35h_K_Lav?usp=sharing
