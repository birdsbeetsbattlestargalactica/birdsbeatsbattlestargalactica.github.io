---
title: Notebook Code Overview 
author: Adina Tung, James Froelich
date: 2023-06-03
category: Jekyll
layout: post
---

The following code was adapted from [Joseph Redmon's tutorial][1]. 

Libraries used:
```
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
from torchvision.models import resnet18, ResNet18_Weights
```

We used custom datasets in Kaggle to import checkpoints from previous runs
```
### set up directories
prev_cpts = '/kaggle/input/bbbg_cpts/'
checkpoints = '/kaggle/working/
```




[1]: https://colab.research.google.com/drive/1kHo8VT-onDxbtS3FM77VImG35h_K_Lav?usp=sharing