---
title: Experiments with RESNET34
author: Adina Tung, James Froelich
date: 2023-06-03
category: Jekyll
layout: post
cover: https://birdsbeetsbattlestargalactica.github.io/assets/birds_better.gif
---

After many trials with resnet18, we noticed the loss plateaued at around 0.5 even with increased number of epochs, so we went ahead and experimented with resnet34. We saw better results from the training with more layers and more epochs in the training model. Additionally, more image augmentation seemed to be correlated to higher loss based on our model with resnet18, which lead us to limit the numbers of augmentation used and also tune down the probability of their occurrence.

<div class="table-wrapper" markdown="block">

|epoch|schedule|horizontal/vertical flip (p)|random color jitter (p)|normalize|invert (p)|final loss|
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
|25|{0:.01, 8:.001, 15:0.0001}|0.2/0.2|-|-|-|0.194| <!--- 2394.png v'lower probability...' --->
|32|{0:.01, 12:.001, 19:0.0001}|0.2/0.2|-|-|-|still running|  <!--  v9 -->

</div>

#### Plots:  

###### epoch = 25, schedule = {0:.01, 8:.001, 15:0.0001}, horizontal flip (p) = 0.2, vertical flip (p) = 0.2, color jitter (p) = 0 normalize (mean, std) = (0, 0), invertion (p) = 0, final loss = 0.194  

![Resnet34 with 25 epochs, 0.2 horizontal and vertical flip](https://birdsbeetsbattlestargalactica.github.io/assets/graphs/2394.png)

```python
transform_train = transforms.Compose([
        transforms.Resize(input_size),
        transforms.RandomCrop(input_size, padding=8, padding_mode='edge'), # Take 256x256 crops from padded images
        transforms.RandomHorizontalFlip(),    # 50% of time flip image along y-axis
        transforms.ToTensor(),
    ])

```



[1]: https://birdsbeetsbattlestargalactica.github.io/assets/graphs/resnet34_epoch25.png
