---
title: Birds, Beets, Battlestar Galactica
author: Adina Tung, James Froelich
date: 2023-06-03
category: Jekyll
layout: post
cover: https://birdsbeetsbattlestargalactica.github.io/jekyll-gitbook/assets/birds.gif
---


```yaml

transform_train = transforms.Compose([
        transforms.Resize(input_size),
        transforms.RandomCrop(input_size, padding=8, padding_mode='edge'), # Take 256x256 crops from padded images
        transforms.RandomHorizontalFlip(),    # 50% of time flip image along y-axis
        transforms.ToTensor(),
    ])

```



![Loss over 25 epochs with resnet34](1)


[1]: https://birdsbeetsbattlestargalactica.github.io/jekyll-gitbook/assets/resnet34_epoch25.png
