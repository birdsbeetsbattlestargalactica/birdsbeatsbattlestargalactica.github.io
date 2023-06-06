---
title: Future Work
author: Adina Tung, James Froelich
date: 2023-06-03
category: Jekyll
layout: post
cover: https://birdsbeetsbattlestargalactica.github.io/assets/birds_better.gif
---

Given the time and hardware constraint, we didn't get to try out other ideas that could potentially give us a better-performing model. Below are the list of adjustment to our current model that we think would improve the accuracy of our model on a more general test dataset.

- Make a composite train set from augmenting images of the original dataset.
- Try averaging the weights of two models, one trained on the base train set and one trained on a very augmented data set. 
- Try RESNET50 or maybe even RESNET101 (on colab pro).
- Try VGG16 model.
- Hire a team of 50+ professional birdwatchers to classify the birds for us.