---
layout: home
title: Birds, Beets, Battlestar Galactica
cover: https://birdsbeetsbattlestargalactica.github.io/assets/birds_better.gif
permalink: /
---

Welcome to the project site for Adina & James' bird classifier submission. This
gitbook serves as documentation for our participation in the Spring 2023 CSE 
455 bird classification competition hosted on [Kaggle][1].

### Summary
  [video here]

  <!-- <iframe width="420" height="315"
    src="https://www.youtube.com/watch?v=WaaANll8h18">
  </iframe> -->
  <iframe width="917" height="516" 
    src="https://www.youtube.com/embed/WaaANll8h18" title="The Office US - Jim vs Dwight - Jim Impersonates Dwight" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" allowfullscreen>
  </iframe>
  


### Dataset
  We were provided a training dataset consisting of 30 thousand bird images 
  divided into 535 separate classes. We were provided names for each class and 
  a .csv file mapping the training images to their classes.

  We were also provided a test dataset consisting of 10 thousand images. We 
  were not provided labels for the test dataset.

### Code
  We were provided starter code by our professor Joseph Redmon. You can see
  our complete code [here][2]. The changes we made will be highlighted in
  the experiment pages.  

### Techniques
  We trained two versions of RESNET as our model. RESNET
  comes pre-trained on IMAGENET but we fine tuned the model for
  classifying birds. Our process consisted of essentially of trial
  and error, tweaking the hyperparameters and image transforms on the
  training set. Once we stopped seeing an improvement in RESNET18,
  we switched to RESNET34.

<!-- This summary should mention the problem setup, data used, techniques, etc. 
It should include a description of which components were from preexisting work (
  i.e. code from github) and which components were implemented for the project 
  (i.e. new code, gathered dataset, etc). -->




[1]: https://kaggle.com/competitions/birds23sp
[2]: https://birdsbeetsbattlestargalactica.github.io/jekyll/2023-06-03-NotebookCodeOverview.html