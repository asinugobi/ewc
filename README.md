# EWC Research Directory

This project focuses on transfer learning. More specifically, we would like to explore the effects of applying an elastic weight penalty to a subset of layers in a convolutional neural network (CNN), otherwise known as layer-wise elastic weight consolidation. We hypothesize that such a technique may provide a boost in forward-transfer performance, while also mitigating the effects of catastrophic interference. 

## Transfer between Permuted MNIST 

This module focuses on transfer learning within MNIST. Please see the `sequntial` directory for more details. 

## Transfer from ImageNet to CIFAR  

This module focuses on transfer learning from ImageNet to CIFAR. This module is currently incomplete, but there is a working Keras + Tensorflow implementation, which you can find [here](https://github.com/asinugobi/layerwise_ewc). 