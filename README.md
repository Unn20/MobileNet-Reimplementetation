# MobileNet-reimplementation
## Maciej Leszczyk
## Piotr Linkowski

Reimplementation of MobileNetV1 network for educational purposes

### MobileNet.py
This file contains an author's implementation of MobileNet network's architecture with own Depthwise Separable Convolution layer.

### train_stanford_dogs_.py
This file contains a script to build MobileNet's network, with a linear layer on top of it. Then a training is started on Stanford Dogs dataset, with previous preprocessing on input images. Training contains lambda callback, which is able to save trained weights every one epoch of training.

