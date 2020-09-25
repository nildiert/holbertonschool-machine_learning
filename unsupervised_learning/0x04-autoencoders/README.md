# 0x04. Autoencoders

> An autoencoder is a type of artificial neural network used to learn efficient data codings in an unsupervised manner. The aim of an autoencoder is to learn a representation (encoding) for a set of data, typically for dimensionality reduction, by training the network to ignore signal “noise”. Along with the reduction side, a reconstructing side is learnt, where the autoencoder tries to generate from the reduced encoding a representation as close as possible to its original input, hence its name. ... [See more](https://en.wikipedia.org/wiki/Autoencoder)

<img src="https://miro.medium.com/max/600/1*nqzWupxC60iAH2dYrFT78Q.png">

## Learning Objectives

* What is an autoencoder?
* What is latent space?
* What is a bottleneck?
* What is a sparse autoencoder?
* What is a convolutional autoencoder?
* What is a generative model?
* What is a variational autoencoder?
* What is the Kullback-Leibler divergence?

## Requirements

### General

* Allowed editors: vi, vim, emacs
* All your files will be interpreted/compiled on Ubuntu 16.04 LTS using python3 (version 3.5)
* Your files will be executed with numpy (version 1.15) and tensorflow (version 1.12)
* All your files should end with a new line
* The first line of all your files should be exactly #!/usr/bin/env python3
* A README.md file, at the root of the folder of the project, is mandatory
* Your code should use the pycodestyle style (version 2.4)
* All your modules should have documentation (python3 -c 'print(__import__("my_module").__doc__)')
* All your classes should have documentation (python3 -c 'print(__import__("my_module").MyClass.__doc__)')
* All your functions (inside and outside a class) should have documentation (python3 -c 'print(__import__("my_module").my_function.__doc__)' and python3 -c 'print(__import__("my_module").MyClass.my_function.__doc__)')
* Unless otherwise noted, you are not allowed to import any module except import tensorflow.keras as keras
* All your files must be executable