# 0x05. Regularization


<img src="https://miro.medium.com/max/771/1*cdvfzvpkJkUudDEryFtCnA.png">

## Learning Objectives

At the end of this project, you are expected to be able to explain to anyone, without the help of Google:

## General

What is regularization? What is its purpose?
What is are L1 and L2 regularization? What is the difference between the two methods?
What is dropout?
What is early stopping?
What is data augmentation?
How do you implement the above regularization methods in Numpy? Tensorflow?
What are the pros and cons of the above regularization methods?

## Requirements

### General

* Allowed editors: vi, vim, emacs
* All your files will be interpreted/compiled on Ubuntu 16.04 LTS using python3 (version 3.5)
* Your files will be executed with numpy (version 1.15)
* All your files should end with a new line
* The first line of all your files should be exactly #!/usr/bin/env python3
* A README.md file, at the root of the folder of the project, is mandatory
* Your code should use the pycodestyle style (version 2.4)
* All your modules should have documentation (python3 -c 'print(__import__("my_module").__doc__)')
* All your classes should have documentation (python3 -c 'print(__import__("my_module").MyClass.__doc__)')
* All your functions (inside and outside a class) should have documentation (python3 -c 'print(__import__("my_module").my_function.__doc__)' and python3 -c 'print(__import__("my_module").MyClass.my_function.__doc__)')
* Unless otherwise noted, you are not allowed to import any module except import numpy as np and import tensorflow as tf
* You are not allowed to use the keras module in tensorflow
* You should not import any module unless it is being used
* All your files must be executable
* The length of your files will be tested using wc
* When initializing layer weights, use tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG").