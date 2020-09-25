# 0x03. Optimization


<img src="https://littleml.files.wordpress.com/2018/05/optiml.png?w=584&h=339">

## Learning Objectives

At the end of this project, you are expected to be able to explain to anyone, without the help of Google:

### General

* What is a hyperparameter?
* How and why do you normalize your input data?
* What is a saddle point?
* What is stochastic gradient descent?
* What is mini-batch gradient descent?
* What is a moving average? How do you implement it?
* What is gradient descent with momentum? How do you implement it?
* What is RMSProp? How do you implement it?
* What is Adam optimization? How do you implement it?
* What is learning rate decay? How do you implement it?
* What is batch normalization? How do you implement it?

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
* Unless otherwise noted, you are not allowed to import any module except import numpy as np and/or import tensorflow as tf
* You should not import any module unless it is being used
* You are not allowed to use the keras module in tensorflow
* All your files must be executable
* The length of your files will be tested using wc


## More Info

### Testing

Please use the following checkpoints for to accompany the following tensorflow main files. You do not need to push these files to GitHub. Your code will not be tested with these files.

* graph.ckpt.data-00000-of-00001
* graph.ckpt.index
* graph.ckpt.meta