# 0x03. Hyperparameter Tuning

<img src="https://pvsmt99345.i.lithium.com/t5/image/serverpage/image-id/74545i97245FDAA10376E9?v=1.0">

> In machine learning, hyperparameter optimization or tuning is the problem of choosing a set of optimal hyperparameters for a learning algorithm. A hyperparameter is a parameter whose value is used to control the learning process. By contrast, the values of other parameters (typically node weights) are learned. ... [See more](https://en.wikipedia.org/wiki/Hyperparameter_optimization)


# Learning Objectives

* What is Hyperparameter Tuning?
* What is random search? grid search?
* What is a Gaussian Process?
* What is a mean function?
* What is a Kernel function?
* What is Gaussian Process Regression/Kriging?
* What is Bayesian Optimization?
* What is an Acquisition function?
* What is Expected Improvement?
* What is Knowledge Gradient?
* What is Entropy Search/Predictive Entropy Search?
* What is GPy?
* What is GPyOpt?

## Requirements

### General

* Allowed editors: vi, vim, emacs
* All your files will be interpreted/compiled on Ubuntu 16.04 LTS using python3 (version 3.5)
* Your files will be executed with numpy (version 1.15)
* All your files should end with a new line
* The first line of all your files should be exactly #!/usr/bin/env python3
* A README.md file, at the root of the folder of the project, is mandatory
* Your code should use the pycodestyle style (version 2.4), but can ignore error E741
* All your modules should have documentation (python3 -c 'print(__import__("my_module").__doc__)')
* All your classes should have documentation (python3 -c 'print(__import__("my_module").MyClass.__doc__)')
* All your functions (inside and outside a class) should have documentation (python3 -c 'print(__import__("my_module").my_function.__doc__)' and python3 -c 'print(__import__("my_module").MyClass.my_function.__doc__)')
* Unless otherwise noted, you are not allowed to import any module except import numpy as np
* All your files must be executable

## Install GPy and GPyOpt

```python
pip install --user GPy
pip install --user gpyopt
```