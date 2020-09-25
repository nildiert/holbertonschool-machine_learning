# 0x00. Binary Classification


<img src="https://3qeqpr26caki16dnhd19sv6by6v-wpengine.netdna-ssl.com/wp-content/uploads/2019/10/Scatter-Plot-of-Binary-Classification-Dataset.png" width="200">

## Background Context

Welcome to your first project on supervised learning! At the end of this project, you should be able to build your own binary image classifier from scratch using numpy. As you might already see, there are a LOT of resources for you to read/watch. It may be tempting to dive into the projects right away, but it is HIGHLY RECOMMENDED that you spend AT LEAST 1 whole day going over the following materials. You should only start the project once you have a decent understanding of all the topics mentioned in Learning Objectives. You may also notice that there are multiple resources that cover the same topic, with some more technical than others. If you find yourself getting lost in a resource, move on to another and come back to the more technical one after you intuitively understand that topic. Good luck and have fun!

## Learning Objectives

At the end of this project, you are expected to be able to explain to anyone, without the help of Google:

## General

* What is a model?
* What is supervised learning?
* What is a prediction?
* What is a node?
* What is a weight?
* What is a bias?
* What are activation functions?
    * Sigmoid?
    * Tanh?
    * Relu?
    * Softmax?
* What is a layer?
* What is a hidden layer?
* What is Logistic Regression?
* What is a loss function?
* What is a cost function?
* What is forward propagation?
* What is Gradient Descent?
* What is back propagation?
* What is a Computation Graph?
* How to initialize weights/biases
* The importance of vectorization
* How to split up your data


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
* Unless otherwise noted, you are not allowed to import any module except import numpy as np
* Unless otherwise noted, you are not allowed to use any loops (for, while, etc.)
* All your files must be executable
* The length of your files will be tested using wc


## More Info

### Matrix Multiplications

For all matrix multiplications in the following tasks, please use numpy.matmul

### Testing your code

In order to test your code, you’ll need DATA! Please download these datasets (Binary_Train.npz, Binary_Dev.npz) to go along with all of the following main files. You do not need to upload these files to GitHub. Your code will not necessarily be tested with these datasets. All of the following code assumes that you have stored all of your datasets in a separate data directory.

```python
alexa@ubuntu-xenial:0x00-binary_classification$ cat show_data.py
#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np

lib_train = np.load('../data/Binary_Train.npz')
X_3D, Y = lib_train['X'], lib_train['Y']

fig = plt.figure(figsize=(10, 10))
for i in range(100):
    fig.add_subplot(10, 10, i + 1)
    plt.imshow(X_3D[i])
    plt.title(Y[0, i])
    plt.axis('off')
plt.tight_layout()
plt.show()
alexa@ubuntu-xenial:0x00-binary_classification$ ./show_data.py
```