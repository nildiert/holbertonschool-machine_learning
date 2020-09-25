# 0x01-multiclass_classification

<img src="https://3qeqpr26caki16dnhd19sv6by6v-wpengine.netdna-ssl.com/wp-content/uploads/2020/01/Scatter-Plot-of-Multi-Class-Classification-Dataset.png">

## Learning Objectives

At the end of this project, you are expected to be able to explain to anyone, without the help of Google:

### General

* What is multiclass classification?
* What is a one-hot vector?
* How to encode/decode one-hot vectors
* What is the softmax function and when do you use it?
* What is cross-entropy loss?
* What is pickling in Python?

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
* All your files must be executable
* The length of your files will be tested using wc

## More Info

### Testing your code
In order to test your code, you’ll need DATA! Please download this dataset to go along with all of the following main files. You do not need to upload this file to GitHub. Your code will not necessarily be tested with this dataset.

```python
alexa@ubuntu-xenial:0x01-multiclass_classification$ cat show_data.py
#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np

lib = np.load('../data/MNIST.npz')
print(lib.files)
X_train_3D = lib['X_train']
Y_train = lib['Y_train']

fig = plt.figure(figsize=(10, 10))
for i in range(100):
    fig.add_subplot(10, 10, i + 1)
    plt.imshow(X_train_3D[i])
    plt.title(str(Y_train[i]))
    plt.axis('off')
plt.tight_layout()
plt.show()
alexa@ubuntu-xenial:0x01-multiclass_classification$ ./show_data.py
```