# 0x02. Tensorflow

<img src="https://www.marketing-branding.com/wp-content/uploads/2020/05/tensorflow-beneficios.jpg">

## Learning Objectives

At the end of this project, you are expected to be able to explain to anyone, without the help of Google:

### General

What is tensorflow?
What is a session? graph?
What are tensors?
What are variables? constants? placeholders? How do you use them?
What are operations? How do you use them?
What are namespaces? How do you use them?
How to train a neural network in tensorflow
What is a checkpoint?
How to save/load a model with tensorflow
What is the graph collection?
How to add and get variables from the collection

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
* Unless otherwise noted, you are not allowed to import any module except import tensorflow as tf
* You are not allowed to use the keras module in tensorflow
* All your files must be executable
* The length of your files will be tested using wc

## More Info


### Installing Tensorflow 1.12

```python
$ pip install --user tensorflow==1.12
```

### Optimize Tensorflow (Optional)

In order to get full use of your computer’s hardware, you will need to build tensorflow from source.

Here are some extra reading on why/how to do this:

* How to compile Tensorflow with SSE4.2 and AVX instructions?
* Installing Bazel on Ubuntu
* Build from Source
* Performance
* Python Configuration Error: ‘PYTHON_BIN_PATH’ environment variable is not set
The following instructions assume you already have tensorflow (version 1.12) installed and that you do not have access to a GPU:

**0. Install All Dependencies:**

```bash
$ sudo apt-get install pkg-config zip g++ zlib1g-dev unzip python python3-dev
```

**1. Install Bazel**

```bash
$ wget https://github.com/bazelbuild/bazel/releases/download/0.18.1/bazel-0.18.1-installer-linux-x86_64.sh
$ chmod +x bazel-0.18.1-installer-linux-x86_64.sh
$ sudo ./bazel-0.18.1-installer-linux-x86_64.sh --bin=/bin
```

Add the line source /usr/local/lib/bazel/bin/bazel-complete.bash to your ~/.bashrc if you want bash to tab complete bazel.

**2. Clone Tensorflow Repository**

$ git clone https://github.com/tensorflow/tensorflow.git
$ cd tensorflow
$git checkout r1.12


**3. Build and Install Tensorflow**

```bash
$ export PYTHON_BIN_PATH=/usr/bin/python3 # or wherever python3 is located
$ bazel build --config=mkl --config=opt //tensorflow/tools/pip_package:build_pip_package
$ ./bazel-bin/tensorflow/tools/pip_package/build_pip_package /tmp/tensorflow_pkg
$ pip install --user /tmp/tensorflow_pkg/tensorflow-1.12.0-cp35-cp35m-linux_x86_64.whl
4. Remove Tensorflow Repository
$ cd ..
$ rm -rf tensorflow
```

Now tensorflow will be able to fully utilize the parallel processing capabilities of your computer’s hardware, which will make your training MUCH faster!