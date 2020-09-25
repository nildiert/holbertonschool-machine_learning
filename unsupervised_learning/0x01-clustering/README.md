# Clustering

> Cluster analysis or clustering is the task of grouping a set of objects in such a way that objects in the same group (called a cluster) are more similar (in some sense) to each other than to those in other groups (clusters). It is a main task of exploratory data mining, and a common technique for statistical data analysis, used in many fields, including pattern recognition, image analysis, information retrieval, bioinformatics, data compression, computer graphics and machine learning. ... [See more](https://en.wikipedia.org/wiki/Cluster_analysis)

<img src="https://miro.medium.com/max/561/0*ff7kw5DRQbs_uixR.jpg">


## Learning Objectives

* What is a multimodal distribution?
* What is a cluster?
* What is cluster analysis?
* What is “soft” vs “hard” clustering?
* What is K-means clustering?
* What are mixture models?
* What is a Gaussian Mixture Model (GMM)?
* What is the Expectation-Maximization (EM) algorithm?
* How to implement the EM algorithm for GMMs
* What is cluster variance?
* What is the mountain/elbow method?
* What is the Bayesian Information Criterion?
* How to determine the correct number of clusters
* What is Hierarchical clustering?
* What is Agglomerative clustering?
* What is Ward’s method?
* What is Cophenetic distance?
* What is scikit-learn?
* What is scipy?

## Requirements

### General

* Allowed editors: vi, vim, emacs
* All your files will be interpreted/compiled on Ubuntu 16.04 LTS using python3 (version 3.5)
* Your files will be executed with numpy (version 1.15), sklearn (version 0.21), and scipy (version 1.3)
* All your files should end with a new line
* The first line of all your files should be exactly #!/usr/bin/env python3
* A README.md file, at the root of the folder of the project, is mandatory
* Your code should use the pycodestyle style (version 2.4)
* All your modules should have documentation (python3 -c 'print(__import__("my_module").__doc__)')
* All your classes should have documentation (python3 -c 'print(__import__("my_module").MyClass.__doc__)')
* All your functions (inside and outside a class) should have documentation (python3 -c 'print(__import__("my_module").my_function.__doc__)' and python3 -c 'print(__import__("my_module").MyClass.my_function.__doc__)')
* Unless otherwise noted, you are not allowed to import any module except import numpy as np
* All your files must be executable
* Your code should use the minimum number of operations


## Installing Scikit-Learn 0.21.x

```python
pip install --user scikit-learn==0.21
```

## Installing Scipy 1.3.x

scipy should have already been installed with matplotlib and numpy, but just in case:

```python
pip install --user scipy==1.3
```

## Resources

- https://brilliant.org/wiki/gaussian-mixture-model/
- https://www.python-course.eu/expectation_maximization_and_gaussian_mixture_models.php

- http://www.cse.iitm.ac.in/~vplab/courses/DVP/PDF/gmm.pdf
