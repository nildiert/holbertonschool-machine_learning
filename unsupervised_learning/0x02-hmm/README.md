# 0x02. Hidden Markov Models


> Hidden Markov Model (HMM) is a statistical Markov model in which the system being modeled is assumed to be a Markov process – call it {\displaystyle X}X – with unobservable ("hidden") states. HMM assumes that there is another process {\displaystyle Y}Y whose behavior "depends" on {\displaystyle X}X. The goal is to learn about {\displaystyle X}X by observing {\displaystyle Y}Y.

<img src="https://upload.wikimedia.org/wikipedia/commons/thumb/8/8a/HiddenMarkovModel.svg/300px-HiddenMarkovModel.svg.png">


## Learning Objectives

* What is the Markov property?
* What is a Markov chain?
* What is a state?
* What is a transition probability/matrix?
* What is a stationary state?
* What is a regular Markov chain?
* How to determine if a transition matrix is regular
* What is an absorbing state?
* What is a transient state?
* What is a recurrent state?
* What is an absorbing Markov chain?
* What is a Hidden Markov Model?
* What is a hidden state?
* What is an observation?
* What is an emission probability/matrix?
* What is a Trellis diagram?
* What is the Forward algorithm and how do you implement it?
* What is decoding?
* What is the Viterbi algorithm and how do you implement it?
* What is the Forward-Backward algorithm and how do you implement it?
* What is the Baum-Welch algorithm and how do you implement it?

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
* All your files must be executable


## Task Documentation (Latex equations)

### 0. Markov Chain

[Properties of Markov Chains ](http://www3.govst.edu/kriordan/files/ssc/math161/pdf/Chapter10ppt.pdf)

<a href="https://www.codecogs.com/eqnedit.php?latex=S_{x}=S_{k-1}P=S_{0}P^{k}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?S_{x}=S_{k-1}P=S_{0}P^{k}" title="S_{x}=S_{k-1}P=S_{0}P^{k}" /></a>

### 3. The Forward Algorithm 

[Forward and Backward algorithms | | UPV](https://www.youtube.com/watch?v=u45KR9eCvJs)

*if* 

\[t = 1\]

*then*:

\[\pi_q\beta_q, x1\]

---
*if* 

\[t > 1\]

*then*:
\[\sum_{q'}^{}\alpha_{q',t-1}A_{q',q}B_{q,x_t}\]

---

\[P(x) = \sum_{q}^{}\alpha_{q,T}A_{q,F}\]


### 4. The Viretbi Algorithm 
---
*if* 

\[t = 1\]

*then*:

\[\pi_q\beta_q, x1\]

---
*if* 

$t > 1$

*then*:
\[max V(q',t-1)A_{q',q}B_{q,x_t} \rightarrow q'\in Q\]