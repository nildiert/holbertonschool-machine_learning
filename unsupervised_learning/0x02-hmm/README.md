# 0x02. Hidden Markov Models


## Task Documentation

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