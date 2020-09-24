#!/usr/bin/env python3
"""
5. The Backward Algorithm
"""

import numpy as np


def backward(Observation, Emission, Transition, Initial):
    """Function that performs the backward algorithm for a hidden
        markov model
    Args:
        Observation is a numpy.ndarray of shape (T,) that contains the index
            of the observation
        T is the number of observations
        Emission is a numpy.ndarray of shape (N, M) containing the emission
            probability of a specific observation given a hidden state
            * Emission[i, j] is the probability of observing j given the hidden
                state i
        N is the number of hidden states
        M is the number of all possible observations
        Transition is a 2D numpy.ndarray of shape (N, N) containing the
            transition probabilities
            * Transition[i, j] is the probability of transitioning from the
                hidden state i to j
        Initial a numpy.ndarray of shape (N, 1) containing the probability of
            starting in a particular hidden state
    Returns: P, B, or None, None on failure
        Pis the likelihood of the observations given the model
        B is a numpy.ndarray of shape (N, T) containing the backward path
            probabilities
            * B[i, j] is the probability of generating the future observations
                from hidden state i at time j
    Documentation:
        https://github.com/jiaeyan/Hidden-Markov-Model/blob/master/HMM.py
        https://web.stanford.edu/~jurafsky/slp3/A.pdf
        http://www.adeveloperdiary.com/data-science/machine-learning/\
        forward-and-backward-algorithm-in-hidden-markov-model/


    """
    T = Observation.shape[0]
    N, M = Emission.shape

    if ((len(Observation.shape)) != 1) or (
            not isinstance(Observation, np.ndarray)):
        return None, None
    if ((len(Emission.shape)) != 2) or (not isinstance(Emission, np.ndarray)):
        return None, None
    N1_T, N2_T = Transition.shape
    if ((len(Transition.shape)) != 2) or (N != N1_T) or (N != N2_T):
        return None, None
    if (N1_T != N2_T) or (not isinstance(Transition, np.ndarray)):
        return None, None
    probability = np.ones((1, N1_T))
    if not (np.isclose((np.sum(Transition, axis=1)), probability)).all():
        return None, None
    if ((len(Initial.shape)) != 2) or (not isinstance(Initial, np.ndarray)):
        return None, None
    if (N != Initial.shape[0]):
        return None, None

    beta = np.zeros((N, T))
    beta[:, T - 1] = np.ones((N))
    # $ \pi_q\beta_,x_1 $

    # $\\sum_{q'}^{}\alpha_{q',t-1}A_{q',q}B_{q,x_t}\$
    for i in range(T - 2, -1, -1):
        for j in range(N):
            first_part = beta[:, i + 1] * Transition[j, :]
            second_part = Emission[:, Observation[i + 1]]
            beta[j, i] = np.sum(first_part * second_part)

    P = np.sum(np.sum(Initial.T * Emission[:, Observation[0]] * beta[:, 0]))

    return P, beta
