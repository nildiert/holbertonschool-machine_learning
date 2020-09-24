#!/usr/bin/env python3
"""
4. The Viterbi Algorithm
"""

import numpy as np


def viterbi(Observation, Emission, Transition, Initial):
    """ Function that calculates the most likely sequence of hidden
        states for a hidden markov model
    Args:
        Observation:
            is a numpy.ndarray of shape (T,) that contains
            the index of the observation
        T is the number of observations
        Emission is a numpy.ndarray of shape (N, M) containing the emission
        probability of a specific observation given a hidden state

            Emission[i, j] is the probability of observing j given the hidden
            state i
        N is the number of hidden states
        M is the number of all possible observations
        Transition is a 2D numpy.ndarray of shape (N, N) containing the
            transition probabilities
            Transition[i, j] is the probability of transitioning from the
            hidden state i to j
        Initial a numpy.ndarray of shape (N, 1) containing the probability of
            starting in a
            particular hidden state
    Returns: path, P, or None, None on failure
        path is the a list of length T containing the most likely sequence of
            hidden states
        P is the probability of obtaining the path sequence
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

    viterbi = np.zeros((N, T))
    backpointer = np.zeros((N, T))

    # $ \pi_q\beta_,x_1 $
    viterbi[:, 0] = Initial.T * Emission[:, Observation[0]]
    backpointer[:, 0] = 0

    # $\\sum_{q'}^{}\alpha_{q',t-1}A_{q',q}B_{q,x_t}\$
    for t in range(1, T):
        for s in range(N):
            first_part = viterbi[:, t - 1] * Transition[:, s]
            second_part = Emission[s, Observation[t]]
            viterbi[s, t] = np.max(first_part * second_part)
            backpointer[s, t] = np.argmax(first_part * second_part)

    bestpathprob = [0 for i in range(T)]
    bestpathprob[-1] = np.argmax(viterbi[:, T - 1])
    for t in range(T - 1, 0, -1):
        bestpathprob[t - 1] = int(backpointer[bestpathprob[t], t])

    P = np.max(viterbi[:, -1])

    return bestpathprob, P
