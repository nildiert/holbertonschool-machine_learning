#!/usr/bin/env python3
"""
3. The Forward Algorithm
"""

import numpy as np


def forward(Observation, Emission, Transition, Initial):
    """ Function that performs the forward algorithm for a
        hidden markov model
    Args:
        Observation is a numpy.ndarray of shape (T,) that contains the index
            of the observation
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
            starting in a particular hidden state
    Returns: P, F, or None, None on failure
        P is the likelihood of the observations given the model
        F is a numpy.ndarray of shape (N, T) containing the forward path
            probabilities
            F[i, j] is the probability of being in hidden state i at time j
                given the previous observations
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

    F = np.zeros((N, T))

    # $ \pi_q\beta_,x_1 $
    F[:, 0] = Initial.T * Emission[:, Observation[0]]

    # $\\sum_{q'}^{}\alpha_{q',t-1}A_{q',q}B_{q,x_t}\$
    for t in range(1, T):
        for s in range(N):
            first_part = F[:, t - 1] * Transition[:, s]
            second_part = Emission[s, Observation[t]]
            F[s, t] = np.sum(first_part * second_part)

    P = np.sum(F[:, -1])

    return P, F
