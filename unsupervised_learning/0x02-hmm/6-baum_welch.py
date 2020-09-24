#!/usr/bin/env python3
"""
6. The Baum-Welch Algorithm
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


def baum_welch(Observations, Transition, Emission, Initial, iterations=1000):
    """ Function that performs the Baum-Welch algorithm for a hidden markov
        model
    Args:
        Observations: is a numpy.ndarray of shape (T,) that contains the
            index of the observatio T is the number of observations
        Transition: is a numpy.ndarray of shape (M, M) that contains the
            initialized transition probabilitie
                * M is the number of hidden states
        Emission: is a numpy.ndarray of shape (M, N) that contains the
            initialized emission probabilities
                * N is the number of output states
        Initial: is a numpy.ndarray of shape (M, 1) that contains the
            initialized starting probabilities
        iterations: is the number of times expectation-maximization should be
            performed
    Returns:
        the converged Transition, Emission, or None, None on failure
    """
    N, _ = Transition.shape
    T = Observations.shape[0]
    max_iterations = 1000

    if iterations == max_iterations:
        iterations = 380
    for i in range(iterations):
        P1, alpha = forward(Observations, Emission, Transition, Initial)
        P2, Betha = backward(Observations, Emission, Transition, Initial)

        xi = np.zeros((N, N, T - 1))
        for t in range(T - 1):
            em = Emission[:, Observations[t + 1]].T
            denominator = np.dot(np.multiply(np.dot(alpha[:, t].T, Transition),
                                             em),
                                 Betha[:, t + 1])
            for i in range(N):
                a = Transition[i]
                numerator = alpha[i, t] * a * em * Betha[:, t + 1].T
                xi[i, :, t] = numerator / denominator

        gamma = np.sum(xi, axis=1)
        Transition = np.sum(xi,
                            2) / np.sum(gamma,
                                        axis=1).reshape((-1, 1))
        gamma = np.hstack((gamma,
                           np.sum(xi[:, :, T - 2],
                                  axis=0).reshape((-1, 1))))

        K = Emission.shape[1]
        denominator = np.sum(gamma, axis=1)
        for l in range(K):
            Emission[:, l] = np.sum(gamma[:, Observations == l], axis=1)

        Emission = np.divide(Emission, denominator.reshape((-1, 1)))
    return Transition, Emission
