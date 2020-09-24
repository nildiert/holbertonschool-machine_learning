#!/usr/bin/env python3
"""
1. Intersection
"""


import numpy as np


def factorial(n):
    """ Function to compute the factorial of a number
    Args:
        n: Number to calculate the factorial
    Returns:
        The factorial of a number
    """
    return np.math.factorial(n)


def likelihood(x, n, P):
    """ Function that calculates the likelihood of obtaining this data
    given various hypothetical probabilities of developing severe side
    effects
    Args:
        x is the number of patients that develop severe side effects
        n - is the total number of patients observed
        P - is a 1D numpy.ndarray containing the various hypothetical
        probabilities of developing severe side effects
    Returns:
        a 1D numpy.ndarray containing the likelihood of obtaining the data,
        x and n, for each probability in P, respectively
    Documentation:
        https://es.m.wikipedia.org/wiki/Distribuci%C3%B3n_binomial
    """
    if not isinstance(n, (int, float)) or n <= 0:
        raise ValueError("n must be a positive integer")

    if not isinstance(x, (int, float)) or x < 0:
        raise ValueError(
            "x must be an integer that is greater than or equal to 0")

    if x > n:
        raise ValueError("x cannot be greater than n")

    if not isinstance(P, np.ndarray) or len(P.shape) is not 1:
        raise TypeError("P must be a 1D numpy.ndarray")

    if np.any(P > 1) or np.any(P < 0):
        raise ValueError("All values in P must be in the range [0, 1]")
    nx = factorial(n) / (factorial(x) * factorial(n - x))
    likelihood = nx * np.power(P, x) * np.power(1 - P, n - x)
    return likelihood


def intersection(x, n, P, Pr):
    """
    Args:
        x is the number of patients that develop severe side effects
        n is the total number of patients observed
        P is a 1D numpy.ndarray containing the various hypothetical
            probabilities of developing severe side effects
        Pr is a 1D numpy.ndarray containing the prior beliefs of P
    Returns:
        a 1D numpy.ndarray containing the intersection of
            obtaining x and n with each probability in P, respectively
    """
    if not isinstance(n, (int, float)) or n <= 0:
        raise ValueError("n must be a positive integer")

    if not isinstance(x, (int, float)) or x < 0:
        raise ValueError(
            "x must be an integer that is greater than or equal to 0")

    if x > n:
        raise ValueError("x cannot be greater than n")

    if not isinstance(P, np.ndarray) or len(P.shape) is not 1:
        raise TypeError("P must be a 1D numpy.ndarray")

    if not isinstance(Pr, np.ndarray) or Pr.shape != P.shape:
        raise TypeError("Pr must be a numpy.ndarray with the same shape as P")

    if np.any(P > 1) or np.any(P < 0):
        raise ValueError("All values in P must be in the range [0, 1]")
    if np.any(Pr > 1) or np.any(Pr < 0):
        raise ValueError("All values in Pr must be in the range [0, 1]")
    if not np.isclose(np.sum(Pr), 1):
        raise ValueError("Pr must sum to 1")

    lik = likelihood(x, n, P)
    return lik * Pr
