#!/usr/bin/env python3
"""
3. Entropy
"""

import numpy as np


def HP(Di, beta):
    """ Calculates the Shannon entropy and P affinities relative
        to a data point
    Args:
        Di - is a numpy.ndarray of shape (n - 1,) containing the
        pariwise distances between a data point and all other points
        except itself
        n - is the number of data points
        beta - is the beta value for the Gaussian distribution
    Returns:
        (Hi, Pi)
        Hi: the Shannon entropy of the points
        Pi: a numpy.ndarray of shape (n - 1,) containing the P affinities
        of the points
    """
    # Apply p_ij equation
    # https://miro.medium.com/max/1400/1*jzRNIXK3jAyKAZrkOEhOfA.png
    Pi = np.exp(-Di * beta) / np.sum(np.exp(-Di * beta))

    # Apply shannon entropy equation
    Hi = - np.sum(Pi * np.log2(Pi))

    return Hi, Pi
