#!/usr/bin/env python3
"""
5. Definiteness
"""
import numpy as np


def definiteness(matrix):
    """ Calculates the definiteness of a matrix
    Args:
        matrix: is a numpy.ndarray of shape (n, n) whose definiteness
        should be calculated
    Returns:
        string - Positive definite, Positive semi-definite, Negative
        semi-definite, Negative definite, or Indefinite if the matrix
        is positive definite, positive semi-definite, negative semi-definite,
        negative definite of indefinite, respectively
        None - If matrix does not fit any of the above categories
    """

    if type(matrix) is not np.ndarray:
        raise TypeError("matrix must be a numpy.ndarray")
    if len(matrix.shape) == 1:
        return None
    if (matrix.shape[0] != matrix.shape[1]):
        return None
    if not np.linalg.eig(matrix):
        return None
    if not (matrix.transpose() == matrix).all():
        return None

    alpha, v = np.linalg.eig(matrix)

    if np.all(alpha == 0):
        return None
    if np.all(alpha > 0):
        return "Positive definite"
    if np.all(alpha >= 0):
        return "Positive semi-definite"
    if np.all(alpha < 0):
        return "Negative definite"
    if np.all(alpha <= 0):
        return "Negative semi-definite"
    else:
        return "Indefinite"
