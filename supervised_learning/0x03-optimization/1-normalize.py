#!/usr/bin/env python3
""" Function that normalizes (standardizes) a matrix """


def normalize(X, m, s):
    """ Normalize function """
    normalized_matrix = (X - m) / s
    return normalized_matrix
