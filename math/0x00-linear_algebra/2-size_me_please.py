#!/usr/bin/env python3
""" Size of matrix """


def check_shape(matrix, size):
    """ Auxiliar function to find the shape """
    if isinstance(matrix, list):
        size.append(len(matrix))
        if (len(matrix) > 0):
            return check_shape(matrix[0], size)
        else:
            return [0]
    else:
        return size


def matrix_shape(matrix):
    """ Function to find the shape of matrix """
    size = []
    return check_shape(matrix, size)
