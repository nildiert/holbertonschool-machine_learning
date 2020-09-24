#!/usr/bin/env python3
""" Return the transpose of Matrix """


def matrix_transpose(matrix):
    """ Find the transpose of matrix """
    return [[row[i] for row in matrix] for i in range(len(matrix[0]))]
