#!/usr/bin/env python3
"""
0. Determinant
"""


def determinant_2D(matrix):
    """ calculates the determinant of a 2D matrix
    Args:
        matrix - is a list of lists whose determinant should be calculated
    Returns:
        the determinant of 2D matrix
    """
    return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]


def recursive_determinant(matrix, total=0):
    """ calculates recursively the determinant of a matrix
    Args:
        matrix - is a list of lists whose determinant should be calculated
        total - summatory of determinants
    Returns:
        the determinant of 2D matrix
    """
    # Extract all indices of matrix
    indices = list(range(len(matrix)))

    # This method works recursively, always we gonna calculate the
    # determinant of a 2D matrix
    if len(matrix) == 2:
        return determinant_2D(matrix)

    for index in indices:
        cp_matrix = matrix.copy()
        # Remove the first column
        cp_matrix = cp_matrix[1:]

        rows_length = len(cp_matrix)

        for i in range(rows_length):
            # Removes column
            cp_matrix[i] = cp_matrix[i][0:index] + cp_matrix[i][index + 1:]

        # Change the sign of all pairs indices
        sign = (-1) ** (index % 2)

        sub_det = recursive_determinant(cp_matrix)

        total += sign * matrix[0][index] * sub_det

    return total


def determinant(matrix):
    """ calculates the determinant of a matrix
    Args:
        matrix - is a list of lists whose determinant should be calculated
    Returns:
        the determinant of matrix
    """
    if matrix == [[]]:
        return 1

    # Check if matrix is a list of lists and matrix is a square matrix
    len_col = [len(row) for row in matrix]
    if isinstance(matrix, list) and len(matrix) is not 0:
        if not all(isinstance(row, list) for row in matrix):
            raise TypeError("matrix must be a list of lists")
    else:
        raise TypeError("matrix must be a list of lists")
    if not all(len(matrix) == col for col in len_col):
        raise ValueError('matrix must be a square matrix')

    if len(matrix) == 1 and len(matrix[0]) == 1:
        return matrix[0][0]

    return recursive_determinant(matrix)
