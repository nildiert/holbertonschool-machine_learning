#!/usr/bin/env python3
"""
4. Inverse
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


def get_minor(matrix, i, j):
    """ Returns the minors without the determinant
    Args:
        matrix - list of list of cofactors
        i - coordenate in i
        j - coordenate in j
    Returns:
        The minor with the determinant calculate
    """
    minor = [row[:j] + row[j+1:] for row in (matrix[:i] + matrix[i+1:])]
    return determinant(minor)


def minor(matrix):
    """ calculates the minor matrix of a matrix
    Args:
        matrix - is a list of lists whose minor matrix should be calculated
    Returns:
        the minor matrix of matrix
    """
    len_col = [len(row) for row in matrix]
    if not isinstance(matrix, list):
        raise TypeError("matrix must be a list of lists")
    if isinstance(matrix, list) and len(matrix) is not 0:
        if not all(isinstance(row, list) for row in matrix):
            raise TypeError("matrix must be a list of lists")
    else:
        raise TypeError("matrix must be a list of lists")
    if matrix == [[]]:
        raise ValueError('matrix must be a non-empty square matrix')
    if not all(len(matrix) == col for col in len_col):
        raise ValueError('matrix must be a non-empty square matrix')

    if len(matrix) == 1:
        return [[1]]

    minors_list = []
    for row in range(len(matrix)):
        minors = []
        for col in range(len(matrix)):
            minors.append(get_minor(matrix, row, col))
        minors_list.append(minors)
    return minors_list


def cofactor(matrix):
    """ Calculates the cofactor matrix of a matrix
    Args:
        matrix - is a list of lists whose cofactor matrix should be calculated
    Returns:
        The cofactor matrix of matrix
    """
    minors = minor(matrix)
    cofactor = minors.copy()
    for i in range(len(minors)):
        for j in range(len(minors)):
            cofactor[i][j] = cofactor[i][j] * (-1)**(i+j)
    return cofactor


def transpose(matrix):
    """ Transpose
    Returns the transpose of a matrix
    Args:
        Matrix - Is a list of lists
    Returns:
        The transpose of a matrix
    """
    return [[row[i] for row in matrix] for i in range(len(matrix[0]))]


def adjugate(matrix):
    """ calculates the adjugate matrix of a matrix
    Args:
        matrix - is a list of lists whose adjugate matrix should be calculated
    Returns:
        the adjugate matrix of matrix
    """
    cofactors = cofactor(matrix)
    return transpose(cofactors)


def inverse(matrix):
    """
    Calculates the inverse of a matrix
    Args:
        matrix - is a list of lists whose inverse should be calculated
    Returns:
        the inverse of matrix, or None if matrix is singular
    """

    len_col = [len(row) for row in matrix]
    if not all(len(matrix) == col for col in len_col):
        raise ValueError('matrix must be a non-empty square matrix')
    if not isinstance(matrix, list):
        raise TypeError("matrix must be a list of lists")
    if isinstance(matrix, list) and len(matrix) is not 0:
        if not all(isinstance(row, list) for row in matrix):
            raise TypeError("matrix must be a list of lists")

    det = determinant(matrix)
    if det is 0:
        return None
    adj = adjugate(matrix)

    for i in range(len(adj)):
        for j in range(len(adj)):
            adj[i][j] = adj[i][j]/det
    return adj
