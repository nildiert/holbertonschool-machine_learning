#!/usr/bin/env python3
""" Function that performs matrix multiplication """


def mat_mul(mat1, mat2):
    """ Function that performs matrix multiplication """

    rows1 = len(mat1)
    cols1 = len(mat1[0])
    rows2 = len(mat2)
    cols2 = len(mat2[0])

    if cols1 != rows2:
        return None
    else:
        new_matrix = []
        for x in range(rows1):
            aux_row = []
            for y in range(cols2):
                aux_sum = []
                for z in range(cols1):
                    aux_sum.append(mat1[x][z] * mat2[z][y])
                aux_row.append(sum(aux_sum))
            new_matrix.append(aux_row)

        return new_matrix
