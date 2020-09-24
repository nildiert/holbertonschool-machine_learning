#!/usr/bin/env python3
""" Function that adds two matrices element-wise """


def add_arrays(arr1, arr2):
    """ Function that adds two arrays element-wise """
    if (len(arr1) == len(arr2)):
        return [x+y for x, y in zip(arr1, arr2)]
    else:
        return


def add_matrices2D(mat1, mat2):
    """ Function that adds two matrices element-wise """
    if (len(mat1) == len(mat2)):
        new_list = [add_arrays(x, y) for x, y in zip(mat1, mat2)]
        if new_list[0] is not None:
            return new_list
    else:
        return None
