#!/usr/bin/env python3
""" Function that adds two arrays element-wise """


def add_arrays(arr1, arr2):
    """ Function that adds two arrays element-wise """
    if (len(arr1) == len(arr2)):
        return [x+y for x, y in zip(arr1, arr2)]
    else:
        return None
