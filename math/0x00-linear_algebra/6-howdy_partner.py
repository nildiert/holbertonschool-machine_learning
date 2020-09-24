#!/usr/bin/env python3
""" Function that concatenates two arrays """


def cat_arrays(arr1, arr2):
    """ Function that concatenates two arrays """
    new = arr1[:]
    for row in arr2:
        new.append(row)
    return new
