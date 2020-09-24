#!/usr/bin/env python3
""" Function that calculates Summatory  """


def summation_i_squared(n):
    """ Function that calculates Summatory  """

    if n > 0 and n is not None:
        myLIst = list(range(1, n+1))
        anotherList = list(map(lambda x: x**2, myLIst))
        return (sum(anotherList))
    else:
        return None
