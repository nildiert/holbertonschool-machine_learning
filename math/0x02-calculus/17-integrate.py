#!/usr/bin/env python3
""" Function that calculates the integral of a polynomial """


def poly_integral(poly, C=0):
    """ Function that calculates the integral of a polynomial """

    if type(poly) is not list or len(poly) == 0 or type(C) is not int:
        return None
    derivative = [C]
    if poly == [0]:
        return derivative

    if type(poly) is list:
        for i in range(len(poly)):
            if type(poly[i]) is int or type(poly[i]) is float:
                val = poly[i] / (i + 1)
                derivative.append(int(val) if val.is_integer() else val)
            else:
                return None
        for summatory in range(len(derivative)):
            if (sum(derivative[summatory:]) is 0):
                return derivative[:summatory]
        return derivative
    else:
        return None
