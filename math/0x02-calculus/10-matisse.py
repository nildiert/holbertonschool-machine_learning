#!/usr/bin/env python3
""" Function that calculates the derivative of a polynomial """


def poly_derivative(poly):
    """ Function that calculates the derivative of a polynomial """

    derivative = []

    if type(poly) is list:
        if len(poly) is 0:
            return None
        for i in range(len(poly)):
            if type(poly[i]) is int:
                if i is not 0:
                    derivative.append(poly[i] * i)
            else:
                return None
        if sum(derivative) is 0:
            return [0]
        else:
            return derivative
    else:
        return None
