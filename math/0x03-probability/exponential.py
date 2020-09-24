#!/usr/bin/env python3
""" This class represents an exponential distribution """


class Exponential():
    """ This class represents an exponential distribution """

    e = 2.7182818285

    def __init__(self, data=None, lambtha=1.):
        " The initialized the class "

        if data is not None:
            if type(data) is not list:
                raise TypeError("data must be a list")
            if len(data) <= 2:
                raise ValueError("data must contain multiple values")
            self.lambtha = float(1 / (sum(data) / len(data)))
        else:
            if lambtha <= 0:
                raise ValueError("lambtha must be a positive value")
            self.lambtha = float(lambtha)

    def pdf(self, x):
        """ Calculates the pdf for a given time period """

        value = Exponential.e ** ((-self.lambtha)*x)
        return 0 if x < 0 else self.lambtha * value

    def cdf(self, x):
        """ Calculates the value of the CDF for a given time period """
        if x < 0:
            return 0
        return 1 - Exponential.e**((-self.lambtha) * x)
