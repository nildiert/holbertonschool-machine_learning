#!/usr/bin/env python3
""" This class represents a poisson distribution """


class Poisson():
    """ This class represents a poisson distribution """

    e = 2.7182818285

    def __init__(self, data=None, lambtha=1.):
        """ Initialize class """
        if data is not None:
            if type(data) is not list:
                raise TypeError("data must be a list")
            if len(data) <= 2:
                raise ValueError("data must contain multiple values")
            self.lambtha = float(sum(data) / len(data))
        else:
            if lambtha <= 0:
                raise ValueError("lambtha must be a positive value")
            self.lambtha = float(lambtha)

    def pmf(self, k):
        """Calculates the value of the PMF for a given number of “successes”"""

        powFact = 1
        k = int(k)

        fact = 1

        if k < 0:
            return 0
        base = (Poisson.e**(-self.lambtha))
        successes = self.lambtha**k

        for x in range(1, k+1):
            fact = fact * x

        pmf = (base * successes) / fact

        return pmf

    def cdf(self, k):
        """Calculates the value of the CDF for a given number of “successes”"""

        k = int(k)

        if k < 0:
            return 0

        return sum([self.pmf(x) for x in range(0, k + 1)])
