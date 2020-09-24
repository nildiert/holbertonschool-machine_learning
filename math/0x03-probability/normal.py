#!/usr/bin/env python3
""" This class represents an normal distribution """


class Normal():
    """ This class represents a normal distribution """

    e = 2.7182818285
    pi = 3.1415926536

    def __init__(self, data=None, mean=0., stddev=1.):
        " The initialized the class "

        if data is not None:
            if type(data) is not list:
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")

            self.mean = sum(data) / len(data)
            summatory = sum([(abs((i - self.mean)**2)) for i in data])
            self.stddev = (summatory / len(data))**(1/2)
        else:

            if stddev <= 0:
                raise ValueError("stddev must be a positive value")
            self.mean = float(mean)
            self.stddev = float(stddev)

    def z_score(self, x):
        """ Calculates the z-score of a given x-value """
        return (x - self.mean) / self.stddev

    def x_value(self, z):
        """Calculates the x-value of a given z-score"""
        return (z * self.stddev) + self.mean

    def erf(self, x):
        """Calculates the error of function"""
        a = ((4/Normal.pi)**0.5)
        b = (x-(x**3)/3+(x**5)/10-x**7/42+(x**9)/216)
        return a * b

    def pdf(self, x):
        """ Calculates the value of the PDF for a given x-value """

        a = (1 / (self.stddev*((2 * Normal.pi)**(1/2))))
        b = (-1/2) * ((x - self.mean) / self.stddev)**2
        return a * (Normal.e**b)

    def cdf(self, x):
        """Calculates the value of the CDF for a given x-value"""
        a = (x - self.mean) / (self.stddev * (2**0.5))
        erf = self.erf(a)
        return (1 + erf) / 2
