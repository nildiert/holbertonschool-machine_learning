#!/usr/bin/env python3
""" This class defines a single neuron performing binary classification """


import numpy as np


class Neuron():
    """ Neuron class """

    def __init__(self, nx):
        """
        This class defines a single neuron performing
        binary classification
        """

        if type(nx) is not int:
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")

        self.__W = np.random.normal(size=(1, nx))
        self.__b = 0
        self.__A = 0

    @property
    def W(self):
        """ Method to get the value of W """
        return self.__W

    @property
    def b(self):
        """ Method to get the value of b """
        return self.__b

    @property
    def A(self):
        """ Method to get the value of A """
        return self.__A
