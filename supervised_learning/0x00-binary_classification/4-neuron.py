#!/usr/bin/env python3
""" This class defines a single neuron performing binary classification """


import numpy as np


class Neuron():
    """Neuron class"""

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

    @A.setter
    def A(self, value):
        """ Method to set the value of A """
        self.__A = value

    def sigmoid(self, x):
        """ Calculates the sigmoid function """
        return (1 / (1 + np.exp(-x)))

    def forward_prop(self, X):
        """ Calculates the forward propagation of the neuron """
        value = np.matmul(self.__W, X) + self.__b
        self.__A = self.sigmoid(value)

        return self.__A

    def cost(self, Y, A):
        """ Calculates the cost of the model using logistic regression """
        m = Y.shape[1]
        cost = (-1/m) * np.sum(Y * np.log(A) + (1-Y) * (np.log(1.0000001-A)))
        return cost

    def evaluate(self, X, Y):
        """ Evaluates the neuron’s predictions """
        self.__A = self.forward_prop(X)
        eva = np.where(self.__A >= 0.5, 1, 0)
        return (eva, self.cost(Y, self.__A))
