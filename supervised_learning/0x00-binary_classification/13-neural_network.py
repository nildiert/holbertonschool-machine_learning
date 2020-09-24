#!/usr/bin/env python3
""" This class defines a neural network with one hidden layer """

import numpy as np


class NeuralNetwork():
    """
    This class defines a neural network with one hidden
    layer performing binary classification
    """

    def __init__(self, nx, nodes):
        """ Constructor method """
        if type(nx) is not int:
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if type(nodes) is not int:
            raise TypeError("nodes must be an integer")
        if nodes < 1:
            raise ValueError("nodes must be a positive integer")
        self.__W1 = np.random.normal(size=(nodes, nx))
        self.__b1 = np.zeros((nodes, 1))
        self.__A1 = 0
        self.__W2 = np.random.normal(size=((1, nodes)))
        self.__b2 = 0
        self.__A2 = 0

    @property
    def W1(self):
        """Getter method for W1 attribute"""
        return self.__W1

    @property
    def b1(self):
        """Getter method for b1 attribute"""
        return self.__b1

    @property
    def A1(self):
        """Getter method for A1 attribute"""
        return self.__A1

    @property
    def W2(self):
        """Getter method for W2 attribute"""
        return self.__W2

    @property
    def b2(self):
        """Getter method for b2 attribute"""
        return self.__b2

    @property
    def A2(self):
        """Getter method for A2 attribute"""
        return self.__A2

    def sigmoid(self, x):
        """ Calculates the sigmoid function """
        return (1 / (1 + np.exp(-x)))

    def forward_prop(self, X):
        """Calculates the forward propagation of the neural network"""
        value = np.matmul(self.__W1, X) + self.__b1
        self.__A1 = self.sigmoid(value)
        value2 = np.matmul(self.__W2, self.__A1) + self.__b2
        self.__A2 = self.sigmoid(value2)

        return (self.__A1, self.__A2)

    def cost(self, Y, A):
        """ Calculates the cost of the model using logistic regression """
        m = Y.shape[1]
        cost = (-1/m) * np.sum(Y * np.log(A) + (1-Y) * (np.log(1.0000001 - A)))
        return cost

    def evaluate(self, X, Y):
        """ Evaluates the neuron’s predictions """
        self.__A = self.forward_prop(X)
        eva = np.where(self.__A2 >= 0.5, 1, 0)
        cost = self.cost(Y, self.__A2)
        return (eva, cost)

    def gradient_descent(self, X, Y, A1, A2, alpha=0.05):
        """ Calculates one pass of gradient descent on the network """
        m = Y.shape[1]

        dz_2 = A2 - Y
        dw_2 = (1/m) * np.matmul(A1, dz_2.T)
        db_2 = (1/m) * np.sum(dz_2, axis=1, keepdims=True)

        dz1_1 = np.matmul(self.__W2.T, dz_2)
        dz1_2 = A1 * (1 - A1)
        dz_1 = dz1_1 * dz1_2
        dw_1 = (1/m) * np.matmul(dz_1, X.T)
        db_1 = (1/m) * np.sum(dz_1, axis=1, keepdims=True)
        self.__W1 = self.__W1 - (alpha * dw_1)
        self.__b1 = self.__b1 - (alpha * db_1)
        self.__W2 = self.__W2 - (alpha * dw_2).T
        self.__b2 = self.__b2 - (alpha * db_2)
