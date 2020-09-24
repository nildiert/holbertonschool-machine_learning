#!/usr/bin/env python3
""" This class defines a single neuron performing binary classification """


import numpy as np
import matplotlib.pyplot as plt


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

    def gradient_descent(self, X, Y, A, alpha=0.05):
        """ Calculates one pass of gradient descent on the neuron """
        m = Y.shape[1]
        dz = A - Y
        dw = (1/m) *np.matmul(X, dz.T)
        db = (1 / m) * np.sum(dz)
        self.__W = self.__W - (alpha * dw).T
        self.__b = self.__b - (alpha * db)

    def train(self, X, Y, iterations=5000, alpha=0.05, verbose=True,
              graph=True, step=100):
        """ Trains the neuron """
        if type(iterations) is not int:
            raise TypeError('iterations must be an integer')
        if iterations < 0:
            raise ValueError("iterations must be a positive integer")
        if type(alpha) is not float:
            raise TypeError("alpha must be a float")
        if alpha < 0:
            raise ValueError("alpha must be positive")
        cost_list = []
        for i in range(iterations + 1):
            self.__A = self.forward_prop(X)
            self.gradient_descent(X, Y, self.__A, alpha)
            cost = self.cost(Y, self.__A)
            if (verbose):
                if (i % step == 0 or step == iterations):
                    print("Cost after {} iterations: {}".format(i, cost))
                    if i < iterations:
                        cost_list.append(cost)
        if (graph):
            x_list = np.arange(0, iterations, step)
            y_list = cost_list
            plt.plot(x_list, y_list)
            plt.xlabel('iteration')
            plt.ylabel('cost')
            plt.title('Training Cost')
            plt.show()

        return self.evaluate(X, Y)
