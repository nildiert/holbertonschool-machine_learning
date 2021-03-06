#!/usr/bin/env python3
""" This class defines a neural deep network with one hidden layer """

import numpy as np


class DeepNeuralNetwork():
    """
    This class defines a deep neural network with some hidden
    layer performing binary classification
    """

    def __init__(self, nx, layers):
        """ Constructor method """
        if type(nx) is not int:
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if type(layers) is not list:
            raise TypeError("layers must be a list of positive integers")
        if len(layers) is 0:
            raise TypeError("layers must be a list of positive integers")

        self.nx = nx
        self.layers = layers
        self.__L = len(layers)
        self.__cache = {}
        self.__weights = {}
        for index, layer in enumerate(layers):
            if type(layer) is not int or layer < 0:
                raise TypeError("layers must be a list of positive integers")
            if index is 0:
                init_value = np.random.randn(layer, nx)*np.sqrt(2/nx)
                self.__weights['W' + str(index + 1)] = init_value
            if index > 0:
                init_value1 = np.random.randn(layer, layers[index - 1])
                init_value2 = np.sqrt(2/layers[index - 1])
                self.__weights['W' + str(index + 1)] = init_value1*init_value2
            self.__weights['b' + str(index + 1)] = np.zeros((layer, 1))

    @property
    def L(self):
        """Getter method for L attribute"""
        return self.__L

    @property
    def cache(self):
        """Getter method for cache attribute"""
        return self.__cache

    @property
    def weights(self):
        """Getter method for weights attribute"""
        return self.__weights

    def forward_prop(self, X):
        """Calculates the forward propagation of the neural network"""
        for index in range(self.__L + 1):
            if index is 0:
                self.__cache['A' + str(index)] = X
            else:
                W_current = self.__weights['W' + str(index)]
                b_current = self.__weights['b' + str(index)]
                A_preview = self.__cache['A' + str(index - 1)]
                mult_mat = (np.matmul(W_current, A_preview)) + b_current
                self.__cache['A' + str(index)] = 1 / (1 + np.exp(-mult_mat))
        return (self.__cache['A' + str(self.__L)], self.__cache)

    def cost(self, Y, A):
        """ Calculates the cost of the model using logistic regression """
        m = Y.shape[1]
        cost = (-1/m) * np.sum(Y * np.log(A) + (1-Y) * (np.log(1.0000001 - A)))
        return cost

    def evaluate(self, X, Y):
        """ Evaluates the neuron’s predictions """
        A, self.__cache = self.forward_prop(X)
        eva = np.where(A >= 0.5, 1, 0)
        cost = self.cost(Y, A)
        return (eva, cost)

    def gradient_descent(self, Y, cache, alpha=0.05):
        """ Calculates one pass of gradient descent on the network """
        weights_copy = self.__weights.copy()

        m = Y.shape[1]

        A3 = self.__cache['A' + str(self.__L)]
        A2 = self.__cache['A' + str(self.__L - 1)]
        W3 = weights_copy['W' + str(self.__L)]
        b3 = weights_copy['b' + str(self.__L)]
        dz_List = {}
        dz3 = A3 - Y

        dz_List['dz'+str(self.__L)] = dz3
        dw3 = (1/m) * np.matmul(A2, dz3.T)
        db3 = (1/m) * np.sum(dz3, axis=1, keepdims=True)
        self.__weights['W'+str(self.__L)] = W3 - (alpha * dw3).T
        self.__weights['b'+str(self.__L)] = b3 - (alpha * db3)

        for index in range(self.__L - 1, 0, -1):
            A_curr = self.__cache['A'+str(index)]
            A_bef = self.__cache['A'+str(index - 1)]
            W_curr = weights_copy['W'+str(index)]
            W_next = weights_copy['W'+str(index + 1)]
            b_curr = weights_copy['b'+str(index)]
            dz1 = np.matmul(W_next.T, dz_List['dz'+str(index + 1)])
            dz2 = A_curr * (1 - A_curr)
            dz = dz1 * dz2
            dw = (1/m) * np.matmul(A_bef, dz.T)
            db = (1/m) * np.sum(dz, axis=1, keepdims=True)
            dz_List['dz'+str(index)] = dz
            self.__weights['W'+str(index)] = W_curr - (alpha * dw).T
            self.__weights['b'+str(index)] = b_curr - (alpha * db)
