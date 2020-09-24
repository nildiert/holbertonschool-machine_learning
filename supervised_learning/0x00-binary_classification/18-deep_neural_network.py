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
