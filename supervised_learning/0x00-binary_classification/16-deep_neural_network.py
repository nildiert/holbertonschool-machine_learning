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
        self.L = len(layers)
        self.cache = {}
        self.weights = {}
        for index, layer in enumerate(layers):
            if type(layer) is not int or layer < 0:
                raise TypeError("layers must be a list of positive integers")
            if index is 0:
                init_value = np.random.randn(layer, nx)*np.sqrt(2/nx)
                self.weights['W' + str(index + 1)] = init_value
            if index > 0:
                init_value1 = np.random.randn(layer, layers[index - 1])
                init_value2 = np.sqrt(2/layers[index - 1])
                self.weights['W' + str(index + 1)] = init_value1 * init_value2
            self.weights['b' + str(index + 1)] = np.zeros((layer, 1))
