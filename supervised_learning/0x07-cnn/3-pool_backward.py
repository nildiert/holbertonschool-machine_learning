#!/usr/bin/env python3
"""
Pooling Back Prop mandatory
"""

import numpy as np


def pool_backward(dA, A_prev, kernel_shape, stride=(1, 1), mode='max'):
    """
    Function that performs back propagation over
    a pooling layer of a neural network
    """
    m, h_new, w_new, c_new = dA.shape
    m, h_prev, w_prev, c_prev = A_prev.shape
    k_h, k_w = kernel_shape
    sh, sw = stride

    dx = np.zeros_like(A_prev)

    for i in range(m):
        for h in range(h_new):
            for w in range(w_new):
                for k in range(c_new):
                    if mode == 'max':
                        aux = A_prev[i, h*sh:k_h+(h*sh),
                                     w*sw:k_w+(w*sw), k]
                        temp = (aux == np.max(aux))
                        dx[i,
                           h*(sh):(h*(sh))+k_h,
                           w*(sw):(w*(sw))+k_w,
                           k] += dA[i, h, w, k] * temp
                    if mode == 'avg':
                        dx[i,
                           h*(sh):(h*(sh))+k_h,
                           w*(sw):(w*(sw))+k_w,
                           k] += (dA[i, h, w, k])/k_h/k_w

    return dx
