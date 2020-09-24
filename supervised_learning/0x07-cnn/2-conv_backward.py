#!/usr/bin/env python3
"""
Convolutional Back Prop
"""

import numpy as np


def conv_backward(dZ, A_prev, W, b, padding="same", stride=(1, 1)):
    """
    Function that performs back propagation
    over a convolutional layer of a neural network:
    """
    _, h_new, w_new, c_new = dZ.shape
    m, h_prev, w_prev, c_prev = A_prev.shape
    kh, kw, c_prev, c_new = W.shape
    sh, sw = stride
    x = A_prev

    if padding == 'valid':
        p_h = 0
        p_w = 0

    if padding == 'same':
        p_h = int(np.ceil(((sh*h_prev) - sh + kh - h_prev) / 2))
        p_w = int(np.ceil(((sw*w_prev) - sw + kw - w_prev) / 2))

    db = np.sum(dZ, axis=(0, 1, 2), keepdims=True)

    x_padded = np.pad(x, [(0, 0), (p_h, p_h), (p_w, p_w), (0, 0)],
                      mode='constant', constant_values=0)

    dW = np.zeros_like(W)
    dx = np.zeros(x_padded.shape)

    for i in range(m):
        for h in range(h_new):
            for w in range(w_new):
                for k in range(c_new):
                    dx[i,
                       h*(sh):(h*(sh))+kh,
                       w*(sw):(w*(sw))+kw,
                       :] += dZ[i, h, w, k] * W[:, :, :, k]

                    dW[:, :,
                       :, k] += x_padded[i,
                                         h*(sh):(h*(sh))+kh,
                                         w*(sw):(w*(sw))+kw,
                                         :] * dZ[i, h, w, k]
    if padding == 'same':
        dx = dx[:, p_h:-p_h, p_w:-p_w, :]
    else:
        dx = dx

    return dx, dW, db
