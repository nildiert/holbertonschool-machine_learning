#!/usr/bin/env python3
"""
Convolutional Forward Prop
"""

import numpy as np


def conv_forward(A_prev, W, b, activation, padding="same", stride=(1, 1)):
    """
    Function that performs forward
    propagation over a convolutional layer
    of a neural network
    """
    m, h_prev, w_prev, c_prev = A_prev.shape
    kh, kw, c_prev, c_new = W.shape
    sh, sw = stride

    if padding is 'same':
        p_h = int(((sh * h_prev)
                   - sh + kh - h_prev) / 2)
        p_w = int(((sw * w_prev)
                   - sw + kw - w_prev) / 2)
    if padding is 'valid':
        p_h, p_w = 0, 0

    A_prev = np.pad(
        array=A_prev,
        pad_width=[(0, 0), (p_h, p_h), (p_w, p_w), (0, 0)],
        mode='constant',
        constant_values=0)
    out_h = int(((h_prev + (2 * p_h) - kh) / sh) + 1)
    out_w = int(((w_prev + (2 * p_w) - kw) / sw) + 1)
    out_conv = np.zeros((m, out_h, out_w, c_new))
    prev_m_A = np.arange(0, m)

    for h in range(out_h):
        for w in range(out_w):
            for k in range(c_new):
                out_conv[prev_m_A, h, w, k] = activation(
                    (np.sum(np.multiply(
                        A_prev[prev_m_A, h*sh:kh+h*sh, w*sw:kw+w*sw],
                        W[:, :, :, k]),
                            axis=(1, 2, 3))) + b[0, 0, 0, k])

    return out_conv
