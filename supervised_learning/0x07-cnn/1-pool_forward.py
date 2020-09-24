#!/usr/bin/env python3
"""
Function that performs pooling on images:
"""

import numpy as np


def pool_forward(A_prev, kernel_shape, stride=(1, 1), mode='max'):
    """
    Function that performs pooling on images
    """

    m, h_prev, w_prev, c_prev = A_prev.shape
    kh, kw = kernel_shape

    sh, sw = stride
    out_h = int(((h_prev - kh) / sh) + 1)
    out_w = int(((w_prev - kw) / sw) + 1)
    conv_output = np.zeros((m, out_h, out_w, c_prev))

    img_m = np.arange(m)

    for i in range(out_h):
        for j in range(out_w):
            if mode == 'max':
                conv_output[img_m, i, j] = np.max(
                    A_prev[
                        img_m,
                        i*sh:kh+(i*sh),
                        j*sw:kw+(j*sw)], axis=(1, 2))
            if mode == 'avg':
                conv_output[img_m, i, j] = np.mean(
                    A_prev[
                        img_m,
                        i*sh:kh+(i*sh),
                        j*sw:kw+(j*sw)], axis=(1, 2))
    return conv_output
