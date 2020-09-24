#!/usr/bin/env python3
"""
Function that performs a convolution on images using multiple kernels
"""

import numpy as np


def convolve(images, kernels, padding='same', stride=(1, 1)):
    """
    Function that performs a convolution on images using multiple kernels
    """
    m, image_h, image_w, channels = images.shape
    kernel_h, kernel_w, kernel_c, kernel_nc = kernels.shape
    stride_h, stride_w = stride

    if isinstance(padding, tuple):
        padding_h, padding_w = padding
    if padding is 'same':
        padding_h = int(((stride_h * image_h)
                        - stride_h + kernel_h - image_h) / 2) + 1
        padding_w = int(((stride_w * image_w)
                        - stride_w + kernel_w - image_w) / 2) + 1
    if padding is 'valid':
        padding_h, padding_w = 0, 0

    output_h = int(((image_h + (2 * padding_h) - kernel_h) / stride_h) + 1)
    output_w = int(((image_w + (2 * padding_w) - kernel_w) / stride_w) + 1)
    conv_output = np.zeros((m, output_h, output_w, kernel_nc))

    img_m = np.arange(0, m)

    images = np.pad(
        images,
        [(0, 0), (padding_h, padding_h), (padding_w, padding_w), (0, 0)],
        mode='constant',
        constant_values=0)
    for i in range(output_h):
        for j in range(output_w):
            for k in range(kernel_nc):
                s_h = (stride_h)
                s_w = (stride_w)
                multiply = images[
                    img_m,
                    i*s_h:kernel_h+i*s_h,
                    j*s_w:kernel_w+j*s_w]
                conv_output[img_m, i, j, k] = np.sum(
                    np.multiply(multiply, kernels[:, :, :, k]), axis=(1, 2, 3))
    return conv_output
