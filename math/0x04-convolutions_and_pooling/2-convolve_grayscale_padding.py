#!/usr/bin/env python3
"""
Function that performs a convolution on grayscale images with custom padding
"""

import numpy as np


def convolve_grayscale_padding(images, kernel, padding):
    """
    Function that performs a convolution on grayscale
    images with custom padding
    """
    m, image_h, image_w = images.shape
    kernel_h, kernel_w = kernel.shape

    padding_h, padding_w = padding
    output_h = image_h + (2 * padding_h) - kernel_h + 1
    output_w = image_w + (2 * padding_w) - kernel_w + 1

    conv_output = np.zeros((m, output_h, output_w))

    img_m = np.arange(0, m)
    images = np.pad(
        images,
        [(0, 0), (padding_h, padding_h), (padding_w, padding_w)],
        mode='constant',
        constant_values=0)

    for i in range(output_h):
        for j in range(output_w):
            multiply = images[img_m, i:kernel_h+i, j:kernel_w+j]
            conv_output[img_m, i, j] = np.sum(
                np.multiply(multiply, kernel), axis=(1, 2))
    return conv_output
