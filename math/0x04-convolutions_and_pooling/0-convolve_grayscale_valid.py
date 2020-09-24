#!/usr/bin/env python3
"""
Function that performs a valid convolution on grayscale images
"""

import numpy as np


def convolve_grayscale_valid(images, kernel):
    """
    Function that performs a valid convolution on grayscale images
    """
    m = images.shape[0]
    image_h = images.shape[1]
    image_w = images.shape[2]
    kernel_h = kernel.shape[0]
    kernel_w = kernel.shape[1]

    output_h = image_h - kernel_h + 1
    output_w = image_w - kernel_w + 1

    conv_output = np.zeros((m, output_h, output_w))

    img = np.arange(m)
    for i in range(output_h):
        for j in range(output_w):
            multiply = images[img, i:kernel_h+i, j:kernel_w+j]
            conv_output[img, i, j] = np.sum(
                np.multiply(multiply, kernel), axis=(1, 2))
    return conv_output
