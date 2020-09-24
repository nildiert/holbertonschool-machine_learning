#!/usr/bin/env python3
"""
Function that performs a same convolution on grayscale images
"""

import numpy as np


def convolve_grayscale_same(images, kernel):
    """
    Function that performs a same convolution
    on grayscale images
    """
    m = images.shape[0]
    image_h = images.shape[1]
    image_w = images.shape[2]
    kernel_h = kernel.shape[0]
    kernel_w = kernel.shape[1]

    if (kernel_h % 2) == 0:
        padding_h = int((kernel_h) / 2)
        output_h = image_h - kernel_h + (2 * padding_h)
    else:
        padding_h = int((kernel_h - 1) / 2)
        output_h = image_h - kernel_h + 1 + (2 * padding_h)

    if (kernel_w % 2) == 0:
        padding_w = int((kernel_w) / 2)
        output_w = image_w - kernel_w + (2 * padding_w)
    else:
        padding_w = int((kernel_w - 1) / 2)
        output_w = image_w - kernel_w + 1 + (2 * padding_w)

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
