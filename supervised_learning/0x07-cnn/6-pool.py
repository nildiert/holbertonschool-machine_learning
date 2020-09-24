#!/usr/bin/env python3
"""
Function that performs pooling on images:
"""

import numpy as np


def pool(images, kernel_shape, stride, mode='max'):
    """
    Function that performs pooling on images
    """

    m, image_h, image_w, channels = images.shape
    kernel_h, kernel_w = kernel_shape

    s_h, s_w = stride
    output_h = int(((image_h - kernel_h) / s_h) + 1)
    output_w = int(((image_w - kernel_w) / s_w) + 1)
    conv_output = np.zeros((m, output_h, output_w, channels))

    img_m = np.arange(m)

    for i in range(output_h):
        for j in range(output_w):
            if mode == 'max':
                conv_output[img_m, i, j] = np.max(
                    images[
                        img_m,
                        i*s_h:kernel_h+(i*s_h),
                        j*s_w:kernel_w+(j*s_w)], axis=(1, 2))
            if mode == 'avg':
                conv_output[img_m, i, j] = np.mean(
                    images[
                        img_m,
                        i*s_h:kernel_h+(i*s_h),
                        j*s_w:kernel_w+(j*s_w)], axis=(1, 2))
    return conv_output
