#!/usr/bin/env python3
"""
NST Class
"""

import numpy as np
import tensorflow as tf


class NST:
    """ class NST that performs tasks for neural style transfer
    style_layers
    """
    


    style_layers = ['block1_conv1',
                    'block2_conv1',
                    'block3_conv1',
                    'block4_conv1',
                    'block5_conv1']
    content_layer = 'block5_conv2'

    def __init__(self, style_image, content_image, alpha=1e4, beta=1):
        """ Constructor method
        Args:
            style_image - the image used as a style reference, stored
                as a numpy.ndarray
            content_image - the image used as a content reference,
                stored as a numpy.ndarray
            alpha - the weight for content cost
            beta - the weight for style cost
        """
        if not isinstance(style_image, np.ndarray) or len(style_image.shape) is not 3:
            raise TypeError('style_image must be a numpy.ndarray with shape (h, w, 3)')

        if not isinstance(content_image, np.ndarray) or len(style_image.shape) is not 3:
            raise TypeError('content_image must be a numpy.ndarray with shape (h, w, 3)')
        if alpha < 0:
            raise TypeError('alpha must be a non-negative number')
        if beta < 0:
            raise TypeError('beta must be a non-negative number')
        tf.enable_eager_execution()
        

        self.alpha = alpha
        self.beta = beta
        self.style_image = style_image
        self.content_image = content_image

    @staticmethod
    def scale_image(image):
        """ rescales an image such that its pixels values are between 0
            and 1 and its largest side is 512 pixels
        Args:
            image - a numpy.ndarray of shape (h, w, 3) containing the
                image to be scaled
        Returns: the scaled image
        """
        tf.enable_eager_execution()
        
        if not isinstance(image, np.ndarray) or len(image.shape) is not 3:
            raise TypeError('image must be a numpy.ndarray with shape (h, w, 3)')

        h, w, _ = image.shape

        # Height es el mas grande
        if h > w:
            h_new = 512
            w_new = (h_new * w) / h
        else:
            # w > h
            w_new = 512
            h_new = (w_new * h) / w

        resized_image = tf.image.resize_images(image, (h_new, w_new), method=tf.image.ResizeMethod.BICUBIC)
        return resized_image


