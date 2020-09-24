#!/usr/bin/env python3
"""
TripletLoss class that inherits from tensorflow.keras.layers.Layer
"""

import tensorflow
import tensorflow.keras as K


class TripletLoss(tensorflow.keras.layers.Layer):
    """
    custom layer class TripletLoss
    """

    def __init__(self, alpha, **kwargs):
        """ Constructor method
        Args:
            alpha - is the alpha value used to calculate the triplet loss
        """
        self.alpha = alpha
        super(TripletLoss, self).__init__(**kwargs)

    def triplet_loss(self, inputs):
        """ Calculate Triplet Loss
        Args:
            inputs -  is a list containing the anchor, positive and
                negative output tensors from the last layer of the model
        Returns:
            a tensor containing the triplet loss values
        Calc:
            max( Σ|| f(A)-f(P) ||²  - Σ|| f(A)-f(N) ||² + α, 0 )
        References:
            coursera.org/lecture/convolutional-neural-networks/triplet-loss-HuUtN
        """
        # Anchor, positive, negative
        a, p, n = inputs
        # Σ|| f(A)-f(P) ||²
        dist_a_p = K.backend.sum(K.backend.square(a - p), axis=-1)
        # Σ|| f(A)-f(N) ||²
        dist_a_n = K.backend.sum(K.backend.square(a - n), axis=-1)

        # max( Σ|| f(A)-f(P) ||²  - Σ|| f(A)-f(N) ||² + α, 0 )
        return (K.backend.maximum(dist_a_p - dist_a_n + self.alpha, 0))

    def call(self, inputs):
        """ Call Triplet Loss
        Args:
            inputs -  is a list containing the anchor, positive, and negative
                output tensors from the last layer of the model, respectively
            Returns:
                the triplet loss tensor
        """
        loss = self.triplet_loss(inputs)
        self.add_loss(loss)
        return loss
