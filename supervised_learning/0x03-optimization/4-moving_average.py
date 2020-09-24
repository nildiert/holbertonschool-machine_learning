#!/usr/bin/env python3
"""
Function that calculates the weighted
moving average of a data set
"""

import numpy as np


def moving_average(data, beta):
    """
    Calculates the weighted moving average of a data set
    """
    moving_avg = []
    epsilon = 1 - beta
    V_t = 0
    for index in range(len(data)):
        V_t = (beta * V_t) + (epsilon * data[index])
        bias_correction = 1 - (beta ** (index + 1))
        V_t_corrected = V_t / bias_correction
        moving_avg.append(V_t_corrected)
    return moving_avg
