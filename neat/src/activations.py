"""
* Author : Batyr Nuryyev
* Date   : Feb 22, 2019
"""

import numpy as np

""" Contains the list of activation functions for nodes. """

def sigmoid(x):
    return 1. / (1. + np.exp(-x))

def tanh(x):
    return np.tanh(x)

def relu(x):
    return np.maximum(0, x)
