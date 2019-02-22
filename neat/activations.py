"""
* Author : Batyr Nuryyev
* Date   : Feb 22, 2019
"""

import numpy as np

class Activations:
    """
        Contains the list of activation functions for nodes.
    """

    def sigmoid(self, x):
        return 1. / (1. + np.exp(-x))
    
    def tanh(self, x):
        return np.tanh(x)

    def relu(self, x):
        return np.max(0, x)