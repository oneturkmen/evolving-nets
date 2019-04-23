"""
* Author : Batyr Nuryyev
* Date   : Feb 22, 2019
"""

import numpy as np
from itertools import count

class Connection:
    """ Class that keeps the edges of neural network here along with 
        weights, and input and output nodes. The connection can either
        be enabled or disabled (for each run), and each weight can be
        either mutated or not.
    """

    # Innovation number counter
    innov_counter = count(1)

    # Table to keep the history of genes
    innovations = dict()


    def __init__(self, in_node, out_node, isEnabled = True):        
        self.in_node = in_node
        self.out_node = out_node
        self.weight = np.random.uniform(low = -2.0, high = 2.0)
        self.isEnabled = isEnabled

        # Historical markings: check if the connection gene has already been
        # somewhere.
        if (in_node, out_node) in self.innovations.keys():
            self.innov = self.innovations[(in_node, out_node)]
        else:
            innov_new = next(self.innov_counter)
            self.innovations[(in_node, out_node)] = innov_new
            self.innov = innov_new

    def is_enabled(self):
        return self.isEnabled

    def get_innov(self):
        """ Returns unique innovation number. """
        return self.innov

    def get_weight(self):
        """ Returns weight of the connection. """
        return self.weight
    
    def set_weight(self, w):
        """ Sets weight (not by mutation). Called by Genome.py."""
        self.weight = w
        return

    def get_in_node(self):
        """ Returns the starting node of the connection. """
        return self.in_node    

    def get_out_node(self):
        """ Returns the ending node of the connection. """
        return self.out_node

    def mutate_weight(self):
        """ Mutates the weight (assigns from a uniform distribution). """
        self.weight += np.random.uniform(low = -3.0, high = 3.0)
        return
    
    def toggle_enabled(self):
        """ Disables the connection if enabled; enables otherwise.
            This is only called when the new node is added to the
            genome.
        """
        self.isEnabled = not self.isEnabled        
        return 
