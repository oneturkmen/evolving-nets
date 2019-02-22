"""
* Author : Batyr Nuryyev
* Date   : Feb 3, 2019
"""

import numpy as np
from itertools import count

class Connection:
    # Innovation number counter
    _ids = count(0)

    def __init__(self, in_node, out_node, weight, isEnabled = True):
        self.id = next(self._ids)
        self.in_node = in_node
        self.out_node = out_node
        self.weight = weight # TODO: change to numpy normal distrib
        self.isEnabled = True

    def mutateWeight(self):
        # TODO
        return 

    
    def mutateEnabled(self):
        # TODO
        return 
