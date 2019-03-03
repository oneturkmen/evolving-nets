"""
* Author : Batyr Nuryyev
* Date   : Feb 22, 2019
"""
from itertools import count

class Node:

    def __init__(self, category = 1):
        self.category = category # 0 - input, 1 - hidden, 2 - output
    
    def get_category(self):
        return self.category