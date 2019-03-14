"""
* Author : Batyr Nuryyev
* Date   : March 14, 2019
"""

"""
The following class will contain the population of genomes,
evaluation of each through fitness function (openAI gym in this case),

SELECTION: Reproduction happens in a separate class
MUTATION:  Each genome will mutate on its own. Function is called in Selection

* Population
* Variables:
    - genomes
    - generation
    - environment
    - constants c1/c2 for speciation?
"""

# TODO
class Population:
    def __init__(self):
        self.genomes = []
        self.best_genome = []
