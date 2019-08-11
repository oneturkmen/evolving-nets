"""
* Author : Batyr Nuryyev
* Date   : March 14, 2019
"""

"""
The following file will contain the reproduction class/methods such as
crossover. Selection will be done in population.py and mutation will be
called here after selection.
"""

import operator
import numpy as np
from copy import deepcopy
from genome import Genome

# For testing
from connectionGene import Connection



# For comparisons of floats
EPSILON = 0.0000001

# Helper for comparison
def equal(a, b):
    return abs(a - b) < EPSILON


# TODO: have to thoroughly test this function (edge cases and load testing)
def crossover(p, q):
    """ This function defines the crossover operation. For more details,
        please refer to NEAT paper (cited in the report).

        It expects p and q to be instances of Genome.
    """
    assert isinstance(p, Genome), "First parent is not an instance of Genome!"
    assert isinstance(q, Genome), "Second parent is not an instance of Genome!"
    assert p.get_inputs() == q.get_inputs(), "Input nodes of parents do not match!"
    assert p.get_outputs() == q.get_outputs(), "Output nodes of parents do not match!"

    # P - parent 1 (an instance of Genome)
    # Q - parent 2 (an instance of Genome)

    # Process genes
    A_score = p.get_score()
    B_score = q.get_score()

    # Sort the connection genes by innovation number
    A_genes = sorted(p.get_connections(), key = operator.attrgetter('innov'))
    B_genes = sorted(q.get_connections(), key = operator.attrgetter('innov'))

    # Single child
    offspring = []

    # Pointers for left and right genomes
    i = 0
    j = 0

    while i < len(A_genes) and j < len(B_genes):
        A_i_innov = A_genes[i].get_innov()
        B_j_innov = B_genes[j].get_innov()

        if A_i_innov == B_j_innov:
            # Choose randomly
            if np.random.rand() > 0.5:
                #print("Accepting A")
                offspring.append(deepcopy(A_genes[i]))
            else:
                #print("Accepting B")
                offspring.append(deepcopy(B_genes[j]))

            # Increment both pointers
            i += 1
            j += 1
        elif A_i_innov < B_j_innov:
            # Append left disjoint connection only if left genome scored higher
            # If scored equally, choose randomly
            if A_score > B_score:
                offspring.append(A_genes[i])
            elif equal(A_score, B_score):
                if np.random.rand() > 0.5:
                    offspring.append(deepcopy(A_genes[i]))
                else:
                    offspring.append(deepcopy(B_genes[j]))

            # Increment only left pointer
            i += 1
        elif A_i_innov > B_j_innov:
            # Append right disjoint connection only if right genome scored higher
            # If scored equally, choose randomly
            if B_score > A_score:
                offspring.append(deepcopy(B_genes[j]))
            elif equal(A_score, B_score):
                if np.random.rand() > 0.5:
                    offspring.append(deepcopy(A_genes[i]))
                else:
                    offspring.append(deepcopy(B_genes[j]))

            # Increment only right pointer
            j += 1

    # Choose excess numbers randomly if scored equally
    # Otherwise, choose from the fitter parent
    if A_score > B_score and i < len(A_genes):
        offspring += deepcopy(A_genes[i:])
    elif B_score > A_score and j < len(B_genes):
        offspring += deepcopy(B_genes[j:])
    elif equal(A_score, B_score) and abs(len(A_genes) - len(B_genes)) > 0:
        # If scores are the same yet one of the genomes is bigger, choose
        # excess connections randomly
        while i < len(A_genes):
            if np.random.rand() > 0.5:
                offspring.append(deepcopy(A_genes[i]))

            i += 1

        while j < len(B_genes):
            if np.random.rand() > 0.5:
                offspring.append(deepcopy(B_genes[j]))

            j += 1

    # Prepare inputs, outputs, and connections and make a new genome
    # Does not matter which parent's inputs/outputs we take (all constant)
    inputs = p.get_inputs()
    outputs = p.get_outputs()

    genome_offspring = Genome()
    genome_offspring.set_genome_crossover(inputs, outputs, offspring)

    return genome_offspring
