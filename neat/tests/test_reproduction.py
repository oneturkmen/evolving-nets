import unittest

import sys
sys.path.append('../src')

from genome import Genome
from connectionGene import Connection
from reproduction import crossover

class TestReproductionMethods(unittest.TestCase):

    def test_crossover(self):
        """ Tests crossover operation. Can only be verified manually. """

        # Setup
        genome1 = Genome()
        genome1.initialize(3, 1)

        genome2 = Genome()
        genome2.initialize(3, 1)

        connections = [
            Connection(1, 5),
            Connection(2, 5),
            Connection(3, 5),
            Connection(5, 4),
            Connection(2, 4),
            Connection(3, 4)
        ]
        connections[0].weight = 2
        connections[1].weight = 3
        connections[2].weight = 4
        connections[3].weight = 5
        connections[4].weight = 6
        connections[5].weight = 7

        genome2.connection_genes = connections

        genome1.add_score(25)
        genome2.add_score(25)

        # Run
        offspring = crossover(genome1, genome2)

        # Verify
        genome1_connections = genome1.get_connections()
        genome2_connections = genome2.get_connections()

        offspring_connections = offspring.get_connections()

        self.assertTrue(len(genome1_connections) + len(genome2_connections) >= len(offspring_connections))


if __name__ == '__main__':
    unittest.main()
