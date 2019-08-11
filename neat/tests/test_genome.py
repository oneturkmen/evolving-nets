import unittest

import sys
sys.path.append('../src')

from genome import Genome
from connectionGene import Connection
from graph import Graph
from activations import relu, sigmoid
from copy import deepcopy

class TestGenomeMethods(unittest.TestCase):

    def test_setting_layers(self):
        """ Tests correctness of layer formation operation. """

        # Setup
        genome = Genome()
        genome.initialize(2, 1)

        inputs = [1, 2]
        outputs = [6]
        connections = [
            Connection(1, 5),
            Connection(2, 5),
            Connection(4, 6),
            Connection(3, 6),
            Connection(5, 4),
            Connection(5, 3)
        ]

        genome.input_nodes = inputs
        genome.output_nodes = outputs
        genome.connection_genes = connections

        # Run
        graph = Graph()
        layers = graph.set_up_layers(
            genome.get_inputs(),
            genome.get_outputs(),
            genome.get_connections()
        )

        # Assert
        self.assertEqual(layers[0], { 5 })    # first non-input layer
        self.assertEqual(layers[1], { 3, 4 }) # second non-input layer
        self.assertEqual(layers[2], { 6 })    # output layer

    def test_adding_node(self):
        """ Tests adding node operation. """

        # Setup
        genome = Genome()
        genome.initialize(3, 1)

        genome_connections = genome.get_connections()

        # List of enabled (respectively, disabled) connections only.
        init_true_connections = list(
            filter(lambda a : a.is_enabled() == True, genome_connections)
        )
        init_false_connections = list(
            filter(lambda a : a.is_enabled() == False, genome_connections)
        )

        init_connections_length = len(genome_connections)

        # Run
        genome.add_node()

        # Assert
        self.assertTrue(
            len(list(filter(lambda x : x.is_enabled() == True, genome.get_connections()))),
            len(init_true_connections) + 1
        )
        self.assertTrue(
            len(list(filter(lambda x : x.is_enabled() == False, genome.get_connections()))),
            len(init_false_connections) + 1
        )
        self.assertEqual(len(genome.get_connections()), init_connections_length + 2)


    def test_adding_connection(self):
        """ Tests new connection addition operation. """

        # Setup
        genome = Genome()
        genome.initialize(3, 2)

        connections_new = [(1, 4), (1, 5), (2, 4), (3, 4)]
        genome.connection_genes = [ Connection(a, b) for a, b in connections_new]

        # Run
        connection = genome.add_connection()

        # Verify
        genome_connections = genome.get_connections()
        self.assertEqual(len(genome_connections), len(connections_new) + 1)
        
        self.assertTrue(all([isinstance(c, Connection) for c in genome_connections]))

    def test_mutating_weight(self):
        """ Tests connection weight mutation operation. """

        # Setup
        genome = Genome()
        genome.initialize(3, 2)

        genome_connections = deepcopy(genome.get_connections())

        # Run
        genome.mutate_connection_weights()

        # Verify
        self.assertEqual(len(genome_connections), len(genome.get_connections()))
        
        ok = False
        for i, conn in enumerate(genome.get_connections()):
            if (abs(genome_connections[i].get_weight() - conn.get_weight()) >= 0.000001):
                ok = True
                break

        self.assertTrue(ok)

        
    def test_distance_function(self):
        """ Tests the genome distance function 
            used for fitness normalization.
        """

        # Setup
        genome_a = Genome()
        genome_b = Genome()

        genome_a.initialize(3, 2)
        genome_b.initialize(3, 2)
        
        connections_new = [
            (1, 4), (2, 4), (1, 6), (2, 6), (3, 7), (6, 4), (6, 5), (7, 5)
        ]
        genome_b.connection_genes = [ Connection(a, b) for a, b in connections_new]

        # Run
        distance = genome_a.distance(genome_b)
        
        # Verify
        self.assertEqual(distance, (4, 6, 8))   # Manually calculated


if __name__ == '__main__':
    unittest.main()

