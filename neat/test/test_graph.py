import unittest

import sys
sys.path.append('../src')

from connectionGene import Connection
from graph import Graph
from activations import relu, sigmoid

class TestGraphMethods(unittest.TestCase):

    def test_forward_propagation(self):
        # Tests forward propagation operation on a direct acyclic graph.
        inputs = [1, 2]
        outputs = [ 3 ]
        connections = [
            Connection(1, 5),
            Connection(2, 5),
            Connection(5, 3),
            Connection(2, 3)
        ]

        # Set connection weights
        connections[0].weight = 2
        connections[1].weight = 3
        connections[2].weight = 4
        connections[3].weight = 5

        # Input data
        data = [10, -10]

        graph = Graph()
        result = graph.forward_propagate(data, inputs, outputs, connections)

        self.assertEqual(result[0], 3)
        self.assertTrue(abs(result[1] - 5.242885663363464e-22) < 1e-6)

    def test_forward_propagation_manual(self):
        # Tests forward propagation by comparing to manual calculations.

        # Setup
        data = [2, 2, 2]
        inputs = [1, 2, 3]
        outputs = [7]
        connections = [
            Connection(1, 5),
            Connection(2, 5),
            Connection(2, 4),
            Connection(3, 4),
            Connection(4, 5),
            Connection(4, 6),
            Connection(5, 6),
            Connection(6, 7)
        ]

        w0 = 1.5
        w1 = 2
        w2 = 2.5
        w3 = -1.5
        w4 = -2
        w5 = 3
        w6 = 3
        w7 = 0.5

        connections[0].weight = w0
        connections[1].weight = w1
        connections[2].weight = w2
        connections[3].weight = w3
        connections[4].weight = w4
        connections[5].weight = w5
        connections[6].weight = w6
        connections[7].weight = w7


        # Manual calculations
        a4 = relu((w2 * data[1] + 1) + (w3 * data[2] + 1))
        a5 = relu((w0 * data[0] + 1) + (w1 * data[1] + 1) + (w4 * a4 + 1))
        a6 = relu((w6 * a5 + 1) + (w5 * a4 + 1))
        a7 = sigmoid((w7 * a6 + 1))

        # Run
        graph = Graph()
        a7_actual = graph.forward_propagate(data, inputs, outputs, connections)

        # Assert
        # TODO: Precision problem
        self.assertTrue(abs(a7_actual[1] - a7) < 0.0005)


if __name__ == '__main__':
    unittest.main()
