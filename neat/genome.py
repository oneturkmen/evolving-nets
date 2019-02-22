"""
* Author : Batyr Nuryyev
* Date   : Feb 22, 2019
"""

from node import Node
from connectionGene import Connection
from itertools import product
from operator import itemgetter
import numpy as np

class Genome:
    """
        You can think of genome as a class containing 
        two lists: node genes and connection genes;
        this class also provides methods for mutation
        of each of those.
        Pretty much the same as neural network, but
        quite dynamic (i.e. can change weights and topology)
    """

    def __init__(self):
        self.node_genes = []
        self.connection_genes = []
        self.score = 0

    ############ - Private methods - ############

    def __addNode(self):
        # TODO
        return

    def __addConnection(self):
        # TODO
        return

    ############ - Public methods - ############

    def forward_propagate(self):
        # Align (sort) connections
        connections = [(1,5),(2,5),(3,4),(1,4),(5,4)]

        sorted_connections = sorted(
            connections, key = lambda c : (c[0])
        )

        # Propagate the weights
        # Get the output
        # Return!
        return sorted_connections



    def reset_score():
        """
            Resets the score
        """
        self.score = 0

    def get_score():
        """
            Returns evaluated fitness score
        """
        return self.score    

    def initialize(self, num_inputs, num_outputs):
        """
            Initializes the input and output nodes along
            with weights.
        """
        assert num_inputs > 0, "Dimension of inputs cannot be less than 1"
        assert num_outputs > 0, "Dimension of outputs cannot be less than 1"
        assert num_inputs >= num_outputs, "Dimensions of inputs and outputs are incorrect"

        # Append inputs and outputs
        self.node_genes = [Node(0) for i in range(num_inputs)]
        self.node_genes = self.node_genes + [Node(2) for i in range(num_outputs)]

        # Initialize connections
        sensor_nodes = list(filter(lambda n : n.category == 0, self.node_genes))
        output_nodes = list(filter(lambda n : n.category == 2, self.node_genes))
        connections = [
            Connection(sensor, output) 
            for sensor in range(1,len(sensor_nodes)+1) for output in range(1,len(output_nodes)+1)
        ]
        self.connection_genes = connections

        return connections # TODO: remove after testing


    def mutate(self):
        # TODO
        # Mutate add connection
        # Mutate add node
        #
        # In adding a new node, the
        # connection gene being split is disabled, and two new connection genes are added to the
        # end of the genome. The new node is between the two new connections. A new node gene
        # representing this new node is added to the genome as well.
        #
        return
    

# For testing purposes
genome = Genome()
#xs = [(connection.get_in_node(), connection.get_out_node()) for connection in genome.initialize(3, 1)]
#print(xs)
print(genome.forward_propagate())