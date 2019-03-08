"""
* Author : Batyr Nuryyev
* Date   : Feb 22, 2019
"""

from node import Node
from connectionGene import Connection
from itertools import product
from operator import itemgetter
from graph import Graph
from visual import visualize
import numpy as np

class Genome:
    """
        You can think of genome as a class containing 
        two lists: node genes and connection genes;
        this class also provides methods for mutation
        of each of those.
        Pretty much the same as neural network, but
        quite dynamic (i.e. can change weights and topology).
    """

    def __init__(self):
        self.input_nodes = []
        self.output_nodes = []
        self.connection_genes = []
        self.score = 0

    ############ - Private methods - ############

    def __addNode(self):
        # TODO
        # MAKE SURE NOT TO ALLOW STALE NEURON IN
        # i.e. one with a single incoming/outgoing connection
        # so each neuron should have at least 2 incoming/outgoing
        # connections
        return

    def __addConnection(self):
        # TODO
        # Make sure cycle is not created either
        return

    ############ - Public methods - ############

    def get_inputs(self):
        return self.input_nodes
    

    def get_outputs(self):
        return self.output_nodes

    def get_connections(self):
        return self.connection_genes

    def forward_propagate(self):
        # TODO: Integrate with graph.py here

        # Return!
        return

    def reset_score():
        """
            Resets the score.
        """
        self.score = 0

    def get_score():
        """
            Returns evaluated fitness score.
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
        self.input_nodes  = [i+1 for i in range(num_inputs)]
        self.output_nodes = [i+1 for i in range(num_inputs, num_outputs + num_inputs)]

        # Initialize connections
        connections = [
            Connection(sensor, output)
            for sensor in self.input_nodes
                for output in self.output_nodes
        ]
        self.connection_genes = connections

        return connections

    def mutate(self):
        # TODO
        # Mutate add connection
        # Mutate add node
        #
        # In adding a new node, the connection gene being split is disabled, 
        # and two new connection genes are added to the end of the genome. 
        # The new node is between the two new connections. A new node gene
        # representing this new node is added to the genome as well.
        #
        return

    def phenotype(self):
        return visualize(self.input_nodes, self.output_nodes, self.connection_genes)
    


# -------------------------- TESTING ----------------------------
# TODO: Delegate to unit testing framework
testing = False
if testing:
    # For testing purposes
    genome = Genome()
    xs = [
        (connection.get_in_node(), connection.get_out_node()) 
        for connection in genome.initialize(3, 1)
    ]

    inputs = [1,2]
    outputs = [6]
    connections = [(1,5),(2,5),(4,6),(3,6),(5,4),(5,3)]

    graph = Graph()
    print(graph.set_up_layers(genome.get_inputs(), genome.get_outputs(), genome.get_connections()))
