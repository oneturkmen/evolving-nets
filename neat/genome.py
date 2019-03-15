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
    """ You can think of genome as a class containing
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
        self.initialized = False

    ############ - Private methods - ############

    def mutate_connection_weights(self):
        """Weight mutation: perturbs the weights of all connections"""
        for connection in self.connection_genes:
            connection.mutate_weight()
        return

    def add_node(self):
        """ Structural mutation: Adds node in-between some edge """
        assert self.initialized == True, "Genome should be first initialized!"
        assert len(self.connection_genes) > 0, "Genome cannot not have connections!"
        
        # Randomly select an edge
        c = np.random.choice(self.connection_genes)

        # Make sure connection is enabled
        while not c.is_enabled():
            c = np.random.choice(self.connection_genes)
        
        # Create new node with the id = max(greatest_node_id) + 1
        nodes = list(set(
            [z.get_in_node()  for z in self.connection_genes] + 
            [z.get_out_node() for z in self.connection_genes]
        ))
        new_node = max(nodes) + 1
        
        # Create incoming connection and append to the list of existing ones
        left_connection = Connection(c.get_in_node(), new_node)
        left_connection.set_weight(1)
        self.connection_genes.append(left_connection)

        # Create outgoing connection and append to the list of existing ones
        right_connection = Connection(new_node, c.get_out_node())
        right_connection.set_weight(c.get_weight())
        self.connection_genes.append(right_connection)

        # Disable old connection
        c.toggle_enabled()

        return

    def add_connection(self):
        """ Structural mutation: Adds an edge between previously unconnected nodes """
        assert self.initialized == True, "Genome should be first initialized!"
        assert len(self.connection_genes) > 0, "Genome cannot not have connections!"

        # Input and output nodes of this genome
        inputs = self.input_nodes
        outputs = self.output_nodes

        # Get connections as a tuple of incoming and outcoming edge
        connections = [
            (c.get_in_node(), c.get_out_node()) for c in self.connection_genes
        ]

        # Conditions for availability:
        # Let a be input node, b be output node.
        # Find all (a, b) such that:
        # - a and b are not both input nodes
        # - a and b are not both output nodes
        # - a is not an output node
        # - b is not an input node

        # Find all available pairs of nodes where a connection can be established        
        available_pairs = list(set( 
            (a, b) for a in inputs for b in outputs
            if a not in outputs and b not in inputs and (a, b) not in connections 
        ))
        
        # TODO: this is ugly. Either make static class w/ static methods
        # or just keep bare functions in the file.
        graph = Graph()

        # For all pairs, make sure cycle is not created
        valid_pairs = list(set(
            pair for pair in available_pairs if not graph.creates_cycle(connections, pair)
        ))

        # If no connection can be established, just return False (for testing).        
        if len(valid_pairs) <= 0:
            return False

        # Select random pair and make a connection out of it        
        choice = valid_pairs[ np.random.choice(len(valid_pairs)) ]
        new_connection = Connection(choice[0], choice[1])

        # Add to the list of existing connections of this genome
        self.connection_genes.append(new_connection)
        
        return new_connection

    ############ - Public methods - ############

    def get_inputs(self):
        assert self.initialized == True, "Genome should be first initialized!"
        return self.input_nodes

    def get_outputs(self):
        assert self.initialized == True, "Genome should be first initialized!"
        return self.output_nodes

    def get_connections(self):
        assert self.initialized == True, "Genome should be first initialized!"
        return self.connection_genes

    def forward_propagate(self):
        assert self.initialized == True, "Genome should be first initialized!"
        # TODO: Integrate with graph.py here

        # Return!
        return

    def reset_score():
        """ Resets the score. """
        # Make sure it's initialized
        assert self.initialized == True, "Genome should be first initialized!"

        self.score = 0

    def get_score():
        """ Returns evaluated fitness score. """
        # Make sure it's initialized
        assert self.initialized == True, "Genome should be first initialized!"

        return self.score

    def initialize(self, num_inputs, num_outputs):
        """ Initializes the input and output nodes along
            with weights. 
        """
        assert num_inputs > 0, "Dimension of inputs cannot be less than 1"
        assert num_outputs > 0, "Dimension of outputs cannot be less than 1"
        assert num_inputs >= num_outputs, "Dimensions of inputs and outputs are incorrect"

        # Append inputs and outputs
        self.input_nodes = [i+1 for i in range(num_inputs)]
        self.output_nodes = [
            i+1 for i in range(num_inputs, num_outputs + num_inputs)
        ]

        # Initialize connections
        connections = [
            Connection(sensor, output)
            for sensor in self.input_nodes
            for output in self.output_nodes
        ]
        self.connection_genes = connections

        # Set it as initialized
        self.initialized = True

        return connections

    def mutate(self):
        """ Runs both structural and weight mutations. """
        assert self.initialized == True, "Genome should be first initialized!"

        # Weight mutation
        if np.random.rand() < 0.10:
            self.mutate_connection_weights()
        
        # Structural mutation: adding a node
        if np.random.rand() < 0.10:
            self.add_node()

        # Structural mutation: adding a connection (edge)
        if np.random.rand() < 0.10:
            self.add_connection()

        return

    def phenotype(self):
        """ Genotype to phenotype mapping (gets a graph visualization) """
        assert self.initialized == True, "Genome should be first initialized!"
        assert len(self.input_nodes) > 0, "There are no input nodes!"
        assert len(self.output_nodes) > 0, "There are no output nodes!"
        assert len(self.connection_genes) > 0, "There are no connection genes!"

        return visualize(self.input_nodes, self.output_nodes, self.connection_genes)


# -------------------------- TESTING ----------------------------
# TODO: Delegate to unit testing framework
testing = True
if testing:
    # ------------- TESTS ------------------
    def test_setting_layers():
        genome = Genome()
        xs = [
            (connection.get_in_node(), connection.get_out_node())
            for connection in genome.initialize(3, 1)
        ]

        inputs = [1, 2]
        outputs = [6]
        connections = [(1, 5), (2, 5), (4, 6), (3, 6), (5, 4), (5, 3)]

        graph = Graph()
        print(graph.set_up_layers(genome.get_inputs(),
                                genome.get_outputs(), genome.get_connections()))
    
    def test_adding_node():
        print("Testing adding node")

        genome = Genome()
        genome.initialize(3,1)

        # 1 adding node
        print(genome.get_inputs())
        print(genome.get_outputs())
        print([
            (c.get_in_node(), c.get_out_node(), c.is_enabled(), c.get_weight())
            for c in genome.get_connections()
        ])

        genome.add_node()
        print(genome.get_inputs())
        print(genome.get_outputs())
        print([
            (c.get_in_node(), c.get_out_node(), c.is_enabled(), c.get_weight())
            for c in genome.get_connections()
        ])        

    def test_adding_connection():
        genome = Genome()
        genome.initialize(3,2)

        print("Inputs = {0}".format(genome.get_inputs()))
        print("Outputs = {0}".format(genome.get_outputs()))
        print("Old connections = {0}".format([
            (c.get_in_node(), c.get_out_node(), c.is_enabled())
            for c in genome.get_connections()
        ]))

        connections_new = [(1, 4), (1, 5), (2, 4), (3, 4)]        
        genome.connection_genes = [ Connection(a, b) for a, b in connections_new]

        # Print the ids so far
        print("Connections IDs = {0}".format([
            c.get_innov()
            for c in genome.get_connections()
        ]))
        
        # add a connection
        connection = genome.add_connection()

        # print new connection's details
        print(connection.get_innov())
        print(connection.get_in_node())
        print(connection.get_out_node())
        print(connection.get_weight())
        
        # verify addition of a connection
        print([
            (c.get_in_node(), c.get_out_node(), c.is_enabled())
            for c in genome.get_connections()
        ])

        # TODO: good. now we just have to process it a bit and make sure ids of connections are unique!

    def test_mutating_weight():
        genome = Genome()
        genome.initialize(3,2)
        print("Connections (before mutation) \n {0}".format([
            (c.get_in_node(), c.get_out_node(), c.is_enabled(), c.get_weight())
            for c in genome.get_connections()
        ]))

        genome.mutate_connection_weights()

        print("Connections (after mutation) \n {0}".format([
            (c.get_in_node(), c.get_out_node(), c.is_enabled(), c.get_weight())
            for c in genome.get_connections()
        ]))

        print("Connections IDs = {0}".format([
            c.get_innov()
            for c in genome.get_connections()
        ]))
    

    # ----------- RUNNING TESTS -----------    
    test_adding_node()
    test_adding_connection()
    test_mutating_weight()
