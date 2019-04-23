"""
* Author : Batyr Nuryyev
* Date   : Feb 22, 2019
"""

from connectionGene import Connection
from itertools import product
from operator import itemgetter
from graph import Graph
from visual import visualize
import numpy as np

from activations import relu, sigmoid

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
        self.graph = Graph()

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
        falsity_flag = 0
        while not c.is_enabled():
            c = np.random.choice(self.connection_genes)
            falsity_flag += 1

            if falsity_flag > 20:
                print(">>> [WARNING]: Cannot choose connection!")
                return
                
        # Create new node with the i = max(greatest_node_id) + 1
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
        
        # For all pairs, make sure cycle is not created
        valid_pairs = list(set(
            pair for pair in available_pairs 
            if not self.graph.creates_cycle(connections, pair)
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

    def action(self, data):
        assert self.initialized == True, "Genome should be first initialized!"
        assert len(self.input_nodes) > 0, "List of inputs is empty!"
        assert len(self.output_nodes) > 0, "List of outputs is empty!"
        assert len(self.connection_genes) > 0, "List of connections is empty!"

        # Pass relay to Graph
        output = self.graph.forward_propagate(
            data, self.input_nodes, self.output_nodes, self.connection_genes
        )

        # Output === aggregate activation from the last layer
        #print("OUTPUT = {0}".format(output))

        # Get only the output part
        return 1 if output[1] >= 0.5 else 0

    def reset_score(self):
        """ Resets the score. """        
        assert self.initialized == True, "Genome should be first initialized!"

        self.score = 0

    def get_score(self):
        """ Returns evaluated fitness score. """        
        assert self.initialized == True, "Genome should be first initialized!"

        return self.score

    def add_score(self, score):
        """ Adds the score (reward) from the environment. """
        assert self.initialized == True, "Genome should be first initialized!"
        self.score += score

        return
    
    def set_genome_crossover(self, inputs, outputs, connections):
        """ Gets a new genome through crossover. Here, initialization is not 
            necessary since we will for sure have some connections. It assures
            that this function gets a list of connections """        
        assert type(connections) == list, "Connections should be of type list!"
        assert type(inputs) == list, "Inputs should be a list"
        assert type(outputs) == list, "Inputs should be a list"        
        assert len(inputs) > 0, "List of inputs cannot be non-empty!"
        assert len(outputs) > 0, "List of outputs cannot be non-empty!"
        assert len(inputs) >= len(outputs), ("Dimensions of inputs"
            "and outputs are incorrect")
        assert len(connections) > 0, "List of connections should be non-empty!"        
        assert all([isinstance(c, Connection) for c in connections]), ("Connections "
            "should be a list of instances of Connection!")

        # Set the inputs, outputs, and connections
        self.connection_genes = connections
        #print("xoxoxo {0}".format([c.is_enabled() for c in self.connection_genes]))
        self.input_nodes = inputs
        self.output_nodes = outputs
        self.score = 0

        # For anomalies (how come non-existent genome passed initialization?!)
        if self.initialized == True:
            print("[ WARNING ] in set_connections_crossover() \
                >>> Genome is already initialized!")        

        # Initialize in the end
        self.initialized = True

        return

    def initialize(self, num_inputs, num_outputs):
        """ Initializes the input and output nodes along with weights. """
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
        # if np.random.rand() < 0.10:
        #     self.mutate_connection_weights()        

        # Structural mutation: adding a node
        if np.random.rand() < 0.10:
            self.add_node()

        # Structural mutation: adding a connection (edge)
        if np.random.rand() < 0.10:
            self.add_connection()

        # Toggle enabled/disabled
        for conn in self.connection_genes:
            if np.random.rand() < 0.05:
                conn.toggle_enabled()

        return

    def phenotype(self):
        """ Genotype to phenotype mapping (gets a graph visualization) """
        assert self.initialized == True, "Genome should be first initialized!"
        assert len(self.input_nodes) > 0, "There are no input nodes!"
        assert len(self.output_nodes) > 0, "There are no output nodes!"
        assert len(self.connection_genes) > 0, "There are no connection genes!"

        return visualize(self.input_nodes, self.output_nodes, self.connection_genes)

    def distance(self, other):
        """ Computes the representation distance between two genomes """
        assert self.initialized == True, "Genome should be first initialized!"
        assert isinstance(other, Genome), "Other genome is not an instance of Genome!"
        # Get their innovations
        innovations_a = [c.get_innov() for c in self.get_connections()]
        innovations_b = [c.get_innov() for c in other.get_connections()]

        # Find the excess level
        N = max(len(innovations_a), len(innovations_b))
        excess_threshold = min(max(innovations_a), max(innovations_b))

        # Find number of disjoint nodes
        num_disjoint = 0
        num_excess = 0

        # Accumulator weight differences
        W_bar = 0
        W_c = 0

        # Count number of disjoint nodes from parent B
        for id in innovations_a:
            if id not in innovations_b:
                if id > excess_threshold:
                    num_excess += 1
                else:
                    num_disjoint += 1

        # Count number of disjoint nodes from parent A
        for id in innovations_b:
            if id not in innovations_a:
                if id > excess_threshold:
                    num_excess += 1
                else:
                    num_disjoint += 1
        
        # Get the weight differences: TODO
        return (num_disjoint, num_excess, N)


# -------------------------- TESTING ----------------------------
# TODO: Delegate to unit testing framework
testing = False
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
    
    def test_forward_prop_delegation():
        print("------------------- TEST 2 ------------------------")
        genome = Genome()
        genome.initialize(3, 1)

        data = [2, 2, 2]
        inputs = [1, 2, 3]
        outputs = [7]
        connections = [Connection(1,5), Connection(2,5), Connection(2,4), Connection(3,4),
                    Connection(4,5), Connection(4,6), Connection(5,6), Connection(6,7)]        

        genome.input_nodes = inputs
        genome.output_nodes = outputs
        genome.connection_genes = connections

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

        a4 = relu((w2 * data[1]) + (w3 * data[2]) + 1)
        a5 = relu((w0 * data[0]) + (w1 * data[1]) + (w4 * a4) + 1)
        a6 = relu((w6 * a5) + (w5 * a4) + 1)
        a7 = sigmoid((w7 * a6) + 1)
        
        a7_actual = genome.action(data)
        print("Actual: {0}".format(a7_actual))
        print("Expected: {0}".format(a7))
        
        status = True if (abs(a7_actual[1] - a7) < 0.000001) else False

        print("Status: {0}".format(status))

    
    def test_distance_function():
        genome_a = Genome()
        genome_b = Genome()

        genome_a.initialize(3,2)
        genome_b.initialize(3,2)

        print("Inputs A = {0}".format(genome_a.get_inputs()))
        print("Outputs A = {0}".format(genome_a.get_outputs()))
        print("Connections A = {0}".format([
            (c.get_in_node(), c.get_out_node(), c.is_enabled())
            for c in genome_a.get_connections()
        ]))

        connections_new = [
            (1, 4), (2, 4), (1, 6), (2, 6), (3, 7), (6, 4), (6, 5), (7, 5)
        ]
        genome_b.connection_genes = [ Connection(a, b) for a, b in connections_new]

        # Print the ids so far
        print("Connections IDs for genome A = {0}".format([
            c.get_innov()
            for c in genome_a.get_connections()
        ]))
        print("Connections IDs for genome B = {0}".format([
            c.get_innov()
            for c in genome_b.get_connections()
        ]))
        
        # Find the distance
        distance = genome_a.distance(genome_b)
        print("Distance between 2 nodes is {0}".format(distance))


    # ----------- RUNNING TESTS -----------    
    # test_adding_node()
    # test_adding_connection()
    # test_mutating_weight()
    # test_forward_prop_delegation()
    # test_distance_function()
