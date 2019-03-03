"""
* Author : Batyr Nuryyev
* Date   : Feb 22, 2019
"""

from connectionGene import Connection

class Graph:
    """
        This class dynamically constructs a directed graph, 
        implements forward propagation of data and keeps 
        the evaluated fitness score.
    """
    
    def required_for_output(self, inputs, outputs, connections):
        """
            Returns a set of nodes that are required to compute
            the output of a neural network.
        """
        # Process the connections first
        edges = [
            (connection.get_in_node(), connection.get_out_node())
            for connection in connections
        ]
        
        # edges = connections

        # Get the set with all required nodes
        required = set(outputs)
        s = set(outputs)

        # Start from the back (i.e. from output nodes)
        while True:
            # Goes backward (i.e. from output layer to input layer)
            t = set(
                a for (a,b) in edges if b in s and a not in s
            )
            if not t:
                break
            
            # Keep all non-input nodes (including output nodes)
            layer_nodes = set(x for x in t if x not in inputs)
            if not layer_nodes:
                break
            
            required = required.union(layer_nodes)
            s = s.union(t)

        return required


    def set_up_layers(self, inputs, outputs, connections):
        """
            Sets up layers of a neural network given input nodes, 
            output nodes, and edges between them (if any).

            Returns a list of sets, where each set in position *i* 
            represents a layer *i + 1*.
        """

        # Process the connections first
        edges = [
            (connection.get_in_node(), connection.get_out_node())
            for connection in connections
        ]
        
        # edges = connections

        # Gets nodes that are required for computing output
        required = self.required_for_output(inputs, outputs, connections)

        layers = []
        s = set(inputs)

        while True:
            # Get next nodes
            c = set(b for (a,b) in edges if a in s and b not in s)
            t = set()

            # Add to the layers only if a node is required
            for n in c:
                if n in required and all(a in s for (a,b) in edges if b == n):
                    t.add(n)
            
            if not t:
                break
            
            # Append newly-constructed layer (as a set) to the list
            layers.append(t)
            s = s.union(t)

        return layers


    def forward_propagate(self, data, inputs, outputs, connections):
        """
            Dynamically constructs a directed graph and forward 
            propagates the data to get the output (e.g. dog or cat).
        """
        # Sanity check
        assert len(data) == len(inputs), "Data and input layer have different dimensions"

        # Contains initially 0th (input) layer
        layer_activations = [set(zip(inputs, data))]

        # Get a list of sets of nodes (set of nodes at 
        # position i represents single layer i)
        layers = self.set_up_layers(inputs, outputs, connections)

        if not layers:
            return layer_activations[0]

        for i, layer in enumerate(layers, start = 1):
            
            # Current layer activation
            curr_layer_activation = set()
            
            # For each node in a current layer
            for node in layer:

                # Get the node's activation in total
                node_activation = sum([
                    prev_a * c.get_weight() + 1
                    for (prev_node, prev_a) in layer_activations[i-1]
                    for c in connections
                    if prev_node == c.get_in_node()
                        and node == c.get_out_node()
                        and c.is_enabled()
                ])

                # Add to the list of activations of the current layer
                curr_layer_activation.add(                    
                    (node, node_activation)
                )
            
            # Keep each layer's activations together
            layer_activations.append(curr_layer_activation)
        
        # Return last layer's (output layer) activations
        return layer_activations[-1]
        
    def show_thyself():
        # TODO : kerascheto za tuk molya zavkshti
        # Visualizes a NN's architecture
        return


# TESTING
# TODO: remove when testing is done
inputs = [1,2]
outputs = [3,4]
connections = [Connection(1,3), Connection(2,3), Connection(2,4), Connection(1,4)]
data = [10, -10]
print([c.get_weight() for c in connections])
graph = Graph()
#testing = graph.forward_propagate(data, inputs, outputs, [], connections)
#print(testing)

# print("Actual calculations ... ")
# calculi1 = data[0] * connections[0].get_weight() + 1 + data[1] * connections[1].get_weight() + 1
# calculi2 = data[0] * connections[3].get_weight() + 1 + data[1] * connections[2].get_weight() + 1
# print(calculi1)
#print(calculi2)

print(graph.forward_propagate(data, inputs, outputs, connections))