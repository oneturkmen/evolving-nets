"""
* Author : Batyr Nuryyev
* Date   : Feb 22, 2019
"""

from activations import sigmoid, relu

class Graph:
    """ This class dynamically constructs a directed graph, 
        implements forward propagation of data and keeps 
        the evaluated fitness score.
    """    

    def creates_cycle(self, connections, new_connection):
        """ Checks if a new connection introduces a cycle
            into the graph (graph without cycles is assumed)

            Ref:
            https://github.com/CodeReclaimers/neat-python/blob/master/neat/graphs.py
        """
        i, o = new_connection
        if i == o:
            return True
        
        visited = {o}
        while True:
            num_added = 0
            for a, b in connections:
                if a in visited and b not in visited:                    
                    if b == i:
                        return True
                    
                    visited.add(b)
                    num_added += 1
            
            if num_added == 0:
                return False

    
    def required_for_output(self, inputs, outputs, connections):
        """ Returns a set of nodes that are required to compute
            the output of a neural network.

            Ref:
            https://github.com/CodeReclaimers/neat-python/blob/master/neat/graphs.py
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
                a for (a, b) in edges if b in s and a not in s
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
        """ Sets up layers of a neural network given input nodes, 
            output nodes, and edges between them (if any).

            Returns a list of sets, where each set in position *i* 
            represents a layer *i + 1*.

            Ref:
            https://github.com/CodeReclaimers/neat-python/blob/master/neat/graphs.py
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
            c = set(b for (a, b) in edges if a in s and b not in s)
            t = set()

            # Add to the layers only if a node is required
            for n in c:
                if n in required and all(a in s for (a, b) in edges if b == n):
                    t.add(n)
            
            if not t:
                break
            
            # Append newly-constructed layer (as a set) to the list
            layers.append(t)
            s = s.union(t)

        return layers


    def forward_propagate(self, data, inputs, outputs, connections):
        """ Dynamically constructs a directed graph and forward 
            propagates the data to get the output (e.g. dog or cat).
        """
        # Sanity check
        assert len(data) == len(inputs), "Data and input layer have different dimensions"

        # Contains initially 0th (input) layer
        layer_activations = list(zip(inputs, data))

        # Get a list of layers (each layer as a set of nodes)
        layers = self.set_up_layers(inputs, outputs, connections)
        
        # If direct network, then return
        if not layers:
            print("WARNING: No layers in the network!")
            print([(c.get_in_node(), c.get_out_node(), c.is_enabled()) for c in connections])
            return layer_activations[0]

        # Accumulate activations from the first (input) layer
        for i, layer in enumerate(layers, start = 1):

            # For each node in the current layer
            for node in layer:         
                # Add bias (==1) to the sum of previous activations
                node_activation = 1 + sum([
                    prev_a * c.get_weight()
                    for (prev_node, prev_a) in layer_activations
                    for c in connections
                    if prev_node == c.get_in_node()
                        and node == c.get_out_node()
                        and c.is_enabled()
                ])

                # Run through activation function and add to the list of activations
                if i == len(layers):
                   layer_activations.append((node, sigmoid(node_activation)))
                else:
                   layer_activations.append((node, relu(node_activation)))

        # Return last layer's (output layer) activations
        return layer_activations[-1]    



