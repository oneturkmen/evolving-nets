"""
* Author : Batyr Nuryyev
* Date   : Feb 22, 2019
"""

from connectionGene import Connection

class Graph:
    """
        This class dynamically constructs a directed graph, 
        implements forward propagation of data and keeps 
        the evaluated fitness score
    """
    
    def required_for_output(self, inputs, outputs, connections):
        # Process the connections first
        # edges = [
        #     (connection.get_in_node(), connection.get_out_node())
        #     for connection in connections
        # ]
        edges = connections

        # Get the set with all required nodes
        required = set(outputs)
        s = set(outputs)

        while 1:
            t = set(
                a for (a,b) in edges if b in s and a not in s
            )
            if not t:
                break
            
            layer_nodes = set(x for x in t if x not in inputs)
            if not layer_nodes:
                break
            
            required = required.union(layer_nodes)
            s = s.union(t)

        return required


    def set_up_layers(self, inputs, outputs, connections):
        # Process the connections first
        # edges = [
        #     (connection.get_in_node(), connection.get_out_node())
        #     for connection in connections
        # ]
        edges = connections

        required = self.required_for_output(inputs, outputs, connections)

        layers = []
        s = set(inputs)

        while 1:
            c = set(b for (a,b) in edges if a in s and b not in s)
            t = set()

            for n in c:
                if n in required and all(a in s for (a,b) in edges if b == n):
                    t.add(n)
            
            if not t:
                break
            
            layers.append(t)
            s = s.union(t)

        return layers


    def forward_propagate(self, data, inputs, outputs, layers, connections):
        # TODO
        # TODO: revise the name for this
        assert len(data) == len(inputs), "Data and input layer have different dimensions"

        # Contains initially 0th (input) layer
        layer_activations = [set(zip(inputs, data))]

        # Get a list of sets of nodes (set of nodes at 
        # position i represents single layer i)
        layers = self.set_up_layers(inputs, outputs, connections)

        for i, layer in enumerate(layers, start = 1):
            
            activation = set()
            
            # example: layer = [{4},{5,6},{3}]
            for node in layer:
                activation.add(
                    (node, sum(
                        [
                            prev_a * 2 + 1
                            for (prev_node, prev_a) in layer_activations[i-1]
                            for c in connections
                            if prev_node == c[0]
                                and node == c[1]
                                and True
                        ]
                    ))
                )
            
            layer_activations.append(activation)
        
        return layer_activations
        #return layer_activations[-1]
        # # For direct (input-output) propagation
        # if not layers:
        #     output_activations = []

        #     for out in outputs:
        #         out_activation = [
        #             c.get_weight() * a + 1
        #             for c in connections
        #             for (n,a) in input_activations
        #             if c.get_in_node() == n and c.get_out_node() == out
        #                 and c.is_enabled()
        #         ]
        #         out_activation = sum(out_activation)

        #         # Append one output activation
        #         output_activations.append((out, out_activation))

        #     return output_activations
    
    def show_thyself():
        # TODO : kerascheto za tuk molya zavkshti
        # Visualizes a NN's architecture
        return


# TESTING
# TODO: remove when testing is done
"""inputs = [1,2]
outputs = [3,4]
connections = [Connection(1,3), Connection(2,3), Connection(2,4), Connection(1,4)]
data = [10, -10]
print([c.get_weight() for c in connections])
graph = Graph()
testing = graph.forward_propagate(data, inputs, outputs, [], connections)
print(testing)

print("Actual calculations ... ")
calculi1 = data[0] * connections[0].get_weight() + 1 + data[1] * connections[1].get_weight() + 1
calculi2 = data[0] * connections[3].get_weight() + 1 + data[1] * connections[2].get_weight() + 1
print(calculi1)
print(calculi2)"""

graph = Graph()
inputs = [1,2]
data = [10, -10]
outputs = [6]
connections = [(1,5),(2,5),(4,6),(3,6),(5,4),(5,3)]
layers = [{5}, {3,4}, {6}]

print(graph.forward_propagate(data, inputs, outputs, layers, connections))