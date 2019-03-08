"""
* Author : Batyr Nuryyev
* Date   : March 8, 2019 (Happy International Women's Day! :) )
"""

import networkx as nx
import matplotlib.pyplot as plt

"""
    Visualizes the neural network architecture along with weights
"""

def visualize(inputs, outputs, connections):
    DG = nx.DiGraph()

    # Add input and output nodes
    for n in (inputs + outputs):
        DG.add_node(n)
    
    # Add edges (corresponding nodes are automatically added)
    # and get hidden layer nodes
    hiddens = []

    for c in connections:
        DG.add_edge(c.get_in_node(), c.get_out_node())

        if c.get_in_node() not in inputs + hiddens + outputs:
            hiddens.append(c.get_in_node())

        if c.get_in_node() not in inputs + hiddens + outputs:
            hiddens.append(c.get_in_node())
    
    # Match the colors (green for inputs, red for outputs, the rest is blue)
    colors = (['g'] * len(inputs)) + (['r'] * len(outputs)) + (['b'] * len(hiddens))
    
    nx.draw(DG,
        node_color = colors,
        with_labels = True
        font_weight = 'bold'
    )

    plt.show()
    #plt.savefig("hi.png") is necessary?

    return


# Only for testing purposes
def test():
    DG = nx.DiGraph()
    inputs = [1, 2, 3, 4, 5, 6]
    outputs = [ 7 ]
    for i in (inputs + outputs):
        DG.add_node(i)

    connections = [
        #(1,5),(2,5),(2,4),(3,4),(4,5),(4,6),(5,6),(6,7)
        (1,8),(2,8),(3,9),(3,10),(3,11),(4,11),
        (5,11),(5,12),(5,13),(6,13),(8,9),(10,9),
        (10,11),(13,12),(12,11),(9,7),(11,7),(12,7)
    ]
    for c in connections:
        DG.add_edge(*c)

    nx.draw(DG, 
        node_color = ['g','g','g','g','g','g','r','b','b','b','b','b','b'], 
        with_labels = True,
        font_weight = 'bold'        
    )    
    plt.show()