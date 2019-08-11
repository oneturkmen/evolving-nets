"""
* Author : Batyr Nuryyev
* Date   : March 8, 2019 (Happy International Women's Day! :) )
"""

import networkx as nx
import matplotlib.pyplot as plt

def visualize(inputs, outputs, connections):
    """ Visualizes the neural network architecture along with weights. """
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
        with_labels = True,
        font_weight = 'bold'
    )

    plt.show()
    #plt.savefig("hi.png") is necessary?

    return
