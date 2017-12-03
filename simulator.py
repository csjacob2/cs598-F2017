import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

def generate_points(num_points):
    '''
    Generates a specified number of points with two attributes in [0, 1].
    '''
    G = nx.random_geometric_graph(num_points, 0.1) # TODO: 0.1 is the prob. of creating an edge.
    return G

def transform(G):
    '''
    Transforms the graph for an iteration.
    '''
    # Examples:
    # G.add_edge(1, 2)
    # G.remove_edge()
    #G.add_edge(1,
    print('transforming')

def plot_points(G):
    '''
    Plots the graph.
    '''
    pos=nx.get_node_attributes(G, 'pos')
    plt.figure(figsize=(8,8))
    nx.draw_networkx_edges(G,pos,nodelist=G.nodes(), alpha=0.4)
    nx.draw_networkx_nodes(G,pos,nodelist=G.nodes(), node_size=50)

    plt.xlim(-0.05,1.05)
    plt.ylim(-0.05,1.05)
    plt.axis('off')
    plt.savefig('random_geometric_graph.png')
    plt.show()

def main():
    G = generate_points(500) # TODO: Number of nodes.

    for i in range(10000): # TODO: Number of iterations here.
        transform(G)

    plot_points(G)

if __name__ == '__main__':
    main()
