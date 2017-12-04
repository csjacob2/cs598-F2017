import networkx as nx
import numpy as np
import matplotlib


matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pylab

class Agent(object):
    """An agent in the network. It has the following properties:

    Attributes:
        label: An int specifying the node number.
        resource: A variable U(0, 1), the resource constraint of the agent.
        state: A float, determining how much the agent is documenting.
    """

    def __init__(self, label):
        """Return an Agent object whose resource is *resource* and state is
        *state*."""
        self.label = label
        self.resource = np.random.random_sample()
        # State is at most the resource.
        self.state = min(self.resource, np.random.uniform(-1, 1))

    def set_state(self, new_state):
        # New state is at most the resource.
        self.state = max(-1, min(self.resource, self.state + new_state))

def generate_points(num_points):
    '''
    Generates a specified number of points with two attributes in [0, 1].
    '''
    # TODO: 0.05 is the probability of forming an edge.
    G = nx.fast_gnp_random_graph(num_points, 0.05, seed=111111)

    # Remap the nodes to Agent objects.
    mapping = {}
    for label in list(G.nodes):
        mapping[label] = Agent(label)
    G = nx.relabel_nodes(G, mapping)

    return G

def transform(G):
    '''
    Transforms the graph for an iteration.
    '''
    new_states = {}
    # Compute the states.
    for node in G:
        # Get the state of the neighbors.
        state_lst = [n.state for n in G.neighbors(node)]
        degree = len(state_lst)
        # Each neighbor gets a vote of 1 / degree multiplied by its state.
        new_states[node] = sum([state / degree for state in state_lst])

    # Set the states.
    for node in new_states:
        node.set_state(new_states[node])

def plot_points(G, pos, i):
    '''
    Plots the graph.
    '''
    plt.figure(figsize=(8,8))

    nx.draw_networkx(G, node_size=100, with_labels=False, pos=pos,
        node_color=[n.state for n in G.nodes], cmap='Reds')
    print [n.state for n in G.nodes]

    plt.axis('off')
    plt.savefig('random_graph_%d.png' % i)
    plt.show()

def main():
    G = generate_points(100) # TODO: Number of nodes.

    pos = nx.spring_layout(G)
    plot_points(G, pos, 0)
    for i in range(10): # TODO: Number of iterations.
        transform(G)
        plot_points(G, pos, i)

if __name__ == '__main__':
    main()
