import networkx as nx
import numpy as np
import matplotlib
import os

matplotlib.use('Agg')
import matplotlib.pyplot as plt

np.random.seed(0)

# Setting variables.
NUM_POINTS = 200 # Number of points in the graph.
P = 0.05 # Probability of creating edge between two nodes in graph.
MIN_STATE = -0.5 # Minimum state an agent can get to.
RESOURCE = 'random' # Resource = 'random' or 1.

DIR = './results/np_%d_p_%g_minstate_%g_res_%s' % (NUM_POINTS, P, MIN_STATE,
    RESOURCE)
if not os.path.exists(DIR):
    os.makedirs(DIR)

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

        assert RESOURCE in [1, 'random']

        if RESOURCE == 1:
            self.resource = 1
        elif RESOURCE == 'random':
            self.resource = np.random.random_sample()

        # State cannot exceed resource, but can be as low as MIN_STATE.
        self.state = min(self.resource, np.random.uniform(MIN_STATE, 1))

    def set_state(self, new_state):
        # New state is at most the resource.
        self.state = max(MIN_STATE, min(self.resource, self.state + new_state))

def generate_points():
    '''
    Generates a specified number of points with two attributes in [0, 1].
    '''
    G = nx.fast_gnp_random_graph(NUM_POINTS, P, seed=0)

    # Remap the nodes to Agent objects.
    mapping = {}
    for label in list(G.nodes):
        mapping[label] = Agent(label)
    return nx.relabel_nodes(G, mapping)

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
    plt.savefig('%s/random_graph_%d.png' % (DIR, i))
    plt.show()

def main():

    G = generate_points()

    pos = nx.spring_layout(G)
    plot_points(G, pos, 0)
    for i in range(15):
        transform(G)
        plot_points(G, pos, i)

if __name__ == '__main__':
    main()
