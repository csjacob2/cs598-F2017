import networkx as nx
import numpy as np
import matplotlib
import os

matplotlib.use('Agg')
import matplotlib.pyplot as plt

np.random.seed(111)

# Setting variables.
NUM_POINTS = 200 # Number of points in the graph.
NUM_EDGES = 3 # Number of edges from new node to existing nodes.
RESOURCE = 1 # Resource = 'random' or 1.
BETA = 0.5 # How much an agent discounts the utility of documentation.
DECAY = 0.1 # Fraction of documentation quality that decays each day.

assert RESOURCE in [1, 'random']

# Create results directory.
DIR = './results/np_%d_ne_%g_res_%s_b_%g_d_%g' % (NUM_POINTS, NUM_EDGES,
    RESOURCE, BETA, DECAY)
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

        if RESOURCE == 1:
            self.resource = 1
        elif RESOURCE == 'random':
            self.resource = np.random.random_sample()

        # State cannot exceed resource.
        self.state = np.random.uniform(0, self.resource)

    def set_state(self, new_state):
        # New state is at most the resource.
        # self.state = max(MIN_STATE, min(self.resource, self.state + new_state))
        self.state = min(self.resource, new_state)

def generate_points():
    '''
    Generates a specified number of points with two attributes in [0, 1].
    '''
    G = nx.barabasi_albert_graph(NUM_POINTS, NUM_EDGES, seed=111)

    # Remap the nodes to Agent objects.
    mapping = {}
    for label in list(G.nodes):
        mapping[label] = Agent(label)
    return nx.relabel_nodes(G, mapping)

def isWorking(node):
    '''
    The agent determines whether or not to work in this iteration.

        1. If the agent chooses to work, then it incurs a cost of 0.1. The
           benefit is BETA * 0.1, saving time in the future, but with present
           bias.
        2. If the agent chooses not to work, then it incurs a cost of its
           current progress * DECAY, since the documentation quality decays each
           day its held off. The benefit is 0.1, since the agent gets that time
           to do other things.
    '''
    work_utility = BETA * 0.1 - 0.1
    lazy_utility = 0.1 - DECAY * node.state

    if work_utility > lazy_utility:
        return True
    return False

def transform(G):
    '''
    Transforms the graph for an iteration.
    '''
    working_status = {}
    # Compute the states.
    for node in G:
        working_status[node] = isWorking(node)

    for node in working_status:
        neighbor_working_status = [working_status[n] for n in G.neighbors(node)]
        if neighbor_working_status.count(True) >= 0.5 * len(neighbor_working_status):
            node.set_state(node.state + 0.1)
        else:
            node.set_state((1 - DECAY) * node.state)

def plot_points(G, pos, i):
    '''
    Plots the graph.
    '''
    norm = matplotlib.colors.Normalize(vmin=0, vmax=1)
    cmap = matplotlib.cm.Reds
    m = matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap)

    nx.draw_networkx(G, node_size=100, with_labels=False, pos=pos,
        node_color=[m.to_rgba(n.state) for n in G.nodes])

    plt.axis('off')
    plt.savefig('%s/random_graph_%d.png' % (DIR, i))
    plt.close()

def get_fraction_documenting(G):
    '''
    Gets the fraction of agents who are documenting. Agents are documenting well
    if their state is >= 0.9.
    '''
    node_lst = G.nodes
    return len([n for n in node_lst if n.state >= 0.9]) / float(len(node_lst))

def main():
    G = generate_points()

    fraction_documenting_lst = []

    # Get the layout of the graph.
    pos = nx.spring_layout(G)

    for i in range(15):
        plot_points(G, pos, i)
        fraction_documenting_lst += [get_fraction_documenting(G)]

        transform(G)
        # print [n.state for n in G.nodes()][:5]

    # Plot fraction of documenting agents at each iteration.
    plt.plot(fraction_documenting_lst)
    plt.savefig('%s/fraction_documenting.png' % DIR)

if __name__ == '__main__':
    main()