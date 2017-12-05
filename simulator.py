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
# MIN_STATE = 0 # Minimum state an agent can get to.
RESOURCE = 'random' # Resource = 'random' or 1.
BETA = 0.5
DECAY = 1

assert RESOURCE in [1, 'random']

# Create results directory.
# DIR = './results/np_%d_p_%g_minstate_%g_res_%s' % (NUM_POINTS, P, MIN_STATE,
#     RESOURCE)
DIR = './results/np_%d_p_%g_res_%s_b_%g_d_%g' % (NUM_POINTS, P, RESOURCE, BETA,
    DECAY)
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

        # State cannot exceed resource, but can be as low as MIN_STATE.
        # Randomly sample the state, since agents may be at various stages of
        # documentation.
        # self.state = min(self.resource, np.random.uniform(MIN_STATE, 1))
        self.state = min(self.resource, np.random.uniform(0, 1))

    def set_state(self, new_state):
        # New state is at most the resource.
        # self.state = max(MIN_STATE, min(self.resource, self.state + new_state))
        self.state = min(self.resource, new_state)

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
        # node.set_state(node.state + 0.1)
        return True
    return False
    # else:
        # node.set_state(DECAY * node.state)

def transform(G):
    '''
    Transforms the graph for an iteration.
    '''
    # new_states = {}
    working_status = {}
    # Compute the states.
    for node in G:
        # Get the state of the neighbors.
        # state_lst = [n.state for n in G.neighbors(node)]
        # Each neighbor gets a vote of 1 / degree multiplied by its state.
        # new_states[node] = sum([state / degree for state in state_lst])
        # pressure = sum([s for s in state_lst]) / len(state_lst)
        # The agent can only choose to work if its neighbors are on average more
        # than halfway done.
        working_status[node] = isWorking(node)

    for node in working_status:
        neighbor_working_status = [working_status[n] for n in G.neighbors(node)]

        if sum(neighbor_working_status) >= 0.1 * 0.5 * len(neighbor_working_status):
            node.set_state(node.state + 0.1)
        else:
            node.set_state(DECAY * node.state)

    print [n.state for n in G.nodes()]
    # # Set the states.
    # for node in new_states:
    #     node.set_state(new_states[node])

def plot_points(G, pos, i):
    '''
    Plots the graph.
    '''
    plt.figure(figsize=(8, 8))

    nx.draw_networkx(G, node_size=100, with_labels=False, pos=pos,
        node_color=[n.state for n in G.nodes], cmap='Reds')

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

    plot_points(G, pos, 0)
    fraction_documenting_lst += [get_fraction_documenting(G)]
    for i in range(15):
        transform(G)
        plot_points(G, pos, i)
        fraction_documenting_lst += [get_fraction_documenting(G)]

    # Plot fraction of documenting agents at each iteration.
    plt.plot(fraction_documenting_lst)
    plt.savefig('%s/fraction_documenting.png' % DIR)

if __name__ == '__main__':
    main()