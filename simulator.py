import networkx as nx
import numpy as np
import matplotlib
import os

matplotlib.use('Agg')
import matplotlib.pyplot as plt

np.random.seed(111)

# Setting variables.
NUM_POINTS = 200 # Number of points in the graph.
NUM_EDGES = 8 # Number of edges from new node to existing nodes.
RESOURCE = 'random' # Resource = 'random' or 1.

MAX_NUM_DOCS = 8 # Number of Documenters (maximum)
ENG_TO_DOC = 5 #Engineer to Documentor ratio

BETA = 0.5 # How much an agent discounts the utility of documentation.
DECAY = 0.1 # Fraction of documentation quality that decays each day.

assert RESOURCE in [1, 'random']

# Create results directory.
DIR = './results/np_%d_ne_%g_res_%s_mnd_%d_etd_%d_b_%g_d_%s' % (NUM_POINTS, NUM_EDGES,
    RESOURCE, MAX_NUM_DOCS, ENG_TO_DOC, BETA, DECAY)
if not os.path.exists(DIR):
    os.makedirs(DIR)

class Agent(object):
    """An agent in the network. It has the following properties:

    Attributes:
        label: An int specifying the node number.
        resource: A variable U(0, 1), the resource constraint of the agent.
        type: A variable, Documenter (D), Engineer (E), or Combo (C), which determines the type of job and constrains the state
            Documenter: only Documents, has state 1 always, has positive influence
            Engineer: only Engineers/works, has state 0 always, can have positive or negative influence
            Combo: works and documents, can have 1 or 0 state, can increase or decay and can have positive or negative influence but only on Engineers, not Documenters
        state: A float, determining how much the agent is documenting.
    """

    def __init__(self, label, agentType):
        """Return an Agent object whose resource is *resource* and state is
        *state*."""
        self.label = label
        self.agentType = agentType

        if RESOURCE == 1:
            self.resource = 1
        elif RESOURCE == 'random':
            self.resource = np.random.random_sample()

        if self.agentType == 'D' or self.agentType == 'E':
            self.state = 1
        else:
            # State cannot exceed resource.
            self.state = np.random.uniform(0, self.resource)
        #print (str(self.label) + ' ' + self.agentType + ' ' + str(self.state))
    
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

    ''' let's assume one Documentor replaces 10 Engineers (we can adjust this)
    assign the first Agent as a D, then the next DtoE Agents as Engineers, the rest as Combos
    until we run out of Agents/nodes
    we have a MAX_NUM_DOCS because a company may not have replacements for all workers
    '''
    
    DOCUMENTERS = 0
    ENGINEERS = 0
    TOTAL_DOCS = 0
    
    for label in list(G.nodes):
    
        if DOCUMENTERS == 0 and MAX_NUM_DOCS != TOTAL_DOCS:
            agentType = 'D'
            DOCUMENTERS += 1
            TOTAL_DOCS += 1
        elif ENGINEERS != ENG_TO_DOC:
            agentType = 'E'
            ENGINEERS+=1
        else:
            agentType = 'C'

        mapping[label] = Agent(label, agentType)
    
        # reset when we have to keep relabeling instead of filling up with Combo     
        if ENGINEERS == ENG_TO_DOC and MAX_NUM_DOCS != TOTAL_DOCS:  
            DOCUMENTERS = 0
            ENGINEERS = 0
    return nx.relabel_nodes(G, mapping)

def isWorking(node):
    '''
    The agent determines whether or not to work in this iteration.
        1. If the agent chooses to work, then it incurs a cost of 0.1. The
           benefit is BETA * 0.1, saving time in the future, but with present
           bias.
        2. If the agent chooses not to work, then it incurs a cost of its
           current progress * DECAY, since the documentation quality decays each
           day its held off. The benefit is 0.1, since the agent continues to work
           on their current tasks. Lazy utility is more desireable, but it will be influenced
           by the social network in the transform function.
    Documenters/Engineers always return True (isWorking == isDocumenting)
    '''

    work_utility = BETA * 0.1 - 0.1
    lazy_utility = 0.1 - DECAY * node.state

    if node.agentType == 'D' or node.agentType == 'E':
        return True
    elif work_utility > lazy_utility:
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
        #can only change self working status if C type
        #D and E agents are always "working"
        #get working neighbour status in an array

        neighbor_working_status = [working_status[n] for n in G.neighbors(node)]
        neighbor_agent_type = [n.agentType for n in G.neighbors(node)]

        if node.agentType == 'C':
            if neighbor_working_status.count(True) >= 0.5 * len(neighbor_working_status):
                #working neighbours directly influence positively to work
                node.set_state(node.state + 0.1)
            else:
                #not enough neighbours working nearby, current node decays
                node.set_state(node.state - DECAY)
                
def plot_points(G, pos, i):
    '''
    Plots the graph.
    '''
    norm = matplotlib.colors.Normalize(vmin=0, vmax=1)
    cmap = matplotlib.cm.Reds
    m = matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap)

    nx.draw_networkx(G, node_size=100, with_labels=False, pos=pos, node_color=[m.to_rgba(n.state) for n in G.nodes])

    plt.axis('off')
    plt.savefig('%s/random_graph_%d.png' % (DIR, i))
    plt.close()

def get_fraction_documenting(G):
    '''
    Gets the fraction of agents who are documenting. Agents are documenting well
    if their state is >= 0.5.
    '''
    node_lst = G.nodes
    return len([n for n in node_lst if n.state >= 0.5]) / float(len(node_lst))

def main():
    G = generate_points()
    fraction_documenting_lst = []
    # Get the layout of the graph.
    pos = nx.spring_layout(G)

    for i in range(15):
        plot_points(G, pos, i)

        fraction_documenting_lst += [get_fraction_documenting(G)]

        transform(G)
        #print [n.state for n in G.nodes()][:5]

    # Plot fraction of documenting agents at each iteration.
    plt.plot(fraction_documenting_lst)
    plt.savefig('%s/fraction_documenting.png' % DIR)

if __name__ == '__main__':
    main()