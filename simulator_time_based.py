import argparse
import matplotlib
from multiprocessing import Pool
import networkx as nx
import numpy as np
import operator
import os

matplotlib.use('Agg')
import matplotlib.pyplot as plt

# np.random.seed(111)

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

        # if RESOURCE == 1:
        #     self.resource = 1
        # elif RESOURCE == 'random':
        #     self.resource = np.random.random_sample()

        # State cannot exceed resource.
        # self.state = np.random.uniform(0, self.resource)
        self.state = np.random.random_sample()

        self.threshold = np.random.random_sample()

    def set_state(self, new_state):
        # New state is at most the resource.
        # self.state = max(MIN_STATE, min(self.resource, self.state + new_state))
        self.state = min(1, new_state)

def generate_points(num_pts):
    '''
    Generates a specified number of points with two attributes in [0, 1].
    '''
    # G = nx.barabasi_albert_graph(num_pts, NUM_EDGES, seed=9305)
    G = nx.barabasi_albert_graph(num_pts, NUM_EDGES)

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

def force_work(working_status, G, k):
    '''
    Force nodes to alter their working states for different methods.
    '''
    if args.method == 'nntu':
        deg_lst = nx.classes.function.degree(G)
        top = sorted(deg_lst, key=lambda x:x[1], reverse=True)
        # Here, force top k nodes with highest degree to work.
        counter = 0
        for node, degree in top:
            # Skip nodes already working.
            if working_status[node]:
                continue
            working_status[node] = True
            counter += 1
            if counter == k:
                break
    # Degree and threshold ranked heuristic.
    elif args.method == 'dtrh':
        below_avg_neighbor_dct = {}
        for node in G:
            if working_status[node]:
                continue
            num_below_avg_thresh = 0
            for neighbor in G.neighbors(node):
                # Count number of below-average threshold neighbors.
                num_working_neighbors = 0
                for neigh_neigh in G.neighbors(neighbor):
                    if working_status[neigh_neigh]:
                        num_working_neighbors += 1
                if num_working_neighbors / float(G.degree[neighbor]) < 0.5:
                    num_below_avg_thresh += 1
                # if not working_status[neighbor]:
                    # num_below_avg_thresh += 1
                # if neighbor.threshold <= 0.5:
                    # num_below_avg_thresh += 1
            below_avg_neighbor_dct[node] = num_below_avg_thresh
        top_k = sorted(below_avg_neighbor_dct.items(), key=operator.itemgetter(1),
            reverse=True)[:k]
        # Force the top nodes to work.
        for node, num in top_k:
            working_status[node] = True
    elif args.method == 'rciw':
        influence_dct = {}
        for node in G:
            if working_status[node]:
                continue
            neighbor_influence = 0
            for neighbor in G.neighbors(node):
                # Add up social influence on neighbors.
                neighbor_influence += 1.0 / G.degree[neighbor]
            influence_dct[node] = neighbor_influence
        top_k = sorted(influence_dct.items(), key=operator.itemgetter(1),
            reverse=True)[:k]
        # Force top nodes to work.
        for node, num in top_k:
            working_status[node] = True
    elif args.method == 'mmciw':
        seed_nodes = []

        while len(seed_nodes) != k :
            influence_dct = {}
            for node in G:
                if node in seed_nodes or working_status[node]:
                    continue
                neighbor_influence = 0
                for neighbor in G.neighbors(node):
                    if neighbor not in seed_nodes:
                        # Add up social influence on neighbors that aren't seeds.
                        neighbor_influence += 1.0 / G.degree[neighbor]
                influence_dct[node] = neighbor_influence
            top = sorted(influence_dct.items(), key=operator.itemgetter(1),
                reverse=True)[0]
            seed_nodes += [top[0]]
        for node in seed_nodes:
            working_status[node] = True
    elif args.method == 'eia':
        seed_nodes = []

        # Compute the EIA value for each node.
        while len(seed_nodes) != k:
            eia_dct = {}
            for node in G:
                if node in seed_nodes or working_status[node]:
                    continue
                neighbor_eia = 0.0
                for neighbor in G.neighbors(node):
                    # Skip if neighbor is a seed node or is already working.
                    if neighbor in seed_nodes or working_status[neighbor]:
                        continue
                    # Add the fraction of its neighbors that are working.
                    neighbor_degree = G.degree[neighbor]
                    for neighbors_neighbors in G.neighbors(neighbor):
                        if neighbors_neighbors in seed_nodes or working_status[neighbors_neighbors]:
                            continue
                        neighbor_eia += 1.0 / neighbor_degree
                eia_dct[node] = neighbor_eia
            top = sorted(eia_dct.items(), key=operator.itemgetter(1),
                reverse=True)[0]
            seed_nodes += [top[0]]
        for node in seed_nodes:
            working_status[node] = True

def transform(G, k=None):
    '''
    Transforms the graph for an iteration.
    '''
    working_status = {}
    # Compute the states.
    for node in G:
        working_status[node] = isWorking(node)

    force_work(working_status, G, k)

    for node in working_status:
        neighbor_working_status = [working_status[n] for n in G.neighbors(node)]
        # If node is working, or the fraction of working neighbors > threshold,
        # update the state.
        if working_status[node] or neighbor_working_status.count(True) >= node.threshold * len(neighbor_working_status):
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

# def get_fraction_documenting(G):
#     '''
#     Gets the fraction of agents who are documenting. Agents are documenting well
#     if their state is >= 0.9.
#     '''

def parse_args():
    global args
    parser = argparse.ArgumentParser()
    # parser.add_argument('-n', '--num_pts', help='Number of points in network.',
    #     required=True, type=int)
    parser.add_argument('-m', '--method', type=str, required=True,
        help='Method of incentivizing nodes.', choices=['none', 'nntu', 'dtrh',
        'rciw', 'mmciw', 'eia'])
    # parser.add_argument('-k', '--top_k', type=int,
    #     help='Number of high degree nodes to incentivize.')
    args = parser.parse_args()

    # if args.method == 'nntu' and args.top_k is None:
    #     parser.error('nntu requires --top_k.')

    # Setting other variables.
    # NUM_POINTS = 200 # Number of points in the graph.
    global NUM_EDGES, RESOURCE, BETA, DECAY, METHOD, DIR
    NUM_EDGES = 5 # Number of edges from new node to existing nodes.
    RESOURCE = 1 # Resource = 'random' or 1.
    BETA = 0.5 # How much an agent discounts the utility of documentation.
    DECAY = 0.1 # Fraction of documentation quality that decays each day.
    # METHOD = 'nntu' # Perturbation method type.
    # K = 50 # Top K. Only used for naive node degree methods.

    assert RESOURCE in [1, 'random']
    # assert METHOD in ['none', 'nntu']

    # Create results directory.
    DIR = './results/ne_%g_res_%s_b_%g_d_%g_m_%s' % (NUM_EDGES, RESOURCE, BETA,
        DECAY, args.method)
    # if args.method != 'none':
    #     DIR += '_k_%d' % args.top_k
    if not os.path.exists(DIR):
        os.makedirs(DIR)

def observe_period(num_pts, k=None):
    G = generate_points(num_pts)

    fraction_documenting_lst = []

    # Get the layout of the graph.
    if args.method == 'none':
        pos = nx.spring_layout(G)

    node_lst = G.nodes
    for i in range(15):
        if args.method == 'none':
            plot_points(G, pos, i)
        # fraction_documenting_lst += [get_fraction_documenting(G)]
        frac_doc = len([n for n in node_lst if n.state >= 0.9]) / float(len(
            node_lst))
        fraction_documenting_lst += [frac_doc]

        # Check if the last iteration has at least 75% of people documenting.
        if fraction_documenting_lst[-1] >= 0.75:
            if args.method == 'none':
                plt.plot(fraction_documenting_lst)
                plt.savefig('%s/fraction_documenting.png' % DIR)
            return k

        transform(G, k)
        # print [n.state for n in G.nodes()][:5]

def plot_tuning(min_lst, pt_lst):
    plt.figure(figsize=(10, 7.5))
    # plt.plot(pt_lst, min_lst, '-o', color='#7CAE00')
    plt.plot(pt_lst, np.mean(min_lst, axis=0), '-o', color='#00BFC4')
    print np.mean(min_lst, axis=0)
    if args.method == 'nntu':
        title = 'Naive node degree'
    elif args.method == 'dtrh':
        title = 'Degree and threshold ranked heuristic'
    elif args.method == 'rciw':
        title = 'Rank-based social influence weight'
    elif args.method == 'mmciw':
        title = 'Max margin social influence weight'
    elif args.method == 'eia':
        title = 'Expected Immediate Adoption (EIA) heuristic'

    plt.title(title, fontsize=17, ha='center')

    plt.ylabel('Minimum k')
    plt.xlabel('Number of nodes')
    plt.xlim(0, 600)

    # Experimenting with plot prettiness.
    # Remove the plot frame lines. They are unnecessary chartjunk.
    ax = plt.subplot(111)
    ax.spines["top"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.errorbar(pt_lst, np.mean(min_lst, axis=0), yerr=np.std(min_lst, axis=0))
    # Ensure that the axis ticks only show up on the bottom and left of the plot.
    # Ticks on the right and top of the plot are generally unnecessary chartjunk.
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()

    plt.savefig('%s/%s_min_k.pdf' % (DIR, args.method))

def pool_observations(pt_lst):
    iter_lst = []
    for num_pts in pt_lst:
        for k in range(num_pts):
            if observe_period(num_pts, k):
                iter_lst += [k]
                break
    return iter_lst

def main():
    parse_args()

    if args.method == 'none':
        observe_period(100)
    else:
        # nntu needs to run 100 times.
        pool = Pool(40)

        # TODO
        pt_lst = [100, 200, 300, 400, 500]
        # pt_lst = [100]
        min_lst = pool.map(pool_observations, [pt_lst] * 100)

        pool.close()
        pool.join()

        plot_tuning(min_lst, pt_lst)
        # Write out the values of the last element.
        largest_network_k = [val[-1] for val in min_lst]
        out = open('%s/min_k.txt' % DIR, 'w')
        out.write('\t'.join(map(str, largest_network_k)))
        out.close()

if __name__ == '__main__':
    main()