# run like: python simulator.py --workers=200 --tasks=1000
# TODO: bug at tasks=2000
import networkx as nx
import numpy as np
import matplotlib
import os
import argparse
import logging

matplotlib.use('Agg')
import matplotlib.pyplot as plt

np.random.seed(111)

# Setting variables.
DEFAULT_NUM_TASKS = 15
DEFAULT_NUM_WORKERS = 200

NUM_POINTS = 200 # Number of points in the graph.
NUM_EDGES = 3 # Number of edges from new node to existing nodes.
RESOURCE = 1 # Resource = 'random' or 1.
BETA = 0.5 # How much an agent discounts the utility of documentation.
DECAY = 0.1 # Fraction of documentation quality that decays each day.

AVERAGE_DOCUMENTATION_FEQUENCY = .5 # mean of possible results
HIGH_QUALITY_THRESHOLD = .7 # mean of possible results
NUM_INITIAL_HIGH_QUALITY = .05 # 1 = 100% .05 = 5%

# Parameters that come from the survey results
FLAT_DOCUMENTATION_COST = .39 # from question 1

assert RESOURCE in [1, 'random']

# Create results directory.
DIR = './results/np_%d_ne_%g_res_%s_b_%g_d_%g' % (NUM_POINTS, NUM_EDGES,
    RESOURCE, BETA, DECAY)
if not os.path.exists(DIR):
    os.makedirs(DIR)

class Employee(object):
    """An employee in the network. It has the following properties:

    Attributes:
        label: An int specifying the node number.
        resource: A variable U(0, 1), the resource constraint of the agent.
        state: A float, determining how well the agent is documenting.
        tasks_completed: An int, denoting the number of tasks completed.
    """

    def __init__(self, label):
        """Return an Agent object whose resource is *resource* and state is
        *state*."""
        self.label = label
        self.resource = max(0.1, np.random.random_sample()) # let's assume everyone has at least a tenth of the max resource
        if np.random.uniform(0,1) <= NUM_INITIAL_HIGH_QUALITY:
            self.state = np.random.uniform(HIGH_QUALITY_THRESHOLD, 1)
        else:
            self.state = np.random.uniform(0, AVERAGE_DOCUMENTATION_FEQUENCY)
        self.tasks_completed = 0

    def set_state(self, new_state):
        self.state = new_state

    def update_tasks_completed(self):
        self.tasks_completed = self.tasks_completed + 1

def generate_points(num_points,degree_probability):
    '''
    Generates a specified number of points with two attributes in [0, 1].
    '''
    G = nx.random_geometric_graph(num_points, degree_probability)

    # Remap the nodes to Agent objects.
    # Initialization is done here
    mapping = {}
    for label in list(G.nodes):
        mapping[label] = Employee(label)
    return nx.relabel_nodes(G, mapping)

def complete_task(G, num_tasks):
    '''
        Each iteration one task is generated and completed within an organization by a single employee
        That task has a difficulty that is selected uniformly and at random from a normal distribution
        The employee decides whether to change his/her documentation behavior (state) after completing the task
        state definition defined below
            Task Difficulty
                - chosen at random. from 0 to 1. The more difficult the task the higher the reward for documenting
            Neighbor State
                - property of the graph. If my teammates all document, there's social pressure for me to document as well
            Personal Resource/Threshold
                - chosen at random. from 0 to 1. If I don't have time, even if I want to document, my documentation will be limited by my resource
            Number of tasks completed
                - depends on how many tasks the user gets
                - numTasksCompleted / numTotalTasks
            Document Cost
                - from 0 to 1, where 1 produces the highest quality document

            Cost of documenting (max of 2, min 0)
                - flatDocumentCostFactor
            Perceived Reward (max of 2, min 0)
                - teammateStateFactor + TaskDifficultyFactor * personalResourceFactor
            Expected Utility (max 1, min 0)
                - (Perceived Reward - Cost of documenting)

            State Change Equation
                - new_state = Current State +  numTasksCompletedFactor * (Expected Utility) / 4
                    - 0 if new_state < 0 #can't go below zero
                    - 1 if new_state > 1 #can't go above 1
    '''
    # generate a task difficulty
    task_difficulty = np.random.uniform(0,1)

    # assign task to an employee
    selected_employee = int(round(np.random.uniform(0, len(G))))

    for node in G:
        if node.label == selected_employee:
            #get teammate states
            teammate_states = [teammate.state for teammate in G.neighbors(node)]
            teammate_average_state = sum(teammate_states) / len(teammate_states)

            #update number of tasks completed
            node.update_tasks_completed()

            #update number of tasks completed
            task_completion_factor = node.tasks_completed / num_tasks

            #calculate cost
            cost = FLAT_DOCUMENTATION_COST

            #calculate perceived reward
            reward = teammate_average_state + task_difficulty * node.resource

            #calculate expected utility
            utility = reward - cost

            #update state
            new_state = node.state + task_completion_factor + utility / 4
            new_state = 0 if new_state < 0 else new_state
            new_state = 1 if new_state > 1 else new_state
            node.set_state(new_state)


def plot_points(G, pos, i):
    '''
    Plots the graph.
    '''
    norm = matplotlib.colors.Normalize(vmin=0, vmax=1)
    cmap = matplotlib.cm.Reds
    m = matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap)

    nx.draw_networkx(G, node_size=70, with_labels=False, width=0.2, pos=pos,
        node_color=[m.to_rgba(n.state) for n in G.nodes])

    plt.axis('off')
    if i == 0:
        plt.savefig('%s/random_graph_initial.png' % DIR)
    else:
        plt.savefig('%s/random_graph_%d.png' % (DIR, i))
    plt.close()

def get_fraction_documenting(G):
    '''
    Gets the fraction of agents who are documenting. Agents are documenting well
    if their state is >= 0.9.
    '''
    node_lst = G.nodes
    return len([n for n in node_lst if n.state >= 0.9]) / float(len(node_lst))

def plot_team_info(numWorkers, G):
    degree_histogram = nx.degree_histogram(G)
    averageTeamSize = len(degree_histogram) / 2
    logging.warn('the company has %s workers' % numWorkers)
    logging.warn('with an average team size is %s' % averageTeamSize)
    plt.hist(degree_histogram)
    plt.savefig('%s/degree.png' % DIR)
    plt.close()

"""
    Flow:
    In the initialization phase the company is constructed:
        - the employees are created as a random geometric graph that resembles a social network
        - the average team size is ~8
        - employees are given an initial personal resource/threshold
        - employees are given an initial state
        - employees start with 0 tasks complete

    In the simulation phase we run the following code a fixed ('numTasks') number of times
        - Each iteration one task is generated and completed within an organization by a single employee
        - That task has a difficulty that is selected uniformly and at random from a normal distribution
        - The employee decides whether to change his/her documentation behavior (state) after completing the task
        - state definition defined below
            Task Difficulty
                - chosen at random. from 0 to 1. The more difficult the task the higher the reward for documenting
            Neighbor State
                - property of the graph. If my teammates all document, there's social pressure for me to document as well
            Personal Resource/Threshold
                - chosen at random. from 0 to 1. If I don't have time, even if I want to document, my documentation will be limited by my resource
            Number of tasks completed
                - depends on how many tasks the user gets
            Document Cost
                - from 0 to 1, where 1 produces the highest quality document

            Cost of documenting (max of 1, min 0)
                - flatDocumentCostFactor / personalResourceFactor
            Percieved Reward (max of 1, min 0)
                - ((teammateStateFactor / numberTasksCompletedFactor) + TaskDifficultyFactor) / 2
            Expected Utility (max 1, min 0)
                - (Perceived Reward - Cost of documenting)
                - normalized = .5 + (reward - cost)/2

            State Change Equation
                - new_state = (Current State + Expected Utility) / 2
"""
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--tasks', required=False)
    parser.add_argument('-w', '--workers', required=False)
    args = parser.parse_args()
    return args

def main():
    # Input parameters
    args = get_args()
    num_tasks = int(args.tasks) if args and args.tasks else DEFAULT_NUM_TASKS
    numWorkers = int(args.workers) if args and args.workers else DEFAULT_NUM_WORKERS
    fraction_documenting_lst = []

    # Initial configuration
    G = generate_points(numWorkers, 0.1115)
    #print [n.state for n in G.nodes()]

    # Get the layout of the graph.
    pos = nx.spring_layout(G,None,None,None,30,'weight',200)

    # Plot the team size
    plot_team_info(numWorkers, G)

    plot_points(G, pos, 0)
    plot_frequency = num_tasks * .1

    #run the simulation
    for i in range(num_tasks):
        if i % plot_frequency == 0:
            plot_points(G, pos, i)
        #fraction_documenting_lst += [get_fraction_documenting(G)]

        complete_task(G, num_tasks)


    # Plot fraction of documenting agents at each iteration.
    plt.plot(fraction_documenting_lst)
    plt.savefig('%s/fraction_documenting.png' % DIR)

if __name__ == '__main__':
    main()