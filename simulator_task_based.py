# run like: python simulator.py --workers=200 --tasks=1000
# TODO: bug at tasks=2000
import networkx as nx
import numpy as np
import matplotlib
import os
import argparse
import logging
from math import pow as power
from os import system
import csv

matplotlib.use('Agg')
import matplotlib.pyplot as plt

np.random.seed(111)

# Setting variables.
DEFAULT_NUM_TASKS = 15
DEFAULT_NUM_WORKERS = 200
DEFAULT_HIGH_QUALITY_CONFIGURATION = 'random'

AVERAGE_DOCUMENTATION_FEQUENCY = .5 # mean of possible results
HIGH_QUALITY_THRESHOLD = .9 # defined arbitrarily
PERCENT_INITIAL_HIGH_QUALITY = .398 # 1 = 100% .05 = 5%
NUM_INITIAL_HIGH_QUALITY_WEAKEST = 90
NUM_INITIAL_HIGH_QUALITY_STRONGEST = 135
NUM_INITIAL_HIGH_QUALITY_PER_TEAM = 2
NUM_INITIAL_HIGH_QUALITY_SPARSE = 10

# Parameters that come from the survey results
FLAT_DOCUMENTATION_COST = .54 # from question 1 originally, but tuned, average state should be about the same throughout the simulation from start to end.


class Employee(object):
    """An employee in the network. It has the following properties:

    Attributes:
        label: An int specifying the node number.
        resource: A variable U(0, 1), the resource constraint of the agent.
        state: A float, determining how well the agent is documenting.
        tasks_completed: An int, denoting the number of tasks completed.
    """

    def __init__(self, label, select_high_quality):
        """Return an Agent object whose resource is *resource* and state is
        *state*."""
        self.label = label
        self.resource = max(0.1, np.random.random_sample()) # let's assume everyone has at least a tenth of the max resource

        # put x one in each team
        if select_high_quality and np.random.uniform(0,1) <= PERCENT_INITIAL_HIGH_QUALITY:
            self.state = np.random.uniform(HIGH_QUALITY_THRESHOLD, 1)
        else:
            self.state = np.random.uniform(0, AVERAGE_DOCUMENTATION_FEQUENCY)
        self.tasks_completed = 0

    def set_state(self, new_state):
        self.state = new_state

    def update_tasks_completed(self):
        self.tasks_completed = self.tasks_completed + 1

def generate_points(num_points,degree_probability, high_quality_configuration):
    '''
    Generates a specified number of points with two attributes in [0, 1].
    '''
    print high_quality_configuration
    G = nx.random_geometric_graph(num_points, degree_probability)

    # Remap the nodes to Agent objects.
    # Initialization is done here
    mapping = {}

    for label in list(G.nodes):
        should_randomly_add_high_quality = True if high_quality_configuration == 'random' else False
        mapping[label] = Employee(label, should_randomly_add_high_quality)

    new_G = nx.relabel_nodes(G, mapping)

    for node in new_G:
        teammate_states = [n.state for n in new_G.neighbors(node)]
        teammates = [n.label for n in new_G.neighbors(node)]
        node.initial_team_strength = sum(teammate_states) / len(teammate_states) if len(teammate_states) > 0 else 0
        node.initial_team_size = len(teammates)
        if high_quality_configuration == 'per_team':
            states = list(teammate_states)
            for i in range(NUM_INITIAL_HIGH_QUALITY_PER_TEAM):
                if i > 0:
                    max_index = states.index(max(states)) if states else -1
                    if max_index > -1:
                        del states[max_index]

                # meaning no teamates produce high quality documentation
                if len(states) > 0 and max(states) < HIGH_QUALITY_THRESHOLD:
                    high_quality_state = np.random.uniform(HIGH_QUALITY_THRESHOLD, 1)
                    node.set_state(high_quality_state)
        if high_quality_configuration == 'sparse':
          teammates = new_G.neighbors(node)
          for teammate in teammates:
             teammate_states = teammate_states + [n.state for n in new_G.neighbors(teammate)]
             states = list(teammate_states)
             for i in range(NUM_INITIAL_HIGH_QUALITY_SPARSE):
                if i > 0:
                    max_index = states.index(max(states)) if states else -1
                    if max_index > -1:
                        del states[max_index]

          # meaning no teamates produce high quality documentation
          if len(states) > 0 and max(states) < HIGH_QUALITY_THRESHOLD:
                high_quality_state = np.random.uniform(HIGH_QUALITY_THRESHOLD, 1)
                node.set_state(high_quality_state)

    if high_quality_configuration == 'weakest':
        # find node with weakest teammates and train him
        team_strengths = [node.initial_team_strength for node in new_G]

        for i in range(NUM_INITIAL_HIGH_QUALITY_WEAKEST):
            weakest = min(team_strengths)
            del team_strengths[team_strengths.index(weakest)]

            weakest_link = [node for node in new_G if node.initial_team_strength == weakest][0]
            high_quality_state = np.random.uniform(HIGH_QUALITY_THRESHOLD, 1)
            weakest_link.set_state(high_quality_state)

    if high_quality_configuration == 'strongest':
        # find node with strongest teammates and train him
        team_strengths = [node.initial_team_strength for node in new_G]

        for i in range(NUM_INITIAL_HIGH_QUALITY_STRONGEST):
            strongest = max(team_strengths)
            del team_strengths[team_strengths.index(strongest)]

            strongest_link = [node for node in new_G if node.initial_team_strength == strongest][0]
            high_quality_state = np.random.uniform(HIGH_QUALITY_THRESHOLD, 1)
            strongest_link.set_state(high_quality_state)

    if high_quality_configuration == 'largest':
        # find node with largest number teammates and train him

        # if we have 10 workers, and 7 team sizes
        # we give 3 to the largest, 2 to the other, 1 to the bottom 4
        for node in new_G:
            teammate_states = [n.state for n in new_G.neighbors(node)]
            states = list(teammate_states)
            allowed_per_teamsize = int( .3 * node.initial_team_size)
            allowed_per_teamsize = 1 if allowed_per_teamsize < 1 else allowed_per_teamsize
            for i in range(allowed_per_teamsize):
                if i > 0:
                    max_index = states.index(max(states)) if states else -1
                    if max_index > -1:
                        del states[max_index]
            # meaning no teamates produce high quality documentation
            if len(states) > 0 and max(states) < HIGH_QUALITY_THRESHOLD:
                high_quality_state = np.random.uniform(HIGH_QUALITY_THRESHOLD, 1)
                node.set_state(high_quality_state)

    if high_quality_configuration == 'smallest':
        # find node with largest number teammates and train him

        max_team_size = max([node.initial_team_size for node in new_G])
        # if we have 10 workers, and 7 team sizes
        # we give 3 to the largest, 2 to the other, 1 to the bottom 4
        for node in new_G:
            teammate_states = [n.state for n in new_G.neighbors(node)]
            states = list(teammate_states)
            allowed_per_teamsize = int( max_team_size / (0.91 * node.initial_team_size) ) if node.initial_team_size > 0 else 0
            allowed_per_teamsize = 1 if allowed_per_teamsize < 1 else allowed_per_teamsize
            for i in range(allowed_per_teamsize):
                if i > 0:
                    max_index = states.index(max(states)) if states else -1
                    if max_index > -1:
                        del states[max_index]
            # meaning no teamates produce high quality documentation
            if len(states) > 0 and max(states) < HIGH_QUALITY_THRESHOLD:
                high_quality_state = np.random.uniform(HIGH_QUALITY_THRESHOLD, 1)
                node.set_state(high_quality_state)

    return new_G




def complete_task(G):
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
            teammate_average_state = sum(teammate_states) / len(teammate_states) if len(teammate_states) > 0 else 0

            #update number of tasks completed
            node.update_tasks_completed()

            task_completion_factor = 1  / (0.1 + 0.9 * node.tasks_completed)

            #calculate cost
            cost = FLAT_DOCUMENTATION_COST

            #calculate perceived reward
            reward = teammate_average_state + task_difficulty * node.resource

            #calculate expected utility
            utility = reward - cost

            #update state
            new_state = node.state + task_completion_factor * utility / 4
            new_state = 0 if new_state < 0 else new_state
            new_state = 1 if new_state > 1 else new_state
            node.set_state(new_state)


def plot_points(G, pos, i, DIR):
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
        plt.savefig('%s/random_graph_0.png' % DIR)
    else:
        plt.savefig('%s/random_graph_%d.png' % (DIR, i))
    plt.close()

def plot_team_info(num_workers, G, DIR):
    degree_histogram = nx.degree_histogram(G)
    averageTeamSize = len(degree_histogram) / 2
    logging.warn('the company has %s workers' % num_workers)
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
    parser.add_argument('-c', '--configuration', required=False)
    parser.add_argument('-i', '--iterations', required=False)
    args = parser.parse_args()
    return args

def main():
    # Input parameters
    args = get_args()
    num_tasks = int(args.tasks) if args and args.tasks else DEFAULT_NUM_TASKS
    num_workers = int(args.workers) if args and args.workers else DEFAULT_NUM_WORKERS
    high_quality_configuration = str(args.configuration) if args and args.configuration else DEFAULT_HIGH_QUALITY_CONFIGURATION
    iterations = int(args.iterations) if args and args.iterations else 1

    # Create results directory.
    DIR = './results/np_%d' % (num_workers)
    if not os.path.exists(DIR):
        os.makedirs(DIR)
    # Initial configuration, 0.1115 from parameter tuning
    team_likelihood = {}
    team_likelihood[2000] = 0.0255
    team_likelihood[200] = 0.1115
    team_likelihood[50] = 0.2515
    team_likelihood[25] = 0.4215

    # for collecting stats
    stats = {}
    initial_high_quality_percent = []
    for i in range(num_tasks):
        stats[i] = ([],[])

    for i in range(iterations):
        is_first_iteration = True if i < 1 else False
        G = generate_points(num_workers, team_likelihood[num_workers], high_quality_configuration)

        # Get the layout of the graph.
        pos = nx.spring_layout(G,None,None,None,30,'weight',200)

        # Plot the team size
        if is_first_iteration:
            plot_team_info(num_workers, G, DIR)

        if is_first_iteration:
            plot_points(G, pos, 0, DIR)

        # initial stats
        initial_average_state = sum([node.state for node in G]) / num_workers
        initial_high_quality = len([node for node in G  if node.state >= HIGH_QUALITY_THRESHOLD ])
        initial_high_quality_percent = initial_high_quality_percent + [float(initial_high_quality) / num_workers]

        #run the simulation
        for j in range(num_tasks):
            if is_first_iteration:
                plot_frequency = power(10, len(str(j)) -1)
                if j % plot_frequency == 0:
                    plot_points(G, pos, j, DIR)
            complete_task(G)

            # collect stats
            intermediate_average_state = [1.0 * sum([node.state for node in G]) / num_workers]
            intermediate_high_quality = [1.0 * len([node for node in G  if node.state >= HIGH_QUALITY_THRESHOLD ])]
            stats[j] = (stats[j][0] + intermediate_average_state, stats[j][1] + intermediate_high_quality)

    # final stats
    output_sample = 100

    file_name = '%s_employees_%s_training_strategy_%s_tasks' % (num_workers, high_quality_configuration,num_tasks)
    with open('./%s_average_state.csv' % file_name, 'w') as csvfile:
        fieldnames = ['iteration', 'state', 'num_high_quality', 'high_quality_percent']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for it, values in stats.items():
            state_list = values[0]
            num_high_quality_list = values[1]
            state = float(sum(state_list) / len(state_list))
            num_high_quality = float(sum(num_high_quality_list) / len(num_high_quality_list))
            high_quality_percent = 100.0 * num_high_quality / num_workers

            if it % output_sample == 0:
                writer.writerow({'iteration': it, 'state': state, 'num_high_quality': num_high_quality, 'high_quality_percent': high_quality_percent})

    with open('./initial_high_quality_%s.csv' % file_name, 'w') as csvfile:
        fieldnames = ['configuration', 'initial_percent']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        average = sum(initial_high_quality_percent) / len(initial_high_quality_percent)
        writer.writerow({'configuration': high_quality_configuration, 'initial_percent': average})

    # make the gif, requires imageMagick
    system('/opt/local/bin/convert -delay 50 results/*/random_graph_* -loop 1 ./%s.gif' % file_name)

if __name__ == '__main__':
    main()