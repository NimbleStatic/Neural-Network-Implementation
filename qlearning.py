import gymnasium as gym
from layers_class import LayerBase
from layer_functions import (
    ReluFunction,
    SigmoidFunction,
    TanHFunction,
    BaseLayerFunction,
)
from adam_gradient_descent import (
    AdamOptimiser,
    TrainableNeuralNetwork,
    BinaryResultOverlay,
)
from neural_net_class import CostFunction

import pandas as pd
from typing import List, Tuple, Callable
import numpy as np
from numpy import array
from matplotlib import pyplot as plt


# Create Taxi environment
# env = gym.make("Taxi-v3", render_mode="human")
env = gym.make("Taxi-v3")

env.action_space.seed(42)


# Reset environment to initial state


def score_possible_moves(
    scoring_function, state_observes: List, possible_moves: List[int]
) -> List[Tuple[int, float]]:
    move_score = []
    for move in possible_moves:
        val = array([*state_observes, move])
        move_score.append((move, float(scoring_function(val))))
    return move_score


def get_best_move(moves_scores: List[Tuple[int, float]]):
    best_score = max([i[1] for i in moves_scores])
    for m, s in moves_scores:
        if s == best_score:
            return m
    raise ValueError("shit")


def choose_based_on_propabilities(moves_scores: List[Tuple[int, float]]):
    scores = [i[1] for i in moves_scores]
    sm = softmax(scores)
    ix = choose_from_probabilities(sm)
    return moves_scores[ix]


# initial_state = env.reset()
def softmax(scores):
    scores = scores + (np.ones(shape=np.array(scores).shape) * (abs(np.min(scores))))
    exp_scores = np.exp(scores)
    probabilities = exp_scores / np.sum(exp_scores)
    return probabilities


def choose_from_probabilities(probabilities):
    chosen_index = np.random.choice(len(probabilities), p=probabilities)
    return chosen_index


# Maximum number of steps in the environment
max_steps = 10
# max_steps = 100

# random_steps = 100000
# possible_moves = env.action_space()
# Interact with the environment
rel = ReluFunction()
sig = SigmoidFunction()
tan = TanHFunction()
lin = BaseLayerFunction()
nr_of_neurons = 12
mid_function = tan
nr_of_layers = 3

layers = [
    *[LayerBase(nr_of_neurons, mid_function) for _ in range(nr_of_layers)],
    LayerBase(1, lin),
]


possible_moves = [0, 1, 2, 3, 4, 5]
aopt = AdamOptimiser()
cf = CostFunction()
observation, info = env.reset(seed=42)
observes = [*env.decode(observation)]
from random import choice

nn = TrainableNeuralNetwork(layers, (len(observes) + 1))
nn.init_one_data_training()
cost_hist = []
from tqdm import tqdm
from time import sleep

for i in tqdm(range(max_steps)):
    poss_scores = score_possible_moves(nn.calculate_output, observes, possible_moves)
    best_move_score = choose_based_on_propabilities(poss_scores)
    best_move = best_move_score[0]

    observation, reward, terminated, truncated, info = env.step(best_move)
    print(observation, reward, terminated, truncated, info)
    cost_hist.append(reward)
    observes = [*env.decode(observation)]
    nn.train_on_one_data(aopt, cf, array([*observes, best_move]), array([reward]))

    if terminated:
        observation, info = env.reset(seed=42)
        observes = [*env.decode(observation)]
env.close()
plt.plot(cost_hist)
plt.show()
plt.plot(nn.cost_hist)
plt.semilogy()
plt.show()

game_steps = 10000
env = gym.make("Taxi-v3", render_mode="human")
observation, info = env.reset(seed=42)
observes = [*env.decode(observation)]
for i in tqdm(range(game_steps)):
    poss_scores = score_possible_moves(nn.calculate_output, observes, possible_moves)
    best_move_score = choose_based_on_propabilities(poss_scores)
    best_move = best_move_score[0]

    observation, reward, terminated, truncated, info = env.step(best_move)
    cost_hist.append(reward)
    observes = [*env.decode(observation)]
    nn.train_on_one_data(aopt, cf, array([*observes, best_move]), array([reward]))
    # sleep(1)
    if terminated:
        observation, info = env.reset(seed=42)
        observes = [*env.decode(observation)]
env.close()
