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
from sklearn.model_selection import train_test_split


def get_x_y_train_data(path: str = "wine_data.csv") -> Tuple[List[array], List[float]]:
    x_name = "x_" + path
    y_name = "y_" + path

    df_x = pd.read_csv(x_name)
    disc_df_y = pd.read_csv(y_name)

    X_tries = df_x.to_numpy()
    Y_tries = disc_df_y.to_numpy()

    Y_tries_form = []
    X_tries_form = []
    for i in range(len(Y_tries)):
        Y_tries_form.append(float(Y_tries[i][1]))
        X_tries_form.append(X_tries[i][1:])

    return X_tries_form, Y_tries_form


def check_precision(Y, Yp):
    accuracy = 0
    avg_in_corect = 0
    for i in range(len(Y)):
        y = Y[i]
        p = Yp[i]
        avg_in_corect += abs(y - p)
        if (y - p) == 0:
            accuracy += 1

    accuracy = accuracy / len(Y)
    avg_in_corect = avg_in_corect / len(Y)
    return accuracy, avg_in_corect


def get_function_training_data(func, nr_of_samples, randomization, min_lim, max_lim):
    X = np.linspace(min_lim, max_lim, nr_of_samples)
    Y = []
    for x in X:
        val = func(x) + randomization * np.random.normal()
        Y.append(val)
    return X, Y


def randomise_order_of_train_data(X, Y):
    ixes = [*range(len(X))]
    np.random.shuffle(ixes)
    Xp = []
    Yp = []
    for i in ixes:
        Xp.append(X[i])
        Yp.append(Y[i])
    return Xp, Yp


rel = ReluFunction()
sig = SigmoidFunction()
tan = TanHFunction()
lin = BaseLayerFunction()
if __name__ == "__main__":

    min_max = (-1, 4)
    testing_range = (min_max[0] * 2, min_max[1] * 2)
    rand = 10

    def func(x):
        return np.sin(x)

    training_posts_nr = 10000
    X, Y = get_function_training_data(
        func, training_posts_nr, rand, min_max[0], min_max[1]
    )
    X, Y = randomise_order_of_train_data(X, Y)

    # def binarise_y(Y):
    #     avg = np.average(Y)
    #     y2 = []
    #     for y in Y:
    #         if y > int(avg):
    #             y2.append(1)
    #         # elif y < int(avg):
    #         #     y2.append(-1)
    #         else:
    #             y2.append(0)
    #     return y2

    Xt, Xtest, Yt, Ytest = train_test_split(X, Y, test_size=0.2, random_state=42)

    # y_size = len(Yt[0])
    y_size = 1

    x_size = 1
    hidden_function = rel
    nr_of_neurons = 10
    last_func = lin
    iters = 100000
    batch_size = 100
    layers = [
        LayerBase(nr_of_neurons, hidden_function),
        LayerBase(nr_of_neurons, sig),
        LayerBase(y_size, last_func),
    ]

    nn = TrainableNeuralNetwork(layers, x_size)
    Ycheck = nn.calculate_output_for_many_values(Xtest)

    adam = AdamOptimiser()
    cf = CostFunction()

    nn.train_on_data(adam, cf, Xt, Yt, iters, (len(Xt) // batch_size))
    Ycheck = nn.calculate_output_for_many_values(
        np.linspace(testing_range[0], testing_range[1], 100)
    )
    Ycheck = [float(y) for y in Ycheck]
    # plt.plot(X, Y)
    plt.plot(
        np.linspace(testing_range[0], testing_range[1], 100),
        func(np.linspace(testing_range[0], testing_range[1], 100)),
    )
    plt.plot(np.linspace(testing_range[0], testing_range[1], 100), Ycheck)
    plt.show()
    plt.plot(nn.cost_hist)
    plt.show()
