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


rel = ReluFunction()
sig = SigmoidFunction()
tan = TanHFunction()
lin = BaseLayerFunction()
if __name__ == "__main__":
    X, Y = get_x_y_train_data()

    def binarise_y(Y):
        avg = np.average(Y)
        y2 = []
        for y in Y:
            if y > int(avg):
                y2.append(1)
            # elif y < int(avg):
            #     y2.append(-1)
            else:
                y2.append(0)
        return y2

    Xt, Xtest, Yt, Ytest = train_test_split(
        X, binarise_y(Y), test_size=0.2, random_state=42
    )

    nr_of_neurons = 8
    nr_of_layers = 3
    batch_size = 100
    iters = 10000
    mid_function = tan
    last_func = sig

    nr_of_possible_values = int(max(Yt) - min(Yt)) + 1
    bro = BinaryResultOverlay(nr_of_possible_values, int(min(Yt)))
    Ytc = bro.convert_list_int_Y_to_grays_code_arrays(Yt)

    y_size = len(Ytc[0])
    x_size = len(Xt[0])
    layers = [
        *[LayerBase(nr_of_neurons, mid_function) for _ in range(nr_of_layers)],
        LayerBase(y_size, last_func),
    ]

    nn = TrainableNeuralNetwork(layers, x_size)
    Ycheck = nn.calculate_output_for_many_values(Xtest)
    a = bro.convert_activations_to_result(Ycheck)
    print(check_precision(Ytest, a))

    adam = AdamOptimiser()
    cf = CostFunction()

    nn.train_on_data(adam, cf, Xt, Ytc, iters, (len(Xt) // batch_size))
    Ycheck = nn.calculate_output_for_many_values(Xtest)
    print(check_precision(Ytest, bro.convert_activations_to_result(Ycheck)))
    ybcheck = bro.convert_activations_to_result(Ycheck)
    print("testing 1", np.sum([1 for i in Ytest if i == 1]))
    print("testing -1", np.sum([-1 for i in Ytest if i == 0]))
    print("nn 1", np.sum([1 for i in ybcheck if i == 1]))
    print("nn -1", np.sum([-1 for i in ybcheck if i == 0]))
    plt.plot(nn.cost_hist)
    plt.show()
