import numpy as np
from numpy import array
from typing import Callable, List, Tuple
from functools import partial
from tqdm import tqdm
from copy import deepcopy
from scipy.optimize import approx_fprime, minimize
import time
from joblib import Parallel, delayed
import timeit
from sklearn.model_selection import train_test_split
from sklearn.metrics import average_precision_score
import pandas as pd
from layers_class import LayerBase, CoveredLayer
from numpy import array
import numpy as np
from typing import Callable, Tuple, List
from neural_net_class import Neural_net, CostFunction
from layer_functions import (
    SigmoidFunction,
    BaseLayerFunction,
    ReluFunction,
    TanHFunction,
)
from matplotlib import pyplot as plt
import math
from type_converters import *

from adam_gradient_descent import AdamOptimiser, TrainableNeuralNetwork


def save_data_to_csv(data_id: int = 186, path: str = "wine_data.csv"):
    from ucimlrepo import fetch_ucirepo
    import pandas as pd

    from functools import partial

    # fetch dataset
    wine_quality = fetch_ucirepo(id=data_id)

    # data (as pandas dataframes)
    X_data = wine_quality.data.features
    y_data = wine_quality.data.targets

    df_x = pd.DataFrame(X_data)
    df_y = pd.DataFrame(y_data)

    # def discretise_item(value: float, middle_value: float):
    #     if value < middle_value:
    #         return -1
    #     else:
    #         return 1

    # discretise = partial(discretise_item, middle_value=6)

    def floaten_value(value):
        return np.average(value)

    disc_df_y = df_y.applymap(floaten_value)

    x_name = "x_" + path
    y_name = "y_" + path
    df_x.to_csv(x_name)
    disc_df_y.to_csv(y_name)
    # df_y.to_csv(y_name)


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


def compare_predicts(y_data: List[int], y_predicts: List[float]) -> float:
    right_predicts = 0
    all_val = 0
    for i, _ in enumerate(y_data):
        all_val += 1
        if y_data[i] == np.sign(y_predicts[i]):
            right_predicts += 1
    return right_predicts / all_val


def time_func(f: Callable, args: list, loop_amount: int = 10) -> tuple:
    # returns time in sec0nds
    times = []

    for i in range(loop_amount):
        t1 = timeit.default_timer()
        result = f(*args)

        t_delta = timeit.default_timer() - t1
        times.append(t_delta)
    return (result, times)


def get_current_time() -> str:
    current_time = time.localtime()
    hour = current_time.tm_hour
    minute = current_time.tm_min
    time_string = f"{hour:02d}_{minute:02d}"
    return time_string


def evaluate_predictions(Y, Y_predicted) -> dict:
    lab_note = {}
    p_one_percantage = sum([1 for i in Y_predicted if i > 0]) / len(Y_predicted)
    p_m_one_percentage = sum([1 for i in Y_predicted if i < 0]) / len(Y_predicted)
    lab_note["1 Predict %"] = p_one_percantage * 100
    lab_note["-1 Predict %"] = p_m_one_percentage * 100
    rel_one_percentage = sum([1 for i in Y if i > 0]) / len(Y)
    rel_m_one_percentage = sum([1 for i in Y if i < 0]) / len(Y)
    lab_note["1 Real %"] = rel_one_percentage * 100
    lab_note["-1 Real %"] = rel_m_one_percentage * 100
    pred_percent = compare_predicts(Y, Y_predicted)
    lab_note["Accuracy"] = pred_percent * 100

    wrong_one_pred_corr_min_one = 0
    wron_min_one_pred_corr_one = 0
    corr_preds = 0
    all_preds = 0
    for i, y in enumerate(Y):
        all_preds += 1
        y_p = Y_predicted[i]
        if y_p == y:
            corr_preds += 1
        elif y_p == -1:
            wron_min_one_pred_corr_one += 1
        else:
            wrong_one_pred_corr_min_one += 1

    lab_note["pred 1 rel -1 %"] = wrong_one_pred_corr_min_one / all_preds
    lab_note["pred -1 rel 1%"] = wron_min_one_pred_corr_one / all_preds

    return lab_note


def fix_up_data_to_1_row_matrix(data: List[array]):
    n_data = []
    for d in data:
        dn = array(d)
        n_data.append(dn.reshape(1, -1))
    return n_data


def fix_up_tuple_data(data):
    d2 = []
    for d in data:
        d2.append(fix_up_data_to_1_row_matrix(d))
    return tuple(d2)


def check_precision(Y, Yp):
    pos_score = 0
    accuracy = 0
    amp = max(Y) - min(Y)
    avg = np.average(Y)

    avg_in_corect = 0
    for i in range(len(Y)):
        y = Y[i]
        p = Yp[i]
        # if (y > avg and p > avg) or (y < avg and p < avg):
        #     pos_score += 1
        avg_in_corect += abs(y - p)
        if abs(y - p) <= 1:
            accuracy += 1

    accuracy = accuracy / len(Y)
    avg_in_corect = avg_in_corect / len(Y)
    return accuracy, avg_in_corect


# def discretise_data_by_avg(Y):
#     avg = np.average(Y)
#     new_y = []
#     for y in Y:
#         if y < avg:
#             new_y.append(-1)
#         else:
#             new_y.append(1)
#     return new_y


if __name__ == "__main__":
    # save_data_to_csv()
    X, Y = get_x_y_train_data()
    # get_x_y_train_data()
    # Y = discretise_data_by_avg(Y)
    d_nr = -1
    X = X[:d_nr]
    Y = Y[:d_nr]
    name = "100000_iters.json"
    # Xt, Xcheck, Yt, Ycheck = (X, X, Y, Y)
    Xt, Xcheck, Yt, Ycheck = fix_up_tuple_data(
        train_test_split(X, Y, test_size=0.2, random_state=42)
    )

    all_neurons = 16
    nr_of_releases = 5
    neurons_to_cover = 10
    release_nr = neurons_to_cover // (nr_of_releases - 1)
    iters = 10000 // nr_of_releases
    train_size = 100
    nr_of_batches = len(X) // train_size
    # cover = [i for i in range(neurons_to_cover)]
    cover = None
    # print(nr_of_batches)
    alpha = 0.001
    b1 = 0.9
    b2 = 0.999
    nr_of_calcs = 1
    # t_it = 5
    # log_calcs = 10
    # lin_calcs = 50
    cf = CostFunction()

    fs = SigmoidFunction()
    frl = ReluFunction()
    ftan = TanHFunction()
    fl = BaseLayerFunction()

    cl = CoveredLayer(
        all_neurons,
        fs,
        cover,
        cover,
    )
    cl2 = CoveredLayer(
        all_neurons,
        ftan,
        cover,
        cover,
    )
    layers = [
        # LayerBase(1, fs),
        cl,
        cl,
        # LayerBase(32, ftan),
        # LayerBase(32, ftan),
        # LayerBase(32, ftan),
        LayerBase(1, frl),
    ]
    # spg = GradientSolverParams(
    #     iters,
    #     Xt,
    #     Yt,
    #     cf,
    #     initial_config_iterations=t_it,
    #     log_configuration_passes=log_calcs,
    #     lin_configuration_passes=lin_calcs,
    #     log_limits=(-5, 5),
    # )
    # nn = Neural_net(layers, len(X[0]))
    nn = TrainableNeuralNetwork(layers, len(X[0]))
    print(Xcheck[0])
    Y_pred = [float(y) for y in nn.calculate_output_for_many_values(Xcheck)]
    prec, accuracy = check_precision([float(y) for y in Ycheck], Y_pred)

    print(0, prec, accuracy)

    opt = AdamOptimiser(alpha, b1, b2)
    for i in range(nr_of_releases):
        best_cost = nn.train_on_data(opt, cf, Xt, Yt, iters, nr_of_batches)
        Y_pred = [float(y) for y in nn.calculate_output_for_many_values(Xcheck)]
        prec, accuracy = check_precision([float(y) for y in Ycheck], Y_pred)
        print(print(nn.layers[0].covered_neurons), prec, accuracy)
        nn.layers[0].uncover_neurons(release_nr)
        nn.layers[1].uncover_neurons(release_nr)

        # print(nn.layers[0].w)
    Y_pred = [float(y) for y in nn.calculate_output_for_many_values(Xcheck)]
    prec, accuracy = check_precision([float(y) for y in Ycheck], Y_pred)
    print(f"after {iters}", prec, accuracy)

    # print(Y_pred)
    Y_pred = [float(y) for y in nn.calculate_output_for_many_values(Xcheck)]
    glob_cost_hist = nn.cost_hist

    plt.plot(glob_cost_hist, label="Gradient descent cost")

    plt.legend()
    plt.semilogy()
    plt.xlabel("iteration")
    plt.ylabel("cost")
    plt.title("Training")
    plt.show()

    import json

    with open(name, "w") as f:
        json.dump(list(get_flattened_ws_bs(nn.layers)), f)
