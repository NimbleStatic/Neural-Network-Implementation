from numpy import array
import numpy as np
from layers_class import LayerBase
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
from type_converters import *
import math
from tqdm import tqdm


class AdamOptimiser:
    def __init__(
        self, alpha: float = 0.001, beta1: float = 0.9, beta2: float = 0.999, eps=1e-8
    ):
        self.init_alpha = alpha
        self.alpha = alpha
        self.init_b1 = beta1
        self.init_b2 = beta2
        self.b1 = beta1
        self.b2 = beta2
        self.eps = eps
        self.m = 0
        self.v = 0
        self.beta_history = []
        self.iterations = 0

    def calculate_descented_x(self, x0: array, gradient: array) -> array:
        self.iterations += 1
        m = self.b1 * self.m + (1 - self.b1) * gradient
        v = self.b2 * self.v + (1 - self.b2) * (np.power(gradient, 2))
        mh = m / (1 - self.b1)
        vh = v / (1 - self.b2)

        self.b1 = self.b1**self.iterations
        self.b2 = self.b2**self.iterations
        # self.b1 = self.init_b1**self.iterations
        # self.b2 = self.init_b2**self.iterations
        value = (self.alpha * mh) / (np.sqrt(vh) + self.eps)
        self.m = m
        self.v = v
        return x0 - value


class TrainableNeuralNetwork(Neural_net):
    def __init__(self, list_of_layers: List[LayerBase], input_size) -> None:
        super().__init__(list_of_layers, input_size)
        self.cost_hist = []

    def __compose_many_array_lists(self, array_lists: List[List[array]]) -> List[array]:
        new_arrays = []
        nr_of_d = len(array_lists)
        for i in range(len(array_lists[0])):
            comp_array = 0
            for j in range(nr_of_d):
                comp_array += array_lists[j][i]
            new_arrays.append((comp_array / nr_of_d))
        return new_arrays

    def __get_gradients_cost_per_sample(
        self,
        X: List[array],
        Y: List[array],
        cf: CostFunction,
    ) -> Tuple[List[array], List[array], float]:
        dweights_list = []
        dbiases_list = []
        cost_sum = 0
        for ts in range(len(X)):
            nout = self.calculate_output(X[ts])
            dC = cf.get_deriv_cost_to_a_output(Y[ts], nout)
            cost_sum += cf.get_float_cost(Y[ts], nout)
            dws, dbs = self.backpropagate(dC)
            dweights_list.append(dws)
            dbiases_list.append(dbs)
        cost = cost_sum / len(X)
        we_comp = self.__compose_many_array_lists(dweights_list)
        bs_comp = self.__compose_many_array_lists(dbiases_list)
        return (we_comp, bs_comp, cost)

    def __combine_to_flat_array(self, array_list: List[array]) -> array:
        all_arrays = []
        for a in array_list:
            all_arrays.append(a.flatten())
        return np.concatenate(array_list)

    def create_n_batches(self, nr_of_batches, X, Y):
        self.X_batches = [[] for _ in range(nr_of_batches)]
        self.Y_batches = [[] for _ in range(nr_of_batches)]
        reseting_i = 0
        for i in range(len(X)):
            self.X_batches[reseting_i].append(X[i])
            self.Y_batches[reseting_i].append(Y[i])
            reseting_i += 1
            if reseting_i == nr_of_batches:
                reseting_i = 0

    def choose_batch_data(self, iter_number, nr_of_batches):
        current_batch = iter_number % nr_of_batches
        return self.X_batches[current_batch], self.Y_batches[current_batch]

    def choose_from_score_param_hist(self, score_hist, param_hist):
        best_score_ix = 0
        best_score = score_hist[0]
        for i in range(len(score_hist)):
            if score_hist[i] < best_score:
                best_score_ix = i
        return score_hist[best_score_ix], param_hist[best_score_ix]

    def train_on_data(
        self,
        optimiser: AdamOptimiser,
        cf: CostFunction,
        X: List[array],
        Y: List[array],
        nr_of_iterations: int,
        nr_of_batches: int = 1,
    ):
        self.X = self.fix_up_data_to_1_row_matrix(X)
        self.Y = self.fix_up_data_to_1_row_matrix(Y)
        self.last_flat_wsbs = get_flattened_ws_bs(self.layers)
        self.create_n_batches(nr_of_batches, self.X, self.Y)
        self.score_hist = []
        self.wsbs_hist = []

        for i in tqdm(range(nr_of_iterations), leave=False):
            X_b, Y_b = self.choose_batch_data(i, nr_of_batches)
            gradient_w, gradient_b, cost_sum = self.__get_gradients_cost_per_sample(
                X_b, Y_b, cf
            )
            self.cost_hist.append(float(cost_sum))
            self.score_hist.append(float(cost_sum))
            self.wsbs_hist.append(self.last_flat_wsbs)
            flat_gradients = get_flattened_ws_bs_from_arrays(gradient_w, gradient_b)
            new_flat_wsbs = optimiser.calculate_descented_x(
                self.last_flat_wsbs, flat_gradients
            )

            self.update_with_flattened_w_and_b(new_flat_wsbs)
            self.last_flat_wsbs = new_flat_wsbs
        best_s, best_par = self.choose_from_score_param_hist(
            self.score_hist, self.wsbs_hist
        )
        self.update_with_flattened_w_and_b(best_par)
        return best_s

    def init_one_data_training(self):
        self.last_flat_wsbs = get_flattened_ws_bs(self.layers)

    def train_on_one_data(
        self,
        optimiser: AdamOptimiser,
        cf: CostFunction,
        xt: array,
        yt: array,
    ):
        x = array(xt).reshape(1, -1)
        y = array(yt).reshape(1, -1)

        gradient_w, gradient_b, cost_sum = self._get_gradient_cost_for_sample(x, y, cf)
        self.cost_hist.append(float(cost_sum))

        flat_gradients = get_flattened_ws_bs_from_arrays(gradient_w, gradient_b)
        new_flat_wsbs = optimiser.calculate_descented_x(
            self.last_flat_wsbs, flat_gradients
        )
        self.update_with_flattened_w_and_b(new_flat_wsbs)
        self.last_flat_wsbs = new_flat_wsbs

        self.update_with_flattened_w_and_b(new_flat_wsbs)

    def _get_gradient_cost_for_sample(
        self,
        x: array,
        y: array,
        cf: CostFunction,
    ) -> Tuple[List[array], List[array], float]:
        nout = self.calculate_output(x)
        dC = cf.get_deriv_cost_to_a_output(y, nout)
        cost_sum = cf.get_float_cost(y, nout)
        dws, dbs = self.backpropagate(dC)

        return (dws, dbs, cost_sum)


# @TODO
class BinaryResultOverlay:
    def __init__(self, number_of_possible_results: int, minimal_value: int = 0):
        self.nr_of_bin_neurons = number_of_possible_results
        self.offset = minimal_value

    def convert_activation_to_result(self, activation: array):
        a_list = activation.flatten()
        most_powerfull_activation = max(a_list)
        for i, a in enumerate(a_list):
            if float(a) == float(most_powerfull_activation):
                return i + self.offset

    def convert_activations_to_result(self, activations: List[array]):
        return [self.convert_activation_to_result(a) for a in activations]

    def convert_list_int_Y_to_grays_code_arrays(self, Y: List[array]):
        Yar = []
        print(Y[0])
        for y in Y:
            ya = np.zeros(self.nr_of_bin_neurons)
            ya[int(y - self.offset)] = 1
            Yar.append(ya)
        return Yar
