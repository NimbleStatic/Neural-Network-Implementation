import numpy as np
from numpy import array
from typing import List, Tuple, Callable
from layer_functions import BaseLayerFunction


class LayerBase:
    def __init__(
        self,
        neuron_size,
        wraping_function: BaseLayerFunction,
    ):
        self.neuron_size = neuron_size
        self.activation_function = wraping_function
        self.b = np.random.normal(loc=0, scale=1, size=self.neuron_size)

        self.b_size = self.neuron_size
        self.last_activation = None
        self.last_output = None

    def set_previous_size(self, size):
        self.previous_layer_size = size
        self.w_size = (self.previous_layer_size, self.neuron_size)

        self.w = np.random.normal(loc=0, scale=1, size=self.w_size)
        self.full_size_of_w = self.neuron_size * self.previous_layer_size

    def compute_deriv_cost_after_w(self, deriv_cost_after_next_layer: array) -> array:
        func_prim = self.activation_function.get_function_derivative(self.last_z)

        v1 = np.multiply(func_prim, deriv_cost_after_next_layer)
        v2 = np.dot(self.last_activation.T, v1)

        return v2

    def compute_deriv_cost_after_b(self, deriv_cost_after_next_layer: array) -> array:
        func_prim = self.activation_function.get_function_derivative(self.last_z)
        cost_after_b = np.multiply(func_prim, deriv_cost_after_next_layer)
        return cost_after_b

    def compute_deriv_cost_after_this_layer(
        self, deriv_cost_after_next_layer: array
    ) -> array:
        func_prim = self.activation_function.get_function_derivative(self.last_z)
        # wz_prim = np.dot(self.w, func_prim.T)
        # cost_after_layer = np.dot(deriv_cost_after_next_layer.T, wz_prim)
        v1 = np.multiply(func_prim, deriv_cost_after_next_layer)
        v2 = np.dot(v1, self.w.T)
        return v2

    def compute_output(self, a0: array):
        self.last_activation = a0
        z = np.dot(a0, self.w)
        z += self.b
        self.last_z = z
        a_out = self.activation_function.get_output(z)
        self.last_output = a_out
        return a_out

    def update_w(self, new_w: array):
        self.w = new_w

    def update_b(self, new_bias: array):
        self.b = new_bias


class CoveredLayer(LayerBase):
    def __init__(
        self,
        neuron_size,
        wraping_function: BaseLayerFunction,
        neurons_to_zero: List[int] = None,
        neurons_to_cover: List[int] = None,
    ):
        super().__init__(neuron_size, wraping_function)
        self.zeroed_neurons = neurons_to_zero
        self.covered_neurons = neurons_to_cover
        zerobmask = self.create_mask_columns((1, neuron_size), self.zeroed_neurons)
        self.b = self.mask_matrix(self.b, zerobmask)

        self.bmask = self.create_mask_columns((1, neuron_size), self.covered_neurons)

    def uncover_neurons(self, nr_of_neurons: int = 1):
        if self.covered_neurons is None or self.covered_neurons == []:
            self.covered_neurons = None
            self.update_masks()
            return None
        for i in range(nr_of_neurons):
            if len(self.covered_neurons) != 0:
                self.covered_neurons.pop()
        self.update_masks()

    def update_masks(self):
        self.bmask = self.create_mask_columns(
            (1, self.neuron_size), self.covered_neurons
        )
        self.wmask = self.create_mask_columns(self.w_size, self.covered_neurons)

    def mask_matrix(self, matrix, mask):
        # return np.multiply(matrix, mask)
        return matrix * mask

    def set_previous_size(self, size):
        self.previous_layer_size = size
        self.w_size = (self.previous_layer_size, self.neuron_size)

        self.w = np.random.normal(loc=0, scale=1, size=self.w_size)
        mask = self.create_mask_columns(self.w_size, self.zeroed_neurons)
        self.w = self.mask_matrix(self.w, mask)
        self.full_size_of_w = self.neuron_size * self.previous_layer_size
        self.wmask = self.create_mask_columns(self.w_size, self.covered_neurons)

    def create_mask_columns(self, size: tuple, columns_to_mask: List[int]):
        rn = size[0]
        cn = size[1]
        mask = np.ones(shape=size)
        if columns_to_mask is None:
            return mask
        for ir in range(rn):
            for ic in range(cn):
                if ic in columns_to_mask:
                    mask[ir][ic] = 0
        return mask

    def compute_deriv_cost_after_w(self, deriv_cost_after_next_layer: array) -> array:
        func_prim = self.activation_function.get_function_derivative(self.last_z)

        v1 = np.multiply(func_prim, deriv_cost_after_next_layer)
        v2 = np.dot(self.last_activation.T, v1)
        v2 = self.mask_matrix(v2, self.wmask)

        return v2

    def compute_deriv_cost_after_b(self, deriv_cost_after_next_layer: array) -> array:
        func_prim = self.activation_function.get_function_derivative(self.last_z)
        cost_after_b = np.multiply(func_prim, deriv_cost_after_next_layer)
        cost_after_b = self.mask_matrix(cost_after_b, self.bmask)
        return cost_after_b

    def compute_deriv_cost_after_this_layer(
        self, deriv_cost_after_next_layer: array
    ) -> array:
        func_prim = self.activation_function.get_function_derivative(self.last_z)
        # wz_prim = np.dot(self.w, func_prim.T)
        # cost_after_layer = np.dot(deriv_cost_after_next_layer.T, wz_prim)
        v1 = np.multiply(func_prim, deriv_cost_after_next_layer)
        masked_w = self.mask_matrix(self.w, self.wmask)
        v2 = np.dot(v1, masked_w.T)
        # v2 = self.mask_matrix(v2, self.bmask)
        return v2

    def compute_output(self, a0: array):
        self.last_activation = a0
        z = np.dot(a0, self.w)
        z += self.b
        z = self.mask_matrix(z, self.bmask)
        self.last_z = z
        a_out = self.activation_function.get_output(z)
        a_out = self.mask_matrix(a_out, self.bmask)
        self.last_output = a_out
        return a_out


if __name__ == "__main__":
    pass
