import numpy as np
from numpy import array
from typing import List, Tuple, Callable


class BaseLayerFunction:
    def get_output(self, z: array):
        return z

    def get_function_derivative(self, z: array):
        return 1


class ReluFunction(BaseLayerFunction):
    def get_output(self, z: array):
        return np.where(z < 0, 0, z)

    def get_function_derivative(self, z: array):
        return np.where(z < 0, 0.1, 1)


class TanHFunction(BaseLayerFunction):
    def get_output(self, z: array):
        return 2 / (1 + np.exp(-2 * z)) - 1

    def get_function_derivative(self, z: array):
        fout = 2 / (1 + np.exp(-2 * z)) - 1
        return 1 - (fout**2)


class SigmoidFunction(BaseLayerFunction):
    def get_output(self, z: array):
        val_exp = 1 / (1 + np.exp(-z))
        return np.array(val_exp)

    def get_function_derivative(self, z: array):
        val_exp = 1 / (1 + np.exp(-z))
        return np.array(val_exp * (1 - val_exp))

    # def get_derivative_after_w(self, weights: array, bias: array, dC: array):
    #     value =
    #     return dC
