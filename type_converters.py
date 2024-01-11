from numpy import array
import numpy as np
from typing import Callable, Tuple, List


def convert_w_to_list(w: array) -> array:
    list_w_val = np.ravel(w)
    return list_w_val


def convert_list_to_w(list_w: array, w_size: Tuple[int, int]) -> array:
    # zmienia vector wartości wag na macierz wag
    w = list_w.reshape(w_size)
    return w


def get_flattened_weights(layers):
    # zwraca vector wszystkich płaskich wag dla wszystkich warstw
    full_weights = []
    for layer in layers:
        full_weights.append(get_flat_layer_weights(layer.w))
    return np.concatenate(full_weights)


def get_weights(layers):
    weights = []
    for layer in layers:
        weights.append(layer.w)
    return weights


def get_biases(layers):
    biases = []
    for layer in layers:
        biases.append(layer.b)
    return biases


def get_flattened_bias(layers):
    # zwraca vector wszystkich płaskich biasów dla wszystkich warstw
    full_bias = []
    for layer in layers:
        full_bias.append(layer.b.flatten())
    return np.concatenate(full_bias)


def get_flattened_ws_bs(layers):
    # Zwraca wektor wszystkich wag i biasów dla wszystkich warstw (wykorzystywany do optymalizacji parametrów całego neural network)
    ws = get_flattened_weights(layers)
    bs = get_flattened_bias(layers)
    all_s = []
    for w in ws:
        all_s.append(w)
    for b in bs:
        all_s.append(b)
    return array(all_s)


def get_flattened_ws_bs_from_arrays(ws, bs):
    # Zwraca wektor wszystkich wag i biasów dla wszystkich warstw (wykorzystywany do optymalizacji parametrów całego neural network)
    ws = get_flattened_weights_arrays(ws)
    bs = get_flattened_weights_arrays(bs)
    all_s = []
    for w in ws:
        all_s.append(w)
    for b in bs:
        all_s.append(b)
    return array(all_s)


def get_flattened_weights_arrays(ws):
    # zwraca vector wszystkich płaskich wag dla wszystkich warstw
    full_weights = []
    for w in ws:
        full_weights.append(get_flat_layer_weights(w))
    return np.concatenate(full_weights)


def get_flat_layer_weights(weights):
    return np.ravel(weights)


def get_normal_w(layer, weights):
    return weights.reshape(layer.w_size)
