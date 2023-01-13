import numpy as np
np.set_printoptions(threshold=np.inf, precision=16, linewidth=10000)

import scipy.sparse as sps
import scipy.sparse.linalg as ssl

import matplotlib
matplotlib.rcParams['text.usetex'] = True
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt

import time

import pickle

from numba import njit
from numba import types
from numba.typed import Dict

@njit
def parity(x):
    n = 1
    while x != 0:
        n *= -1
        x = x & (x - 1)
    return n

# Find the position of the last 1 in binary representation of number x (non negative int)
@njit
def position_binary(x): 
    pos = 0
    one_digit = x - (x & (x - 1))
    while one_digit != 1:
        one_digit = (one_digit >> 1)
        pos += 1
    return pos


# Generate the binary representation of a given sector, with total momentum Kt.
@njit
def shift(Nsite, input_array):
    output_array = np.zeros_like(input_array, dtype=np.uint64)
    for i in range(len(input_array)):
        output_array[i] = input_array[i] + (1 << (Nsite - 1))
    return output_array

# Using Just-In-Time complier RECursion to get the hilbert space binary representation of given quantum numbers.

@njit
def hilbert_space_sector_rec(N_site, N_particle, L_target, L_current, L_values):
    if N_site < N_particle or N_particle < 0:
        return np.empty(0, dtype=np.uint64)
    elif N_particle == 0:
        if L_current == L_target:
            return np.array([0], dtype=np.uint64)
        else:
            return np.empty(0, dtype=np.uint64)
    else:
        return auxiliary_0(N_site, N_particle, L_target, L_current, L_values)

@njit
def auxiliary_0(N_site, N_particle, L_target, L_current, L_values):
    return auxiliary_1(N_site, N_particle, L_target, L_current, L_values)

@njit
def auxiliary_1(N_site, N_particle, L_target, L_current, L_values):
    return np.append(hilbert_space_sector_rec(N_site - 1, N_particle, L_target, L_current, L_values), shift(N_site, hilbert_space_sector_rec(N_site - 1, N_particle - 1, L_target, L_current + L_values[N_site - 1], L_values)))

@njit
def hilbert_space(N_size, N_particle, L_target, L_values):
    return hilbert_space_sector_rec(N_size, N_particle, L_target, L_current=0, L_values=L_values)

@njit
def hilbert_space_dict(hilbert):
    result_dict = Dict.empty(key_type=types.uint64, value_type=types.int64)
    for i in range(len(hilbert)):
        result_dict[hilbert[i]] = i
    return result_dict

@njit
def find_target(hs_dict, tar):
    if tar in hs_dict:
        j = hs_dict[tar]
        return j
    else:
        return -1