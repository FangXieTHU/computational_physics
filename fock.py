import numpy as np
np.set_printoptions(threshold=np.inf, precision=16, linewidth=10000)

import scipy.sparse as sps
import scipy.sparse.linalg as ssl

import time

import pickle

from numba import njit
from numba import types
from numba.experimental import jitclass
from numba.typed import Dict

@njit
def parity(x):
    n = 1
    y = x
    while y != 0:
        n *= -1
        y = y & (y - np.uint64(1))
    return n

@njit
def num_of_1(x):
    n = 0
    y = x
    while y != 0:
        n += 1
        y = y & (y - np.uint64(1))
    return n

@njit
def binary_position(x):
    state = x - (x & (x - np.uint64(1)))
    pos = 0
    while True:
        state = state >> 1
        if state == 0:
            break
        pos += 1
    return pos


@jitclass([('state', types.uint64), ('sign', types.int64), ('length', types.int64)])
class fermion_fock(object):
    
    def __init__(self, l):
        self.state = np.uint64(0)
        self.sign = 1
        self.length = l

    def set_state(self, x):
        self.state = np.uint64(x)
        self.sign = 1

    def annihilation_operator(self, n):
        if (self.state >> n) & 1 == 1:
            if n == 63:
                jordan_wigner_coef = 1
            else:
                jordan_wigner_coef = parity(self.state >> (n + 1))
            self.state = (self.state) ^ (1 << n)
            self.sign *= jordan_wigner_coef
        else:
            self.sign = 0

    def creation_operator(self, n):
        if (self.state >> n) & 1 == 1:
            self.sign = 0
        else:
            if n == 63:
                jordan_wigner_coef = 1
            else:
                jordan_wigner_coef = parity(self.state >> (n + 1))
            self.state = self.state | (1 << n)
            self.sign *= jordan_wigner_coef