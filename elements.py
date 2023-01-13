import numpy as np
np.set_printoptions(threshold=np.inf)

from numba import njit
from numba import types
from numba.experimental import jitclass
from numba.typed import Dict

import time
import pickle

# A single body state is labeled by kc=0, 1,..., Nc-1 for c-electron, and s=0, 1 for d-electron. The single-body index is ind = kc or ind = Nc + s

# Kinetic energy and quadratic Hamiltonian

@njit
def factorial(n):
    if n == 0 or n == 1:
        return 1.0
    elif n < 0:
        return 1.0
    else:
        return factorial(n - 1) * n

@njit
def choose(n, m):
    if n >= 0 and m >= 0:
        return factorial(n)/(factorial(m) * factorial(n - m))
    else:
        return 0.0

@njit
def pseudopotential(m1, m2, m):
    result = 0.0
    factor = np.sqrt(choose(m1 + m2, m)/(choose(m1 + m2, m1) * (2.0 ** (m1 + m2))))

    for k in range(m1 + m2 - m + 1):
        if m + k - m2 >= 0 and k <= m2:
            result += (-1) ** (m + k - m2) * choose(m1 + m2 - m, k) * choose(m, m + k - m2)
    result *= factor
    return result

@njit
def four_fermion_terms(Vm_array, N_max):
    U_values_arr = []
    op1_int_arr = []
    op2_int_arr = []
    op3_int_arr = []
    op4_int_arr = []

    for m1 in range(N_max):
        for m2 in range(N_max):
            for n1 in range(N_max):
                n2 = m1 + m2 - n1
                if n2 < 0 or n2 >= N_max:
                    continue
                else:
                    if m1 == m2 or n1 == n2:
                        continue
                    else:
                        U = 0.0
                        for m in range(len(Vm_array)):
                            U += Vm_array[m] * pseudopotential(m1, m2, m + 1) * pseudopotential(n1, n2, m + 1)
                        op1_int_arr.append(m2)
                        op2_int_arr.append(m1)
                        op3_int_arr.append(n1)
                        op4_int_arr.append(n2)
                        U_values_arr.append(U/2)

    U_values_arr = np.array(U_values_arr, dtype=np.float64)
    op1_int_arr = np.array(op1_int_arr, dtype=np.int64)
    op2_int_arr = np.array(op2_int_arr, dtype=np.int64)
    op3_int_arr = np.array(op3_int_arr, dtype=np.int64)
    op4_int_arr = np.array(op4_int_arr, dtype=np.int64)

    return U_values_arr, op1_int_arr, op2_int_arr, op3_int_arr, op4_int_arr