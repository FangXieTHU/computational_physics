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

import matplotlib
matplotlib.rcParams['text.usetex'] = True
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
from matplotlib import cm


N_max = 45
N_particle = 5
potential_slope = 0.0
edge = 15

L_values = np.arange(N_max)

import hilbert as hlbt
import elements as ele

import fock

Vm_arr = np.array([1.0, 0.0, 0.0])
U_values_arr, op1_int_arr, op2_int_arr, op3_int_arr, op4_int_arr = ele.four_fermion_terms(Vm_arr, N_max)

Ek_values_arr = np.zeros(N_max, dtype=np.float64)
op1_arr = L_values
op2_arr = L_values

for l in range(N_max):
    if l >= edge:
        Ek_values_arr[l] = potential_slope * (l - edge + 1)

@njit
def manybody_hamiltonian_length(hilbert_space_arr, hilbert_dict, U_values_arr, op1_int_arr, op2_int_arr, op3_int_arr, op4_int_arr, Ek_values_arr, op1_arr, op2_arr):
    length_of_data = 0
    num_of_int = len(U_values_arr)
    num_of_Ek = len(Ek_values_arr)
    dim = len(hilbert_space_arr)

    fock_state = fock.fermion_fock(N_max)

    for i in range(dim):
        x = hilbert_space_arr[i]

        U_term = 0
        while U_term < num_of_int:
            fock_state.set_state(x)
            pos4 = op4_int_arr[U_term]
            pos3 = op3_int_arr[U_term]
            pos2 = op2_int_arr[U_term]
            pos1 = op1_int_arr[U_term]

            fock_state.annihilation_operator(pos4)
            if fock_state.sign == 0:
                U_term += 1
                continue
            
            fock_state.annihilation_operator(pos3)	
            if fock_state.sign == 0:
                U_term += 1
                continue
            
            fock_state.creation_operator(pos2)
            if fock_state.sign == 0:
                U_term += 1
                continue
            
            fock_state.creation_operator(pos1)
            if fock_state.sign != 0:
                length_of_data += 1
            U_term += 1

        for Ek_term in range(num_of_Ek):
            fock_state.set_state(x)
            op1 = op1_arr[Ek_term]
            op2 = op2_arr[Ek_term]
            
            fock_state.annihilation_operator(op2)
            if fock_state.sign == 0:
                continue
            
            fock_state.creation_operator(op1)
            if fock_state.sign == 0:
                continue
            length_of_data += 1
    return length_of_data

@njit
def manybody_hamiltonian_array(length, hilbert_space_arr, hilbert_dict, U_values_arr, op1_int_arr, op2_int_arr, op3_int_arr, op4_int_arr, Ek_values_arr, op1_arr, op2_arr):
    dim = len(hilbert_space_arr)
    indptr = np.zeros(dim + 1, dtype=np.int32)
    indices = np.zeros(length, dtype=np.int32)
    data = np.zeros(length, dtype=np.float64)

    data_index = 0

    num_of_int = len(U_values_arr)
    num_of_Ek = len(Ek_values_arr)

    fock_state = fock.fermion_fock(N_max)

    for i in range(dim):
        x = hilbert_space_arr[i]

        U_term = 0
        while U_term < num_of_int:
            fock_state.set_state(x)
            pos4 = op4_int_arr[U_term]
            pos3 = op3_int_arr[U_term]
            pos2 = op2_int_arr[U_term]
            pos1 = op1_int_arr[U_term]
            U_value = U_values_arr[U_term]

            fock_state.annihilation_operator(pos4)
            if fock_state.sign == 0:
                U_term += 1
                continue
            
            fock_state.annihilation_operator(pos3)	
            if fock_state.sign == 0:
                U_term += 1
                continue
            
            fock_state.creation_operator(pos2)
            if fock_state.sign == 0:
                U_term += 1
                continue
            
            fock_state.creation_operator(pos1)
            if fock_state.sign != 0:
                j = hlbt.find_target(hilbert_dict, fock_state.state)
                indices[data_index] = j
                data[data_index] = fock_state.sign * U_value
                data_index += 1
            
            U_term += 1

        for Ek_term in range(num_of_Ek):
            fock_state.set_state(x)
            op1 = op1_arr[Ek_term]
            op2 = op2_arr[Ek_term]
            Ek = Ek_values_arr[Ek_term]
            
            fock_state.annihilation_operator(op2)
            if fock_state.sign == 0:
                continue
            fock_state.creation_operator(op1)
            if fock_state.sign == 0:
                continue
            j = hlbt.find_target(hilbert_dict, fock_state.state)
            indices[data_index] = j
            data[data_index] = fock_state.sign * Ek
            data_index += 1

        indptr[i + 1] = data_index
    return indptr, indices, data

def manybody_hamiltonian(N_particle, L):
    hilbert_space_arr = hlbt.hilbert_space(N_max, N_particle, L, L_values)
    hilbert_dict = hlbt.hilbert_space_dict(hilbert_space_arr)
    length = manybody_hamiltonian_length(hilbert_space_arr, hilbert_dict, U_values_arr, op1_int_arr, op2_int_arr, op3_int_arr, op4_int_arr, Ek_values_arr, op1_arr, op2_arr)
    dim = len(hilbert_space_arr)
    indptr, indices, data = manybody_hamiltonian_array(length, hilbert_space_arr, hilbert_dict, U_values_arr, op1_int_arr, op2_int_arr, op3_int_arr, op4_int_arr, Ek_values_arr, op1_arr, op2_arr)
    mat = sps.csr_matrix((data, indices, indptr), shape=(dim, dim))
    return mat

L_arr = []
E_arr = []

for L in range(41):
    H = manybody_hamiltonian(N_particle, L)
    H = H.toarray()
    u, v = np.linalg.eigh(H)
    for E in u:
        E_arr.append(E)
        L_arr.append(L)

fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(L_arr, E_arr, 'r_', markersize=10, mew=1)
ax.set_xlabel(r'$L$', fontsize=20)
ax.set_ylabel(r'$E$', fontsize=20)
ax.tick_params(labelsize=15)
ax.set_ylim(-0.2, 6)
fig.set_tight_layout(True)

name_str = f'V1_{Vm_arr[0]:.5f}_V3_{Vm_arr[2]:.5f}_N_{N_particle}_Nmax_{N_max}_edge_{edge}_slope_{potential_slope:.5f}'
name_str = name_str.replace('.', '_')
plt.savefig(f'./figures/Laughlin_state_{name_str}.pdf')