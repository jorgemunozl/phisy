#!/usr/bin/env python3
import numpy as np


def gram_schmidt(A, dim):
    """
    Devuelve una lista de vectores ortonormales
    y una matriz triangular superior.
    """
    # Trabajamos sobre una copia para no modificar la matriz original.
    A = A.copy()
    a0_norm = np.linalg.norm(A[0])
    e_1 = A[0]/a0_norm
    orthonormal_set = [e_1]
    triangular_sup = np.zeros([dim, dim])
    triangular_sup[0, 0] = a0_norm
    for i in range(1, dim):
        v_ij = A[i]
        for j in range(i):
            e_j = orthonormal_set[j]
            coef = np.dot(e_j, A[i])
            triangular_sup[j, i] = coef
            v_ij -= coef*e_j
        norm_vij = np.linalg.norm(v_ij)
        triangular_sup[i, i] = norm_vij
        orthonormal_set.append(v_ij/norm_vij)
    return orthonormal_set, triangular_sup


def absolute_error(A, E):
    a = 0
    e = 0
    for i in range(A.shape[0]):
        a += np.linalg.norm(A[i])**2
        e += np.linalg.norm((A-E)[i])**2
    return np.sqrt(e)/np.sqrt(a)


# Norma de la matriz A usando metrica euclidea
A = np.array([[7.0, -4.0, -5.0, -7.0],
              [-6.0, -1.0, -1.0, -3.0],
              [5.0, 5.0, 0.0,  -4.],
              [-1.0, 4.0, -4.0, 4.0]])


A_clone = A.copy()  # tranpose modify the original matrix
print("A:\n", A)
e, R = gram_schmidt(A.transpose(), A.shape[0])
Q = np.array(e)
print("Q: \n", Q)
print("R: \n", R)
E = Q.transpose()@R
print("QR:\n", E)
print("Print Absolute Error:", absolute_error(A_clone, E))
