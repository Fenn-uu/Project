# -*- coding: utf-8 -*-
"""
Created on Thu Feb 18 15:46:25 2021

@author: ferde233
"""

from math import exp
import numpy as np

def cy_rbf_network(X, beta, theta):

    N = X.shape[0]
    D = X.shape[1]
    Y = np.zeros(N)

    for i in range(N):
        for j in range(N):
            r = 0
            for d in range(D):
                r += (X[j, d] - X[i, d]) ** 2
            r = r**0.5
            Y[i] += beta[j] * exp(-(r * theta)**2)

    return Y