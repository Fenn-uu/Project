# -*- coding: utf-8 -*-
"""
Created on Thu Feb 18 15:08:31 2021

@author: ferde233
"""
import numpy as np
import primes
import cy_primes

from rbf import rbf_network, rbf_scipy
import cy_np_rbf
import cy_sci_rbf


import time

t0 = time.time()
primes.primes(1000)
print("Python primes : %s ms "%(time.time() - t0))


t0 = time.time()
cy_primes.primes(1000)
print("Cython primes : %s ms "%(time.time() - t0))



D = 5
N = 1000
X = np.array([np.random.rand(N) for d in range(D)]).T
beta = np.random.rand(N)
theta = 10

t0 = time.time()
rbf_network(X, beta, theta)
print("Python rbf: %s ms"%(time.time() - t0))

t0 = time.time()
rbf_scipy(X, beta)
print("Scipy rbf: %s ms"%(time.time() - t0))

t0 = time.time()
cy_np_rbf.cy_rbf_network(X, beta, theta)
print("Python rbf: %s ms"%(time.time() - t0))

t0 = time.time()
cy_sci_rbf.cy_rbf_scipy(X, beta)
print("Python rbf: %s ms"%(time.time() - t0))