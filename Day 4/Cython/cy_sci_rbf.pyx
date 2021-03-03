# -*- coding: utf-8 -*-
"""
Created on Thu Feb 18 15:46:26 2021

@author: ferde233
"""


from scipy.interpolate import Rbf



def cy_rbf_scipy(X, beta):

    N = X.shape[0]
    D = X.shape[1]    
    rbf = Rbf(X[:,0], X[:,1], X[:,2], X[:,3], X[:, 4], beta)
    #Xtuple = tuple([X[:, i] for i in range(D)])
    Xtuple = tuple([X[:, i] for i in range(D)])

    return rbf(*Xtuple)

