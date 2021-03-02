#!/usr/bin/env python3
# -*- coding: utf-8 -*-


"""
A collection of simple math operations
"""

def simple_add(a,b):
    return a+b

def simple_sub(a,b):
    return a-b

def simple_mult(a,b):
    return a*b

def simple_div(a,b):
    return a/b

def poly_first(x, a0, a1):
    return a0 + a1*x

def poly_second(x, a0, a1, a2):
    return poly_first(x, a0, a1) + a2*(x**2)

def power(a,b):
    return a**b

def factorial (n) : # n : positive integer
    n0 = 1
    
    if n == 0 :
        pass
    else :
        for k in range(n) :
            n0 *= k+1
    return n0
        
        