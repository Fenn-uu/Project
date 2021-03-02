#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  2 22:10:00 2021

@author: fernand
"""

import simple_math


def test_simple_math():
    assert simple_math.simple_add(99.9999 , 0.0001) == 100.
    assert simple_math.simple_add(1,2) == 3
    

    assert simple_math.simple_mult(11,11) == 121

    assert simple_math.simple_div(16,4) == 4

    assert simple_math.poly_first(2,1,1) == 3

    assert simple_math.poly_second(2,1,1,1) == 7

    assert simple_math.power(1,1000) == 1
    
    assert simple_math.factorial(0) == 1
    assert simple_math.factorial(1) == 1
    assert simple_math.factorial(6) == 720
    

