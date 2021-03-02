#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  2 17:56:19 2021

@author: fernand
"""

#hello.py
from mpi4py import MPI 

comm = MPI.COMM_WORLD 
rank = comm.Get_rank()
print("hello world from process ", rank)