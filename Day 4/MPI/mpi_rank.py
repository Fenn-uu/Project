#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  2 20:49:03 2021

@author: fernand
"""

from mpi4py import MPI 

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

print('Process %s is ready.'%rank)