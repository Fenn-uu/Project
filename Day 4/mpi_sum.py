#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  2 18:04:12 2021

@author: fernand
"""
import numpy as np
from mpi4py import MPI 

comm = MPI.COMM_WORLD
rank = comm.Get_rank()


hey = np.array([])

    
    
if rank == 1 :
    print('Process %s ready !'%(rank))
    hey = np.append(hey, np.random.randint(0,100,(1)))
    print('I send a %s'%(hey[0]))
    comm.Send(hey, dest= 0)

elif rank == 0 :
    print('Process %s ready !'%(rank))
    comm.Recv(hey, source=1)
    hey0 = np.random.randint(0,100,(1))
    
    sum_num = hey0[0]-hey[0]
    print('Received the number %s from process %s \nI got a %s'%(hey[0],1,sum_num))
    
    if hey0[0] > hey[0] :
        print('Process 0 won !')
    elif hey0[0] < hey[0] :
        print('Process 1 won !')
    else :
        print('it\'s a tie.')
