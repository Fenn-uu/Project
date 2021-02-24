# -*- coding: utf-8 -*-
"""
Created on Tue Feb 23 15:48:18 2021

@author: ferde233
"""


from scipy.optimize import curve_fit
from numpy import arange, linspace, loadtxt, array, zeros, arctan, diag, sqrt, mean
from pandas import read_table
import numpy as np
import matplotlib.pyplot as plt

import Vdipole_functions


### https://qdusa.com/siteDocs/appNotes/1014-213.pdf 
### for more information.

############### Parameters ################

center = -0.25
# assumed sample position shift

Tmin = 250
Tmax = 310
#Temperature range assessed for background extrapolation


filename_1 = "Almax_pressure_cell (upright) _Gasket+TAS100+Nujol+Sn+Ruby_background_25_Oe_cooling_AmbientPressure_3cmfromtop_Pressure_Reapplied.dc.raw"


############### MPMS settings ################

N_points = 32 # number of points taken per scan
N_scans = 2 # number of scans
cut = N_points*N_scans # unless some points are rejected (e.g error during measurements), each scan should be separated by "cut" number of lines

L_scan = 4 # in cm

##############################################





class MagAnalysis() :
    ''' 
    Enter the MPMS raw data filepath of a '.dc.raw' and its raw data will be converted to ndarrays. If a diagnostic file '.dc.diag' exists it
    will also be stored.
    Multiple functions described below can be used to subtract the background and plot data
    '''
    
    def __init__(self, filepath ) :
        
        self.raw_data = read_table( filepath , dtype=float,skiprows =31 , header=None, sep=',' , usecols=(2,3,7,10) ).fillna(0).values
        # imports four columns : 0-field, 1-temperature, 2-sample position,3- averaged gradiometer voltage
        print('Raw file stored.')
        try:
            self.diag_data = read_table( filepath.replace('dc.raw','dc.diag') , dtype=float,skiprows =27 , header=None, sep=',' , usecols=(7,8) ).fillna(0).values
            # imports two columns from the diagnostic file : 0- "range code", 1- "gain code"
            print('Diagnostic file stored.')
        except:
            print('Diagnostic file not found.')
        self.n_r = arange( self.raw_data.shape[0] )
        # array of the total number of rows
        

        
    
    def findV( T , acc ) :
        '''
        Returns the position and voltage arrays of the first measurement found at T within accuracy 'acc'
        '''
        bg0_pos = np.array([])
        bg0_V = np.array([])
        
        
        
        
        
        
        
        