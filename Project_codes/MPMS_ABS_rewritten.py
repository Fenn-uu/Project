# -*- coding: utf-8 -*-
"""
Created on Tue Feb 23 15:48:18 2021

@author: ferde233
"""


from scipy.interpolate import interp1d, Rbf
from scipy.optimize import curve_fit
from numpy import arange, linspace, array, polyfit
from pandas import read_table
import numpy as np
import matplotlib.pyplot as plt

import Vdipole_functions


### https://qdusa.com/siteDocs/appNotes/1014-213.pdf 
### for more information.

############### Parameters ################
global center
center = -0.5 # in cm
# assumed sample position shift 

Tmin = 250
Tmax = 310
#Temperature range assessed for background extrapolation


filename_1 = "Almax_pressure_cell (upright) _TAS0+Nujol+Ruby_MT_10_Oe_4K-2K_PressureApplied_2.4cmfromtop.dc.raw"

sample_name = 'TAS(0) in HP cell'

############### MPMS settings ################

N_points = 32 # number of points taken per scan
N_scans = 2 # number of scans
cut = N_points*N_scans # unless some points are rejected (e.g error during measurements), each scan should be separated by "cut" number of lines

L_scan = 4 # in cm

##############################################

gain = [1.,2.,5.,10.] # table of the gain code values  in the diagnostic file to evaluate the moment, see Table 1 in the ref below
M_emu = lambda m0, r, g : (m0*1.825)/(8890.6*(10**-r)*gain[int(g)]*0.9125)
# calculates the magnetization in emu units from the gain 'g' and range 'r' tables in the .dc.diag file 
# m0 is the amplitude from the functions defined in the Vdipole_functions
# See ### https://qdusa.com/siteDocs/appNotes/1014-213.pdf    p.2

##############################################



class MagAnalysis() :
    ''' 
    Enter the MPMS raw data filepath of a '.dc.raw' and its raw data will be converted to ndarrays. If a diagnostic file '.dc.diag' exists it
    will also be stored.
    Multiple functions described below can be used to subtract the background and plot data
    '''
    
    def __init__( self, filepath ) :
        
        self.raw_data = read_table( filepath , dtype=float,skiprows =31 , header=None, sep=',' , usecols=(2,3,7,10) ).fillna(0).values
        # imports four columns : 0-field, 1-temperature, 2-sample position,3- averaged gradiometer voltage
        print('Raw file stored successfully.')
        try:
            self.diag_data = read_table( filepath.replace('dc.raw','dc.diag') , dtype=float,skiprows =27 , header=None, sep=',' , usecols=(7,8) ).fillna(0).values
            # imports two columns from the diagnostic file : 0- "range code", 1- "gain code"
            print('Diagnostic file stored successfully.')
        except:
            print('Diagnostic file not found...')
        self.n_r = arange( self.raw_data.shape[0] )
        # array of the total number of rows
        
        self.BG_data = array([])
        # can be filled with background data (1)
        self.BG_f = array([])
        # can be filled with an interpolation function (2)
        self.treated_data = array([])
        
        
        
        self.Vdip0 = Vdipole_functions.Vdipole_lin_centered
        # function to be used for plotting  and magnetization extraction (see Vdipole_functions for the complete list and number of arguments)
        self.Vdip0_guess = array([ 0. , 0., 1.])
        # A start guess array with parameters of the Vdip0 function  

    
    def findV( self , T , acc ) :
        '''
        Returns the position and voltage arrays of the first measurement found at T within accuracy 'acc'.
        '''
        bg0_pos = array([])
        bg0_V = array([])
        for i in self.n_r :
            if T - acc < self.raw_data[i,1] < T + acc and self.raw_data[i,3] != 0. :
                print('Matching T value found.\n\t T= %a'%(self.raw_data[i,1]))
                # for p in range(N_points):
                bg0_pos = np.append( bg0_pos , self.raw_data[ int(i) : int(i+N_points) , 2] - 2 ) # appends the positions (between -2 and 2 cm)
                bg0_V = np.append( bg0_V , self.raw_data[ int(i) : int(i+N_points) , 3]) # appends the last gradiometer voltage scan (averaged)
                
                    # bg0_pos = np.append(bg0_pos, self.raw_data[i+(N_scans-1)*(N_points)+p,2] - 2 ) # appends the last position (between -2 and 2 cm)
                    # bg0_V = np.append(bg0_V, self.raw_data[i+(N_scans-1)*(N_points)+p,3] ) # appends the last of the averaged gradiometer voltage
                break
        if bg0_pos.size == 0 : # if no value is found then the arrays should remain empty 
            print('No matching T value found...')
        return array(bg0_pos), array(bg0_V)
    
        
    
    def BG_polyfit_T_coeff( self , Tmin , Tmax , n_deg) :
        ''' 
        Uses the raw data of a temperature-dependent  acquisition and fits the position and voltage arrays 
        with polynomials of degree 'n_deg' for each point.
        A new point starts every 'cut' rows, see l.38
        
        Returns n the number of magnetic measurements present, t the list of temperatures
                h the list of fields and M_param a matrix of polynomial coefficients (descending order of the columns)
        
        The number of voltage points per dataset is self.raw_data.shape[0], so the number of 
        magnetization points is self.raw_data.shape[0]/cut
        '''
        
        bg_pos = array([]) # position list of a single scan  for the polynomial fit (reset at every temperature point) 
        bg_V = array([]) # gradiometer voltage list for the polynomial fit (reset at every temperature point)
        
        t = array([]) # array of temperatures encountered
        h = array([]) # array of magnetic field values
        M_param = array([[]]) # matrix of the fitted parameters with dimensions (n_rows) x (n_deg + 1)
        
        for i in arange(self.raw_data.shape[0]/cut) :
            
            if Tmin <= self.raw_data[int(i*cut),1] <= Tmax :
                
                bg_pos = np.append( bg_pos , self.raw_data[ int( i*cut+(N_scans-1)*(N_points)) : int(i*cut+(N_scans)*(N_points)) , 2] - 2 ) # appends the positions (between -2 and 2 cm)
                bg_V = np.append( bg_V , self.raw_data[ int( i*cut+(N_scans-1)*(N_points)) : int(i*cut+(N_scans)*(N_points)) , 3]) # appends the last gradiometer voltage scan (averaged)
                     
                param = polyfit(bg_pos , bg_V , n_deg) # A parameter array with n_deg + 1 elements
                                                       # arranged from highest order to lower
                                                       
                M_param = np.append( M_param , param )
                
                t = np.append( t , self.raw_data[int(i*cut),1] ) # We also record the temperature
                h = np.append( h , self.raw_data[int(i*cut),0] ) # and field
                
                bg_pos = array([])
                bg_V = array([])
                
        print('Whole dataset analysed.')
        
        n = t.shape[0] # The number of figures fitted (i.e number of final magnetization points).
        
        M_param = M_param.reshape( ( n , n_deg+1 ) ) # Reshapes the parameters collected into rows
                                                     # from highest order (M_param[:,0]) to lowest (M_param[:,n_deg])
        
        return n , t , h , M_param
    
    
    
    def interpol_1D(self, n_deg) :
        '''
        Interpolation of T-dependent OR H-dependent 1D array with a polynomial of degree 'n_deg'.
        returns an array of length (n_deg+1) with interpolation functions for the coefficients of the
        gradiometer voltage response along L_scan of (i.e returns f(T) or f(H)).
        '''
        
        T_yn = input('Interpolate for temperature ? [y/n] \n(H = cst)')
        
        if T_yn == 'Y' or T_yn == 'y' :
            
            Tmin_cut = float( input('Tmin = ') ) # Thresholds
            Tmax_cut = float( input('Tmax = ') )
            n_points , T , H , M_start = self.BG_polyfit_T_coeff(Tmin_cut,Tmax_cut,n_deg)
            
            f_interp = array([]) # empty array to be filled with n_deg+1 rows of coefficients interpolation
                                 # functions from highest order to lowest, computed for the interval [Tmin,Tmax]
            
            for i in arange(int(n_deg+1)) :
                
                f_interp = np.append( f_interp , interp1d(T,M_start[:,i]) ) # interpolation function of the ith highest order
            
            print('Temperature interpolation at H = %s Oe completed.'%(H[0]))
                
        else :
            
            H_yn = input('Interpolate for magnetic field ? [y/n] \n(T = cst)')
            
            if H_yn == 'Y' or H_yn == 'y' :
                
                n_points , T , H , M_start = self.BG_polyfit_T_coeff(Tmin_cut,Tmax_cut,n_deg)
                
                f_interp = array([]) # empty array to be filled with n_deg+1 rows of coefficients interpolation
                                     # functions from highest order to lowest.
                
                for i in arange(int(n_deg+1)) :
                
                    f_interp = np.append( f_interp , interp1d(H,M_start[:,i]) ) # interpolation function of the ith highest order
                
                print('Magnetic field interpolation at T = %s Oe completed.'%(T[0]))
                
            else :
                
                print('Try the self.interpol_2D to have a grid interpolation along both T and H.')
                
        return f_interp
    
    
    
    def interpol_2D(self, n_deg) :
        '''
        Interpolation of T- and H-dependent dataset with a polynomial of degree 'n_deg' performed on the raw data.
        returns an array of length (n_deg+1) with interpolation functions for the coefficients of the
        gradiometer voltage response over the position range L_scan (i.e returns f(T,H)).
        '''
        
        Hmin = min( self.raw_data[:,0] )
        Hmax = max( self.raw_data[:,0] )
        
        Tmin = min( self.raw_data[:,1] ) # Thresholds
        Tmax = max( self.raw_data[:,1] )
        
        n_points , T , H , M_start = self.BG_polyfit_T_coeff(Tmin,Tmax,n_deg)
        
        f_interp = array([]) # empty array to be filled with n_deg+1 rows of coefficients interpolation
                             # functions from highest order to lowest, computed for the interval [Tmin,Tmax]
        
        for i in arange(int(n_deg+1)) :
            
            f_interp = np.append( f_interp , Rbf(T,H,M_start[:,i]) ) # interpolation function of the ith highest order
        
        print('Interpolation completed.\nTemperature range : [ %s K , %s K ]\nMagnetic field range : [ %s Oe , %s Oe ]'%((Tmin,Tmax,Hmin,Hmax)))       
              
        return f_interp
    
    
    
    def BG_interpol_2D(self) :
        '''
        Interpolation of T- and H-dependent dataset with a polynomial of degree 'n_deg' performed on 
        the data stored in the matrix self.BG_data (imported from other files).
        returns an array of length (n_deg+1) with interpolation functions for the coefficients of the
        gradiometer voltage response over the position range L_scan (i.e returns f(T,H)).
        '''
        if self.BG_data.shape[0] == 0 :
         
            print('Empty background data file. Cannot perform interpolation (try self.BG_add first).')
        
        else :
            
            n_deg = self.BG_data.shape[1] - 3
            print('Polynomial of order %s found in the background file'%(n_deg))
            
            H = self.BG_data[:,0]
            Hmin = min( self.BG_data[:,0] )
            Hmax = max( self.BG_data[:,0] )
            
            T = self.BG_data[:,1] 
            Tmin = min( self.BG_data[:,1] ) # Thresholds
            Tmax = max( self.BG_data[:,1] )
            
            
            f_interp = array([]) # empty array to be filled with n_deg+1 rows of coefficients interpolation
                                 # functions from highest order to lowest, computed for the interval [Tmin,Tmax]
            
            for i in arange(int(n_deg+1)) :
                
                f_interp = np.append( f_interp , Rbf( T , H , self.BG_data[:,i+2]) ) # interpolation function of the ith highest order
            
            print('Interpolation completed.\nTemperature range : [ %s K , %s K ]\nMagnetic field range : [ %s Oe , %s Oe ]'%((Tmin,Tmax,Hmin,Hmax)))       
            
            self.BG_f = f_interp
            
        return f_interp
    
    def P_BG_coeff(self,T,H) :
        '''
        Returns an array in ascending order of the polynomial coefficients (order n) of the gradiometer voltage at the
        temperature T and magnetic field H
        '''
        
        f = self.BG_f
        n = self.BG_deg
        p = array([])
        
        for i in arange(int(n+1)) :
            p = np.append( p , f[i](T,H)) # Evaluating the function for every coefficient (descending order)
        
        p = p[::-1] # inverting the order (last to first)
        return p # Array of the polynomial coefficients in ASCENDING order
    
    def BG_erase(self) :
        '''
        Empties the self.BG_data matrix of all previously recorded background data. 
        '''
        self.BG_data = array([])
        self.BG_f = array([])
        
    @classmethod
    def BG_import(cls , BG_filename , n_deg) :
        '''
        This function imports the background from filename, runs the BG_polyfit_T_coeff functions on it and 
        returns the T and H coordinates, as well as M_param
        '''
        
        BG_new = cls(BG_filename)
        
        Tmin = min( BG_new.raw_data[:,1] ) # Thresholds
        Tmax = max( BG_new.raw_data[:,1] )
        
        n_BG , T_BG , H_BG , M_param = BG_new.BG_polyfit_T_coeff(Tmin , Tmax , n_deg)
        # The data arrays to be stored in order to subtract background
        
        return n_BG , T_BG , H_BG , M_param
    
    def BG_add(self, BG_filename , n_deg) :
        ''' 
        Uses a background raw datafile data BG_filename with various T and H values to define its Rbf of its 
        polynomial coefficients before subtracting it to raw data.
        If background already exists it is possible to append other files to the dataset (e.g different T and/or H).
        
        The self.BG_data is a 2D matrix of dim (N x n_deg +3) with N the number of magnetization points, and
        the first column [:,0] corresponding to magnetic field, [:,1] to temperature, then polynomial in decreasing power
        order from highest [:,2] to lowest [:,n_deg+3].
        
        Once the Rbf is established from self.BG_data, 
        '''
        
        if self.BG_data.shape[0] == 0 :
            
            print('No background file found.')
            fill_yn = input('Would you like to add a file (.dc.raw) ? [y/n]')
            
            if fill_yn == 'y' or fill_yn == 'Y' :
                
                n_import , T_import , H_import , M_param_import = self.BG_import(BG_filename, n_deg)
                print('%s magnetic measurements present in the background file.'%(n_import))
                self.BG_data = np.concatenate( (array([H_import]).T, array([T_import]).T , M_param_import) ,axis=1)
                # [:,0] : Magnetic field , [:,1] : Temperature , [:,2] : highest order coeff ,..., : [:,n_deg+3] : lowest order (constant)
                print('Data stored in self.BG_data.')
                
                self.BG_deg = n_deg # The polynomial order used is stored
                
            else :
                print('No file added.')
        
        else: 
            
            print('Background file found !')
            append_yn = input('Would you like to append the current file (.dc.raw) ? [y/n] \n\nWarning : Each dataset should have a matching n_deg and should only be added once.')
            
            if append_yn == 'y' or append_yn == 'Y' :
                
                if self.BG_deg == n_deg :
                    
                    n_append , T_append , H_append , M_param_append = self.BG_import(BG_filename, n_deg)
                    print('%s magnetic measurements present in the background file.'%(n_append))
                    M_new = np.concatenate( (array([H_append]).T, array([T_append]).T , M_param_append) ,axis=1)
                    # We first create a matrix of dim (N2 x (n_deg + 3)) in the same way as for the 'import' right above
                    self.BG_data = np.concatenate( (self.BG_data,M_new) , axis=0 )
                    # Then we stitch it to self.BG_data of dim (N1 x (n_deg + 3)), resulting in a ( (N1+N2) x (n_deg + 3)) matrix
                    print('Data stored in self.BG_data.')
                
                else :
                
                    print('Polynomial order n_deg = %s does not match the previous one (%s)'%(n_deg,self.BG_deg))    
                
            else :
                
                print('self.BG_data left untouched.')
                
    
        
    def BG_subtraction_TH (self) :
        '''
        Creates a replica of self.raw_data where the last column's values (Gradiometer voltage) have the
        background subtracted using the Rbf function computed from self.BG_interpol_2D.
        '''
        
        self.treated_data = self.raw_data.copy()
            
        
        if self.BG_f.shape[0] != 0 :
            
            
            P_BG = lambda x , T , H :  np.polynomial.polynomial.Polynomial(self.P_BG_coeff(T,H))(x)
            # x : position , T : temperature, H :magnetic field
                      
            bg_pos = array([]) # To be filled with the position values at each measurement, then emptied
            bg_V = array([])   # To be filled with the gradiometer voltages values at each measurement, then emptied
            
            P_BG_shift_guess = [0.,0.] #very first guess of P_BG_shift described below
            
            for i in arange(self.n_r.shape[0]/cut) :
            
                
                bg_pos = np.append( bg_pos , self.raw_data[ int( i*cut+(N_scans-1)*(N_points)) : int(i*cut+(N_scans)*(N_points)) , 2] - 2 ) # appends the positions (between -2 and 2 cm)
                bg_V = np.append( bg_V , self.raw_data[ int( i*cut+(N_scans-1)*(N_points)) : int(i*cut+(N_scans)*(N_points)) , 3]) # appends the last gradiometer voltage scan (averaged)
                    
                T0 = self.raw_data[ int( i*cut+(N_scans-1)*(N_points)) , 1] # Temperature 
                H0 = self.raw_data[ int( i*cut+(N_scans-1)*(N_points)) , 0] # Magnetic field 
                
                P_BG_shift = lambda x , s, A  : P_BG(x+s,T0,H0) + A
                #The background data can be shifted vertically or horizontally with the variables s and A respectively
                #The polynomial is first fitted to the data before the background subtraction
                
                param, cov = curve_fit(P_BG_shift , bg_pos , bg_V , p0 = P_BG_shift_guess) # parameters and covariance arrays
                
                self.treated_data[ int( i*cut+(N_scans-1)*(N_points)) : int(i*cut+(N_scans)*(N_points)) , 3] -= P_BG_shift( self.treated_data[ int(i*cut+(N_scans-1)*(N_points)):int( i*cut+(N_scans)*(N_points)) , 2] - 2 , *param )
                # Backgroud data subtraction
                
                P_BG_shift_guess = param
                
                bg_pos = array([])
                bg_V = array([])
                
            print('Background subtracted successfully !\nUse self.treated_data to access it.')
        
        else :
            print('No background interpolation function found. Try adding a background then run self.BG_interpol_2D.')
        
        return self.treated_data
            
    
    def MagPlot_raw(self , plot_bool , newdata_bool) :
        '''
        If plot_bool is set to True, plots the gradiometer voltages against the position values for all measurements in self.raw_data,
        then plots the magnetization value in emu and the R^2 value.
        If plot_bool is set to False, then only the magnetization value and R^2 are plotted.
        
        If newdata_bool is set to true, self.treated_data is used if not empty. Else, the self.raw_data is used as default.
        '''
        
        bg_pos = array([]) # To be filled with the position values at each measurement, then emptied
        bg_V = array([])   # To be filled with the gradiometer voltages values at each measurement, then emptied
        
        m = array([]) # To be filled with the magnetization values in emu
        T = array([]) # To be filled with temperature
        H = array([]) # To be filled with magnetic fields
        
        X = linspace(-0.5*L_scan , 0.5*L_scan , 200) # position range for plotting
        
        V_guess = self.Vdip0_guess
        
        if self.treated_data.shape[0] != 0 and newdata_bool :
            M_data = self.treated_data
        else :
            M_data = self.raw_data
        
        
        if plot_bool :
            
            for i in arange(self.n_r.shape[0]/cut) :
                
                bg_pos = np.append( bg_pos , M_data[ int( i*cut+(N_scans-1)*(N_points)) : int(i*cut+(N_scans)*(N_points)) , 2] - 2 ) # appends the positions (between -2 and 2 cm)
                bg_V = np.append( bg_V , M_data[ int( i*cut+(N_scans-1)*(N_points)) : int(i*cut+(N_scans)*(N_points)) , 3]) # appends the last gradiometer voltage scan (averaged)
                
                
                try :
                    param , cov = curve_fit( self.Vdip0 , bg_pos , bg_V , p0=V_guess ) # optimal fit parameters and covariance
                except :
                    print('curve_fit couldn\'t find optimized parameter (T = %s K ; H = %s Oe)'%( M_data[ int( i*cut+(N_scans-1)*(N_points)) , 1] , M_data[ int( i*cut+(N_scans-1)*(N_points)) , 0] ))
                
                m = np.append( m , M_emu( param[-1] , self.diag_data[ int(i*N_scans) , 0] , self.diag_data[ int(i*N_scans) , 1] ) )
                T = np.append( T , M_data[ int( i*cut+(N_scans-1)*(N_points)) , 1])
                H = np.append( H , M_data[ int( i*cut+(N_scans-1)*(N_points)) , 0])
                
                
                plt.figure()
                plt.plot( bg_pos , bg_V , 'bo' , X , self.Vdip0(X,*param), 'r-' )
                # plots the experimental data (blue) and the fitted test dipole function (red)
                plt.xlabel('x (cm)')
                plt.ylabel('Voltage (V)')
                plt.legend( ('Experimental data\nT = %s\nH = %s'%(T[-1],H[-1]),'Dipole fit') )
                plt.show()
                
                V_guess = param
                
                bg_pos = array([])
                bg_V = array([])
            
            print('Whole raw_data matrix analysed.')
            
        else : #same sequence but no plotting
            
            for i in arange(self.n_r.shape[0]/cut) :
            
                bg_pos = np.append( bg_pos , M_data[ int( i*cut+(N_scans-1)*(N_points)) : int(i*cut+(N_scans)*(N_points)) , 2] - 2 ) # appends the positions (between -2 and 2 cm)
                bg_V = np.append( bg_V , M_data[ int( i*cut+(N_scans-1)*(N_points)) : int(i*cut+(N_scans)*(N_points)) , 3]) # appends the last gradiometer voltage scan (averaged)
                
                try :
                    param , cov = curve_fit( self.Vdip0 , bg_pos , bg_V , p0=V_guess ) # optimal fit parameters and covariance
                except :
                    print('curve_fit couldn\'t find optimized parameter (T = %s K ; H = %s Oe)'%( M_data[ int( i*cut+(N_scans-1)*(N_points)) , 1] , M_data[ int( i*cut+(N_scans-1)*(N_points)) , 0] ))
                
                m = np.append( m , M_emu( param[-1] , self.diag_data[ int(i*N_scans),0] , self.diag_data[int(i*N_scans),1] ) )
                T = np.append( T , M_data[ int( i*cut+(N_scans-1)*(N_points)) , 1])
                H = np.append( H , M_data[ int( i*cut+(N_scans-1)*(N_points)) , 0])
                
                V_guess = param
                
                bg_pos = array([])
                bg_V = array([])
                
        MT_yn = input('plot M(T) ? [y,n]')
        
        if MT_yn == 'y' or MT_yn == 'Y' :
            
            plt.figure()
            plt.plot(T,m,'b-')
            plt.xlabel('T (K)')
            plt.ylabel('M (emu)')
            plt.title('Magnetization of %s'%(sample_name))
            plt.legend( [' H = %s Oe'%(H[0])] )
            plt.show()
            
        else :
            
            MH_yn = input('plot M(H) ? [y,n]')
            
            if MH_yn == 'y' or MH_yn == 'Y' :
            
                plt.figure()
                plt.plot(H,m,'b-')
                plt.xlabel('H (Oe)')
                plt.ylabel('M (emu)')
                plt.title('Magnetization of %s'%(sample_name))
                plt.legend( ['T = %s K'%(T[-1])] )
                plt.show()
                
        return T , H , m
