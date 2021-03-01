# -*- coding: utf-8 -*-
"""
Created on Tue Feb 23 17:12:17 2021

@author: ferde233
"""

import numpy as np

R = 0.97 # Longitudinal radius (cm)

G = 1.519 # Longitudinal coil separation (cm)

center = -0.25
# assumed sample position shift


Vdipole = lambda z, C, s, m0 : C + m0 * (2*(R**2 + (z + s)**2)**(-3/2) - (R**2 + (G + (z + s))**2)**(-3/2) - (R**2 + (-G + (z + s))**2)**(-3/2)) 
# in volts
# C : voltage offset constant
# s : z-axis shift of the sample from center position (usually not more than a few millimeters) 
# m0 : amplitude, proportional to the magnetization


Vdipole_lin = lambda  z , C , s , a , m0 : C + m0 * (2*(R**2 + (z + s)**2)**(-3/2) - (R**2 + (G + (z + s))**2)**(-3/2) - (R**2 + (-G + (z + s))**2)**(-3/2)) + a*z


Vdipole_asym = lambda z , C , s , a , b , c , m0    : C + m0 * (2*(R**2 + (z + s)**2)**(-3/2) - (R**2 + (G + (z + s))**2)**(-3/2) - (R**2 + (-G + (z + s))**2)**(-3/2)) + a*z + b*abs((z+s)**4)  + c*z**3





# Note : the 'centered' functions given below assume sample position is fixed at the value given l.25

Vdipole_centered = lambda z, C, m0 : C + m0 * (2*(R**2 + (z - center)**2)**(-3/2) - (R**2 + (G + (z - center))**2)**(-3/2) - (R**2 + (-G + (z - center))**2)**(-3/2)) 


Vdipole_alt = lambda z, C, s, a, b, m0 : C + m0 * (2*(R**2 + (z + s)**2)**(-3/2) - (R**2 + (G + (z + s))**2)**(-3/2) - (R**2 + (-G + (z + s))**2)**(-3/2)) + a*(z+s)**4 + b*z


Vdipole_alt_centered = lambda z, C, s, a, b, m0 : C + m0 * (2*(R**2 + (z-center)**2)**(-3/2) - (R**2 + (G + (z-center))**2)**(-3/2) - (R**2 + (-G + (z-center))**2)**(-3/2)) + a*abs(z+s)**2 + b*z


Vdipole_cubic_centered = lambda z, C, a, b, c, m0 : C + m0 * (2*(R**2 + (z-center)**2)**(-3/2) - (R**2 + (G + (z-center))**2)**(-3/2) - (R**2 + (-G + (z-center))**2)**(-3/2)) + a*z + b*z**2  + c*z**3


Vdipole_quartic_centered = lambda z, C, a, b, c, d, m0 : C + m0 * (2*(R**2 + (z-center)**2)**(-3/2) - (R**2 + (G + (z-center))**2)**(-3/2) - (R**2 + (-G + (z-center))**2)**(-3/2)) + a*z + b*z**2  + c*z**3 + d*z**4


Vdipole_asym_centered = lambda z, C, a, b, c, m0 : C + m0 * (2*(R**2 + (z-center)**2)**(-3/2) - (R**2 + (G + (z-center))**2)**(-3/2) - (R**2 + (-G + (z-center))**2)**(-3/2)) + a*z + b*((z)**4)  + c*z**3


Vdipole_asym_centered_2 = lambda z, C, a, b, c,d , m0 : C + m0 * (2*(R**2 + (z-center)**2)**(-3/2) - (R**2 + (G + (z-center))**2)**(-3/2) - (R**2 + (-G + (z-center))**2)**(-3/2)) + a*z + b*(abs((z)**2))  + c*z**3 + d*((z)**4)





gauss_f = lambda x , A , mu , sig :   A * np.exp( -0.5*( (x-mu)/sig )**2  )

sig = 1

Vdipole_gauss =  lambda z , C , s , a , A , mu  , m0 :     C + m0 * (2*(R**2 + (z + s)**2)**(-3/2) - (R**2 + (G + (z + s))**2)**(-3/2) - (R**2 + (-G + (z + s))**2)**(-3/2)) + a*z + gauss_f(z,A,mu,sig)
#guess = [2,0, -0.25, 3,0.4,1.5, 0]

Vdipole_gauss_centered =  lambda z , C , a , A , mu , m0 :     C + m0 * (2*(R**2 + (z-center)**2)**(-3/2) - (R**2 + (G + (z-center))**2)**(-3/2) - (R**2 + (-G + (z-center))**2)**(-3/2)) + a*z + gauss_f(z,A,mu,sig)
#guess = [2, -0.25, 3,0.4,1.5, 0]

b_w = 1

hyperbol = lambda x, a, h :    a * np.sqrt( 1 + (x-h)**2/(b_w**2) )

Vdipole_lin_hyper_centered = lambda z, C, B,  a, h, m0 :   C + m0 * (2*(R**2 + (z-center)**2)**(-3/2) - (R**2 + (G + (z-center))**2)**(-3/2) - (R**2 + (-G + (z-center))**2)**(-3/2)) + B*z + hyperbol(z,a,h)



##################################################################################################################



