# -*- coding: utf-8 -*-
"""
Created on Tue Jan 19 15:08:59 2021

@author: ferde233
"""


from scipy.optimize import curve_fit
from numpy import linspace, loadtxt, array, zeros, arctan, diag, sqrt, mean
import numpy as np
import matplotlib.pyplot as plt





### https://qdusa.com/siteDocs/appNotes/1014-213.pdf 
### for more information.

R = 0.97 # Longitudinal radius (cm)

G = 1.519 # Longitudinal coil separation (cm)

center = -0.25
# assumed sample position shift

Tmin = 250
Tmax = 310
#Temperature range for analysis of file 1

Tmin2 = 250
Tmax2 = 310
#Temperature range for analysis of file 2

################################################

filename_1 = "Almax_pressure_cell (upright) _Gasket+TAS100+Nujol+Sn+Ruby_background_25_Oe_cooling_AmbientPressure_3cmfromtop_Pressure_Reapplied.dc.raw"
# filename_2 = "Almax_pressure_cell (upright) _Gasket+TAS100+Nujol+Sn+Ruby_background_100_Oe_ZFC-FC_AmbientPressure_3cmfromtop_pressureapplied.dc.raw"
#f3 = 'file:///C:/Users/ferde233/Box Sync/Downloads/phD/Data (Quasicrystals)/Quasicrystals MPMS data/HP Background/11 12 2020/MPMS/Almax_pressure_cell (upright) _Gasket+TAS100+Nujol+Sn+Ruby_background_5000_Oe_warming_AmbientPressure_3cmfromtop_pressureapplied.dc.raw'
#f4 = 'file:///C:/Users/ferde233/Downloads/Almax_pressure_cell (upright) _Gasket+TAS100+Nujol+Sn+Ruby_background_25_Oe_cooling 4K-2Krange_AmbientPressure_3cmfromtop_Pressure_Reapplied.dc.raw'

#################################################

### MEMO : Headings start at row 31 and data at row 32
###        The sample pos. is recorded at column 8 (i.e M[:,7]  ) while the averaged voltage is recorded at column 11 (i.e M[:,10]  ) from the second scan onwards







M_raw = loadtxt(filename_1 ,dtype='str', delimiter =',',skiprows =31, usecols=(2,3,7,10))
# imports four columns : 0-field, 1-temperature, 2-sample position,3- averaged gradiometer voltage
M_d = loadtxt(filename_1.replace('dc.raw','dc.diag'),dtype='str', delimiter =',',skiprows =27, usecols=(7,8))
# imports two columns from the diagnostic file : 0- "range code", 1- "gain code"

# M_raw2 = loadtxt(filename_2 ,dtype='str', delimiter =',',skiprows =31, usecols=(2,3,7,10))
# M_d2 = loadtxt(filename_2.replace('dc.raw','dc.diag'),dtype='str', delimiter =',',skiprows =27, usecols=(7,8))

#M_raw3 = loadtxt(filename_3 ,dtype='str', delimiter =',',skiprows =31, usecols=(2,3,7,10))
#M_d3 = loadtxt(filename_3.replace('dc.raw','dc.diag'),dtype='str', delimiter =',',skiprows =27, usecols=(7,8))

#M_raw4 = loadtxt(filename_4 ,dtype='str', delimiter =',',skiprows =31, usecols=(2,3,7,10))
#M_d4 = loadtxt(filename_4.replace('dc.raw','dc.diag'),dtype='str', delimiter =',',skiprows =27, usecols=(7,8))


##################################################################################################################

Vbackground = lambda x, a, b, c, d, e, f, g, h : a*x**7 + b*x**6 + c*x**5 + d*x**4 + e*x**3 + f*x**2 + g*x + h
# polynomial background (7th degree)

Vdipole = lambda z, C, s, m0 : C + m0 * (2*(R**2 + (z + s)**2)**(-3/2) - (R**2 + (G + (z + s))**2)**(-3/2) - (R**2 + (-G + (z + s))**2)**(-3/2)) 
# in volts
# C : voltage offset constant
# s : z-axis shift of the sample from center position (usually not more than a few millimeters) 
# m0 : amplitude, proportional to the magnetization





# Note : the 'centered' functions given below assume sample position is fixed at the value given l.25

Vdipole_centered = lambda z, C, m0 : C + m0 * (2*(R**2 + (z - center)**2)**(-3/2) - (R**2 + (G + (z - center))**2)**(-3/2) - (R**2 + (-G + (z - center))**2)**(-3/2)) 


Vdipole_alt = lambda z, C, s, a, b, m0 : C + m0 * (2*(R**2 + (z + s)**2)**(-3/2) - (R**2 + (G + (z + s))**2)**(-3/2) - (R**2 + (-G + (z + s))**2)**(-3/2)) + a*(z+s)**4 + b*z


Vdipole_alt_centered = lambda z, C, s, a, b, m0 : C + m0 * (2*(R**2 + (z-center)**2)**(-3/2) - (R**2 + (G + (z-center))**2)**(-3/2) - (R**2 + (-G + (z-center))**2)**(-3/2)) + a*abs(z+s)**2 + b*z


Vdipole_cubic_centered = lambda z, C, a, b, c, m0 : C + m0 * (2*(R**2 + (z-center)**2)**(-3/2) - (R**2 + (G + (z-center))**2)**(-3/2) - (R**2 + (-G + (z-center))**2)**(-3/2)) + a*z + b*z**2  + c*z**3


Vdipole_quartic_centered = lambda z, C, a, b, c, d, m0 : C + m0 * (2*(R**2 + (z-center)**2)**(-3/2) - (R**2 + (G + (z-center))**2)**(-3/2) - (R**2 + (-G + (z-center))**2)**(-3/2)) + a*z + b*z**2  + c*z**3 + d*z**4


Vdipole_lin = lambda  z , C , s , a , m0 : C + m0 * (2*(R**2 + (z + s)**2)**(-3/2) - (R**2 + (G + (z + s))**2)**(-3/2) - (R**2 + (-G + (z + s))**2)**(-3/2)) + a*z


Vdipole_asym = lambda z , C , s , a , b , c , m0    : C + m0 * (2*(R**2 + (z + s)**2)**(-3/2) - (R**2 + (G + (z + s))**2)**(-3/2) - (R**2 + (-G + (z + s))**2)**(-3/2)) + a*z + b*abs((z+s)**4)  + c*z**3


Vdipole_asym_centered = lambda z, C, a, b, c, m0 : C + m0 * (2*(R**2 + (z-center)**2)**(-3/2) - (R**2 + (G + (z-center))**2)**(-3/2) - (R**2 + (-G + (z-center))**2)**(-3/2)) + a*z + b*((z)**4)  + c*z**3


Vdipole_asym_centered_2 = lambda z, C, a, b, c,d , m0 : C + m0 * (2*(R**2 + (z-center)**2)**(-3/2) - (R**2 + (G + (z-center))**2)**(-3/2) - (R**2 + (-G + (z-center))**2)**(-3/2)) + a*z + b*(abs((z)**2))  + c*z**3 + d*((z)**4)





gauss_f = lambda x , A , mu , sig :   A * np.exp( -0.5*( (x-mu)/sig )**2  )

sig = 1

Vdipole_gauss =  lambda z , C , s , a , A , mu  , m0 :     C + m0 * (2*(R**2 + (z + s)**2)**(-3/2) - (R**2 + (G + (z + s))**2)**(-3/2) - (R**2 + (-G + (z + s))**2)**(-3/2)) + a*z + gauss_f(z,A,mu,sig)
#guess = [2,0, -0.25, 3,0.4,1.5, 0]

Vdipole_gauss_centered =  lambda z , C , a , A , mu , m0 :     C + m0 * (2*(R**2 + (z-center)**2)**(-3/2) - (R**2 + (G + (z-center))**2)**(-3/2) - (R**2 + (-G + (z-center))**2)**(-3/2)) + a*z + gauss_f(z,A,mu,sig)
#guess = [2, -0.25, 3,0.4,1.5, 0]

b_w = 1

hyperbol = lambda x, a, h :    a * sqrt( 1 + (x-h)**2/(b_w**2) )

Vdipole_lin_hyper_centered = lambda z, C, B,  a, h, m0 :   C + m0 * (2*(R**2 + (z-center)**2)**(-3/2) - (R**2 + (G + (z-center))**2)**(-3/2) - (R**2 + (-G + (z-center))**2)**(-3/2)) + B*z + hyperbol(z,a,h)



##################################################################################################################







gain = [1.,2.,5.,10.] # table of the gain code values  in the diagnostic file to evaluate the moment, see Table 1 in the ref below
M_emu = lambda m0, r, g : (m0*1.825)/(8890.6*(10**-r)*gain[int(g)]*0.9125)
# calculates the magnetization in emu units from the gain 'g' and range 'r' tables in the .dc.diag file 
# See ### https://qdusa.com/siteDocs/appNotes/1014-213.pdf    p.2

############### MPMS settings ###############

N_points = 32 # number of points taken per scan
N_scans = 2 # number of scans
cut = N_points*N_scans # unless some points are rejected (rare), each scan should be separated by "cut" number of lines

L_scan = 4 # in cm





# Moment = lambda c : m0 * C
# in emu


n_r = M_raw.shape[0] # number of rows
n_c = M_raw.shape[1] # number of columns


def Mfloat(M):  # converts the table to float numbers and fills the empty spaces with "0" in the averaged voltage during 1st scans    
    M0 = zeros((M.shape[0],M.shape[1]))
    
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            if M[i,j] == '' :
                continue
            else :
                M0[i,j] = float(M[i,j])
                
    return M0


M_raw = Mfloat(M_raw)
M_d = Mfloat(M_d)


# M_raw2 = Mfloat(M_raw2)
            
        

T_int = array([270,280,290]) # temperatures targeted for interpolation





bg_pos = list()
bg_V = list() # empty list to be filled with the background data

def findT( temp , acc ): # find the first matching temperature within accuracy p and gives out the positions and voltages arrays
    bg0_pos = list()
    bg0_V = list()
    for i in range(n_r):
        if temp - acc < M_raw[i,1] < temp + acc :
            print('Matching T value found.')
            for p in range(N_points):
                bg0_pos.append(M_raw[i+(N_scans-1)*(N_points)+p,2]-2) #appends the last position (between -2 and 2 cm)
                bg0_V.append(M_raw[i+(N_scans-1)*(N_points)+p,3]) #appends the last of the averaged gradiometer voltage
            break
    return array(bg0_pos), array(bg0_V)

t_bg = 250
s_bg = 2
bg_pos, bg_V = findT(t_bg,s_bg)

print('Gradiometer voltage at T = %s K :'%(t_bg))

plt.figure(0)
plt.plot(bg_pos,bg_V,'bo')
plt.xlabel('position (cm)')
plt.ylabel('Voltage (V)')
plt.title('background at T = %s K'%(t_bg))
plt.savefig('Background_example_%sK.png'%(t_bg),format='png')
np.savetxt('Background_examples%sK.txt'%(t_bg), np.c_[bg_pos,bg_V])

    
        
            
            
        

init_guess = [ 0.01880116,  0.02216215,  0.09606059, -0.1188798 , -2.04078738,  -1.20408988,  4.7904851 ,  3.67130523] 
# first guess for the backgrounds [a,b,c,d,e,f,g,h]
r_param, pcov = curve_fit(Vbackground, bg_pos, bg_V, p0 = init_guess)
# fit of Vbackground from the x coordinates bg_pos and y coordinates bg_V from the initial guess.
# r_param returns the best fitted values while pcov is a covariance vector (i.e reliability) of the parameters 

x = linspace(-L_scan/2 , L_scan/2 , 1000) # replica of the x position for every scan

plt.figure(1)
plt.plot(bg_pos, bg_V ,'ro',  x, Vbackground(x,*r_param) ,'g-')
plt.xlabel('position (cm)')
plt.ylabel('Voltage (V)')
plt.title('background at T =  K')
plt.legend(('experimental background','7th polynomial fit'))
plt.savefig('Background_fitted.png',format='png') 


#for i in range(n_r) :
#    if t-p < M_raw[i,1]< t+p :
       

## The idea : creating a matrix  with the fitted values (a,b,...,g,h) of the 7th degree polynomial obtained 
##            at all the temperatures entered in T_int


def extrapolate(func,Tmin,Tmax,M): # gives the list of parameters for the background (a,...,h) for all the temperatures in T_range
    
    pos = list() # position list of a single scan  for the polynomial fit (reset at every temperature point) 
    volt = list() # gradiometer voltage list for the polynomial fit (reset at every temperature point)
    
    t = list() # list of temperatures encountered
    
    a_fit = list()
    b_fit = list()
    c_fit = list()
    d_fit = list()
    e_fit = list()
    f_fit = list()
    g_fit = list()
    h_fit = list()
    #fits of all the parameters of the polynomial Vbackground
    
    r = 0 # row counter
    n_r = M.shape[0]
    
    while r < n_r :
        
        if M[r,1] < Tmin :
            r += cut
            continue

        
        if M[r,1] > Tmax :
            print('Parameters extraction completed over the whole temperature range (%s - %s K)'%(Tmin, Tmax))
            break
        
        if Tmin < M[r,1] < Tmax :
            for p in range(N_points):
                
                pos.append(M[r+(N_scans-1)*(N_points)+p,2]-2) #appends the last positions (between -2 and 2 cm)
                volt.append(M[r+(N_scans-1)*(N_points)+p,3]) #appends the last of the averaged gradiometer voltage
              
            # at this step both pos and volt have their 32 (N_points) designed values
            param, cov = curve_fit(func, array(pos), array(volt), p0 = init_guess)
            # fit at the specified temperature of a,...,h
            a_fit.append(param[0])
            b_fit.append(param[1])
            c_fit.append(param[2])
            d_fit.append(param[3])
            e_fit.append(param[4])
            f_fit.append(param[5])
            g_fit.append(param[6])
            h_fit.append(param[7])
            
            t.append(M[r,1])
            # we also record the temperature
            
            pos = list()
            volt = list()
            # ...and we reset the fitting point lists
        else :
            print('Error in the temperature range')
        r += cut         
        
    if r >= n_r :
        print('Whole matrix scanned')
    else :
        print('Matrix scanned until row %s, last temperature T = %s K '%(r,t[-1]))
        
    n = len(t)
    print('%s fits were performed in total'%(n))
        
    return n, array(t), array(a_fit), array(b_fit), array(c_fit), array(d_fit), array(e_fit), array(f_fit), array(g_fit), array(h_fit)
        




n, t, a, b, c, d, e, f, g ,h = extrapolate(Vbackground,Tmin,Tmax,M_raw)




# n2, t2, a2, b2, c2, d2, e2, f2, g2 ,h2 = extrapolate(Vbackground,Tmin2,Tmax2,M_raw2)





plt.figure('2')
plt.plot(t,a,'o',t,b,'o',t,c,'o',t,d,'o',t,e,'o',t,f,'o',t,g,'o')
## I didn't include "h"
plt.xlabel('T (K)')
plt.ylabel('fit values')
plt.title('Fitting 7th polynomial parameters temperature dependence')
plt.legend(('a','b','c','d','e','f','g'))
plt.savefig('fitted_parameters.png',format='png') 

# plt.figure('2bis')
# plt.plot(t2,a2,'o',t2,b2,'o',t2,c2,'o',t2,d2,'o',t2,e2,'o',t2,f2,'o',t2,g2,'o')
# ## I didn't include "h"
# plt.xlabel('T (K)')
# plt.ylabel('fit values')
# plt.title('Fitting 7th polynomial parameters temperature dependence')
# plt.legend(('a2','b2','c2','d2','e2','f2','g2'))
# plt.savefig('fitted_parameters_2.png',format='png') 



#_____________________________________________________________________________


linfit = lambda x, a, b : a*x + b
#linear function for parameter fitting

arcfit =  lambda x, A, l, a, b : A*arctan(l*x) + a*x + b
# Alternatively, an arctan function modified function

invfit = lambda x, a, b, s : a*x + b + (s/x)


#_____________________________________________________________________________

a_param, a_cov = curve_fit(linfit ,array(t), array(a), p0= [0,a[-1]])
b_param, b_cov = curve_fit(linfit ,array(t), array(b), p0= [0,b[-1]])
c_param, c_cov = curve_fit(linfit ,array(t), array(c), p0= [0,c[-1]])
d_param, d_cov = curve_fit(linfit ,array(t), array(d), p0= [0,d[-1]])
e_param, e_cov = curve_fit(linfit ,array(t), array(e), p0= [0,e[-1]])
f_param, f_cov = curve_fit(linfit ,array(t), array(f), p0= [0,f[-1]])
g_param, g_cov = curve_fit(linfit ,array(t), array(g), p0= [0,g[-1]])
h_param, h_cov = curve_fit(linfit ,array(t), array(h), p0= [0,h[-1]])

#a_param, a_cov = curve_fit(arcfit ,array(t), array(a), p0= [a[-1],1,0,a[-1]])
#b_param, b_cov = curve_fit(arcfit ,array(t), array(b), p0= [b[-1],1,0,b[-1]])
#c_param, c_cov = curve_fit(arcfit ,array(t), array(c), p0= [c[-1],1,0,c[-1]])
#d_param, d_cov = curve_fit(arcfit ,array(t), array(d), p0= [d[-1],1,0,d[-1]])
#e_param, e_cov = curve_fit(arcfit ,array(t), array(e), p0= [e[-1],1,0,e[-1]])
#f_param, f_cov = curve_fit(arcfit ,array(t), array(f), p0= [f[-1],1,0,f[-1]])
#g_param, g_cov = curve_fit(arcfit ,array(t), array(g), p0= [g[-1],1,0,g[-1]])
#h_param, h_cov = curve_fit(arcfit ,array(t), array(h), p0= [h[-1],1,0,h[-1]])

#a_param, a_cov = curve_fit(invfit ,array(t), array(a), p0= [0,a[-1],0])
#b_param, b_cov = curve_fit(invfit ,array(t), array(b), p0= [0,b[-1],0])
#c_param, c_cov = curve_fit(invfit ,array(t), array(c), p0= [0,c[-1],0])
#d_param, d_cov = curve_fit(invfit ,array(t), array(d), p0= [0,d[-1],0])
#e_param, e_cov = curve_fit(invfit ,array(t), array(e), p0= [0,e[-1],0])
#f_param, f_cov = curve_fit(invfit ,array(t), array(f), p0= [0,f[-1],0])
#g_param, g_cov = curve_fit(invfit ,array(t), array(g), p0= [0,g[-1],0])
#h_param, h_cov = curve_fit(invfit ,array(t), array(h), p0= [0,h[-1],0])


#Tscale = linspace(Tmin, Tmax, 2)
## to be used for linear fits

Tscale = linspace(2, Tmax, 200)
# to be used for arctan fits

plt.figure('2')
plt.plot( Tscale, linfit(Tscale,*a_param),'k-'  ,Tscale,linfit(Tscale,*b_param),'k-',Tscale,linfit(Tscale,*c_param),'k-',Tscale,linfit(Tscale,*d_param),'-k',Tscale,linfit(Tscale,*e_param),'k-',Tscale,linfit(Tscale,*f_param),'-k',Tscale,linfit(Tscale,*g_param),'k-')
plt.savefig('Fitting_parameters_T-dependence.png',format='png') 
### I didn't include "h"
#plt.legend(('a fit','b fit','c fit','d fit','e fit','f fit','g fit'))


    
# a2_param, a2_cov = curve_fit(linfit ,array(t2), array(a2), p0= [0,a2[-1]])
# b2_param, b2_cov = curve_fit(linfit ,array(t2), array(b2), p0= [0,b2[-1]])
# c2_param, c2_cov = curve_fit(linfit ,array(t2), array(c2), p0= [0,c2[-1]])
# d2_param, d2_cov = curve_fit(linfit ,array(t2), array(d2), p0= [0,d2[-1]])
# e2_param, e2_cov = curve_fit(linfit ,array(t2), array(e2), p0= [0,e2[-1]])
# f2_param, f2_cov = curve_fit(linfit ,array(t2), array(f2), p0= [0,f2[-1]])
# g2_param, g2_cov = curve_fit(linfit ,array(t2), array(g2), p0= [0,g2[-1]])
# h2_param, h2_cov = curve_fit(linfit ,array(t2), array(h2), p0= [0,h2[-1]])



    
#a2_param, a2_cov = curve_fit(arcfit ,array(t2), array(a2), p0= [a2[-1]/10,0.25,0,a2[-1]])
#b2_param, b2_cov = curve_fit(arcfit ,array(t2), array(b2), p0= [b2[-1]/10,0.25,0,b2[-1]])
#c2_param, c2_cov = curve_fit(arcfit ,array(t2), array(c2), p0= [c2[-1]/10,0.25,0,c2[-1]])
#d2_param, d2_cov = curve_fit(arcfit ,array(t2), array(d2), p0= [d2[-1]/10,0.25,0,d2[-1]])
#e2_param, e2_cov = curve_fit(arcfit ,array(t2), array(e2), p0= [e2[-1]/10,0.25,0,e2[-1]])
#f2_param, f2_cov = curve_fit(arcfit ,array(t2), array(f2), p0= [f2[-1]/10,0.25,0,f2[-1]])
#g2_param, g2_cov = curve_fit(arcfit ,array(t2), array(g2), p0= [g2[-1]/10,0.25,0,g2[-1]])
#h2_param, h2_cov = curve_fit(arcfit ,array(t2), array(h2), p0= [h2[-1]/10,0.25,0,h2[-1]])


#a2_param, a2_cov = curve_fit(invfit ,array(t2), array(a2), p0= [0,a2[-1],0])
#b2_param, b2_cov = curve_fit(invfit ,array(t2), array(b2), p0= [0,b2[-1],0])
#c2_param, c2_cov = curve_fit(invfit ,array(t2), array(c2), p0= [0,c2[-1],0])
#d2_param, d2_cov = curve_fit(invfit ,array(t2), array(d2), p0= [0,d2[-1],0])
#e2_param, e2_cov = curve_fit(invfit ,array(t2), array(e2), p0= [0,e2[-1],0])
#f2_param, f2_cov = curve_fit(invfit ,array(t2), array(f2), p0= [0,f2[-1],0])
#g2_param, g2_cov = curve_fit(invfit ,array(t2), array(g2), p0= [0,g2[-1],0])
#h2_param, h2_cov = curve_fit(invfit ,array(t2), array(h2), p0= [0,h2[-1],0])

#Tscale2 = linspace(Tmin2, Tmax2, 2)
## to be used for linear fits

Tscale2 = linspace(2, Tmax2, 200)
# to be used for other fits

# plt.figure('2bis')
# plt.plot(Tscale2,linfit(Tscale2,*a2_param),'-k',Tscale2,linfit(Tscale2,*b2_param),'k-',Tscale2,linfit(Tscale2,*c2_param),'k-',Tscale2,linfit(Tscale2,*d2_param),'-k',Tscale2,linfit(Tscale2,*e2_param),'k-',Tscale2,linfit(Tscale2,*f2_param),'-k',Tscale2,linfit(Tscale2,*g2_param),'k-')
# plt.savefig('Fitting_parameters_T-dependence.png',format='png') 



X = linspace(-2,2,1000)
plt.figure(10)
plt.plot(X,Vdipole(X,0,0,1),'ro')
plt.xlabel('position (cm)')
plt.ylabel('Voltage (V)')
plt.title('Typical voltage induced by a point-like dipole')
plt.savefig('dipole_function.png',format='png') 
np.savetxt('dipole.txt', np.c_[X,Vdipole(X,0,0,1)])
plt.figure()



T0 = 150
# fixed variable for the low temperature extrapolation
# Once the value of T0 has been set, the functions defined right below should follow the background at that temperature




# for linfit

V_bg_extrapol = lambda x, A , l, C :  A * ( linfit(T0,a_param[0],a_param[1])*x**7 + linfit(T0,b_param[0],b_param[1])*x**6 + linfit(T0,c_param[0],c_param[1])*x**5 + linfit(T0,d_param[0],d_param[1])*x**4 + linfit(T0,e_param[0],e_param[1])*x**3 + linfit(T0,f_param[0],f_param[1])*x**2 + linfit(T0,g_param[0],g_param[1])*x ) + l*x + C
# T0 : fixed temperature (constant) ; x : position  ; A : amplitude ; C : offset constant

# V_bg2_extrapol = lambda x, A , l, C :  A * ( linfit(T0,a2_param[0],a2_param[1])*x**7 + linfit(T0,b2_param[0],b2_param[1])*x**6 + linfit(T0,c2_param[0],c2_param[1])*x**5 + linfit(T0,d2_param[0],d2_param[1])*x**4 + linfit(T0,e2_param[0],e2_param[1])*x**3 + linfit(T0,f2_param[0],f2_param[1])*x**2 + linfit(T0,g2_param[0],g2_param[1])*x ) + l*x + C



# for invfit
# same but invfit has more parameters (comment/uncomment the right blocks at l. 320 to l. 391)

#V_bg_extrapol = lambda x, A , l, C :  A * ( invfit(T0,a_param[0],a_param[1], a_param[2])*x**7 + invfit(T0,b_param[0],b_param[1], b_param[2])*x**6 + invfit(T0,c_param[0],c_param[1], c_param[2])*x**5 + invfit(T0,d_param[0],d_param[1], d_param[2])*x**4 + invfit(T0,e_param[0],e_param[1], e_param[2])*x**3 + invfit(T0,f_param[0],f_param[1], f_param[2])*x**2 + invfit(T0,g_param[0],g_param[1], g_param[2])*x ) + l*x + C
## T0 : fixed temperature (constant) ; x : position  ; A : amplitude ; C : offset constant
#
#V_bg2_extrapol = lambda x, A , l, C :  A * ( invfit(T0,a2_param[0],a2_param[1], a2_param[2])*x**7 + invfit(T0,b2_param[0],b2_param[1], b2_param[2])*x**6 + invfit(T0,c2_param[0],c2_param[1], c2_param[2])*x**5 + invfit(T0,d2_param[0],d2_param[1], d2_param[2])*x**4 + invfit(T0,e2_param[0],e2_param[1], e2_param[2])*x**3 + invfit(T0,f2_param[0],f2_param[1], f2_param[2])*x**2 + invfit(T0,g2_param[0],g2_param[1], g2_param[2])*x ) + l*x + C




##for the coefficients a,...,g assumed constant

#V_bg_extrapol = lambda x, A , l, C :  A * (( mean(a))*x**7 + mean(b)*x**6 + mean(c)*x**5 + mean(d)*x**4 + mean(e)*x**3 + mean(f)*x**2 + mean(g)*x ) + l*x + C
## T0 : fixed temperature (constant) ; x : position  ; A : amplitude ; C : offset constant
#
#V_bg2_extrapol = lambda x, A , l, C :   A * (( mean(a2))*x**7 + mean(b2)*x**6 + mean(c2)*x**5 + mean(d2)*x**4 + mean(e2)*x**3 + mean(f2)*x**2 + mean(g2)*x ) + l*x + C



## for invfit (without linear background fit)
## same but invfit has more parameters (comment/uncomment the right blocks at l. 320 to l. 391)

#V_bg_extrapol = lambda x, A , C :  A * ( invfit(T0,a_param[0],a_param[1], a_param[2])*x**7 + invfit(T0,b_param[0],b_param[1], b_param[2])*x**6 + invfit(T0,c_param[0],c_param[1], c_param[2])*x**5 + invfit(T0,d_param[0],d_param[1], d_param[2])*x**4 + invfit(T0,e_param[0],e_param[1], e_param[2])*x**3 + invfit(T0,f_param[0],f_param[1], f_param[2])*x**2 + invfit(T0,g_param[0],g_param[1], g_param[2])*x ) + C
## T0 : fixed temperature (constant) ; x : position  ; A : amplitude ; C : offset constant
#
#V_bg2_extrapol = lambda x, A , C :  A * ( invfit(T0,a2_param[0],a2_param[1], a2_param[2])*x**7 + invfit(T0,b2_param[0],b2_param[1], b2_param[2])*x**6 + invfit(T0,c2_param[0],c2_param[1], c2_param[2])*x**5 + invfit(T0,d2_param[0],d2_param[1], d2_param[2])*x**4 + invfit(T0,e2_param[0],e2_param[1], e2_param[2])*x**3 + invfit(T0,f2_param[0],f2_param[1], f2_param[2])*x**2 + invfit(T0,g2_param[0],g2_param[1], g2_param[2])*x ) + C




def M_bg_subtracted(M_start,T_lim,V_bg,Vguess,P):
    # return a matrix with an additional column [:,4] with the previously subtracted background , cutdown temperature can be added
    # V_bg is the extrapolated background function for the set of data chosen and P is the proportion of fitted points, from 0 to 1
    # where the proportion P outside of the center is taken for the fit
    # e.g if P = 0.5, the fit is done from [-2 cm,-1 cm] and [1 cm, 2 cm] instead of the whole [-2 cm, 2 cm] range
    
    
    n_r = M_start.shape[0]
    n_c = M_start.shape[1]
    
    M = zeros((n_r,n_c +1))
    M[:,:-1] = M_start
    # we create a matrix with an added 5th column filled with "0", to be filled with the subtracted backgrounds
    
    T0 = 0 
    #fixed variable for the low temperature extrapolation
    bg0_pos = list()
    bg0_V = list()
    #same trick as before, we create position and voltage lists as we go
    skipped = 0
    
    V_bg_extrapol = lambda x, A , l, C :  A * ( linfit(T0,a_param[0],a_param[1])*x**7 + linfit(T0,b_param[0],b_param[1])*x**6 + linfit(T0,c_param[0],c_param[1])*x**5 + linfit(T0,d_param[0],d_param[1])*x**4 + linfit(T0,e_param[0],e_param[1])*x**3 + linfit(T0,f_param[0],f_param[1])*x**2 + linfit(T0,g_param[0],g_param[1])*x ) + l*x + C
# T0 : fixed temperature (constant) ; x : position  ; A : amplitude ; C : offset constant

    V_bg = V_bg_extrapol

    
    
    for i in range(n_r//cut):
            
            if M_start[i*cut,1] > T_lim :
                skipped += 1
                continue
            
            for p in list(range( int(N_points*0.5*P) )) + list(range(N_points - 1 , int((1-0.5*P)*N_points) - 1,-1)):
                bg0_pos.append(M_start[(i*cut)+(N_scans-1)*(N_points)+p,2]-2) #appends the last position (between -2 and 2 cm)
                bg0_V.append(M_start[(i*cut)+(N_scans-1)*(N_points)+p,3]) #appends the last of the averaged gradiometer voltage
            
            
            T0 = M_start[i*cut,1]
            bg0_param, bg0_cov = curve_fit(V_bg, bg0_pos, bg0_V, p0 = Vguess)
            
            for p in range(N_points):
                M[(i*cut)+(N_scans-1)*(N_points)+p,4]   =   M_start[(i*cut)+(N_scans-1)*(N_points)+p,3] - V_bg(M_start[(i*cut)+(N_scans-1)*(N_points)+p , 2]-2,*bg0_param)
                # Fills in the 5th column with the subtracted points from the experimental data and the previously extrapolated background
                
            bg0_pos = list()
            bg0_V = list()   
            
    
    print('last temperature replaced : %s K'%(T0))
    print('%s points were out of range ( T > %s K)'%(skipped,T_lim))
    
    return M





#_________________________________________________________________________________

guess = [1. , 0. , 0. ]
#change the guess depending on which V_bg_extrapol you chose to have the right number of arguments/a nice guess.
#_________________________________________________________________________________

M = M_bg_subtracted(M_raw, 310, V_bg_extrapol, guess, 1)
# M2 = M_bg_subtracted(M_raw2, 310, V_bg2_extrapol, guess, 1)

### Now is time to fit the dipole equation l. 22 to obtain each point
# --->  Vdipole = lambda z, C, s, m0 : C + m0 * (2*(R**2 + (z + s)**2)**(-3/2) - (R**2 + (G + (z + s))**2)**(-3/2) - (R**2 + (-G + (z + s))**2)**(-3/2)) 

def magnetization(Mf, Md, T_lim, V_extrap, P) :
    # subtract the background extrapolated earlier up to T_lim and allows you to plot both the subtracted background with dipolar fit 
    # as well as the original data with the outline of the extrapolated background if [Y] is answered.
    
    h = Mf[0,0]
    # the field (assumed constant) is stored
    
    t = list()
    # list of recorded temperatures
    
    mz = list()
    # list of recorded amplitudes from the fit (proportional to the magnetization)
    s = list()
    # list of shift from zero position on the longitudinal axis of the SQUID 
    d = list() 
    # standard deviation of the fit

    bg0_pos = list()
    bg0_V = list()
    #same
    
    n_r = Mf.shape[0]
    
    dip_fit_plot = input(print('Do you want to plot the dipole fit (after background subtraction) and the raw background at each step ? (Y/N)'))

    
    if dip_fit_plot == 'Y' or  dip_fit_plot == 'y' :
        
        for i in range(n_r//cut):
                
                if Mf[(i*cut)+(N_scans-1)*N_points,4] == 0 :
                    continue
                if Mf[i*cut,1] > T_lim :
                    continue
                
                for p in range(N_points):
                    bg0_pos.append(Mf[(i*cut)+(N_scans-1)*(N_points)+p,2]-2) #appends the last position (between -2 and 2 cm)
                    bg0_V.append(Mf[(i*cut)+(N_scans-1)*(N_points)+p,4]) #appends the last of the averaged gradiometer voltage (background subtracted)
                
                
                dip_param, dip_cov = curve_fit(Vdipole_centered, bg0_pos, bg0_V, p0 = [0. , (Mf[int((i*cut)+cut*1.4/4),4]-Mf[(i*cut)+ 1 ,4])/Vdipole(0,0,0,1)])
                
                t.append(Mf[i*cut,1])
                mz.append(dip_param[-1])
                s.append(dip_param[-2])
                
                residuals = Vdipole_centered( array(bg0_pos), *dip_param) - array(bg0_V)
                ss_res = np.sum(residuals**2)
                ss_tot = np.sum( (array(bg0_V) - mean(array(bg0_V)))**2 )
                
                d.append( 1 - (ss_res / ss_tot) )
                
                print('T = %s K'%(t[-1]))
                plt.figure()
                plt.plot(Mf[ (i*cut) + (N_scans-1)*N_points: i*cut + cut - 1 ,2]-2 ,Mf[ (i*cut) + (N_scans-1)*N_points: i*cut + cut - 1 ,4] , 'ko', X, Vdipole_centered(X,*dip_param),'r-')
                plt.show()
                
                bg0_pos = list()
                bg0_V = list()
    
        bg_raw_pos = list()
        bg_raw_V = list()
        
        for i in range(n_r//cut):
                
                if Mf[(i*cut)+(N_scans-1)*N_points,4] == 0 :
                    continue
                if Mf[i*cut,1] > T_lim :
                    continue
                
                for p in list(range( int(N_points*0.5*P) )) + list(range(N_points - 1, int((1-0.5*P)*N_points) - 1,-1)):
                    bg_raw_pos.append(Mf[(i*cut)+(N_scans-1)*(N_points)+p,2]-2) #appends the last position (between -2 and 2 cm)
                    bg_raw_V.append(Mf[(i*cut)+(N_scans-1)*(N_points)+p,3]) #appends the last of the raw averaged gradiometer voltage 
                
                
                T0 = Mf[i*cut,1]
                bg_raw_param , bg_raw_cov = curve_fit(V_extrap , bg_raw_pos , bg_raw_V , p0 = [1. , 0.25 , 0. ])
                
                
                print('T = %s K'%(T0))
                plt.figure()
                plt.plot(Mf[ (i*cut) + (N_scans-1)*N_points: i*cut + cut - 1 ,2]-2 ,Mf[ (i*cut) + (N_scans-1)*N_points: i*cut + cut - 1 ,3] , 'ko', X, V_extrap(X,*bg_raw_param),'r-')
                plt.show()
                bg_raw_pos = list()
                bg_raw_V = list()
                
    else :
        for i in range(n_r//cut):
                
                if Mf[(i*cut)+(N_scans-1)*N_points,4] == 0 :
                    continue
                if Mf[i*cut,1] > T_lim :
                    continue
                
                for p in range(N_points):
                    bg0_pos.append(Mf[(i*cut)+(N_scans-1)*(N_points)+p,2]-2) #appends the last position (between -2 and 2 cm)
                    bg0_V.append(Mf[(i*cut)+(N_scans-1)*(N_points)+p,4]) #appends the last of the averaged gradiometer voltage (background subtracted)
                
                
                dip_param, dip_cov = curve_fit(Vdipole_centered, bg0_pos, bg0_V, p0 = [0. , (Mf[(i*cut)+(N_scans-1)*(N_points)+N_points//2,4]-Mf[(i*cut)+cut-1 ,4])/Vdipole(0,0,0,1)])
                
                t.append(Mf[i*cut,1])
                mz.append(dip_param[-1])
                s.append(dip_param[-2])
                
                residuals = Vdipole_centered( array(bg0_pos), *dip_param) - array(bg0_V)
                ss_res = np.sum(residuals**2)
                ss_tot = np.sum( (array(bg0_V) - mean(array(bg0_V)))**2 )
                
                d.append( 1 - (ss_res / ss_tot) )
                
                
#                print('T = %s K'%(t[-1]))
#                plt.figure()
#                plt.plot(Mf[ (i*cut) + (N_scans-1)*N_points: i*cut + cut - 1 ,2]-2 ,Mf[ (i*cut) + (N_scans-1)*N_points: i*cut + cut - 1 ,4] , 'ko', X, Vdipole_centered(X,*dip_param),'r-')
#                plt.show()
                bg0_pos = list()
                bg0_V = list()
    
    
    
    return h, array(t) , array(mz), array(s), array(d)


#
#def magnetization_lin(Mf, T_lim) :
#    
#    h = Mf[0,0]
#    # the field (assumed constant) is stored
#    
#    t = list()
#    # list of recorded temperatures
#    
#    mz = list()
#    # list of recorded amplitudes from the fit (proportional to the magnetization)
#    s = list()
#    # list of shift from zero position on the longitudinal axis of the SQUID 
#
#    bg0_pos = list()
#    bg0_V = list()
#    #same
#    
#    n_r = Mf.shape[0]
#    
#    for i in range(n_r//cut):
#            
#            if Mf[(i*cut)+(N_scans-1)*N_points,4] == 0 :
#                continue
#            if Mf[i*cut,1] > T_lim :
#                continue
#            
#            for p in range(N_points):
#                bg0_pos.append(Mf[(i*cut)+(N_scans-1)*(N_points)+p,2]-2) #appends the last position (between -2 and 2 cm)
#                bg0_V.append(Mf[(i*cut)+(N_scans-1)*(N_points)+p,4]) #appends the last of the averaged gradiometer voltage (background subtracted)
#            
#            
#            dip_param, dip_cov = curve_fit(Vdipole_lin, bg0_pos, bg0_V, p0 = [0., 0.15 , (Mf[(i*cut)+(N_scans-1)*(N_points)+N_points//2,4]-Mf[(i*cut)+cut-1 ,4])/Vdipole_lin(0,0,0,1,0),0])
#            
#            t.append(Mf[i*cut,1])
#            mz.append(dip_param[-1])
#            s.append(dip_param[-2])
#            
#            print('T = %s K'%(t[-1]))
#            plt.figure()
#            plt.plot(Mf[ (i*cut) + (N_scans-1)*N_points: i*cut + cut - 1 ,2]-2 ,Mf[ (i*cut) + (N_scans-1)*N_points: i*cut + cut - 1 ,4] , 'ko', X, Vdipole_lin(X,*dip_param),'r-')
#            plt.show()
#            bg0_pos = list()
#            bg0_V = list()
#            
#    return h, array(t) , array(mz), array(s)

#
#def magnetization_quad(Mf, T_lim) :
#    
#    h = Mf[0,0]
#    # the field (assumed constant) is stored
#    
#    t = list()
#    # list of recorded temperatures
#    
#    mz = list()
#    # list of recorded amplitudes from the fit (proportional to the magnetization)
#    s = list()
#    # list of shift from zero position on the longitudinal axis of the SQUID 
#
#    bg0_pos = list()
#    bg0_V = list()
#    #same
#    
#    n_r = Mf.shape[0]
#    
#    for i in range(n_r//cut):
#            
#            if Mf[(i*cut)+(N_scans-1)*N_points,4] == 0 :
#                continue
#            if Mf[i*cut,1] > T_lim :
#                continue
#            
#            for p in range(N_points):
#                bg0_pos.append(Mf[(i*cut)+(N_scans-1)*(N_points)+p,2]-2) #appends the last position (between -2 and 2 cm)
#                bg0_V.append(Mf[(i*cut)+(N_scans-1)*(N_points)+p,4]) #appends the last of the averaged gradiometer voltage (background subtracted)
#            
#            
#            dip_param, dip_cov = curve_fit(Vdipole_alt, bg0_pos, bg0_V, p0 = [0., 0.5 , -(Mf[(i*cut)+(N_scans-1)*(N_points)+int(N_points*1.3/4),4]-Mf[(i*cut)+1 ,4])/Vdipole_alt(0,0,0,1,0,0),0,0])
#            
#            t.append(Mf[i*cut,1])
#            mz.append(dip_param[-1])
#            s.append(dip_param[-2])
#            
#            print('T = %s K'%(t[-1]))
#            plt.figure()
#            plt.plot(Mf[ (i*cut) + (N_scans-1)*N_points: i*cut + cut - 1 ,2]-2 ,Mf[ (i*cut) + (N_scans-1)*N_points: i*cut + cut - 1 ,4] , 'ko', X, Vdipole_alt(X,*dip_param),'r-')
#            plt.show()
#            bg0_pos = list()
#            bg0_V = list()
#            
#    return h, array(t) , array(mz), array(s)


            







guess = [0,0,0]


def magnetization_raw(Mf, T_lim, Vdip, p_guess) :
# gives the values of magnetization from trying to fit a dipole to the raw data
    
    h = Mf[0,0]
    # the field (assumed constant) is stored
    
    t = list()
    # list of recorded temperatures
    
    mz = list()
    # list of recorded amplitudes from the fit (proportional to the magnetization)
    s = list()
    # list of shift from zero position on the longitudinal axis of the SQUID 

    d = list()
    # standart deviation of the fit
    
    bg0_pos = list()
    bg0_V = list()
    #same
    
    n_r = Mf.shape[0]
    
    for i in range(n_r//cut):
            
            if Mf[(i*cut)+(N_scans-1)*N_points,3] == 0 :
                continue
            if Mf[i*cut,1] > T_lim :
                continue
            
            for p in range(N_points):
                bg0_pos.append(Mf[(i*cut)+(N_scans-1)*(N_points)+p,2]-2) #appends the last position (between -2 and 2 cm)
                bg0_V.append(Mf[(i*cut)+(N_scans-1)*(N_points)+p,3]) #appends the last of the averaged gradiometer voltage (raw data)
            
            p_guess[-1] = (Mf[(i*cut)+(N_scans-1)*(N_points)+N_points//2,3]-Mf[(i*cut)+cut-1 ,3])/Vdipole(0,0,0,1)
            dip_param, dip_cov = curve_fit(Vdip, bg0_pos, bg0_V, p0 = p_guess)
            
            t.append(Mf[i*cut,1])
            mz.append(dip_param[-1])
            s.append(dip_param[-2])
            d.append(sqrt(diag(dip_cov)))
            
            print('T = %s K'%(t[-1]))
            plt.figure()
            plt.plot(Mf[ (i*cut) + (N_scans-1)*N_points: i*cut + cut - 1 ,2]-2 ,Mf[ (i*cut) + (N_scans-1)*N_points: i*cut + cut - 1 ,3] , 'ko', X, Vdipole(X,*dip_param),'r-')
            plt.show()
            bg0_pos = list()
            bg0_V = list()
            
    return h, array(t) , array(mz), array(s), array(d)



def magnetization_raw_diag(Mf,Md, T_lim, Vdip, p_guess) :
# gives the values of magnetization (in emu) from trying to fit a dipole to the raw data (Mf) and diagnostic file (Md)
    
    h = Mf[0,0]
    # the field (assumed constant) is stored
    
    t = list()
    # list of recorded temperatures
    
    mz = list()
    # list of recorded magnetization values from the fit (in emu)
    s = list()
    # list of shift from zero position on the longitudinal axis of the SQUID 
    d = list()
    # standard deviation of the fit
    
    bg0_pos = list()
    bg0_V = list()
    #same
    
    n_r = Mf.shape[0]
#    n_r_d = Md.shape[0]
    
    
    for i in range(n_r//cut):
            
            if Mf[(i*cut)+(N_scans-1)*N_points,3] == 0 :
                continue
            if Mf[i*cut,1] > T_lim :
                continue
            
            for p in range(N_points):
                bg0_pos.append(Mf[(i*cut)+(N_scans-1)*(N_points)+p,2]-2) #appends the last position (between -2 and 2 cm)
                bg0_V.append(Mf[(i*cut)+(N_scans-1)*(N_points)+p,3]) #appends the last of the averaged gradiometer voltage (raw data)
            
#            p_guess[-1] = (Mf[(i*cut)+(N_scans-1)*(N_points)+N_points//2,3]-Mf[(i*cut)+cut-1 ,3])/Vdipole(0,0,0,1)
            dip_param, dip_cov = curve_fit(Vdip, bg0_pos, bg0_V, p0 = p_guess)
            
            t.append(Mf[i*cut,1])
            mz.append( M_emu( dip_param[-1], Md[i*N_scans,0], Md[i*N_scans,1] )) # obtains the emu value of magnetization from the formula
            s.append(dip_param[-2])
            
            residuals = Vdip( array(bg0_pos), *dip_param) - array(bg0_V)
            ss_res = np.sum(residuals**2)
            ss_tot = np.sum( (array(bg0_V) - mean(array(bg0_V)))**2 )
            
            d.append( 1 - (ss_res / ss_tot) )
            
            print('T = %s K'%(t[-1]))
            plt.figure()
            plt.plot(Mf[ (i*cut) + (N_scans-1)*N_points: i*cut + cut - 1 ,2]-2 ,Mf[ (i*cut) + (N_scans-1)*N_points: i*cut + cut - 1 ,3] , 'ko', X, Vdip(X,*dip_param),'r-')
            plt.show()
            bg0_pos = list()
            bg0_V = list()
            
    return h, array(t) , array(mz), array(s), array(d)














#_________________________________________________________________________________
##################################################################################

Vdip0 = Vdipole_alt_centered

##################################################################################
#_________________________________________________________________________________

guess = [0,0,0,0,0]

#_________________________________________________________________________________


##### Method 1 : fitting the raw data but with an altered dipole function #####

h, tf , mzf , sf, d = magnetization_raw_diag(M_raw , M_d , 300 , Vdip0 , guess)

# h2, tf2 , mzf2 , sf2, d2 = magnetization(M2,M_d2,310,V_bg2_extrapol, 0.8) 

plt.figure()
plt.plot(tf, mzf, 'r-')
plt.xlabel('T(K)')
plt.xlim((0,310))
plt.ylabel('Magnetization (a.u)')
plt.title('$TAS(100)$ + $Sn$ cooling under field H = %s Oe'%(h))
plt.savefig('TAS100+Sn_%s Oe_cooling_HTBS.png'%(h),format='png') 
plt.show()


plt.figure()
plt.plot(tf, d, 'g-')
plt.xlabel('T(K)')
plt.xlim((0,310))
plt.ylabel('$R^2$')
plt.title('Reference ferromagnet under field H = %s Oe'%(h))
plt.savefig('TAS100+Sn_%s Oe_cooling_HTBS_R2.png',format='png') 
plt.show()

np.savetxt('TAS100+Sn_%s _Oe_NoPressure_HTBS_method_1.txt'%(h),np.c_[tf,mzf])
np.savetxt('TAS100+Sn_%s _Oe_NoPressure_HTBS_R2_method_1.txt'%(h),np.c_[tf,d])

#________________________________________________________________________________

##### Method 2 : subtracting the background then fitting  the regular dipole #####

h, tf , mzf , sf, d = magnetization(M,M_d,310,V_bg_extrapol, 0.8) 

# h2, tf2 , mzf2 , sf2, d2 = magnetization(M2,M_d2,310,V_bg2_extrapol, 0.8) 

plt.figure()
plt.plot(tf, mzf, 'r-')
plt.xlabel('T(K)')
plt.xlim((0,310))
plt.ylabel('Magnetization (a.u)')
plt.title('$TAS(100)$ + $Sn$ cooling under field H = %s Oe'%(h))
plt.savefig('TAS100+Sn_%s Oe_cooling_HTBS.png'%(h),format='png') 
plt.show()


plt.figure()
plt.plot(tf, d, 'g-')
plt.xlabel('T(K)')
plt.xlim((0,310))
plt.ylabel('$R^2$')
plt.title('Reference ferromagnet under field H = %s Oe'%(h))
plt.savefig('TAS100+Sn_%s Oe_cooling_HTBS_R2.png',format='png') 
plt.show()

np.savetxt('TAS100+Sn_%s _Oe_NoPressure_HTBS_method_2.txt'%(h),np.c_[tf,mzf])
np.savetxt('TAS100+Sn_%s _Oe_NoPressure_HTBS_R2_method_2.txt'%(h),np.c_[tf,d])







# plt.figure()
# plt.plot(tf2, mzf2, 'r-')
# plt.xlabel('T(K)')
# plt.xlim((0,310))
# plt.ylabel('Magnetization (a.u)')
# plt.title('$TAS(100)$ + $Sn$ under pressure cooling under field H = %s Oe'%(h2))
# plt.savefig('TAS100+Sn_%s Oe_cooling_pressureApplied_HTBS.png'%(h2),format='png') 
# plt.show()


# plt.figure()
# plt.plot(tf2, d2, 'g-')
# plt.xlabel('T(K)')
# plt.xlim((0,310))
# plt.ylabel('$R^2$')
# plt.title('$TAS(100)$ + $Sn$ under pressure cooling under field H = %s Oe'%(h2))
# plt.savefig('TAS100+Sn_%s Oe_cooling_pressureApplied_HTBS_R2.png'%(h2),format='png') 
# plt.show()

# np.savetxt('TAS100+Sn_%s _Oe_PressureApplied_HTBS.txt'%(h2),np.c_[tf2,mzf2])
# np.savetxt('TAS100+Sn_%s _Oe_PressureApplied_HTBS_R2.txt'%(h2),np.c_[tf2,d2])
#_____________________________________________________________________________
