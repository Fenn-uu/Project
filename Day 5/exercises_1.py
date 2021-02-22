# -*- coding: utf-8 -*-
"""
Created on Fri Feb 19 09:49:36 2021

@author: ferde233
"""

import numpy as np
import scipy.linalg as lina 
from scipy.stats import poisson, norm
import matplotlib.pyplot as plt 


### Exercise 1 ###

# 1.a
A = np.arange(9).reshape((3,3)) + 1
# 3x3 matrix

# 1.b
b = np.arange(3) + 1
# 1x3 vector

# 1.c
x_sol = lina.solve(A,b)
# Solution of Ax = b

print(x_sol)

# 1-d
print('Ax - b should be -within error- the vector (0,0,0) :')
print('\t%s' % (np.dot(A,x_sol) - b))

# 1.e
B = np.random.random(9).reshape((3,3))
x_sol_rand = lina.solve(B,b)
print(x_sol_rand)
print('Bx - b should be -within error- the vector (0,0,0) :')
print('\t%s' % (np.dot(B,x_sol_rand) - b))


# 1.f
A_eig = lina.eig(A)


print('The eigenvalues of A are :')
for eigenvalues in A_eig[0]:
    print(eigenvalues)
    
print('The respective eigenvectors are :')
for i in np.arange(len(A_eig[0])) :
    print(A_eig[1][:,i])
    
# 1.g

A_inv = lina.inv(A)
A_inv_det = lina.det(A_inv)

print('A =')
print(A)

print('A_inv =')
print(A_inv)

print('The determinant of A_inv is :')
print(A_inv_det)

# 1.h
A_norm = lina.norm(A)
# Frobenius norm (order 2)

for i in np.delete( np.arange(-2,3,1), 2)  :
    
    print('The norm of A (order %s) is :'% (i))
    print('\t%s' % (lina.norm(A, ord = i )))
    #print the norm of orders -2 to 2, skipping 0
    
# Statistics #

# a
mu = 4.
K = np.arange(0,21)
Y = poisson.pmf( K , mu*np.ones(21))
Y2 = poisson.cdf( K , mu*np.ones(21))

plt.figure('1.a')
plt.plot( K , Y , 'ro' )
plt.xlabel('k')
plt.xlim((0,20))
plt.ylabel('P(x=k)')
plt.title(' Poisson probability mass function for mu = %s' %(mu))
plt.show()

plt.figure('1.b')
plt.plot( K , Y2 , 'rv')
plt.xlabel('k')
plt.xlim((0,20))
plt.ylabel('P(x=k)')
plt.title('Poisson cumulative distribution function for mu = %s' %(mu))
plt.show()






# b
sig = 1.
X = np.linspace(-3,3,100)
Y = norm.pdf(X, scale=sig)
Y2 = norm.cdf(X,scale=sig)

plt.figure('2.a')
plt.plot( X , Y , 'g-' )
plt.xlabel('x')
plt.xlim((-3,3))
plt.ylabel('norm(x)')
plt.title('Normal probability distribution function for sigma = %s' % (sig))
plt.show()

plt.figure('2.b')
plt.plot( X , Y2 , 'g--')
plt.xlabel('x')
plt.xlim((-3,3))
plt.ylabel('cdf(x)')
plt.title('Normal cumulative distribution function for sigma = %s' %(sig))
plt.show()

    

