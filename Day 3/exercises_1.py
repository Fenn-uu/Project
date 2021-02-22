# -*- coding: utf-8 -*-
"""
Created on Wed Feb 17 13:43:14 2021

@author: ferde233
"""

import numpy as np

# https://github.com/uu-python/day3-highperformance/blob/master/exersises.md for the exercises' description.

# 1.a
a = np.zeros(10)

a[4] = 1.

print(a)

# 1.b

b = np.arange(10,49,1)

print(b)

# 1.c

c = b[::-1].copy()

print(c)	

# 1.d

d = np.arange(9).reshape(3,3)

print(d)
 
# 1.e

e = np.array([1,2,0,0,4,0]) == 0

print(e)

for i in range(len(e)) :
    if e[i] :
        print("0 at position %s "%(i))
        

# 1.f

f = np.random.random((30))

f2 = np.mean(f)

print(f)
print('The mean of that vector is %s'%(f2))


# 1.g

g = np.zeros((10,10))

g[0,:] = 1.
g[:,0] = 1.
g[-1,:] = 1.
g[:,-1] = 1.

print(g)

# 1.h

h = np.zeros((8,8))
len_h = len(h)

for i in np.arange(len_h) :
    for j in np.arange(len_h) :
        if (i+j) % 2 == 0 :
            h[i,j] = 1.

            
print(h)


# 1.i

i = np.tile([[1.,0.],[0.,1.]],(4,4))

print(i)


# 1.j

Z = np.arange(11)
Z_more = Z > 3 
Z_less = Z < 8
Z_bool = Z_more * Z_less


for i in np.arange(11) :
    if Z_bool[i] :
        Z[i] = -Z[i]
    
print(Z)

# 1.k

Z = np.random.random(10)
Z.sort()

print(Z)

# 1.l

A = np.random.randint(0,2,5)
B = np.random.randint(0,2,5)

print(A)
print(B)

for i in np.arange(5):
    if A[i] == B[i] :
        if i == 4 :
            print('The two vectors are equal !')
        continue
    else :
        print('The two vectors are not equal.')
        break

# 1.m 

Z = np.arange(10, dtype=np.int32)
print(Z.dtype)
Z = Z.astype('float32')
print(Z.dtype)

# 1.n 

A = np.arange(9).reshape(3,3)
B = A + 1
C = np.dot(A,B)
D = np.zeros((3,3))

for i in np.arange(3) :
    D[i,i] = C[i,i]

print(D)
