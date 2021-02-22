# -*- coding: utf-8 -*-
"""
Created on Thu Feb 18 14:05:05 2021

@author: ferde233
"""
import setuptools
from	distutils.core	import	setup	
from	Cython.Build	import	cythonize	

setup(
				ext_modules	= cythonize("cy_sci_rbf.pyx")
)