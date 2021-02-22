# -*- coding: utf-8 -*-
"""
Created on Thu Feb 18 16:52:40 2021

@author: ferde233
"""

class Mammals:
    def __init__(self):
        ''' Constructor for this class. '''
        # Create some member animals
        self.members = ['Tiger', 'Elephant', 'Wild cat']
        self.d = ['Tiger' , 'Elephant']
        self.h = ['Wild cat']
    
    
    def printMembers(self):
        print('Printing members of the Mammals class')
        for member in self.members:
            print('\t%s ' % member)
            
    def dangerous(self): 
        def printMembers(self) :
            print('Printing dangerous members of the Mammals class')
            for member in self.d :
                print('\t%s' % member)
    
    def harmless(self):
        def printMembers(self) :
            print('Printing harmless members of the Mammals class')
            for member in self.h :
                print('\t%s' % member)