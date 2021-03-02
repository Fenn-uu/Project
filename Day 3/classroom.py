#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  2 15:29:39 2021

@author: fernand
"""


class Person(object):
    
    def __init__(self , firstname , lastname) :
        
        self.firstname = str(firstname)
        self.lastname = str(lastname)
        
        print(self.firstname + ' ' + self.lastname)
        


class Student(Person):
    
    def __init__(self , firstname , lastname , subject) :
        Person.__init__(self,firstname,lastname)
        self.subject = str(subject)
        
    def printNameSubject(self) :
        print(self.firstname + ' ' + self.lastname + ', ' + self.subject)
        
    
class Teacher(Person):
    
    def __init__(self , firstname , lastname , course) :
        Person.__init__(self,firstname,lastname)
        self.course = str(course)
        
    def printNameCourse(self) :
        print(self.firstname + ' ' + self.lastname + ', ' + self.course)
        
         
        
        
    