# -*- coding: utf-8 -*-
"""
Created on Thu Feb 18 17:08:19 2021

@author: ferde233
"""

# Import classes from your brand new package
from Animals import Mammals
from Animals import Birds
from Animals import Fish

# Create an object of Mammals class & call a method of it
myMammal = Mammals()
myMammal.printMembers()
myMammal.dangerous.printMembers()

# Create an object of Birds class & call a method of it
myBird = Birds()
myBird.printMembers()
myBird.dangerous.printMembers()

# Create an object of Fish class & call a method of it
myFish = Fish()
myFish.printMembers()
myFish.dangerous.printMembers()