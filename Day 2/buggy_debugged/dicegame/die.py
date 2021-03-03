# Just count the stupid dice
import random
from numpy import arange


die_face_list = ["---------\n|       |\n|   *   |\n|       |\n---------" ,
                 "---------\n|*      |\n|       |\n|      *|\n---------" ,
                 "---------\n|*      |\n|   *   |\n|      *|\n---------" ,
                 "---------\n|*     *|\n|       |\n|*     *|\n---------" ,
                 "---------\n|*     *|\n|   *   |\n|*     *|\n---------" ,
                 "---------\n|*     *|\n|*     *|\n|*     *|\n---------"]
#list af the die faces ready to be printed (see .show_die_face)


class Die:
    """
    Defines a die with an integer value from 1 to 6 which can be displayed using the list above
    """

    def __init__(self):
        """
        Rolls the die and gives a random integer between 1 and 6
        """
        self.value = random.randint(1,6)
        

    def roll(self):
        """
        re-rolls the die and gives a random integer between 1 and 6
        """
        self.value = random.randint(1,6)

    def show_die_face(self):
        """ 
        Prints a representation of the die face at each turn
        """
        print(die_face_list[ self.value - 1 ])
        
    def counts(self, N):
        """
        Rolls N die and displays them. returns the total summed value 
        """
        total = 0

        self.show_die_face()
        total += self.value
        
        for i in arange(N-1) :
            
            self.roll()
            
            self.show_die_face()
            total += self.value
        return total
            
        
            
        



