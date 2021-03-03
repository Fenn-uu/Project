from .die import Die
# from .utils import i_just_throw_an_exception
# deleted since useless



class GameRunner:

    def __init__(self):
        
        
        myDie = Die()
        self.answer = myDie.counts(5)
        # This function prints 5 random dice and returns the value of their sum all in one.
    
        self.round = 1
        self.wins = 0
        self.loses = 0
        # Stores the number of rounds and win-lose points


    def reroll(self):
        
        anotherDie = Die()
        self.answer = anotherDie.counts(5)

    def win(self):
        self.round += 1
        self.wins += 1
        
    def lose(self):
        self.round += 1
        self.loses += 1
    
    def retry(self):
        prompt = input("Would you like to play again?[Y/n]: ")

        if prompt == 'y' or prompt == 'Y' or prompt == '':
            self.game_start()
        else:
            print("See you soon.")
            global end
            end = 1
        
    @staticmethod
    def game_continue():
        prompt = input("Would you like to continue ?[Y/n]: ")

        if prompt == 'y' or prompt == 'Y' or prompt == '':
            pass
        else:
            print("See you soon.")
            global end
            end = 1
        

    @classmethod
    def game_start(cls):
        # Probably counts wins or something.
        # Great variable name, 10/10.
        global end
        end = 0
        
        runner = cls()
        
        while True:
            
            if runner.wins > 5:
                print("You won... Congrats...")
                print("The fact it took you so long is pretty sad")
                runner.retry()
                break

            if runner.loses > 5:
                print("Sorry... You lose...")
                print("Give it another try.")
                runner.retry()
                break
            
            
            if end != 0 : 
                break
            

            print("Round {}\n".format(runner.round))

            # for die in runner.dice:
            #     print(die.show())

            guess = input("Sigh. What is your guess?: ")
            guess = int(guess)

            if guess == runner.answer:
                print("Congrats, you can add like a 5 year old...")
                runner.win()
                print("Wins: {} Loses {}".format(runner.wins, runner.loses))
                if runner.wins == 6 or runner.loses == 6:
                    continue
                runner.game_continue()
                
            else:
                print("Sorry that's wrong")
                print("The answer is: {}".format(runner.answer))
                print("Like seriously, how could you mess that up")
                runner.lose()
                print("Wins: {} Loses {}".format(runner.wins, runner.loses))
                if runner.wins == 6 or runner.loses == 6:
                    continue
                runner.game_continue()
             
                
            runner.reroll()
            # At this point the dice are rerolled and the loop starts again, unless the win or lose number is 6.
            

            
            
            
