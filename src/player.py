import random

class Player:
    def __init__(self, name):
        self.name = name
        self.dice = []
        self.num_dice = 5
    
    def roll_dice(self):
        self.dice = [random.randint(1, 6) for _ in range(self.num_dice)]
    
    def remove_die(self):
        if self.num_dice > 0:
            self.num_dice -= 1
            if self.num_dice > 0:
                self.dice = self.dice[:self.num_dice]
            else:
                self.dice = []
    
    def get_dice_count(self):
        return self.num_dice
