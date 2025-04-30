import random
from typing import List, Optional, Union, Any

class Player:
    def __init__(self, name: str) -> None:
        self.name: str = name
        self.dice: List[int] = []
        self.num_dice: int = 5
    
    def roll_dice(self) -> None:
        self.dice = [random.randint(1, 6) for _ in range(self.num_dice)]
    
    def remove_die(self) -> None:
        if self.num_dice > 0:
            self.num_dice -= 1
            if self.num_dice > 0:
                self.dice = self.dice[:self.num_dice]
            else:
                self.dice = []
    
    def get_dice_count(self) -> int:
        return self.num_dice
