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

class HumanPlayer(Player):
    def __init__(self, name: str) -> None:
        super().__init__(name)

    def get_bid_input(self, game: Any, max_dice_qty: int) -> Optional[tuple[str, Optional[tuple[int, int]]]]:
        """Get bid or liar call from human player."""
        # This method would contain the logic to prompt the human user for their move.
        # For now, it's a placeholder. In a full implementation, you'd handle
        # input parsing, validation against game rules, etc.
        print(f"{self.name}, it's your turn. Your dice: {self.dice}")
        action = input("Enter 'bid' or 'liar': ").lower().strip()
        if action == "bid":
            while True:
                try:
                    qty_str = input(f"Enter quantity (1-{max_dice_qty}): ")
                    if not qty_str: return None # Allow empty input to re-prompt main action
                    quantity = int(qty_str)

                    val_str = input("Enter value (1-6): ")
                    if not val_str: return None # Allow empty input to re-prompt main action
                    value = int(val_str)
                    
                    # Basic validation (more can be added from LiarsDice.is_valid_bid)
                    if not (1 <= quantity <= max_dice_qty and 1 <= value <= 6):
                        print("Invalid quantity or value. Try again.")
                        continue
                    return "bid", (quantity, value)
                except ValueError:
                    print("Invalid input. Please enter numbers for quantity and value.")
                except Exception as e:
                    print(f"An error occurred: {e}")
                    return None # Or handle more gracefully
        elif action == "liar":
            return "liar", None
        else:
            print("Invalid action. Type 'bid' or 'liar'.")
            return None # Re-prompt
