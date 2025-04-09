from src.liars_dice import LiarsDice
from src.batch_runner import GameBatchRunner
from src.utils import print_intro, print_main_menu
import sys

def main():
    print_intro()
    
    try:
        while True:
            choice = print_main_menu()
            
            if choice == "1":
                game = LiarsDice()
                game.play_game()
            elif choice == "2":
                runner = GameBatchRunner()
                runner.run_tournament()
            elif choice == "3":
                print("Thanks for playing!")
                break
            else:
                print("Invalid choice. Please select 1, 2, or 3.")
    except (KeyboardInterrupt, EOFError):
        print("\nExiting Liar's Dice. Goodbye!")
        sys.exit(0)

if __name__ == "__main__":
    main()
