from src.liars_dice import LiarsDice
from src.batch_runner import GameBatchRunner
from src.utils import print_intro, print_main_menu, select_saved_game
import sys
import argparse
import os

def parse_args():
    parser = argparse.ArgumentParser(description='Liar\'s Dice Game')
    parser.add_argument('--load', type=str, help='Load a saved game from file')
    parser.add_argument('--resume-tournament', type=str, help='Resume a tournament from a saved state file')
    return parser.parse_args()

def main():
    args = parse_args()
    
    # If load argument provided, load the game directly
    if args.load and os.path.exists(args.load):
        print(f"Loading saved game from {args.load}...")
        game = LiarsDice(save_file=args.load)
        game.play_game(load_from=args.load)
        return
    
    # If resume-tournament argument provided, resume the tournament
    if args.resume_tournament and os.path.exists(args.resume_tournament):
        print(f"Resuming tournament from {args.resume_tournament}...")
        runner = GameBatchRunner()
        runner.run_tournament(resume_from=args.resume_tournament)
        return
    
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
                # Load a saved game
                save_file = select_saved_game()
                if save_file:
                    game = LiarsDice(save_file=save_file)
                    result = game.play_game(load_from=save_file)
                    if result == "paused":
                        print("Game paused. You can continue later.")
            elif choice == "4":
                print("Thanks for playing!")
                break
            else:
                print("Invalid choice. Please select 1, 2, 3, or 4.")
    except (KeyboardInterrupt, EOFError):
        print("\nExiting Liar's Dice. Goodbye!")
        sys.exit(0)

if __name__ == "__main__":
    main()
