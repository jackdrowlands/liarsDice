from src.liars_dice import LiarsDice
from src.batch_runner import GameBatchRunner
from src.utils import print_intro, print_main_menu, select_saved_game
import sys
import argparse
import os

# Ensure the src directory is in the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))


def parse_args():
    parser = argparse.ArgumentParser(description='Liar\'s Dice Game')
    parser.add_argument('--load', type=str, help='Load a saved game from file')
    parser.add_argument('--resume-tournament', type=str, help='Resume a tournament from a saved state file')
    return parser.parse_args()

def main():
    args = parse_args()

    if args.load:
        game_state = load_game_state(args.load)
        if game_state:
            game = LiarsDice.from_state(game_state)
            print(f"Game loaded from {args.load}")
            game.play_game()
        else:
            print(f"Failed to load game from {args.load}")
        return

    if args.resume_tournament:
        batch_runner = GameBatchRunner() # Create instance
        if batch_runner.load_tournament_state(args.resume_tournament):
            print(f"Tournament resumed from {args.resume_tournament}")
            # The run_tournament method should handle whether to create visualizations based on its internal state
            batch_runner.run_tournament()
        else:
            print(f"Failed to resume tournament from {args.resume_tournament}. Starting new setup.")
            # Fall through to main menu if resume fails
        # return # Decide if we should exit or fall through to menu if resume fails

    while True:
        print("\\n--- Liar's Dice Main Menu ---")
        print("1. Start New Game (Human vs AI)")
        print("2. Start New Game (AI vs AI)")
        print("3. Run AI Model Tournament")
        print("4. Load Game State")
        print("5. Resume Tournament")
        print("6. Exit")
        choice = input("Enter your choice: ")

        if choice == '1':
            game = LiarsDice()
            game.play_game()
        elif choice == '2':
            game = LiarsDice()
            game.play_game()
        elif choice == '3':
            batch_runner = GameBatchRunner() # Create instance for new tournament
            if batch_runner.setup_batch(): # Interactive setup
                # run_tournament should use self.create_visualizations which is set during setup_batch
                batch_runner.run_tournament()
            else:
                print("Tournament setup failed or was cancelled.")
        elif choice == '4':
            filename = input("Enter filename to load game state from (e.g., game_state.json): ").strip()
            game_state = load_game_state(filename)
            if game_state:
                game = LiarsDice.from_state(game_state)
                print(f"Game loaded from {filename}")
                game.play_game()
            else:
                print(f"Failed to load game from {filename}")
        elif choice == '5':
            filename = input("Enter filename to resume tournament from (e.g., tournament_state.json): ").strip()
            batch_runner = GameBatchRunner() # Create instance
            if batch_runner.load_tournament_state(filename):
                print(f"Tournament resumed from {filename}")
                batch_runner.run_tournament()
            else:
                print(f"Failed to resume tournament from {filename}.")
        elif choice == '6':
            print("Exiting Liar's Dice. Goodbye!")
            break
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main()
