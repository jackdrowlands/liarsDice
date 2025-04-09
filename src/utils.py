import os

def print_intro():
    """Print the introduction message for the game"""
    print("=" * 70)
    print("Welcome to Liar's Dice OpenRouter Tournament!")
    print("=" * 70)
    print("\nIn this game, each player has dice that only they can see.")
    print("Players take turns making bids about how many dice of a certain value exist among all players.")
    print("Each bid must be higher than the previous one (either more dice, or same number but higher value).")
    print("When a player thinks the previous bid is a lie, they can call 'Liar'.")
    print("The loser of each round loses one die. The last player with dice is the winner!")
    
    try:
        import requests
        OPENROUTER_AVAILABLE = True
    except ImportError:
        OPENROUTER_AVAILABLE = False
        
    if OPENROUTER_AVAILABLE:
        print("\nOpenRouter integration is AVAILABLE!")
        print("Game Modes Available:")
        print("1. Human Players Only - Play with friends")
        print("2. Mixed Game - Play with a mix of human and AI players") 
        print("3. AI vs AI - Watch different AI models compete against each other")
        
        print("\n🤖 MODEL VS MODEL FEATURE 🤖")
        print("- Pit different AI models against each other")
        print("- Run tournaments to find the best Liar's Dice AI")
        print("- See detailed statistics and performance metrics")
        
        print("\nModels Available Through OpenRouter:")
        print("- Claude 3 (Anthropic)")
        print("- GPT-4 and GPT-4o (OpenAI)")
        print("- Gemini (Google)")
        print("- Mistral, Llama, Command-R, and many others")
        print("- Access to 50+ different models through a single API")
        
        print("\nEach model has its own 'personality' and strategic approach")
        print("Auto mode available to watch games without interruption")
    else:
        print("\nOpenRouter integration is NOT available.")
        print("To enable AI players, install the required package:")
        print("- pip install requests")
        print("\nYou'll also need an OpenRouter API key from openrouter.ai")
    
    print("\nLet's get started!\n")
    print("=" * 70)


def print_main_menu():
    """Print the main menu options and get user selection"""
    print("\n" + "=" * 70)
    print("LIARS DICE MAIN MENU")
    print("=" * 70)
    print("1. Play a single game")
    print("2. Run a model tournament")
    print("3. Exit")
    return input("\nSelect an option (1-3): ").strip()
