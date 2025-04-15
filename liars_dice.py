import random
import os
import time
import json
import sys

# Check for OpenRouter support (required)
try:
    import requests
    OPENROUTER_AVAILABLE = True
except ImportError:
    OPENROUTER_AVAILABLE = False

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


class AIPlayer(Player):
    def __init__(self, name, model, api_key=None):
        super().__init__(name)
        self.model = model
        self.game_history = []
        self.personality = self.generate_personality(model)
        self.api_key = api_key or os.environ.get("OPENROUTER_API_KEY", "")
        
    def generate_personality(self, model):
        """Generate a personality for the AI player based on the model"""
        model_lower = model.lower()
        
        if "claude" in model_lower:
            return "thoughtful and careful"
        elif "gpt-4" in model_lower:
            if "o" in model_lower:
                return "calculated and adaptive"
            else:
                return "analytical and strategic"
        elif "gpt-3.5" in model_lower:
            return "bold and unpredictable"
        elif "gemini" in model_lower or "palm" in model_lower:
            return "creative and unexpected" 
        elif "llama" in model_lower:
            return "determined and focused"
        elif "mistral" in model_lower:
            return "resourceful and practical"
        elif "command" in model_lower:
            return "balanced and consistent"
        else:
            return "balanced and versatile"
    
    def record_game_state(self, game_state):
        self.game_history.append(game_state)
    
    def get_prompt_for_game(self, game_state):
        """Create a prompt for the AI model"""
        # Customize the prompt based on the model type
        model_lower = self.model.lower()
        
        if "claude" in model_lower or "gpt-4" in model_lower or "gemini-pro" in model_lower:
            strategy_style = """
            Advanced strategy tips:
            1. Calculate probability distributions for each value based on visible dice
            2. Track each player's behavior patterns over time
            3. Use strategic bluffing to mislead opponents about your actual dice
            4. Identify when a player is likely bluffing based on their past behavior
            5. Consider the risk/reward of calling "Liar" vs making a higher bid
            """
        else:
            strategy_style = """
            Strategy tips:
            1. Use the move history to understand each player's tendencies
            2. Track which players have been caught bluffing in the past
            3. Consider how many dice are left in the game when calculating probabilities
            4. If a player has lost dice, they are less likely to have high quantities of any value
            5. Be more cautious when making high bids later in the game
            """
            
        # Format the prompt with game information
        system_prompt = f"""
        You are playing Liar's Dice as model {self.model}. In this game, each player has dice that only they can see.
        Players take turns making bids about how many dice of a certain value exist among all players.
        Each bid must be higher than the previous one (either more dice, or same number but higher value).
        When a player thinks the previous bid is a lie, they can call "Liar".
        
        Your personality: You are {self.personality}.
        
        Rules for making decisions:
        1. You can either make a higher bid or call the previous player a liar
        2. A bid consists of a quantity and a value (e.g., "three 4's" means "three dice with value 4")
        3. A bid must increase either the quantity or the value of the previous bid
        4. Be strategic - consider probability and bluffing
        5. Return your decision in JSON format as specified
        
        {strategy_style}
        
        Think step by step about your decision.
        """
        
        # Format move history for prompt
        move_history_text = ""
        if game_state['move_history']:
            move_history_text = "Move history:\n"
            for i, move in enumerate(game_state['move_history']):
                if move["action"] == "bid":
                    move_history_text += f"- {move['player']} bid {move['quantity']} {move['value']}'s\n"
                elif move["action"] == "liar":
                    move_history_text += f"- {move['player']} called 'Liar!' on {move['target_player']}"
                    if "outcome" in move:
                        if move["outcome"] == "success":
                            move_history_text += f" and was right! {move['target_player']} lost a die.\n"
                        else:
                            move_history_text += f" and was wrong! {move['player']} lost a die.\n"
                    else:
                        move_history_text += "\n"
        
        user_prompt = f"""
        Current game state:
        - Current round: {game_state.get('round_number', 1)}
        - Your dice: {sorted(self.dice)}
        - Total dice in game: {game_state['total_dice']}
        - Players and their dice counts: {game_state['player_dice_counts']}
        
        {f"Previous bid: {game_state['last_bid'][0]} dice showing {game_state['last_bid'][1]}" if game_state['last_bid'] else "You are making the first bid."}
        
        {move_history_text}
        
        Please decide:
        1. If you want to make a bid, respond with: {{"action": "bid", "quantity": X, "value": Y}}
        2. If you want to call "Liar" on the previous bid, respond with: {{"action": "liar"}}
        
        Your decision:
        """
        
        return {"system": system_prompt, "user": user_prompt}
    
    def get_ai_decision(self, game_state):
        if not OPENROUTER_AVAILABLE:
            raise ImportError("Requests package is not installed. Install with 'pip install requests'")
        
        if not self.api_key:
            raise ValueError("OpenRouter API key is required")
        
        prompt = self.get_prompt_for_game(game_state)
        
        try:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": ""
            }
            
            data = {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": prompt["system"]},
                    {"role": "user", "content": prompt["user"]}
                ],
                "temperature": 0.7,
                "max_tokens": 500
            }
            
            response = requests.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers=headers,
                json=data
            )
            
            if response.status_code != 200:
                print(f"OpenRouter API Error: {response.status_code} - {response.text}")
                return {"action": "bid", "quantity": 1, "value": 4}
            
            result = response.json()
            content = result["choices"][0]["message"]["content"].strip()
            
            # Try to extract JSON from the response
            try:
                # Find JSON in the response if it's embedded in text
                json_start = content.find('{')
                json_end = content.rfind('}') + 1
                if json_start >= 0 and json_end > json_start:
                    json_str = content[json_start:json_end]
                    decision = json.loads(json_str)
                else:
                    # Default fallback if no JSON found
                    decision = {"action": "bid", "quantity": 1, "value": 4}
            except json.JSONDecodeError:
                # Fallback to a default bid if parsing fails
                decision = {"action": "bid", "quantity": 1, "value": 4}
            
            return decision
        except Exception as e:
            print(f"Error getting AI decision: {e}")
            # Fallback to a default bid
            return {"action": "bid", "quantity": 1, "value": 4}


class LiarsDice:
    def __init__(self):
        self.players = []
        self.current_player_idx = 0
        self.last_bid = None
        self.total_dice_in_game = 0
        self.game_over = False
        self.winner = None
        self.move_history = []  # Track all moves in the game
        self.round_number = 1   # Track game rounds
    
    def clear_screen(self):
        # os.system('cls' if os.name == 'nt' else 'clear')
        print("clear")
    
    def add_player(self, name):
        player = Player(name)
        self.players.append(player)
    
    def add_ai_player(self, name, model, api_key=None):
        if not OPENROUTER_AVAILABLE:
            print("Requests package is not installed. Install with 'pip install requests'")
            return False
        
        ai_player = AIPlayer(name, model, api_key)
        self.players.append(ai_player)
        return True
        
    def get_available_models(self):
        """Return a list of available models from OpenRouter"""
        default_models = [
            "openai/gpt-4-turbo",
            "openai/gpt-4o",
            "anthropic/claude-3-haiku",
            "anthropic/claude-3-opus",
            "anthropic/claude-3-sonnet",
            "google/gemini-pro",
            "mistralai/mistral-large",
            "meta/llama-3-70b-instruct",
            "meta/llama-3-8b-instruct",
            "cohere/command-r-plus",
            "anthropic/claude-3-5-sonnet"
        ]
        
        # Check if we have an OpenRouter API key
        openrouter_api_key = os.environ.get("OPENROUTER_API_KEY", "")
        
        if OPENROUTER_AVAILABLE and openrouter_api_key:
            try:
                headers = {
                    "Authorization": f"Bearer {openrouter_api_key}"
                }
                
                response = requests.get("https://openrouter.ai/api/v1/models", headers=headers)
                
                if response.status_code == 200:
                    data = response.json()
                    models = []
                    
                    for model in data["data"]:
                        models.append(model["id"])
                    
                    # Sort models to group them by provider
                    models.sort()
                    return models
                else:
                    print(f"Error fetching OpenRouter models: {response.status_code}")
                    return default_models
            except Exception as e:
                print(f"Error fetching OpenRouter models: {e}")
                return default_models
        else:
            return default_models
        
    def setup_game(self):
        num_players = 0
        while num_players < 2:
            try:
                num_players = int(input("Enter number of players (2 or more): "))
                if num_players < 2:
                    print("You need at least 2 players.")
            except ValueError:
                print("Please enter a valid number.")
        
        # Determine if using AI players
        game_mode = ""
        while game_mode not in ["1", "2", "3"]:
            print("\nGame Mode Options:")
            print("1. Human players only")
            print("2. Mix of human and AI players")
            print("3. AI players only (model vs model)")
            game_mode = input("Select game mode (1-3): ")
        
        game_mode = int(game_mode)
        ai_count = 0
        all_ai_game = False
        
        # Handle game mode selection
        if game_mode == 1:  # Human only
            ai_count = 0
        elif game_mode == 2:  # Mix of human and AI
            if OPENROUTER_AVAILABLE:
                try:
                    ai_count = int(input("How many AI players? "))
                    if ai_count > num_players:
                        print(f"AI count cannot exceed total players ({num_players})")
                        ai_count = num_players - 1
                except ValueError:
                    print("Please enter a valid number for AI players.")
                    ai_count = 0
            else:
                print("Requests package is not installed. Install with 'pip install requests'")
                ai_count = 0
        elif game_mode == 3:  # AI only (model vs model)
            if OPENROUTER_AVAILABLE:
                ai_count = num_players
                all_ai_game = True
            else:
                print("Requests package is not installed. Install with 'pip install requests'")
                print("Falling back to human players only.")
                ai_count = 0
        
        # Get API key if needed
        api_key = None
        if ai_count > 0:
            api_key = os.environ.get("OPENROUTER_API_KEY", "")
            if not api_key:
                api_key = input("Enter your OpenRouter API key: ")
                os.environ["OPENROUTER_API_KEY"] = api_key
        
        human_count = num_players - ai_count
        
        # Add human players
        for i in range(human_count):
            name = input(f"Enter name for Human Player {i+1}: ")
            self.add_player(name)
        
        # Add AI players
        if ai_count > 0:
            # Get available models
            available_models = self.get_available_models()
            
            if not available_models:
                print("No models available. Check your API key and connection.")
                return
            
            print("\nAvailable Models:")
            for i, model in enumerate(available_models):
                print(f"{i+1}. {model}")
            
            if all_ai_game:
                print("\nSetting up AI vs AI game...")
                
                # In AI vs AI mode, select different models for each player
                for i in range(ai_count):
                    print(f"\nAI Player {i+1}:")
                    
                    # Select model for this AI player
                    model_idx = -1
                    while model_idx < 0 or model_idx >= len(available_models):
                        try:
                            model_idx = int(input(f"Select model number (1-{len(available_models)}): ")) - 1
                        except ValueError:
                            print("Please enter a valid number.")
                    
                    model_choice = available_models[model_idx]
                    
                    # Get a readable model name for the player name
                    model_short_name = model_choice.split('/')[-1] if '/' in model_choice else model_choice
                    
                    # Name the AI based on the model
                    default_name = f"{model_short_name.upper()}-{i+1}"
                    name = input(f"Enter name for this AI (default: {default_name}): ")
                    if not name:
                        name = default_name
                    
                    # Add the AI player
                    self.add_ai_player(name, model_choice, api_key)
                    print(f"Added AI player '{name}' using model: {model_choice}")
            else:
                # In mixed mode, let user choose a model for each AI
                for i in range(ai_count):
                    print(f"\nAI Player {i+1}:")
                    
                    # Select model for this AI player
                    model_idx = -1
                    while model_idx < 0 or model_idx >= len(available_models):
                        try:
                            model_idx = int(input(f"Select model number (1-{len(available_models)}): ")) - 1
                        except ValueError:
                            print("Please enter a valid number.")
                    
                    model_choice = available_models[model_idx]
                    
                    # Get a readable model name for the player name
                    model_short_name = model_choice.split('/')[-1] if '/' in model_choice else model_choice
                    
                    # Name the AI
                    default_name = f"{model_short_name}-{i+1}"
                    name = input(f"Enter name for this AI (default: {default_name}): ")
                    if not name:
                        name = default_name
                    
                    # Add the AI player
                    self.add_ai_player(name, model_choice, api_key)
                    print(f"Added AI player '{name}' using model: {model_choice}")
        
        if self.players:
            self.current_player_idx = random.randint(0, len(self.players) - 1)
            print(f"\n{self.players[self.current_player_idx].name} will go first!")
    
    def roll_all_dice(self):
        self.total_dice_in_game = 0
        for player in self.players:
            player.roll_dice()
            self.total_dice_in_game += player.get_dice_count()
    
    def next_player(self):
        while True:
            self.current_player_idx = (self.current_player_idx + 1) % len(self.players)
            if self.players[self.current_player_idx].get_dice_count() > 0:
                break
    
    def show_dice_to_player(self, player_idx):
        player = self.players[player_idx]
        
        # Skip display for AI players
        if isinstance(player, AIPlayer):
            return
            
        self.clear_screen()
        print(f"\n{player.name}'s turn")
        print("Your dice:", sorted(player.dice))
        
        # Show how many dice each player has
        print("\nDice count:")
        for i, p in enumerate(self.players):
            dice_count = p.get_dice_count()
            if i == player_idx:
                print(f"  {p.name} (YOU): {dice_count}")
            else:
                print(f"  {p.name}: {dice_count}")
        
        if self.last_bid:
            quantity, value = self.last_bid
            last_player = self.players[(player_idx - 1) % len(self.players)]
            print(f"\nLast bid: {last_player.name} bid {quantity} {value}'s")
        
        # Display move history
        self.display_move_history()
    
    def create_game_state_for_ai(self, player_idx):
        """Create a game state dictionary for the AI player"""
        player_dice_counts = {player.name: player.get_dice_count() for player in self.players}
        
        return {
            "total_dice": self.total_dice_in_game,
            "player_dice_counts": player_dice_counts,
            "last_bid": self.last_bid,
            "current_player": self.players[player_idx].name,
            "move_history": self.move_history,
            "round_number": self.round_number
        }
    
    def get_player_bid(self, player_idx):
        player = self.players[player_idx]
        
        # If the player is an AI
        if isinstance(player, AIPlayer):
            print(f"\n{player.name} (AI) is thinking...")
            
            # Create game state for AI
            game_state = self.create_game_state_for_ai(player_idx)
            
            # Add a short delay to make it feel more natural
            time.sleep(random.uniform(1.5, 3.0))
            
            try:
                # Get AI decision
                decision = player.get_ai_decision(game_state)
                
                if decision["action"] == "liar":
                    if not self.last_bid:
                        # AI shouldn't call liar on first turn, make a bid instead
                        print(f"{player.name} decides to make a bid.")
                        self.last_bid = (1, random.randint(3, 6))
                        print(f"{player.name} bids {self.last_bid[0]} {self.last_bid[1]}'s")
                        return False
                    
                    # Record liar call in history
                    self.move_history.append({
                        "round": len(self.move_history) + 1,
                        "player": player.name,
                        "action": "liar",
                        "target_player": self.players[(self.current_player_idx - 1) % len(self.players)].name
                    })
                    print(f"{player.name} calls 'Liar!' on the previous bid.")
                    return True
                else:
                    # AI is making a bid
                    quantity = decision["quantity"]
                    value = decision["value"]
                    
                    # Validate the bid
                    valid_bid = True
                    if quantity < 1 or value < 1 or value > 6:
                        valid_bid = False
                    
                    # Check if bid is higher than the last bid
                    if self.last_bid and valid_bid:
                        last_quantity, last_value = self.last_bid
                        
                        # Bid must be higher
                        if quantity < last_quantity or (quantity == last_quantity and value <= last_value):
                            valid_bid = False
                    
                    if valid_bid:
                        self.last_bid = (quantity, value)
                        # Record move in history
                        self.move_history.append({
                            "round": len(self.move_history) + 1,
                            "player": player.name,
                            "action": "bid",
                            "quantity": quantity,
                            "value": value
                        })
                        print(f"{player.name} bids {quantity} {value}'s")
                        return False
                    else:
                        # If AI made an invalid bid, make a safe valid bid
                        if self.last_bid:
                            last_quantity, last_value = self.last_bid
                            if last_value < 6:
                                self.last_bid = (last_quantity, last_value + 1)
                            else:
                                self.last_bid = (last_quantity + 1, 1)
                        else:
                            self.last_bid = (1, random.randint(3, 6))
                        
                        # Record move in history
                        self.move_history.append({
                            "round": len(self.move_history) + 1,
                            "player": player.name,
                            "action": "bid",
                            "quantity": self.last_bid[0],
                            "value": self.last_bid[1]
                        })
                        
                        print(f"{player.name} bids {self.last_bid[0]} {self.last_bid[1]}'s")
                        return False
            except Exception as e:
                print(f"Error with AI decision: {e}")
                # Fallback to a simple bid
                if self.last_bid:
                    last_quantity, last_value = self.last_bid
                    if last_value < 6:
                        self.last_bid = (last_quantity, last_value + 1)
                    else:
                        self.last_bid = (last_quantity + 1, 1)
                else:
                    self.last_bid = (1, random.randint(3, 6))
                
                # Record move in history
                self.move_history.append({
                    "round": len(self.move_history) + 1,
                    "player": player.name,
                    "action": "bid",
                    "quantity": self.last_bid[0],
                    "value": self.last_bid[1],
                    "error_fallback": True
                })
                
                print(f"{player.name} bids {self.last_bid[0]} {self.last_bid[1]}'s")
                return False
        
        # Human player logic
        while True:
            print("\nOptions:")
            print("1. Make a bid")
            print("2. Call 'Liar!' on the previous bid")
            
            choice = input("Enter your choice (1 or 2): ").strip()
            
            if choice == "1":
                try:
                    quantity = int(input("How many dice? "))
                    value = int(input("What value (1-6)? "))
                    
                    if quantity < 1 or value < 1 or value > 6:
                        print("Invalid bid. Quantity must be positive and value must be between 1 and 6.")
                        continue
                    
                    # Check if bid is higher than the last bid
                    if self.last_bid:
                        last_quantity, last_value = self.last_bid
                        
                        # Bid must be higher
                        if quantity < last_quantity or (quantity == last_quantity and value <= last_value):
                            print("Your bid must be higher than the previous bid.")
                            continue
                    
                    self.last_bid = (quantity, value)
                    
                    # Record move in history
                    self.move_history.append({
                        "round": len(self.move_history) + 1,
                        "player": player.name,
                        "action": "bid",
                        "quantity": quantity,
                        "value": value
                    })
                    
                    return False  # Not calling liar
                except ValueError:
                    print("Please enter valid numbers.")
            
            elif choice == "2":
                if not self.last_bid:
                    print("There is no previous bid to call 'Liar!' on.")
                    continue
                
                # Record liar call in history
                self.move_history.append({
                    "round": len(self.move_history) + 1,
                    "player": player.name,
                    "action": "liar",
                    "target_player": self.players[(self.current_player_idx - 1) % len(self.players)].name
                })
                
                return True  # Calling liar
            
            else:
                print("Invalid choice. Please enter 1 or 2.")
    
    def count_dice(self, value):
        count = 0
        for player in self.players:
            count += player.dice.count(value)
        return count
    
    def display_move_history(self):
        """Display the history of moves in the game"""
        if not self.move_history:
            print("\nNo moves recorded yet.")
            return
            
        print("\n===== MOVE HISTORY =====")
        for i, move in enumerate(self.move_history):
            if move["action"] == "bid":
                print(f"{i+1}. {move['player']} bid {move['quantity']} {move['value']}'s")
            elif move["action"] == "liar":
                print(f"{i+1}. {move['player']} called 'Liar!' on {move['target_player']}")
                if "outcome" in move:
                    if move["outcome"] == "success":
                        print(f"   Result: {move['player']} was right! {move['target_player']} lost a die.")
                    else:
                        print(f"   Result: {move['player']} was wrong! {move['player']} lost a die.")
        print("=======================")
    
    def handle_liar_call(self, auto_continue=False):
        calling_player = self.players[self.current_player_idx]
        previous_player_idx = (self.current_player_idx - 1) % len(self.players)
        previous_player = self.players[previous_player_idx]
        
        quantity, value = self.last_bid
        actual_count = self.count_dice(value)
        
        self.clear_screen()
        print(f"\n{calling_player.name} called 'Liar!' on {previous_player.name}'s bid of {quantity} {value}'s")
        
        # Show all dice
        for player in self.players:
            if player.get_dice_count() > 0:
                print(f"{player.name}'s dice: {sorted(player.dice)}")
        
        print(f"\nActual count of {value}'s: {actual_count}")
        
        if actual_count >= quantity:
            # Bid was valid, calling player loses a die
            print(f"{calling_player.name} was wrong! {previous_player.name}'s bid was valid.")
            print(f"{calling_player.name} loses a die.")
            calling_player.remove_die()
            loser = calling_player
            outcome = "failure"
        else:
            # Bid was a lie, previous player loses a die
            print(f"{calling_player.name} was right! {previous_player.name}'s bid was invalid.")
            print(f"{previous_player.name} loses a die.")
            previous_player.remove_die()
            loser = previous_player
            outcome = "success"
        
        # Update the last liar call with the outcome
        for move in reversed(self.move_history):
            if move["action"] == "liar":
                move["outcome"] = outcome
                move["actual_count"] = actual_count
                break
        
        # Check if loser is out
        if loser.get_dice_count() == 0:
            print(f"{loser.name} is out of the game!")
        
        # Display move history
        self.display_move_history()
        
        # Reset the bid for the next round
        self.last_bid = None
        
        if not auto_continue:
            input("\nPress Enter to continue...")
        else:
            # In auto mode, add a short delay instead of requiring input
            print("\nMoving to next round...")
            time.sleep(2)
    
    def check_game_over(self):
        active_players = [p for p in self.players if p.get_dice_count() > 0]
        
        if len(active_players) == 1:
            self.game_over = True
            self.winner = active_players[0]
            return True
        
        return False
    
    def play_round(self, auto_mode=False, game_num=0):
        # Start by rolling all dice and display round number
        self.roll_all_dice()
        print(f"\n===== GAME {game_num} | ROUND {self.round_number} =====")
        
        while not self.game_over:
            current_player = self.players[self.current_player_idx]
            self.show_dice_to_player(self.current_player_idx)
            
            is_calling_liar = self.get_player_bid(self.current_player_idx)
            
            if is_calling_liar:
                self.handle_liar_call(auto_continue=auto_mode)
                
                if self.check_game_over():
                    break
                
                # Start new round
                self.round_number += 1
                print(f"\n===== GAME {game_num} | ROUND {self.round_number} =====")
                self.roll_all_dice()
                continue
            
            self.next_player()
    
    def display_game_summary(self):
        """Display a summary of the entire game"""
        print("\n========== GAME SUMMARY ==========")
        print(f"Total rounds played: {self.round_number}")
        print(f"Winner: {self.winner.name}")
        
        # Count moves by player
        player_moves = {}
        player_liar_calls = {}
        successful_liar_calls = {}
        
        for player in self.players:
            player_moves[player.name] = 0
            player_liar_calls[player.name] = 0
            successful_liar_calls[player.name] = 0
        
        for move in self.move_history:
            player = move["player"]
            player_moves[player] = player_moves.get(player, 0) + 1
            
            if move["action"] == "liar":
                player_liar_calls[player] = player_liar_calls.get(player, 0) + 1
                if move.get("outcome") == "success":
                    successful_liar_calls[player] = successful_liar_calls.get(player, 0) + 1
        
        # Display player statistics
        print("\nPlayer Statistics:")
        for player in self.players:
            name = player.name
            total_moves = player_moves.get(name, 0)
            liar_calls = player_liar_calls.get(name, 0)
            successful_calls = successful_liar_calls.get(name, 0)
            success_rate = (successful_calls / liar_calls * 100) if liar_calls > 0 else 0
            
            print(f"  {name}:")
            if isinstance(player, AIPlayer):
                print(f"    - Model: {player.model}")
            print(f"    - Total moves: {total_moves}")
            print(f"    - Liar calls: {liar_calls}")
            print(f"    - Successful liar calls: {successful_calls}")
            print(f"    - Success rate: {success_rate:.1f}%")
        
        print("==================================")
    
    def play_game(self):
        self.setup_game()
        
        # Check if this is an all-AI game
        all_ai_game = all(isinstance(player, AIPlayer) for player in self.players)
        
        # For AI-only games, offer auto mode
        auto_mode = False
        if all_ai_game:
            choice = input("\nThis is an AI-only game. Would you like to run in auto mode? (y/n): ").lower().strip()
            auto_mode = choice == 'y'
            
            if auto_mode:
                print("\nRunning in AUTO MODE. Game will play automatically.")
                print("Press Ctrl+C at any time to interrupt.")
                time.sleep(2)
        
        while not self.game_over:
            self.play_round(auto_mode=auto_mode)
        
        self.clear_screen()
        print(f"\nGame over! {self.winner.name} is the winner!")
        
        # Display game summary
        self.display_game_summary()


def print_intro():
    print("=" * 70)
    print("Welcome to Liar's Dice OpenRouter Tournament!")
    print("=" * 70)
    print("\nIn this game, each player has dice that only they can see.")
    print("Players take turns making bids about how many dice of a certain value exist among all players.")
    print("Each bid must be higher than the previous one (either more dice, or same number but higher value).")
    print("When a player thinks the previous bid is a lie, they can call 'Liar'.")
    print("The loser of each round loses one die. The last player with dice is the winner!")
    
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


class GameBatchRunner:
    def __init__(self):
        self.leaderboard = {}
        self.game_results = []
        self.total_games = 0
        self.model_stats = {}
    
    def setup_batch(self):
        """Setup a batch of games to run"""
        if not OPENROUTER_AVAILABLE:
            print("OpenRouter integration is not available.")
            print("Install the required package with: pip install requests")
            return False
        
        print("\n" + "=" * 70)
        print("LIARS DICE MODEL TOURNAMENT")
        print("=" * 70)
        
        # Get OpenRouter API key
        self.api_key = os.environ.get("OPENROUTER_API_KEY", "")
        if not self.api_key:
            self.api_key = input("Enter your OpenRouter API key: ")
            os.environ["OPENROUTER_API_KEY"] = self.api_key
        
        # Get the number of games to run
        while True:
            try:
                self.total_games = int(input("\nHow many games to run in the tournament? "))
                if self.total_games < 1:
                    print("Please enter a positive number.")
                    continue
                break
            except ValueError:
                print("Please enter a valid number.")
        
        # Get the number of models per game
        while True:
            try:
                self.models_per_game = int(input("How many models per game? (2-6 recommended): "))
                if self.models_per_game < 2:
                    print("You need at least 2 models per game.")
                    continue
                break
            except ValueError:
                print("Please enter a valid number.")
        
        # Get available models
        game = LiarsDice()
        self.available_models = game.get_available_models()
        self.selected_models = []
        
        # Show available models
        print("\nAvailable models:")
        for i, model in enumerate(self.available_models):
            print(f"{i+1}. {model}")
        
        # Select models to include
        print("\nSelect models to include (enter model numbers separated by spaces, or 'all'):")
        selection = input("> ").strip()
        
        if selection.lower() == "all":
            self.selected_models = self.available_models.copy()
        else:
            selection = selection.split()
            for idx_str in selection:
                try:
                    idx = int(idx_str) - 1
                    if 0 <= idx < len(self.available_models):
                        self.selected_models.append(self.available_models[idx])
                except ValueError:
                    continue
        
        # Make sure we have at least 2 models
        if len(self.selected_models) < 2:
            print("You need to select at least 2 models. Using the first two available models.")
            self.selected_models = self.available_models[:2]
        
        # Print selected models
        print("\nSelected models for tournament:")
        for model in self.selected_models:
            print(f"- {model}")
        
        # Initialize leaderboard and stats for all models
        for model in self.selected_models:
            self.leaderboard[model] = {
                "wins": 0,
                "games_played": 0,
                "win_rate": 0.0,
                "model": model
            }
            
            self.model_stats[model] = {
                "total_rounds_played": 0,
                "avg_rounds_survived": 0,
                "total_bids": 0,
                "total_liar_calls": 0,
                "successful_liar_calls": 0,
                "unsuccessful_liar_calls": 0,
                "liar_success_rate": 0.0,
                "avg_rounds_per_game": 0.0,
                "early_game_wins": 0,    # Wins in games with fewer than 10 rounds
                "mid_game_wins": 0,      # Wins in games with 10-20 rounds
                "long_game_wins": 0      # Wins in games with more than 20 rounds
            }
        
        # Ask about auto mode
        self.auto_mode = input("\nRun all games in automatic mode? (y/n): ").lower().strip() == 'y'
        
        # Ask about detailed output
        self.verbose_output = input("Show detailed output for each game? (y/n): ").lower().strip() == 'y'
        
        return True
    
    def run_single_game(self, game_num):
        """Run a single game with the selected models"""
        game = LiarsDice()
        
        # Randomly select models for this game
        if len(self.selected_models) <= self.models_per_game:
            game_models = self.selected_models.copy()
        else:
            game_models = random.sample(self.selected_models, self.models_per_game)
        
        # Add AI players with selected models
        for i, model in enumerate(game_models):
            # Get a readable model name for the player name
            model_short_name = model.split('/')[-1] if '/' in model else model
            
            # Generate player name based on model
            player_name = f"{model_short_name.upper()}-{i+1}"
            
            # Add the AI player
            game.add_ai_player(player_name, model, self.api_key)
            
            # Update games played in leaderboard
            self.leaderboard[model]["games_played"] += 1
        
        # Start the game without setup (we already added the players)
        game.current_player_idx = random.randint(0, len(game_models) - 1)
        
        # Hide output if not verbose
        original_stdout = sys.stdout
        if not self.verbose_output:
            sys.stdout = open(os.devnull, 'w')
        
        try:
            # Play the game in auto mode
            while not game.game_over:
                game.play_round(auto_mode=True, game_num=game_num)
            
            # Record the winner and update leaderboard
            winner_model = None
            
            for player in game.players:
                if isinstance(player, AIPlayer) and player.name == game.winner.name:
                    winner_model = player.model
                    
                    # Update wins
                    self.leaderboard[winner_model]["wins"] += 1
                    
                    # Track game length categorized wins
                    if game.round_number < 10:
                        self.model_stats[winner_model]["early_game_wins"] += 1
                    elif game.round_number < 20:
                        self.model_stats[winner_model]["mid_game_wins"] += 1
                    else:
                        self.model_stats[winner_model]["long_game_wins"] += 1
                    
                    break
            
            # Collect detailed statistics for each model
            for player in game.players:
                if isinstance(player, AIPlayer):
                    model = player.model
                    
                    # Count bids and liar calls
                    bids = 0
                    liar_calls = 0
                    successful_liar_calls = 0
                    unsuccessful_liar_calls = 0
                    
                    for move in game.move_history:
                        if move["player"] == player.name:
                            if move["action"] == "bid":
                                bids += 1
                            elif move["action"] == "liar":
                                liar_calls += 1
                                if move.get("outcome") == "success":
                                    successful_liar_calls += 1
                                elif move.get("outcome") == "failure":
                                    unsuccessful_liar_calls += 1
                    
                    # Update model statistics
                    self.model_stats[model]["total_bids"] += bids
                    self.model_stats[model]["total_liar_calls"] += liar_calls
                    self.model_stats[model]["successful_liar_calls"] += successful_liar_calls
                    self.model_stats[model]["unsuccessful_liar_calls"] += unsuccessful_liar_calls
                    self.model_stats[model]["total_rounds_played"] += game.round_number
            
            # Record game result with detailed stats
            result = {
                "game_number": game_num,
                "players": [
                    {
                        "name": p.name, 
                        "model": p.model
                    } for p in game.players if isinstance(p, AIPlayer)
                ],
                "winner": game.winner.name,
                "winner_model": winner_model,
                "rounds": game.round_number,
                "move_history": [move.copy() for move in game.move_history]
            }
            self.game_results.append(result)
            
            # Restore output
            if not self.verbose_output:
                sys.stdout = original_stdout
            
            # Print game result
            print(f"Game {game_num}: Winner is {game.winner.name} ({winner_model}) after {game.round_number} rounds")
            
            return winner_model
        
        except Exception as e:
            # Restore output in case of error
            if not self.verbose_output:
                sys.stdout = original_stdout
            print(f"Error in game {game_num}: {e}")
            return None
    
    def update_leaderboard(self):
        """Update win rates and statistics"""
        for model in self.leaderboard:
            # Update win rate
            wins = self.leaderboard[model]["wins"]
            games = self.leaderboard[model]["games_played"]
            win_rate = (wins / games * 100) if games > 0 else 0
            self.leaderboard[model]["win_rate"] = win_rate
            
            # Update detailed statistics
            stats = self.model_stats[model]
            
            # Calculate liar call success rate
            total_liar_calls = stats["total_liar_calls"]
            successful_calls = stats["successful_liar_calls"]
            stats["liar_success_rate"] = (successful_calls / total_liar_calls * 100) if total_liar_calls > 0 else 0
            
            # Calculate average rounds per game
            total_rounds = stats["total_rounds_played"]
            stats["avg_rounds_per_game"] = total_rounds / games if games > 0 else 0
    
    def display_leaderboard(self):
        """Display the current leaderboard and statistics"""
        print("\n" + "=" * 80)
        print("LIARS DICE MODEL LEADERBOARD")
        print("=" * 80)
        
        # Sort models by win rate
        sorted_models = sorted(
            self.leaderboard.items(),
            key=lambda x: (x[1]["win_rate"], x[1]["wins"]),
            reverse=True
        )
        
        # Basic leaderboard
        print(f"{'Rank':<6}{'Model':<42}{'Win Rate':<15}{'Wins':<10}{'Games':<10}")
        print("-" * 80)
        
        for i, (model, stats) in enumerate(sorted_models):
            # Shorten model name if too long
            model_name = model
            if len(model_name) > 40:
                model_name = model_name[:37] + "..."
                
            print(f"{i+1:<6}{model_name:<42}{stats['win_rate']:.1f}%{' ':<10}{stats['wins']:<10}{stats['games_played']:<10}")
            
        # Detailed statistics
        print("\n" + "=" * 80)
        print("DETAILED MODEL STATISTICS")
        print("=" * 80)
        
        for i, (model, stats) in enumerate(sorted_models):
            model_stats = self.model_stats[model]
            
            # Get provider from model name (usually format is provider/model)
            provider = "unknown"
            if "/" in model:
                provider = model.split("/")[0]
            
            print(f"\n{i+1}. {model}")
            print(f"   Provider: {provider}")
            print(f"   Win Rate: {stats['win_rate']:.1f}%")
            print(f"   Games Won: {stats['wins']} / {stats['games_played']}")
            print(f"   Avg. Rounds per Game: {model_stats['avg_rounds_per_game']:.1f}")
            print(f"   Liar Call Success Rate: {model_stats['liar_success_rate']:.1f}%")
            print(f"   Liar Calls: {model_stats['successful_liar_calls']} successful, {model_stats['unsuccessful_liar_calls']} unsuccessful")
            
            total_moves = model_stats['total_bids'] + model_stats['total_liar_calls']
            liar_call_pct = (model_stats['total_liar_calls'] / total_moves * 100) if total_moves > 0 else 0
            print(f"   Playing Style: {model_stats['total_bids']} bids, {model_stats['total_liar_calls']} liar calls ({liar_call_pct:.1f}% liar calls)")
            
            print(f"   Wins by Game Length: {model_stats['early_game_wins']} early, {model_stats['mid_game_wins']} mid, {model_stats['long_game_wins']} long")
    
    def save_results(self):
        """Save the tournament results to a file"""
        filename = f"liars_dice_tournament_{time.strftime('%Y%m%d_%H%M%S')}.txt"
        
        with open(filename, 'w') as f:
            f.write("LIARS DICE MODEL TOURNAMENT RESULTS\n")
            f.write("=" * 80 + "\n")
            f.write(f"Date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total Games: {self.total_games}\n")
            f.write(f"Models per Game: {self.models_per_game}\n\n")
            
            # Count providers
            providers = set()
            for model in self.leaderboard:
                if "/" in model:
                    provider = model.split("/")[0]
                    providers.add(provider)
            
            f.write(f"Providers Used: {', '.join(providers)}\n\n")
            
            # Basic leaderboard
            f.write("LEADERBOARD\n")
            f.write("=" * 80 + "\n")
            f.write(f"{'Rank':<6}{'Model':<42}{'Win Rate':<15}{'Wins':<10}{'Games':<10}\n")
            f.write("-" * 80 + "\n")
            
            # Sort models by win rate
            sorted_models = sorted(
                self.leaderboard.items(),
                key=lambda x: (x[1]["win_rate"], x[1]["wins"]),
                reverse=True
            )
            
            for i, (model, stats) in enumerate(sorted_models):
                # Shorten model name if too long
                model_name = model
                if len(model_name) > 40:
                    model_name = model_name[:37] + "..."
                    
                f.write(f"{i+1:<6}{model_name:<42}{stats['win_rate']:.1f}%{' ':<10}{stats['wins']:<10}{stats['games_played']:<10}\n")
            
            # Detailed statistics
            f.write("\nDETAILED MODEL STATISTICS\n")
            f.write("=" * 80 + "\n")
            
            for i, (model, stats) in enumerate(sorted_models):
                model_stats = self.model_stats[model]
                
                # Get provider from model name
                provider = "unknown"
                if "/" in model:
                    provider = model.split("/")[0]
                
                f.write(f"\n{i+1}. {model}\n")
                f.write(f"   Provider: {provider}\n")
                f.write(f"   Win Rate: {stats['win_rate']:.1f}%\n")
                f.write(f"   Games Won: {stats['wins']} / {stats['games_played']}\n")
                f.write(f"   Avg. Rounds per Game: {model_stats['avg_rounds_per_game']:.1f}\n")
                f.write(f"   Liar Call Success Rate: {model_stats['liar_success_rate']:.1f}%\n")
                f.write(f"   Liar Calls: {model_stats['successful_liar_calls']} successful, {model_stats['unsuccessful_liar_calls']} unsuccessful\n")
                
                total_moves = model_stats['total_bids'] + model_stats['total_liar_calls']
                liar_call_pct = (model_stats['total_liar_calls'] / total_moves * 100) if total_moves > 0 else 0
                f.write(f"   Playing Style: {model_stats['total_bids']} bids, {model_stats['total_liar_calls']} liar calls ({liar_call_pct:.1f}% liar calls)\n")
                
                f.write(f"   Wins by Game Length: {model_stats['early_game_wins']} early, {model_stats['mid_game_wins']} mid, {model_stats['long_game_wins']} long\n")
            
            # Game-by-game results
            f.write("\nGAME RESULTS\n")
            f.write("=" * 80 + "\n")
            for result in self.game_results:
                f.write(f"Game {result['game_number']}:\n")
                
                # List players with their models
                f.write("  Players:\n")
                for p in result['players']:
                    f.write(f"    - {p['name']} ({p['model']})\n")
                
                winner_model = result.get('winner_model', 'unknown')
                f.write(f"  Winner: {result['winner']} ({winner_model})\n")
                f.write(f"  Rounds: {result['rounds']}\n")
                
                # Don't write the full move history to keep the file manageable
                f.write(f"  Total Moves: {len(result.get('move_history', []))}\n\n")
            
            # Additional analysis
            f.write("\nMODEL ANALYSIS\n")
            f.write("=" * 80 + "\n")
            
            # Most aggressive model (most liar calls)
            most_aggressive = max(self.model_stats.items(), key=lambda x: x[1]['total_liar_calls'] / self.leaderboard[x[0]]['games_played'] if self.leaderboard[x[0]]['games_played'] > 0 else 0)
            # Most cautious model (fewest liar calls)
            most_cautious = min(self.model_stats.items(), key=lambda x: x[1]['total_liar_calls'] / self.leaderboard[x[0]]['games_played'] if self.leaderboard[x[0]]['games_played'] > 0 else float('inf'))
            # Most accurate model (highest liar call success rate)
            most_accurate = max(self.model_stats.items(), key=lambda x: x[1]['liar_success_rate'])
            # Model with shortest games (lowest avg rounds)
            shortest_games = min(self.model_stats.items(), key=lambda x: x[1]['avg_rounds_per_game'] if x[1]['avg_rounds_per_game'] > 0 else float('inf'))
            # Model with longest games (highest avg rounds)
            longest_games = max(self.model_stats.items(), key=lambda x: x[1]['avg_rounds_per_game'])
            
            f.write(f"Most Aggressive Model: {most_aggressive[0]}\n")
            f.write(f"Most Cautious Model: {most_cautious[0]}\n")
            f.write(f"Most Accurate Liar Detector: {most_accurate[0]} ({most_accurate[1]['liar_success_rate']:.1f}% success rate)\n")
            f.write(f"Model with Shortest Games: {shortest_games[0]} ({shortest_games[1]['avg_rounds_per_game']:.1f} rounds avg)\n")
            f.write(f"Model with Longest Games: {longest_games[0]} ({longest_games[1]['avg_rounds_per_game']:.1f} rounds avg)\n")
            
            # Provider comparison if multiple providers are used
            if len(providers) > 1:
                f.write("\nPROVIDER COMPARISON\n")
                f.write("=" * 80 + "\n")
                
                provider_stats = {}
                for provider in providers:
                    provider_stats[provider] = {
                        "wins": 0,
                        "games": 0,
                        "win_rate": 0.0
                    }
                
                # Calculate provider-level statistics
                for model, stats in self.leaderboard.items():
                    if "/" in model:
                        provider = model.split("/")[0]
                        provider_stats[provider]["wins"] += stats["wins"]
                        provider_stats[provider]["games"] += stats["games_played"]
                
                # Calculate win rates
                for provider in provider_stats:
                    if provider_stats[provider]["games"] > 0:
                        provider_stats[provider]["win_rate"] = (
                            provider_stats[provider]["wins"] / provider_stats[provider]["games"] * 100
                        )
                
                # Sort providers by win rate
                sorted_providers = sorted(
                    provider_stats.items(),
                    key=lambda x: (x[1]["win_rate"], x[1]["wins"]),
                    reverse=True
                )
                
                f.write(f"{'Provider':<15}{'Win Rate':<15}{'Wins':<10}{'Games':<10}\n")
                f.write("-" * 50 + "\n")
                
                for provider, stats in sorted_providers:
                    f.write(f"{provider:<15}{stats['win_rate']:.1f}%{' ':<10}{stats['wins']:<10}{stats['games']:<10}\n")
        
        print(f"\nResults saved to {filename}")
    
    def run_tournament(self):
        """Run a tournament of multiple games"""
        if not self.setup_batch():
            return
        
        print("\nStarting tournament...")
        
        for i in range(self.total_games):
            print(f"\nRunning game {i+1} of {self.total_games}...")
            self.run_single_game(i+1)
            self.update_leaderboard()
            
            # Show current standings every 5 games
            if (i+1) % 5 == 0 or i+1 == self.total_games:
                self.display_leaderboard()
        
        print("\nTournament complete!")
        self.display_leaderboard()
        self.save_results()


def print_main_menu():
    print("\n" + "=" * 70)
    print("LIARS DICE MAIN MENU")
    print("=" * 70)
    print("1. Play a single game")
    print("2. Run a model tournament")
    print("3. Exit")
    return input("\nSelect an option (1-3): ").strip()


if __name__ == "__main__":
    import sys
    
    print_intro()
    
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