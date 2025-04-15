import random
import os
import time
import json

# Import requests only if available
try:
    import requests
except ImportError:
    pass

from .player import Player
from .ai_player import AIPlayer, OPENROUTER_AVAILABLE
from .metrics import GameMetrics, BidAnalyzer

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
        self.metrics = GameMetrics()  # Initialize metrics tracking
        self.player_history = {}  # Track player actions across games
        self.bid_analyzer = BidAnalyzer()  # For analyzing bid optimality
    
    def clear_screen(self):
        # os.system('cls' if os.name == 'nt' else 'clear')
        print("clear")
    
    def add_player(self, name):
        player = Player(name)
        self.players.append(player)
        # Initialize player history tracking
        self.player_history[name] = []
    
    def add_ai_player(self, name, model, api_key=None, api_url=None):
        if not OPENROUTER_AVAILABLE:
            print("Requests package is not installed. Install with 'pip install requests'")
            return False
        
        ai_player = AIPlayer(name, model, api_key, api_url)
        self.players.append(ai_player)
        # Initialize AI metrics tracking
        self.metrics.initialize_model(model)
        # Initialize player history tracking
        self.player_history[name] = []
        return True
        
    def get_available_models(self, use_local_endpoint=False):
        """Return a list of available models from OpenRouter and local server"""
        # Hardcoded local endpoint URL
        LOCAL_ENDPOINT_URL = "http://127.0.0.1:1234"
        
        default_models = []
        
        models = []
        
        # Check if we have an OpenRouter API key
        openrouter_api_key = os.environ.get("OPENROUTER_API_KEY", "")
        
        # First try to get OpenRouter models
        if OPENROUTER_AVAILABLE and openrouter_api_key:
            try:
                headers = {
                    "Authorization": f"Bearer {openrouter_api_key}"
                }
                
                response = requests.get("https://openrouter.ai/api/v1/models", headers=headers)
                
                if response.status_code == 200:
                    data = response.json()
                    
                    for model in data["data"]:
                        models.append({
                            "id": model["id"],
                            "provider": "openrouter",
                            "name": model["id"],
                            "api_key": openrouter_api_key,
                            "api_url": "https://openrouter.ai/api/v1/chat/completions"
                        })
                else:
                    print(f"Error fetching OpenRouter models: {response.status_code}")
                    # Fall back to default models
                    for model in default_models:
                        models.append({
                            "id": model,
                            "provider": "openrouter",
                            "name": model,
                            "api_key": openrouter_api_key,
                            "api_url": "https://openrouter.ai/api/v1/chat/completions"
                        })
            except Exception as e:
                print(f"Error fetching OpenRouter models: {e}")
                # Fall back to default models
                for model in default_models:
                    models.append({
                        "id": model,
                        "provider": "openrouter",
                        "name": model,
                        "api_key": openrouter_api_key,
                        "api_url": "https://openrouter.ai/api/v1/chat/completions"
                    })
        else:
            # Fall back to default models
            for model in default_models:
                models.append({
                    "id": model,
                    "provider": "openrouter",
                    "name": model,
                    "api_key": openrouter_api_key,
                    "api_url": "https://openrouter.ai/api/v1/chat/completions"
                })
        
        # Add local models by fetching from the models endpoint
        if use_local_endpoint and OPENROUTER_AVAILABLE:
            try:
                print(f"Fetching models from local server at {LOCAL_ENDPOINT_URL}...")
                
                # Try to get models from the local server
                response = requests.get(f"{LOCAL_ENDPOINT_URL}/v1/models")
                
                if response.status_code == 200:
                    data = response.json()
                    
                    # Process the models data according to API format
                    if "data" in data:
                        # OpenAI-compatible format
                        for model in data["data"]:
                            models.append({
                                "id": f"{model.get('id', 'unknown')}",
                                "provider": "local",
                                "name": model.get('id', 'unknown'),
                                "api_key": None,
                                "api_url": f"{LOCAL_ENDPOINT_URL}/v1/chat/completions"
                            })
                    elif "models" in data:
                        # Alternative format
                        for model in data["models"]:
                            if isinstance(model, str):
                                models.append({
                                    "id": f"local/{model}",
                                    "provider": "local",
                                    "name": model,
                                    "api_key": None,
                                    "api_url": f"{LOCAL_ENDPOINT_URL}/v1/chat/completions"
                                })
                            elif isinstance(model, dict) and "id" in model:
                                models.append({
                                    "id": f"local/{model['id']}",
                                    "provider": "local",
                                    "name": model["id"],
                                    "api_key": None,
                                    "api_url": f"{LOCAL_ENDPOINT_URL}/v1/chat/completions"
                                })
                    else:
                        # If no standard format, try to extract any model identifiers
                        for key, value in data.items():
                            if isinstance(value, list):
                                for item in value:
                                    if isinstance(item, str):
                                        models.append({
                                            "id": f"local/{item}",
                                            "provider": "local",
                                            "name": item,
                                            "api_key": None,
                                            "api_url": f"{LOCAL_ENDPOINT_URL}/v1/chat/completions"
                                        })
                                    elif isinstance(item, dict) and "id" in item:
                                        models.append({
                                            "id": f"local/{item['id']}",
                                            "provider": "local",
                                            "name": item["id"],
                                            "api_key": None,
                                            "api_url": f"{LOCAL_ENDPOINT_URL}/v1/chat/completions"
                                        })
                
                else:
                    print(f"Error fetching local models: {response.status_code} - {response.text}")
                    # Add a default local model
                    models.append({
                        "id": "local/default-model",
                        "provider": "local",
                        "name": "default-model",
                        "api_key": None,
                        "api_url": f"{LOCAL_ENDPOINT_URL}/v1/chat/completions"
                    })
                    
            except Exception as e:
                print(f"Error fetching local models: {e}")
                # Add a default local model
                models.append({
                    "id": "local/default-model",
                    "provider": "local",
                    "name": "default-model",
                    "api_key": None,
                    "api_url": f"{LOCAL_ENDPOINT_URL}/v1/chat/completions"
                })
        
        # Sort models to group them by provider
        models.sort(key=lambda x: x["id"])
        
        return models
        
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
        
        # Ask about local LLM server
        use_local_endpoint = input("\nDo you want to use a local LLM server at http://127.0.0.1:1234? (y/n): ").lower().strip() == 'y'
        
        human_count = num_players - ai_count
        
        # Add human players
        for i in range(human_count):
            name = input(f"Enter name for Human Player {i+1}: ")
            self.add_player(name)
        
        # Add AI players
        if ai_count > 0:
            # Get available models (including local models if enabled)
            available_models = self.get_available_models(use_local_endpoint)
            
            if not available_models:
                print("No models available. Check your API key and connection.")
                return
            
            print("\nAvailable Models:")
            for i, model in enumerate(available_models):
                provider = model["provider"]
                model_name = model["id"]
                print(f"{i+1}. {model_name}{' (Local)' if provider == 'local' else ''}")
            
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
                    
                    model_info = available_models[model_idx]
                    model_id = model_info["id"]
                    
                    # Get a readable model name for the player name
                    model_short_name = model_id.split('/')[-1] if '/' in model_id else model_id
                    
                    # Name the AI based on the model
                    if model_info["provider"] == "local":
                        default_name = f"LOCAL-{model_short_name.upper()}-{i+1}"
                    else:
                        default_name = f"{model_short_name.upper()}-{i+1}"
                        
                    name = input(f"Enter name for this AI (default: {default_name}): ")
                    if not name:
                        name = default_name
                    
                    # Add the AI player with the right config
                    self.add_ai_player(
                        name=name,
                        model=model_id,
                        api_key=model_info["api_key"],
                        api_url=model_info["api_url"]
                    )
                    print(f"Added AI player '{name}' using model: {model_id}")
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
                    
                    model_info = available_models[model_idx]
                    model_id = model_info["id"]
                    
                    # Get a readable model name for the player name
                    model_short_name = model_id.split('/')[-1] if '/' in model_id else model_id
                    
                    # Name the AI
                    if model_info["provider"] == "local":
                        default_name = f"LOCAL-{model_short_name}-{i+1}"
                    else:
                        default_name = f"{model_short_name}-{i+1}"
                        
                    name = input(f"Enter name for this AI (default: {default_name}): ")
                    if not name:
                        name = default_name
                    
                    # Add the AI player with the right config
                    self.add_ai_player(
                        name=name,
                        model=model_id,
                        api_key=model_info["api_key"],
                        api_url=model_info["api_url"]
                    )
                    print(f"Added AI player '{name}' using model: {model_id}")
        
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
                        
                        # Record rule adherence issue - called liar when not allowed
                        if isinstance(player, AIPlayer):
                            self.metrics.record_rule_adherence(player.model, False)
                            
                        return False
                    
                    # Check if calling liar is optimal
                    is_optimal_call = BidAnalyzer.should_call_liar(
                        player.dice, 
                        player.get_dice_count(), 
                        self.total_dice_in_game, 
                        self.last_bid
                    )
                    
                    # Record bid optimality
                    if isinstance(player, AIPlayer):
                        self.metrics.record_bid_optimality(player.model, is_optimal_call)
                    
                    # Record liar call in history
                    move_data = {
                        "round": len(self.move_history) + 1,
                        "player": player.name,
                        "action": "liar",
                        "target_player": self.players[(self.current_player_idx - 1) % len(self.players)].name,
                        "is_optimal": is_optimal_call
                    }
                    self.move_history.append(move_data)
                    
                    # Track in player history
                    self.player_history[player.name].append(move_data)
                    
                    print(f"{player.name} calls 'Liar!' on the previous bid.")
                    
                    # Record rule adherence - valid move
                    if isinstance(player, AIPlayer):
                        self.metrics.record_rule_adherence(player.model, True)
                        
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
                    
                    # Check if this bid is mathematically optimal
                    is_optimal_bid = BidAnalyzer.is_bid_optimal(
                        player.dice, 
                        player.get_dice_count(), 
                        self.total_dice_in_game, 
                        self.last_bid, 
                        (quantity, value)
                    ) if valid_bid else False
                    
                    # Record rule adherence
                    if isinstance(player, AIPlayer):
                        self.metrics.record_rule_adherence(player.model, valid_bid)
                    
                    # Is this likely a bluff?
                    player_dice_count = player.dice.count(value)
                    is_bluff = player_dice_count < quantity / 2
                    
                    if valid_bid:
                        self.last_bid = (quantity, value)
                        
                        # Record bid optimality
                        if isinstance(player, AIPlayer):
                            self.metrics.record_bid_optimality(player.model, is_optimal_bid)
                            
                        # Record move in history with bluff info
                        move_data = {
                            "round": len(self.move_history) + 1,
                            "player": player.name,
                            "action": "bid",
                            "quantity": quantity,
                            "value": value,
                            "bluff": is_bluff,
                            "is_optimal": is_optimal_bid
                        }
                        self.move_history.append(move_data)
                        
                        # Track in player history
                        self.player_history[player.name].append(move_data)
                        
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
                        
                        # Record in metrics that AI made an invalid bid
                        if isinstance(player, AIPlayer):
                            self.metrics.record_rule_adherence(player.model, False)
                        
                        # Record move in history
                        move_data = {
                            "round": len(self.move_history) + 1,
                            "player": player.name,
                            "action": "bid",
                            "quantity": self.last_bid[0],
                            "value": self.last_bid[1],
                            "invalid_bid_corrected": True
                        }
                        self.move_history.append(move_data)
                        
                        # Track in player history
                        self.player_history[player.name].append(move_data)
                        
                        print(f"{player.name} bids {self.last_bid[0]} {self.last_bid[1]}'s")
                        return False
            except Exception as e:
                print(f"Error with AI decision: {e}")
                # Fallback to a simple bid
                if self.last_bid:
                    last_quantity, last_value = self.last_bid
                    if last_quantity > self.total_dice_in_game:
                        # Change decision to call liar due to invalid bid
                        return True
                    if last_value < 6:
                        self.last_bid = (last_quantity, last_value + 1)
                    else:
                        self.last_bid = (last_quantity + 1, 1)
                else:
                    self.last_bid = (1, random.randint(3, 6))
                
                # Record in metrics that there was an error
                if isinstance(player, AIPlayer):
                    self.metrics.record_rule_adherence(player.model, False)
                
                # Record move in history
                move_data = {
                    "round": len(self.move_history) + 1,
                    "player": player.name,
                    "action": "bid",
                    "quantity": self.last_bid[0],
                    "value": self.last_bid[1],
                    "error_fallback": True
                }
                self.move_history.append(move_data)
                
                # Track in player history
                self.player_history[player.name].append(move_data)
                
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

                # Determine the actual previous player who made the last bid
                previous_bidder = None
                for move in reversed(self.move_history):
                    if move["action"] == "bid":
                        previous_bidder = move["player"]
                        break
        
                previous_player = next(p for p in self.players if p.name == previous_bidder)
                
                # Record liar call in history
                self.move_history.append({
                    "round": len(self.move_history) + 1,
                    "player": player.name,
                    "action": "liar",
                    "target_player": previous_player
                })
                
                return True  # Calling liar2
            
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
        
        # Determine the actual previous player who made the last bid
        previous_bidder = None
        for move in reversed(self.move_history):
            if move["action"] == "bid":
                previous_bidder = move["player"]
                break
        
        previous_player = next(p for p in self.players if p.name == previous_bidder)
        
        quantity, value = self.last_bid
        actual_count = self.count_dice(value)
        
        self.clear_screen()
        print(f"\n{calling_player.name} called 'Liar!' on {previous_player.name}'s bid of {quantity} {value}'s")
        
        # Show all dice
        for player in self.players:
            if player.get_dice_count() > 0:
                print(f"{player.name}'s dice: {sorted(player.dice)}")
        
        print(f"\nActual count of {value}'s: {actual_count}")
        
        # Was the previous bid a bluff?
        previous_bid_was_bluff = previous_player.dice.count(value) < quantity
        
        if actual_count >= quantity:
            # Bid was valid, calling player loses a die
            print(f"{calling_player.name} was wrong! {previous_player.name}'s bid was valid.")
            print(f"{calling_player.name} loses a die.")
            calling_player.remove_die()
            loser = calling_player
            outcome = "failure"
            
            # Record lie detection metrics - false positive (called liar when bid was valid)
            if isinstance(calling_player, AIPlayer):
                self.metrics.record_lie_detection(calling_player.model, "false_positive")
            
            # Record bluff success metrics - if the previous player was bluffing
            if previous_bid_was_bluff and isinstance(previous_player, AIPlayer):
                self.metrics.record_bluff(previous_player.model, "bluff", True) 
        else:
            # Bid was a lie, previous player loses a die
            print(f"{calling_player.name} was right! {previous_player.name}'s bid was invalid.")
            print(f"{previous_player.name} loses a die.")
            previous_player.remove_die()
            loser = previous_player
            outcome = "success"
            
            # Record lie detection metrics - true positive (correctly called liar)
            if isinstance(calling_player, AIPlayer):
                self.metrics.record_lie_detection(calling_player.model, "true_positive")
                
            # Record bluff failure - if the previous player was bluffing
            if previous_bid_was_bluff and isinstance(previous_player, AIPlayer):
                self.metrics.record_bluff(previous_player.model, "bluff", False)
        
        # Record the final bid of this round (for calculating average final bid metric)
        if isinstance(previous_player, AIPlayer):
            self.metrics.record_final_bid(previous_player.model, quantity, self.round_number)
        
        # Update the last liar call with the outcome
        for move in reversed(self.move_history):
            if move["action"] == "liar":
                move["outcome"] = outcome
                move["actual_count"] = actual_count
                move["was_bluff"] = previous_bid_was_bluff
                break
                
                # Also update player history
                for name, history in self.player_history.items():
                    for hist_move in reversed(history):
                        if hist_move.get("action") == "liar" and "outcome" not in hist_move:
                            hist_move["outcome"] = outcome
                            hist_move["actual_count"] = actual_count
                            hist_move["was_bluff"] = previous_bid_was_bluff
                            break
        
        # Check if loser is out
        if loser.get_dice_count() == 0:
            print(f"{loser.name} is out of the game!")
        
        # Display move history
        self.display_move_history()
        
        # Check for player adaptation based on history
        if len(self.player_history.get(calling_player.name, [])) > 5:
            for player in self.players:
                if isinstance(player, AIPlayer) and player.name != calling_player.name:
                    # Check if this AI has adapted to the calling player's behavior
                    adapted = BidAnalyzer.check_adaptation(
                        self.player_history.get(player.name, []),
                        self.player_history.get(calling_player.name, [])
                    )
                    if adapted:
                        self.metrics.record_adaptation(player.model, True)
        
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
            
            # Update metrics for the game winner
            if isinstance(self.winner, AIPlayer):
                # Record metrics for the winner
                for player in self.players:
                    if player != self.winner and isinstance(player, AIPlayer):
                        # Update Elo ratings
                        self.metrics.update_elo(self.winner.model, player.model)                      
                        
                        # Record false negatives for lie detection (if applicable)
                        # Find cases where the losing AI didn't call liar when it should have
                        for move in self.move_history:
                            if (move.get("action") == "bid" and 
                                move.get("player") == self.winner.name and 
                                move.get("bluff", False) and
                                isinstance(player, AIPlayer)):
                                    # This was a bluff that wasn't called
                                    self.metrics.record_lie_detection(player.model, "false_negative")
            
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
        player_bluffs = {}
        successful_bluffs = {}
        
        for player in self.players:
            player_moves[player.name] = 0
            player_liar_calls[player.name] = 0
            successful_liar_calls[player.name] = 0
            player_bluffs[player.name] = 0
            successful_bluffs[player.name] = 0
        
        for move in self.move_history:
            player = move["player"]
            player_moves[player] = player_moves.get(player, 0) + 1
            
            if move["action"] == "liar":
                player_liar_calls[player] = player_liar_calls.get(player, 0) + 1
                if move.get("outcome") == "success":
                    successful_liar_calls[player] = successful_liar_calls.get(player, 0) + 1
            elif move["action"] == "bid" and move.get("bluff", False):
                player_bluffs[player] = player_bluffs.get(player, 0) + 1
                # A bluff succeeds if it's not called or if it's called but actually true
                if not any(m.get("action") == "liar" and m.get("target_player") == player for m in self.move_history[self.move_history.index(move):]):
                    successful_bluffs[player] = successful_bluffs.get(player, 0) + 1
        
        # Display player statistics
        print("\nPlayer Statistics:")
        for player in self.players:
            name = player.name
            total_moves = player_moves.get(name, 0)
            liar_calls = player_liar_calls.get(name, 0)
            successful_calls = successful_liar_calls.get(name, 0)
            call_success_rate = (successful_calls / liar_calls * 100) if liar_calls > 0 else 0
            bluffs = player_bluffs.get(name, 0)
            successful_bs = successful_bluffs.get(name, 0)
            bluff_success_rate = (successful_bs / bluffs * 100) if bluffs > 0 else 0
            
            print(f"  {name}:")
            if isinstance(player, AIPlayer):
                print(f"    - Model: {player.model}")
                # Get all detailed metrics if this is an AI player
                if player.model in self.metrics.elo_ratings:
                    all_metrics = self.metrics.get_model_metrics(player.model)
                    print(f"    - Elo rating: {all_metrics['elo_rating']:.1f}")
                    print(f"    - Bluff success rate: {all_metrics['bluff_success_rate']:.1f}%")
                    print(f"    - Lie detection precision: {all_metrics['lie_detection']['precision']:.1f}%")
                    print(f"    - Lie detection recall: {all_metrics['lie_detection']['recall']:.1f}%")
                    print(f"    - Average final bid: {all_metrics['average_final_bid']:.2f}")
                    print(f"    - Bid optimality: {all_metrics['bid_optimality']:.1f}%")
                    print(f"    - Adaptation score: {all_metrics['adaptation_score']:.1f}%")
                    print(f"    - Rule adherence rate: {all_metrics['rule_adherence_rate']:.1f}%")
            print(f"    - Total moves: {total_moves}")
            print(f"    - Liar calls: {liar_calls} (Success rate: {call_success_rate:.1f}%)")
            print(f"    - Bluffs: {bluffs} (Success rate: {bluff_success_rate:.1f}%)")
        
        print("==================================")

    def get_metrics(self):
        """Return all collected metrics"""
        return self.metrics.get_all_metrics()
    
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
            self.play_round(auto_mode=auto_mode, game_num=0)
        
        self.clear_screen()
        print(f"\nGame over! {self.winner.name} is the winner!")
        
        # Display game summary
        self.display_game_summary()
