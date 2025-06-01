import random
import os
import time
import json
import concurrent.futures
import pickle
import signal
import sys
import threading

# Import requests only if available
try:
    import requests
except ImportError:
    pass

from .player import Player
from .ai_player import (
    AIPlayer, REQUESTS_AVAILABLE,
    PROVIDER_OPENROUTER, PROVIDER_LOCAL
)
from .metrics import GameMetrics, BidAnalyzer
from .event_logger import EventLoggerFactory, log_round_start, log_round_end, log_move, log_liar_resolution

class LiarsDice:
    def __init__(self, save_file=None):
        if save_file and os.path.exists(save_file):
            self.load_game_state(save_file)
        else:
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
            self.save_file = save_file
        
        # We'll set the signal handler in the main thread only to avoid 
        # "signal only works in main thread" errors in multi-threaded environments
        if threading.current_thread() is threading.main_thread():
            signal.signal(signal.SIGINT, self._signal_handler)
    
    def clear_screen(self):
        # os.system('cls' if os.name == 'nt' else 'clear')
        print("clear")
    
    def add_player(self, name):
        player = Player(name)
        self.players.append(player)
        # Initialize player history tracking
        self.player_history[name] = []
    
    def add_ai_player(self, name, model, provider=PROVIDER_OPENROUTER, api_key=None, api_url=None, project_id=None, region=None):
        if not REQUESTS_AVAILABLE:
            print("Requests package is not installed. Install with 'pip install requests'")
            return False
        
        ai_player = AIPlayer(
            name=name, 
            model=model, 
            provider=provider,
            api_key=api_key, 
            api_url=api_url,
            project_id=project_id,
            region=region
        )
        self.players.append(ai_player)
        
        # Initialize AI metrics tracking
        self.metrics.initialize_model(model)
        
        # Initialize player history tracking
        self.player_history[name] = []
        return True
        
    def get_available_models(self, use_local_endpoint=False, use_vertex_ai=False):
        """Return a list of available models from OpenRouter, Google Vertex AI, and local server"""
        # Hardcoded local endpoint URL
        LOCAL_ENDPOINT_URL = "http://127.0.0.1:1234"
        
        default_models = []
        models = []
        
        # Check if we have an OpenRouter API key
        openrouter_api_key = os.environ.get("OPENROUTER_API_KEY", "")
        
        # First try to get OpenRouter models
        if REQUESTS_AVAILABLE and openrouter_api_key:
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
                            "provider": PROVIDER_OPENROUTER,
                            "name": model["id"],
                            "api_key": openrouter_api_key,
                            "api_url": "https://openrouter.ai/api/v1/chat/completions",
                            "project_id": None,
                            "region": None
                        })
                else:
                    print(f"Error fetching OpenRouter models: {response.status_code}")
                    # Fall back to default models
                    for model in default_models:
                        models.append({
                            "id": model,
                            "provider": PROVIDER_OPENROUTER,
                            "name": model,
                            "api_key": openrouter_api_key,
                            "api_url": "https://openrouter.ai/api/v1/chat/completions",
                            "project_id": None,
                            "region": None
                        })
            except Exception as e:
                print(f"Error fetching OpenRouter models: {e}")
                # Fall back to default models
                for model in default_models:
                    models.append({
                        "id": model,
                        "provider": PROVIDER_OPENROUTER,
                        "name": model,
                        "api_key": openrouter_api_key,
                        "api_url": "https://openrouter.ai/api/v1/chat/completions",
                        "project_id": None,
                        "region": None
                    })
        else:
            # Fall back to default models
            for model in default_models:
                models.append({
                    "id": model,
                    "provider": PROVIDER_OPENROUTER,
                    "name": model,
                    "api_key": openrouter_api_key,
                    "api_url": "https://openrouter.ai/api/v1/chat/completions",
                    "project_id": None,
                    "region": None
                })
        
        
        # Add local models by fetching from the models endpoint
        if use_local_endpoint and REQUESTS_AVAILABLE:
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
                                "provider": PROVIDER_LOCAL,
                                "name": model.get('id', 'unknown'),
                                "api_key": None,
                                "api_url": f"{LOCAL_ENDPOINT_URL}/v1/chat/completions",
                                "project_id": None,
                                "region": None
                            })
                    elif "models" in data:
                        # Alternative format
                        for model in data["models"]:
                            if isinstance(model, str):
                                models.append({
                                    "id": f"local/{model}",
                                    "provider": PROVIDER_LOCAL,
                                    "name": model,
                                    "api_key": None,
                                    "api_url": f"{LOCAL_ENDPOINT_URL}/v1/chat/completions",
                                    "project_id": None,
                                    "region": None
                                })
                            elif isinstance(model, dict) and "id" in model:
                                models.append({
                                    "id": f"local/{model['id']}",
                                    "provider": PROVIDER_LOCAL,
                                    "name": model["id"],
                                    "api_key": None,
                                    "api_url": f"{LOCAL_ENDPOINT_URL}/v1/chat/completions",
                                    "project_id": None,
                                    "region": None
                                })
                    else:
                        # If no standard format, try to extract any model identifiers
                        for key, value in data.items():
                            if isinstance(value, list):
                                for item in value:
                                    if isinstance(item, str):
                                        models.append({
                                            "id": f"local/{item}",
                                            "provider": PROVIDER_LOCAL,
                                            "name": item,
                                            "api_key": None, 
                                            "api_url": f"{LOCAL_ENDPOINT_URL}/v1/chat/completions",
                                            "project_id": None,
                                            "region": None
                                        })
                                    elif isinstance(item, dict) and "id" in item:
                                        models.append({
                                            "id": f"local/{item['id']}",
                                            "provider": PROVIDER_LOCAL,
                                            "name": item["id"],
                                            "api_key": None,
                                            "api_url": f"{LOCAL_ENDPOINT_URL}/v1/chat/completions",
                                            "project_id": None,
                                            "region": None
                                        })
                
                else:
                    print(f"Error fetching local models: {response.status_code} - {response.text}")
                    # Add a default local model
                    models.append({
                        "id": "local/default-model",
                        "provider": PROVIDER_LOCAL,
                        "name": "default-model",
                        "api_key": None,
                        "api_url": f"{LOCAL_ENDPOINT_URL}/v1/chat/completions",
                        "project_id": None,
                        "region": None
                    })
                    
            except Exception as e:
                print(f"Error fetching local models: {e}")
                # Add a default local model
                models.append({
                    "id": "local/default-model",
                    "provider": PROVIDER_LOCAL,
                    "name": "default-model",
                    "api_key": None,
                    "api_url": f"{LOCAL_ENDPOINT_URL}/v1/chat/completions",
                    "project_id": None,
                    "region": None
                })
        
        # Sort models to group them by provider
        models.sort(key=lambda x: x["provider"] + "/" + x["id"])
        
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
            if REQUESTS_AVAILABLE:
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
            if REQUESTS_AVAILABLE:
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
            available_models = self.get_available_models(use_local_endpoint, False)
            
            if not available_models:
                print("No models available. Check your API key and connection.")
                return
            
            print("\nAvailable Models:")
            for i, model in enumerate(available_models):
                provider = model["provider"]
                model_name = model["id"]
                provider_label = ""
                if provider == PROVIDER_LOCAL:
                    provider_label = " (Local)"
                print(f"{i+1}. {model_name}{provider_label}")
            
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
                    provider = model_info["provider"]
                    
                    # Get a readable model name for the player name
                    model_short_name = model_id.split('/')[-1] if '/' in model_id else model_id
                    
                    # Name the AI based on the model
                    if provider == PROVIDER_LOCAL:
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
                        provider=model_info["provider"],
                        api_key=model_info["api_key"],
                        api_url=model_info["api_url"],
                        project_id=model_info["project_id"],
                        region=model_info["region"]
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
                    provider = model_info["provider"]
                    
                    # Get a readable model name for the player name
                    model_short_name = model_id.split('/')[-1] if '/' in model_id else model_id
                    
                    # Name the AI
                    if provider == PROVIDER_LOCAL:
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
                        provider=model_info["provider"],
                        api_key=model_info["api_key"],
                        api_url=model_info["api_url"],
                        project_id=model_info["project_id"],
                        region=model_info["region"]
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
    
    def _log_move_if_enabled(self, game_num, player, game_state, decision):
        """Log move event if raw logging is enabled"""
        if not EventLoggerFactory.is_enabled():
            return
        
        # Create player snapshot for the move
        player_snapshot = {
            "player_order": [p.name for p in self.players if p.get_dice_count() > 0],
            "dice_by_player": {p.name: p.dice for p in self.players if p.get_dice_count() > 0},
            "counts": {str(face): self.count_dice(face) for face in range(1, 7)},
            "total_dice": self.total_dice_in_game,
            "last_bid": list(self.last_bid) if self.last_bid else None,
            "current_player": player.name,
            "round": self.round_number,
            "game_id": game_num
        }
        
        # Extract data from decision and game_state
        raw_prompt = game_state.get('raw_prompt', '')
        raw_response = game_state.get('raw_response', '')
        response_time = game_state.get('response_time', 0.0)
        token_usage = game_state.get('token_usage', {'prompt_tokens': 0, 'completion_tokens': 0, 'total_tokens': 0})
        
        parsed_action = decision.get('action', 'unknown')
        parsed_quantity = decision.get('quantity', None)
        parsed_face = decision.get('face', None)
        utterance = decision.get('utterance', '')
        
        log_move(
            game_num, self.round_number, player_snapshot,
            player.name, player.model if isinstance(player, AIPlayer) else 'human',
            raw_prompt, raw_response, parsed_action, parsed_quantity, parsed_face,
            utterance, response_time, token_usage
        )
    
    def get_ai_decisions_parallel(self, players, max_workers=None):
        """Get AI decisions for multiple players in parallel"""
        if not players:
            return {}
            
        # Set a reasonable number of workers if not specified
        if max_workers is None:
            cpu_count = os.cpu_count() or 4
            max_workers = min(cpu_count * 2, 16)
            
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Prepare the futures - don't make actual API calls yet
            futures = {}
            for player in players:
                if isinstance(player, AIPlayer):
                    # Create game state for this player
                    game_state = self.create_game_state_for_ai(self.players.index(player))
                    
                    # Get request parameters without making API call
                    request_params = player.get_prompt_and_params(game_state)
                    
                    # Submit API call to the thread pool
                    future = executor.submit(
                        requests.post,
                        request_params["endpoint_url"],
                        headers=request_params["headers"],
                        json=request_params["data"]
                    )
                    futures[player] = {
                        "future": future,
                        "game_state": game_state, 
                        "start_time": time.time()
                    }
                    
            # Collect the results
            results = {}
            for player, future_data in futures.items():
                try:
                    future = future_data["future"]
                    game_state = future_data["game_state"]
                    start_time = future_data["start_time"]
                    
                    # Get the response
                    response = future.result()
                    response_time = time.time() - start_time
                    
                    # Process the response
                    decision = player.process_api_response(response, response_time, game_state)
                    results[player] = decision
                except Exception as e:
                    print(f"Error getting AI decision for {player.name}: {e}")
                    results[player] = None
                    
            return results
    
    def _process_ai_liar_call(self, player, decision):
        """Process an AI player's decision to call liar"""
        # Get additional fields from the new format
        reasoning = decision.get("reasoning", "No reasoning provided")
        utterance = decision.get("utterance", "I call liar!")
        
        if not self.last_bid:
            # AI shouldn't call liar on first turn, make a bid instead
            print(f"{player.name} decides to make a bid instead.")
            print(f"Reasoning: {reasoning}")
            print(f"{player.name} says: \"{utterance}\"")
            
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
            "is_optimal": is_optimal_call,
            "reasoning": reasoning,
            "utterance": utterance
        }
        self.move_history.append(move_data)
        
        # Track in player history
        self.player_history[player.name].append(move_data)
        
        print(f"{player.name} calls 'Liar!' on the previous bid.")
        print(f"Reasoning: {reasoning}")
        print(f"{player.name} says: \"{utterance}\"")
        
        # Record rule adherence - valid move
        if isinstance(player, AIPlayer):
            self.metrics.record_rule_adherence(player.model, True)
            
        return True
        
    def _process_ai_bid(self, player, decision):
        """Process an AI player's decision to make a bid"""
        # Get basic bid data
        quantity = decision["quantity"]
        value = decision["face"]  # Changed from "value" to "face" in new format
        
        # Get additional fields from the new format
        reasoning = decision.get("reasoning", "No reasoning provided")
        utterance = decision.get("utterance", "I make this bid.")
        
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
                
            # Record move in history with bluff info and new fields
            move_data = {
                "round": len(self.move_history) + 1,
                "player": player.name,
                "action": "bid",
                "quantity": quantity,
                "value": value,
                "bluff": is_bluff,
                "is_optimal": is_optimal_bid,
                "reasoning": reasoning,
                "utterance": utterance
            }
            self.move_history.append(move_data)
            
            # Track in player history
            self.player_history[player.name].append(move_data)
            
            print(f"{player.name} bids {quantity} {value}'s")
            print(f"Reasoning: {reasoning}")
            print(f"{player.name} says: \"{utterance}\"")
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
            
            # Record move in history with the invalid bid info
            move_data = {
                "round": len(self.move_history) + 1,
                "player": player.name,
                "action": "bid",
                "quantity": self.last_bid[0],
                "value": self.last_bid[1],
                "invalid_bid_corrected": True,
                "original_quantity": quantity,
                "original_value": value,
                "reasoning": reasoning,
                "utterance": utterance
            }
            self.move_history.append(move_data)
            
            # Track in player history
            self.player_history[player.name].append(move_data)
            
            print(f"{player.name} attempted invalid bid ({quantity} {value}'s), corrected to {self.last_bid[0]} {self.last_bid[1]}'s")
            print(f"Reasoning: {reasoning}")
            print(f"{player.name} says: \"{utterance}\"")
            return False
    
    def get_player_bid(self, player_idx, game_num=0):
        player = self.players[player_idx]
        
        # If the player is an AI
        if isinstance(player, AIPlayer):
            print(f"\n{player.name} (AI) is thinking...")
            
            # Create game state for AI
            game_state = self.create_game_state_for_ai(player_idx)
            
            # Add a short delay to make it feel more natural
            time.sleep(random.uniform(1.5, 3.0))
            
            try:
                # Get AI decision which now includes reasoning and utterance
                decision = player.get_ai_decision(game_state)
                
                # Log the move event
                self._log_move_if_enabled(game_num, player, game_state, decision)
                
                # Record API response time if available
                if isinstance(player, AIPlayer) and 'response_time' in game_state:
                    self.metrics.record_api_response_time(player.model, game_state['response_time'])
                    
                # Record token usage if available
                if isinstance(player, AIPlayer) and 'token_usage' in game_state:
                    usage = game_state['token_usage']
                    self.metrics.record_token_usage(
                        player.model,
                        usage.get('prompt_tokens', 0),
                        usage.get('completion_tokens', 0),
                        usage.get('total_tokens', 0)
                    )
                
                # Process the AI decision based on action type
                if decision["action"] == "liar":
                    return self._process_ai_liar_call(player, decision)
                else:
                    return self._process_ai_bid(player, decision)
                    
            except Exception as e:
                print(f"Error with AI decision: {e}")
                # Fallback to a simple bid
                if self.last_bid:
                    last_quantity, last_value = self.last_bid
                    if last_quantity > self.total_dice_in_game:
                        # Change decision to call liar due to invalid bid
                        print(f"{player.name} calls 'Liar!' (automatic fallback)")
                        move_data = {
                            "round": len(self.move_history) + 1,
                            "player": player.name,
                            "action": "liar",
                            "target_player": self.players[(self.current_player_idx - 1) % len(self.players)].name,
                            "error_fallback": True,
                            "reasoning": "Error processing response, making automatic decision.",
                            "utterance": "I call."
                        }
                        self.move_history.append(move_data)
                        self.player_history[player.name].append(move_data)
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
                
                # Record move in history with fallback reasoning
                move_data = {
                    "round": len(self.move_history) + 1,
                    "player": player.name,
                    "action": "bid",
                    "quantity": self.last_bid[0],
                    "value": self.last_bid[1],
                    "error_fallback": True,
                    "reasoning": "Error processing response, using default bid.",
                    "utterance": "I'll make this bid."
                }
                self.move_history.append(move_data)
                
                # Track in player history
                self.player_history[player.name].append(move_data)
                
                print(f"{player.name} bids {self.last_bid[0]} {self.last_bid[1]}'s (fallback)")
                print("Reasoning: Error processing model response, using default bid.")
                return False
        
        # Human player logic - now handled directly in play_round
        # This is only reached if we're showing command options in play_round
        if not self.last_bid:
            print("You're making the first bid.")
        else:
            last_quantity, last_value = self.last_bid
            print(f"Previous bid: {last_quantity} {last_value}'s")
        
        try:
            # Ask if human player wants to bid or call
            action_choice = input("Do you want to (1) make a bid or (2) call 'liar'? Enter 1 or 2: ")
            
            if action_choice == "2":
                # Player chooses to call liar
                if not self.last_bid:
                    print("There is no previous bid to call 'Liar!' on.")
                    return False
                
                # Get the target player (previous bidder)
                previous_bidder = None
                for move in reversed(self.move_history):
                    if move["action"] == "bid":
                        previous_bidder = move["player"]
                        break
                
                target_player = next(p for p in self.players if p.name == previous_bidder)
                
                # Optional reasoning and utterance for human player
                reasoning = input("Your reasoning (optional): ")
                utterance = input("What would you say (optional): ")
                
                # Record liar call in history with new fields
                move_data = {
                    "round": len(self.move_history) + 1,
                    "player": player.name,
                    "action": "liar",
                    "target_player": target_player.name,
                    "reasoning": reasoning if reasoning else "Human player called liar.",
                    "utterance": utterance if utterance else "I call liar!"
                }
                self.move_history.append(move_data)
                
                # Add to player history
                if player.name not in self.player_history:
                    self.player_history[player.name] = []
                self.player_history[player.name].append(move_data)
                
                return True
                
            else:
                # Player is making a bid
                quantity = int(input("How many dice? "))
                value = int(input("What value (1-6)? "))
                
                # Get optional reasoning and utterance
                reasoning = input("Your reasoning (optional): ")
                utterance = input("What would you say (optional): ")
                
                if quantity < 1 or value < 1 or value > 6:
                    print("Invalid bid. Quantity must be positive and value must be between 1 and 6.")
                    return False
                
                # Check if bid is higher than the last bid
                if self.last_bid:
                    last_quantity, last_value = self.last_bid
                    
                    # Bid must be higher
                    if quantity < last_quantity or (quantity == last_quantity and value <= last_value):
                        print("Your bid must be higher than the previous bid.")
                        return False
                
                self.last_bid = (quantity, value)
                
                # Record move in history with new fields
                move_data = {
                    "round": len(self.move_history) + 1,
                    "player": player.name,
                    "action": "bid",
                    "quantity": quantity,
                    "value": value,
                    "reasoning": reasoning if reasoning else "Human player bid.",
                    "utterance": utterance if utterance else "I make this bid."
                }
                self.move_history.append(move_data)
                
                # Add to player history
                if player.name not in self.player_history:
                    self.player_history[player.name] = []
                self.player_history[player.name].append(move_data)
                
                return False  # Not calling liar
        except ValueError:
            print("Please enter valid numbers.")
            return False
    
    def count_dice(self, value):
        count = 0
        for player in self.players:
            count += player.dice.count(value)
        return count
    
    def display_move_history(self):
        """Display the history of moves in the game with reasoning and utterance"""
        if not self.move_history:
            print("\nNo moves recorded yet.")
            return
            
        print("\n===== MOVE HISTORY =====")
        for i, move in enumerate(self.move_history):
            if move["action"] == "bid":
                print(f"{i+1}. {move['player']} bid {move['quantity']} {move['value']}'s")
                
                # Show utterance if available (shorter display for history)
                if "utterance" in move:
                    print(f"   Said: \"{move['utterance']}\"")
                
            elif move["action"] == "liar":
                print(f"{i+1}. {move['player']} called 'Liar!' on {move['target_player']}")
                
                # Show utterance if available (shorter display for history)
                if "utterance" in move:
                    print(f"   Said: \"{move['utterance']}\"")
                
                # Show outcome if available
                if "outcome" in move:
                    if move["outcome"] == "success":
                        print(f"   Result: {move['player']} was right! {move['target_player']} lost a die.")
                    else:
                        print(f"   Result: {move['player']} was wrong! {move['player']} lost a die.")
        print("=======================")
    
    def handle_liar_call(self, auto_continue=False, game_num=0):
        calling_player = self.players[self.current_player_idx]
        
        # Determine the actual previous player who made the last bid
        previous_bidder = None
        last_bid_move = None
        for move in reversed(self.move_history):
            if move["action"] == "bid":
                previous_bidder = move["player"]
                last_bid_move = move
                break
        
        previous_player = next(p for p in self.players if p.name == previous_bidder)
        
        # Find the liar call move to get reasoning and utterance
        liar_call_move = None
        for move in reversed(self.move_history):
            if move["action"] == "liar":
                liar_call_move = move
                break
        
        quantity, value = self.last_bid
        actual_count = self.count_dice(value)
        
        self.clear_screen()
        print(f"\n{calling_player.name} called 'Liar!' on {previous_player.name}'s bid of {quantity} {value}'s")
        
        # Display reasoning and utterance if available
        if liar_call_move and "reasoning" in liar_call_move:
            print(f"Reasoning: {liar_call_move['reasoning']}")
        
        if liar_call_move and "utterance" in liar_call_move:
            print(f"{calling_player.name} says: \"{liar_call_move['utterance']}\"")
            
        # Show all dice
        for player in self.players:
            if player.get_dice_count() > 0:
                print(f"{player.name}'s dice: {sorted(player.dice)}")
        
        print(f"\nActual count of {value}'s: {actual_count}")
        
        # Was the previous bid a bluff?
        previous_bid_was_bluff = previous_player.dice.count(value) < quantity
        
        # Display previous player's reasoning for their bid if available
        if last_bid_move and "reasoning" in last_bid_move:
            print(f"\n{previous_player.name}'s reasoning for the bid: {last_bid_move['reasoning']}")
        
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
            self.metrics.record_final_bid(previous_player.model, (quantity, value), self.round_number)
        
        # Log liar resolution event
        if EventLoggerFactory.is_enabled():
            all_players_dice = {p.name: p.dice for p in self.players if p.get_dice_count() > 0}
            log_liar_resolution(
                game_num, self.round_number, calling_player.name, previous_player.name,
                self.last_bid, actual_count, outcome, all_players_dice
            )
        
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
        
        # Log round start event
        if EventLoggerFactory.is_enabled():
            log_round_start(game_num, self.round_number, self.total_dice_in_game)
        
        while not self.game_over:
            current_player = self.players[self.current_player_idx]
            self.show_dice_to_player(self.current_player_idx)
            
            is_calling_liar = False
            
            # Offer pause option for human players only (not in auto mode)
            if not auto_mode and not isinstance(current_player, AIPlayer):
                print("\nOptions:")
                print("1. Make a bid")
                print("2. Call 'Liar!' on the previous bid")
                print("3. Pause game")
                
                choice = input("Enter your choice (1, 2, or 3): ").strip()
                
                if choice == "3":
                    save_path = self.save_game_state()
                    print(f"\nGame paused and saved to {save_path}")
                    print("You can resume this game later by running:")
                    print(f"python main.py --load {save_path}")
                    return "paused"
                elif choice == "1":
                    # Player is making a bid, proceed to get_player_bid
                    is_calling_liar = False
                elif choice == "2":
                    # Player is calling liar
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
                    liar_call = {
                        "round": len(self.move_history) + 1,
                        "player": current_player.name,
                        "action": "liar",
                        "target_player": previous_player.name
                    }
                    self.move_history.append(liar_call)
                    
                    # Add to player history
                    if current_player.name not in self.player_history:
                        self.player_history[current_player.name] = []
                    self.player_history[current_player.name].append(liar_call)
                    
                    is_calling_liar = True
                else:
                    print("Invalid choice. Please enter 1, 2, or 3.")
                    continue
            else:
                # For AI players, get decision
                is_calling_liar = self.get_player_bid(self.current_player_idx, game_num)
            
            if is_calling_liar:
                self.handle_liar_call(auto_continue=auto_mode, game_num=game_num)
                
                if self.check_game_over():
                    break
                
                # Log round end before starting new round
                if EventLoggerFactory.is_enabled():
                    surviving_players = [p.name for p in self.players if p.get_dice_count() > 0]
                    log_round_end(game_num, self.round_number, surviving_players)
                
                # Start new round
                self.round_number += 1
                print(f"\n===== GAME {game_num} | ROUND {self.round_number} =====")
                self.roll_all_dice()
                
                # Log new round start
                if EventLoggerFactory.is_enabled():
                    log_round_start(game_num, self.round_number, self.total_dice_in_game)
                
                continue
            
            # Advance to next player
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
        # Calculate token usage summary
        total_prompt_tokens = 0
        total_completion_tokens = 0
        total_tokens = 0
        total_ai_actions = 0
        
        for model_id in self.metrics.token_usage:
            data = self.metrics.token_usage[model_id]
            total_prompt_tokens += data["prompt_tokens"]
            total_completion_tokens += data["completion_tokens"]
            total_tokens += data["total_tokens"]
            total_ai_actions += data["actions"]
            
        # Display token usage summary if any
        if total_tokens > 0:
            print("\nToken Usage Summary:")
            print(f"  Total Tokens: {total_tokens}")
            print(f"  Prompt Tokens: {total_prompt_tokens}")
            print(f"  Completion Tokens: {total_completion_tokens}")
            print(f"  Total AI Actions: {total_ai_actions}")
            print(f"  Avg Tokens per Action: {(total_tokens / total_ai_actions) if total_ai_actions > 0 else 0:.1f}")
            
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
                    print(f"    - Avg API response time: {all_metrics['avg_api_response_time']:.2f} seconds")
                    
                    # Print token usage if available
                    if 'token_usage' in all_metrics:
                        token_usage = all_metrics['token_usage']
                        print(f"    - Token usage:")
                        print(f"      - Total tokens: {token_usage['total_tokens']}")
                        print(f"      - Avg tokens per action: {token_usage['avg_tokens_per_action']:.1f}")
            print(f"    - Total moves: {total_moves}")
            print(f"    - Liar calls: {liar_calls} (Success rate: {call_success_rate:.1f}%)")
            print(f"    - Bluffs: {bluffs} (Success rate: {bluff_success_rate:.1f}%)")
        
        print("==================================")

    def get_metrics(self):
        """Return all collected metrics"""
        return self.metrics.get_all_metrics()
        
    def _serialize_player(self, player):
        """Convert a Player object to a serializable dictionary"""
        if player is None:
            return None
            
        data = {
            "name": player.name,
            "dice": player.dice,
            "num_dice": player.get_dice_count()
        }
        
        # Add AI-specific data if it's an AI player
        if isinstance(player, AIPlayer):
            data.update({
                "is_ai": True,
                "model": player.model,
                "provider": player.provider,
                "api_key": player.api_key,
                "api_url": player.api_url,
                "project_id": player.project_id,
                "region": player.region
            })
        else:
            data["is_ai"] = False
            
        return data
    
    def _deserialize_player(self, data):
        """Create a Player object from serialized data"""
        if data is None:
            return None
            
        if data.get("is_ai", False):
            player = AIPlayer(
                name=data["name"],
                model=data["model"],
                provider=data["provider"],
                api_key=data["api_key"],
                api_url=data["api_url"],
                project_id=data["project_id"],
                region=data["region"]
            )
        else:
            player = Player(data["name"])
            
        player.dice = data["dice"]
        return player
    
    def save_game_state(self, filepath=None):
        """Save the current game state to a file"""
        if filepath is None:
            if hasattr(self, 'save_file') and self.save_file:
                filepath = self.save_file
            else:
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                filepath = f"liars_dice_save_{timestamp}.json"
                self.save_file = filepath
        
        print(f"\nSaving game state to {filepath}...")
        
        # Prepare serializable game state
        game_state = {
            "players": [self._serialize_player(p) for p in self.players],
            "current_player_idx": self.current_player_idx,
            "last_bid": self.last_bid,
            "total_dice_in_game": self.total_dice_in_game,
            "game_over": self.game_over,
            "winner": self._serialize_player(self.winner) if self.winner else None,
            "move_history": self.move_history,
            "round_number": self.round_number,
            "player_history": self.player_history,
            # Save metrics data
            "metrics": {
                "elo_ratings": self.metrics.elo_ratings,
                "bluff_stats": self.metrics.bluff_data,
                "lie_detection_stats": self.metrics.lie_detection_data,
                "final_bids": self.metrics.final_bids,
                "bid_optimality_stats": self.metrics.bid_optimality,
                "rule_adherence_stats": self.metrics.rule_adherence,
                "adaptation_stats": self.metrics.adaptation_scores,
                "api_response_times": self.metrics.api_response_times,
                "token_usage": self.metrics.token_usage
            }
        }
        
        with open(filepath, 'w') as f:
            json.dump(game_state, f, indent=2)
        
        print(f"Game saved successfully to {filepath}")
        return filepath
    
    def load_game_state(self, filepath):
        """Load game state from a file"""
        print(f"\nLoading game from {filepath}...")
        
        try:
            with open(filepath, 'r') as f:
                game_state = json.load(f)
            
            # Reconstruct the players
            self.players = [self._deserialize_player(p) for p in game_state["players"]]
            
            # Restore game state
            self.current_player_idx = game_state["current_player_idx"]
            self.last_bid = game_state["last_bid"]
            self.total_dice_in_game = game_state["total_dice_in_game"]
            self.game_over = game_state["game_over"]
            self.move_history = game_state["move_history"]
            self.round_number = game_state["round_number"]
            self.player_history = game_state["player_history"]
            self.save_file = filepath
            
            # Reconstruct the winner if exists
            if game_state["winner"]:
                winner_name = game_state["winner"]["name"]
                self.winner = next((p for p in self.players if p.name == winner_name), None)
            else:
                self.winner = None
            
            # Reconstruct metrics
            metrics_data = game_state.get("metrics", {})
            self.metrics = GameMetrics()
            self.metrics.elo_ratings = metrics_data.get("elo_ratings", {})
            self.metrics.bluff_data = metrics_data.get("bluff_stats", {})
            self.metrics.lie_detection_data = metrics_data.get("lie_detection_stats", {})
            self.metrics.final_bids = metrics_data.get("final_bids", {})
            self.metrics.bid_optimality = metrics_data.get("bid_optimality_stats", {})
            self.metrics.rule_adherence = metrics_data.get("rule_adherence_stats", {})
            self.metrics.adaptation_scores = metrics_data.get("adaptation_stats", {})
            self.metrics.api_response_times = metrics_data.get("api_response_times", {})
            self.metrics.token_usage = metrics_data.get("token_usage", {})
            
            # Initialize the BidAnalyzer
            self.bid_analyzer = BidAnalyzer()
            
            print("Game loaded successfully!")
            return True
        except Exception as e:
            print(f"Error loading game: {e}")
            return False
    
    def _signal_handler(self, sig, frame):
        """Handle Ctrl+C by saving the game and exiting gracefully"""
        print("\n\nInterrupt received! Saving game before exit...")
        if not self.game_over:
            save_path = self.save_game_state()
            print(f"\nGame was saved to {save_path}")
            print("You can resume this game later by running:")
            print(f"python main.py --load {save_path}")
        print("Exiting...")
        sys.exit(0)
    
    def play_game(self, load_from=None):
        if not load_from:
            self.setup_game()
        else:
            print(f"Resuming saved game from {load_from}...")
        
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
        
        # Setup regular autosave
        should_autosave = True
        autosave_interval = 2  # Autosave every 2 rounds
        
        if not self.game_over:
            print("\nGame controls:")
            print("- At your turn, you can choose option 3 to pause the game")
            print("- Press Ctrl+C at any time to save and exit")
            print(f"- The game will autosave every {autosave_interval} rounds\n")
            time.sleep(1)
        
        while not self.game_over:
            # Autosave at regular intervals
            if should_autosave and self.round_number % autosave_interval == 0:
                self.save_game_state()
                
            result = self.play_round(auto_mode=auto_mode, game_num=0)
            
            # Check if the game was paused
            if result == "paused":
                return "paused"
        
        self.clear_screen()
        print(f"\nGame over! {self.winner.name} is the winner!")
        
        # Display game summary
        self.display_game_summary()
        
        # Delete save file when game completes successfully
        if hasattr(self, 'save_file') and self.save_file and os.path.exists(self.save_file):
            try:
                os.remove(self.save_file)
                print(f"\nSave file {self.save_file} has been deleted as the game is complete.")
            except Exception as e:
                print(f"Could not delete save file: {e}")
                
        return "completed"