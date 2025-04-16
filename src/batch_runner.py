import random
import os
import time
import json
import sys
import csv
import signal
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
import concurrent.futures

from .liars_dice import LiarsDice
from .ai_player import (
    AIPlayer, REQUESTS_AVAILABLE,
    PROVIDER_OPENROUTER, PROVIDER_LOCAL
)
from .metrics import GameMetrics

class GameBatchRunner:
    def __init__(self):
        self.leaderboard = {}
        self.game_results = []
        self.total_games = 0
        self.model_stats = {}
        self.use_local_endpoint = False
        self.local_endpoint_url = None
        self.local_models = []
    
    def setup_batch(self):
        """Setup a batch of games to run"""
        if not REQUESTS_AVAILABLE:
            print("API requests are not available.")
            print("Install the required package with: pip install requests")
            return False
        
        print("\n" + "=" * 70)
        print("LIARS DICE MODEL TOURNAMENT")
        print("=" * 70)
        
        # Ask if user wants to use a local LLM server
        self.use_local_endpoint = input("Do you want to use a local LLM server at http://127.0.0.1:1234? (y/n): ").lower().strip() == 'y'
        
        # Get OpenRouter API key (for OpenRouter models)
        self.openrouter_api_key = os.environ.get("OPENROUTER_API_KEY", "")
        if not self.openrouter_api_key and not self.use_local_endpoint:
            self.openrouter_api_key = input("Enter your OpenRouter API key: ")
            os.environ["OPENROUTER_API_KEY"] = self.openrouter_api_key
        elif not self.openrouter_api_key and self.use_local_endpoint:
            print("No OpenRouter API key provided. Will use local models.")
        
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
        
        # Get available models (including local models if enabled)
        game = LiarsDice()
        self.available_models = game.get_available_models(self.use_local_endpoint, False)
        
        if not self.available_models:
            print("No models available. Check your API key and connection.")
            return False
        
        self.selected_models = []
        
        # Show available models
        print("\nAvailable models:")
        for i, model in enumerate(self.available_models):
            provider = model["provider"]
            model_id = model["id"]
            provider_label = ""
            if provider == PROVIDER_LOCAL:
                provider_label = " (Local)"
            print(f"{i+1}. {model_id}{provider_label}")
        
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
            provider = model["provider"]
            model_id = model["id"]
            print(f"- {model_id}{' (Local)' if provider == 'local' else ''}")
        
        # Initialize leaderboard and stats for all models
        for model in self.selected_models:
            model_id = model["id"]
            self.leaderboard[model_id] = {
                "wins": 0,
                "games_played": 0,
                "win_rate": 0.0,
                "model": model_id
            }
            
            self.model_stats[model_id] = {
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
    
    def setup_game(self, game_num):
        """Set up a single game with selected models but don't run it yet"""
        game = LiarsDice()
        
        # Randomly select models for this game
        if len(self.selected_models) <= self.models_per_game:
            game_models = self.selected_models.copy()
        else:
            game_models = random.sample(self.selected_models, self.models_per_game)
        
        # Add AI players with selected models
        for i, model_info in enumerate(game_models):
            model_id = model_info["id"]
            provider = model_info["provider"]
            
            # Get a readable model name for the player name
            model_short_name = model_id.split('/')[-1] if '/' in model_id else model_id
            
            # Generate player name based on model
            if provider == "local":
                player_name = f"LOCAL-{model_short_name.upper()}-{i+1}"
            else:
                player_name = f"{model_short_name.upper()}-{i+1}"
            
            # Add the AI player with the appropriate configuration
            game.add_ai_player(
                name=player_name, 
                model=model_id,
                provider=model_info["provider"],
                api_key=model_info["api_key"],
                api_url=model_info["api_url"],
                project_id=model_info["project_id"],
                region=model_info["region"]
            )
            
            # Update games played in leaderboard
            self.leaderboard[model_id]["games_played"] += 1
        
        # Start the game without setup (we already added the players)
        game.current_player_idx = random.randint(0, len(game_models) - 1)
        
        return game, game_models

    def run_single_game(self, game_num):
        """Run a single game with the selected models"""
        game, game_models = self.setup_game(game_num)
        
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
            
            # Record game result with detailed stats and game object for metrics access
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
                "move_history": [move.copy() for move in game.move_history],
                "game_obj": game  # Store the game object to access metrics
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
            
    def run_multiple_games_batch(self, game_nums, batch_size=10):
        """Run multiple games in parallel using ThreadPoolExecutor"""
        
        # Determine optimal batch size based on CPU count and available memory
        cpu_count = os.cpu_count() or 4
        optimal_workers = min(max(cpu_count * 2, batch_size), 32)  # Scale by CPU count, but set reasonable limits
        
        print(f"Running with {optimal_workers} parallel workers")
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=optimal_workers) as executor:
            # Submit all games in this batch
            futures = {}
            for game_num in game_nums:
                future = executor.submit(self.run_single_game, game_num)
                futures[future] = game_num
            
            # Collect results as they complete
            results = []
            completed = 0
            total = len(game_nums)
            
            for future in concurrent.futures.as_completed(futures):
                game_num = futures[future]
                try:
                    result = future.result()
                    results.append(result)
                    completed += 1
                    
                    # Print progress as a percentage
                    print(f"Completed game {game_num} in batch ({completed}/{total}, {completed/total*100:.1f}%)")
                    
                    # Autosave tournament state periodically
                    if completed % max(1, total // 10) == 0:  # Save at 10% intervals
                        self.save_tournament_state()
                        
                except Exception as e:
                    print(f"Error in game {game_num}: {e}")
                    results.append(None)
            
        return results
        
    def save_tournament_state(self, filepath=None):
        """Save the current tournament state to a file"""
        if filepath is None:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filepath = f"tournament_autosave_{timestamp}.json"
            
        print(f"Saving tournament state to {filepath}...")
        
        # Create serializable tournament state without game objects
        serializable_game_results = []
        for result in self.game_results:
            # Create a copy without the non-serializable game object
            serializable_result = result.copy()
            if 'game_obj' in serializable_result:
                # Remove the game object which isn't JSON serializable
                del serializable_result['game_obj']
            serializable_game_results.append(serializable_result)
        
        tournament_state = {
            "leaderboard": self.leaderboard,
            "game_results": serializable_game_results,
            "total_games": self.total_games,
            "model_stats": self.model_stats,
            "completed_games": len(self.game_results),
            "models_per_game": self.models_per_game,
            "selected_models": self.selected_models,
            "use_local_endpoint": self.use_local_endpoint
        }
        
        with open(filepath, 'w') as f:
            json.dump(tournament_state, f, indent=2)
            
        print(f"Tournament state saved to {filepath}")
        return filepath
        
    def load_tournament_state(self, filepath):
        """Load tournament state from a file"""
        print(f"Loading tournament state from {filepath}...")
        
        try:
            with open(filepath, 'r') as f:
                tournament_state = json.load(f)
                
            self.leaderboard = tournament_state.get("leaderboard", {})
            self.game_results = tournament_state.get("game_results", [])
            self.total_games = tournament_state.get("total_games", 0)
            self.model_stats = tournament_state.get("model_stats", {})
            self.models_per_game = tournament_state.get("models_per_game", 2)
            self.selected_models = tournament_state.get("selected_models", [])
            self.use_local_endpoint = tournament_state.get("use_local_endpoint", False)
            
            completed_games = tournament_state.get("completed_games", 0)
            print(f"Successfully loaded tournament with {completed_games} completed games")
            return completed_games
        except Exception as e:
            print(f"Error loading tournament state: {e}")
            return 0
    
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
            
            api_times = []
            for game in self.game_results:
                game_obj = game.get("game_obj")
                if game_obj:
                    metrics = game_obj.get_metrics()
                    if model in metrics and "avg_api_response_time" in metrics[model]:
                        api_times.append(metrics[model]["avg_api_response_time"])
            if api_times:
                avg_api_time = sum(api_times) / len(api_times)
                print(f"   Avg API Response Time: {avg_api_time:.2f} seconds")
    
    def save_results(self):
        """Save the tournament results to a file"""
        base_filename = f"liars_dice_tournament_{time.strftime('%Y%m%d_%H%M%S')}"
        txt_filename = f"{base_filename}.txt"
        csv_filename = f"{base_filename}.csv"
        
        # Export traditional summary report to text file
        with open(txt_filename, 'w') as f:
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
            f.write(f"{'Rank':<6}{'Model':<30}{'Elo':<10}{'Win Rate':<15}{'Wins':<10}{'Games':<10}\n")
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
                if len(model_name) > 28:
                    model_name = model_name[:25] + "..."
                
                # Get Elo rating
                elo = "N/A"
                for game in self.game_results:
                    for player in game["players"]:
                        if player["model"] == model:
                            game_obj = game.get("game_obj")
                            if game_obj:
                                metrics = game_obj.get_metrics()
                                if model in metrics:
                                    elo = f"{metrics[model]['elo_rating']:.1f}"
                                    break
                
                f.write(f"{i+1:<6}{model_name:<30}{elo:<10}{stats['win_rate']:.1f}%{' ':<10}{stats['wins']:<10}{stats['games_played']:<10}\n")
            
            # Detailed statistics
            f.write("\nDETAILED MODEL STATISTICS\n")
            f.write("=" * 80 + "\n")
            
            # Collect all advanced metrics
            all_advanced_metrics = {}
            for game in self.game_results:
                game_obj = game.get("game_obj")
                if game_obj:
                    metrics = game_obj.get_metrics()
                    for model, model_metrics in metrics.items():
                        if model not in all_advanced_metrics:
                            all_advanced_metrics[model] = model_metrics
                        else:
                            # Average metrics across games
                            for key in model_metrics:
                                if key != "elo_rating":  # Elo is cumulative, not averaged
                                    if isinstance(model_metrics[key], dict):
                                        for subkey in model_metrics[key]:
                                            if subkey not in all_advanced_metrics[model][key]:
                                                all_advanced_metrics[model][key][subkey] = 0
                                            all_advanced_metrics[model][key][subkey] += model_metrics[key][subkey]
                                    else:
                                        all_advanced_metrics[model][key] += model_metrics[key]
            
            # Average the metrics by dividing by number of games
            for model, metrics in all_advanced_metrics.items():
                games_played = self.leaderboard[model]["games_played"]
                if games_played > 0:
                    for key in metrics:
                        if key != "elo_rating":  # Skip Elo
                            if isinstance(metrics[key], dict):
                                for subkey in metrics[key]:
                                    metrics[key][subkey] /= games_played
                            else:
                                metrics[key] /= games_played
            
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
                
                # Add the new metrics if available
                if model in all_advanced_metrics:
                    metrics = all_advanced_metrics[model]
                    f.write(f"   Elo Rating: {metrics['elo_rating']:.1f}\n")
                    f.write(f"   Bluff Success Rate: {metrics['bluff_success_rate']:.1f}%\n")
                    f.write(f"   Lie Detection (Precision): {metrics['lie_detection']['precision']:.1f}%\n")
                    f.write(f"   Lie Detection (Recall): {metrics['lie_detection']['recall']:.1f}%\n")
                    f.write(f"   Lie Detection (F1): {metrics['lie_detection']['f1_score']:.1f}%\n")
                    f.write(f"   Average Final Bid: {metrics['average_final_bid']:.2f}\n")
                    f.write(f"   Bid Optimality: {metrics['bid_optimality']:.1f}%\n")
                    f.write(f"   Adaptation Score: {metrics['adaptation_score']:.1f}%\n")
                    f.write(f"   Rule Adherence Rate: {metrics['rule_adherence_rate']:.1f}%\n")
                    f.write(f"   Avg API Response Time: {metrics['avg_api_response_time']:.2f} seconds\n")
                    
                    # Add token usage metrics if available
                    if 'token_usage' in metrics:
                        token_usage = metrics['token_usage']
                        f.write(f"   Token Usage:\n")
                        f.write(f"     - Total Tokens: {token_usage['total_tokens']}\n")
                        f.write(f"     - Prompt Tokens: {token_usage['total_prompt_tokens']}\n")
                        f.write(f"     - Completion Tokens: {token_usage['total_completion_tokens']}\n")
                        f.write(f"     - Avg Tokens Per Action: {token_usage['avg_tokens_per_action']:.1f}\n")
                
                # Original metrics
                liar_call_success_rate = model_stats['liar_success_rate']
                f.write(f"   Liar Call Success Rate: {liar_call_success_rate:.1f}%\n")
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
            
            # Add the new advanced metrics-based analysis
            if all_advanced_metrics:
                f.write("\nADVANCED METRICS ANALYSIS\n")
                f.write("=" * 80 + "\n")
                
                # Find best models by each advanced metric
                best_bluffer = max(all_advanced_metrics.items(), key=lambda x: x[1]['bluff_success_rate'])
                best_lie_detector = max(all_advanced_metrics.items(), key=lambda x: x[1]['lie_detection']['f1_score'])
                most_optimal_bidder = max(all_advanced_metrics.items(), key=lambda x: x[1]['bid_optimality'])
                most_adaptive = max(all_advanced_metrics.items(), key=lambda x: x[1]['adaptation_score'])
                most_rule_adherent = max(all_advanced_metrics.items(), key=lambda x: x[1]['rule_adherence_rate'])
                
                f.write(f"Best Bluffer: {best_bluffer[0]} ({best_bluffer[1]['bluff_success_rate']:.1f}% success rate)\n")
                f.write(f"Best Lie Detector: {best_lie_detector[0]} (F1 score: {best_lie_detector[1]['lie_detection']['f1_score']:.1f}%)\n")
                f.write(f"Most Optimal Bidder: {most_optimal_bidder[0]} ({most_optimal_bidder[1]['bid_optimality']:.1f}% optimal bids)\n")
                f.write(f"Most Adaptive Player: {most_adaptive[0]} ({most_adaptive[1]['adaptation_score']:.1f}% adaptation score)\n")
                f.write(f"Most Rule Adherent: {most_rule_adherent[0]} ({most_rule_adherent[1]['rule_adherence_rate']:.1f}% adherence rate)\n")
            
            # Provider comparison if multiple providers are used
            if len(providers) > 1:
                f.write("\nPROVIDER COMPARISON\n")
                f.write("=" * 80 + "\n")
                
                provider_stats = {}
                for provider in providers:
                    provider_stats[provider] = {
                        "wins": 0,
                        "games": 0,
                        "win_rate": 0.0,
                        "avg_elo": 0,
                        "models": 0,
                        "metrics": {
                            "bluff_success_rate": 0,
                            "lie_detection_f1": 0,
                            "bid_optimality": 0,
                            "adaptation_score": 0,
                            "rule_adherence_rate": 0
                        }
                    }
                
                # Calculate provider-level statistics
                for model, stats in self.leaderboard.items():
                    if "/" in model:
                        provider = model.split("/")[0]
                        provider_stats[provider]["wins"] += stats["wins"]
                        provider_stats[provider]["games"] += stats["games_played"]
                        
                        # Add advanced metrics if available
                        if model in all_advanced_metrics:
                            provider_stats[provider]["models"] += 1
                            metrics = all_advanced_metrics[model]
                            provider_stats[provider]["avg_elo"] += metrics["elo_rating"]
                            provider_stats[provider]["metrics"]["bluff_success_rate"] += metrics["bluff_success_rate"]
                            provider_stats[provider]["metrics"]["lie_detection_f1"] += metrics["lie_detection"]["f1_score"]
                            provider_stats[provider]["metrics"]["bid_optimality"] += metrics["bid_optimality"]
                            provider_stats[provider]["metrics"]["adaptation_score"] += metrics["adaptation_score"]
                            provider_stats[provider]["metrics"]["rule_adherence_rate"] += metrics["rule_adherence_rate"]
                
                # Calculate win rates and averages
                for provider, stats in provider_stats.items():
                    if stats["games"] > 0:
                        stats["win_rate"] = stats["wins"] / stats["games"] * 100
                    
                    if stats["models"] > 0:
                        stats["avg_elo"] /= stats["models"]
                        for key in stats["metrics"]:
                            stats["metrics"][key] /= stats["models"]
                
                # Sort providers by win rate
                sorted_providers = sorted(
                    provider_stats.items(),
                    key=lambda x: (x[1]["win_rate"], x[1]["wins"]),
                    reverse=True
                )
                
                f.write(f"{'Provider':<15}{'Win Rate':<15}{'Avg Elo':<15}{'Wins':<10}{'Games':<10}\n")
                f.write("-" * 65 + "\n")
                
                for provider, stats in sorted_providers:
                    f.write(f"{provider:<15}{stats['win_rate']:.1f}%{' ':<10}{stats['avg_elo']:.1f}{' ':<5}{stats['wins']:<10}{stats['games']:<10}\n")
                
                # Also write provider-level metrics comparison
                f.write("\nProvider Advanced Metrics Comparison:\n")
                f.write("-" * 100 + "\n")
                f.write(f"{'Provider':<15}{'Bluff Success':<15}{'Lie Detection':<15}{'Bid Optimality':<15}{'Adaptation':<15}{'Rule Adherence':<15}\n")
                f.write("-" * 100 + "\n")
                
                for provider, stats in sorted_providers:
                    metrics = stats["metrics"]
                    f.write(f"{provider:<15}{metrics['bluff_success_rate']:.1f}%{' ':<10}{metrics['lie_detection_f1']:.1f}%{' ':<10}{metrics['bid_optimality']:.1f}%{' ':<10}{metrics['adaptation_score']:.1f}%{' ':<10}{metrics['rule_adherence_rate']:.1f}%\n")
        
        # Also save results in CSV format for easier analysis
        with open(csv_filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            
            # Write header row
            header = [
                "Model", "Provider", "Elo", "Win Rate", "Wins", "Games", 
                "Bluff Success Rate", "Lie Detection Precision", "Lie Detection Recall", "Lie Detection F1",
                "Average Final Bid", "Bid Optimality", "Adaptation Score", "Rule Adherence Rate",
                "Avg API Response Time", "Avg Rounds per Game", "Early Game Wins", "Mid Game Wins", "Long Game Wins"
            ]
            writer.writerow(header)
            
            # Write data rows
            for model, stats in sorted_models:
                provider = model.split('/')[0] if '/' in model else "unknown"
                model_stats = self.model_stats[model]
                
                # Get advanced metrics if available
                elo = 1000
                bluff_success = 0
                lie_precision = 0
                lie_recall = 0
                lie_f1 = 0
                avg_final_bid = 0
                bid_optimality = 0
                adaptation = 0
                rule_adherence = 0
                avg_api_response_time = 0
                
                if model in all_advanced_metrics:
                    metrics = all_advanced_metrics[model]
                    elo = metrics["elo_rating"]
                    bluff_success = metrics["bluff_success_rate"]
                    lie_precision = metrics["lie_detection"]["precision"]
                    lie_recall = metrics["lie_detection"]["recall"]
                    lie_f1 = metrics["lie_detection"]["f1_score"]
                    avg_final_bid = metrics["average_final_bid"]
                    bid_optimality = metrics["bid_optimality"]
                    adaptation = metrics["adaptation_score"]
                    rule_adherence = metrics["rule_adherence_rate"]
                    avg_api_response_time = metrics["avg_api_response_time"]
                
                # Create data row
                row = [
                    model, provider, elo, stats["win_rate"], stats["wins"], stats["games_played"],
                    bluff_success, lie_precision, lie_recall, lie_f1,
                    avg_final_bid, bid_optimality, adaptation, rule_adherence,
                    avg_api_response_time, model_stats["avg_rounds_per_game"], model_stats["early_game_wins"], 
                    model_stats["mid_game_wins"], model_stats["long_game_wins"]
                ]
                writer.writerow(row)
        
        print(f"\nResults saved to {txt_filename} and {csv_filename}")
        
        # Create visualizations of the results
        try:
            self.generate_visualizations(base_filename, all_advanced_metrics)
        except Exception as e:
            print(f"Could not generate visualizations: {e}")

    def generate_visualizations(self, base_filename, metrics_data):
        """Generate visualizations of the tournament results"""
        if not metrics_data:
            return
            
        # Create directory for visualizations
        vis_dir = f"{base_filename}_visualizations"
        os.makedirs(vis_dir, exist_ok=True)
        
        # 1. Plot Elo ratings
        plt.figure(figsize=(12, 6))
        models = []
        elos = []
        
        for model, data in metrics_data.items():
            models.append(model.split('/')[-1] if '/' in model else model)  # Shorter model names
            elos.append(data["elo_rating"])
        
        # Sort by Elo
        sorted_indices = np.argsort(elos)[::-1]  # Descending order
        sorted_models = [models[i] for i in sorted_indices]
        sorted_elos = [elos[i] for i in sorted_indices]
        
        plt.bar(sorted_models, sorted_elos, color='skyblue')
        plt.title('Model Comparison: Elo Ratings')
        plt.xlabel('Model')
        plt.ylabel('Elo Rating')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(f"{vis_dir}/elo_ratings.png")
        plt.close()
        
        # 2. Create a radar chart for top 5 models comparing all metrics
        top_models = sorted_indices[:5]
        top_model_names = [models[i] for i in top_models]
        
        # Get metrics for radar chart
        metrics_to_plot = {
            "Bluff Success": [metrics_data[model]["bluff_success_rate"] / 100 for model in top_model_names],
            "Lie Detection": [metrics_data[model]["lie_detection"]["f1_score"] / 100 for model in top_model_names],
            "Bid Optimality": [metrics_data[model]["bid_optimality"] / 100 for model in top_model_names],
            "Adaptation": [metrics_data[model]["adaptation_score"] / 100 for model in top_model_names],
            "Rule Adherence": [metrics_data[model]["rule_adherence_rate"] / 100 for model in top_model_names]
        }
        
        # Create radar chart
        categories = list(metrics_to_plot.keys())
        N = len(categories)
        
        # Create angles for each metric
        angles = [n / float(N) * 2 * np.pi for n in range(N)]
        angles += angles[:1]  # Close the loop
        
        # Create plot
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
        
        # Add each model's data
        for i, model in enumerate(top_model_names):
            values = [metrics_to_plot[cat][i] for cat in categories]
            values += values[:1]  # Close the loop
            
            # Plot data and fill area
            ax.plot(angles, values, linewidth=1, label=model)
            ax.fill(angles, values, alpha=0.1)
        
        # Set category labels
        plt.xticks(angles[:-1], categories)
        
        # Add legend
        plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
        plt.title('Advanced Metrics Comparison for Top Models')
        plt.tight_layout()
        plt.savefig(f"{vis_dir}/radar_metrics.png")
        plt.close()
        
        # 3. Correlation between metrics and win rate
        win_rates = []
        metric_values = defaultdict(list)
        
        for model, stats in self.leaderboard.items():
            if model in metrics_data:
                win_rates.append(stats["win_rate"])
                
                # Add each metric
                metric_values["Bluff Success"].append(metrics_data[model]["bluff_success_rate"])
                metric_values["Lie Detection"].append(metrics_data[model]["lie_detection"]["f1_score"])
                metric_values["Bid Optimality"].append(metrics_data[model]["bid_optimality"])
                metric_values["Adaptation"].append(metrics_data[model]["adaptation_score"])
                metric_values["Rule Adherence"].append(metrics_data[model]["rule_adherence_rate"])
        
        # Plot correlation for each metric with win rate
        for metric, values in metric_values.items():
            plt.figure(figsize=(8, 6))
            plt.scatter(values, win_rates)
            
            # Add trend line
            z = np.polyfit(values, win_rates, 1)
            p = np.poly1d(z)
            plt.plot(values, p(values), "r--", alpha=0.8)
            
            plt.title(f'Correlation: {metric} vs Win Rate')
            plt.xlabel(f'{metric} Score (%)')
            plt.ylabel('Win Rate (%)')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(f"{vis_dir}/correlation_{metric.lower().replace(' ', '_')}.png")
            plt.close()
        
        print(f"Visualizations saved to {vis_dir}/")
    
    def run_tournament(self, resume_from=None):
        """Run a tournament of multiple games with optimized parallelism"""
        completed_games = 0
        
        if resume_from:
            completed_games = self.load_tournament_state(resume_from)
            print(f"Resuming tournament with {completed_games} completed games")
        else:
            if not self.setup_batch():
                return
        
        print("\nStarting tournament...")

        # Determine optimal parallelism for this system
        cpu_count = os.cpu_count() or 4
        # For network/API-bound workloads, we can go much higher than CPU count
        max_workers = 10000 # min(cpu_count * 8, 64)  # Higher parallelism for API calls
        
        # Each thread manages one game independently
        print(f"Running with {max_workers} concurrent games for maximum throughput")
        
        # Calculate remaining games
        remaining_games = self.total_games - completed_games
        total = self.total_games
        
        # We'll handle keyboard interrupts in the try/except block instead of using
        # signal handlers, which only work in the main thread
        
        try:
            # Create a pool of workers that directly run individual games
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                # Submit all remaining games at once, letting the executor manage concurrency
                future_to_game = {
                    executor.submit(self.run_single_game, game_num): game_num 
                    for game_num in range(completed_games + 1, self.total_games + 1)
                }
                
                # Process results as they complete
                for i, future in enumerate(concurrent.futures.as_completed(future_to_game)):
                    game_num = future_to_game[future]
                    try:
                        winner_model = future.result()
                        completed_games += 1
                        
                        # Show progress
                        progress_pct = (completed_games / total) * 100
                        print(f"Game {game_num} completed. Progress: {completed_games}/{total} games ({progress_pct:.1f}%)")
                        
                        # Autosave progress at regular intervals
                        if completed_games % max(1, min(10, total // 10)) == 0:  # Save every ~10% or every 10 games
                            self.save_tournament_state()
                            self.update_leaderboard()
                            self.display_leaderboard()
                    
                    except Exception as e:
                        print(f"Error in game {game_num}: {e}")
        
        except KeyboardInterrupt:
            # Handle manual interruption 
            print("\n\nTournament interrupted! Saving current state...")
            save_path = self.save_tournament_state()
            print(f"Tournament state saved to {save_path}")
            print("You can resume this tournament later by running:")
            print(f"python main.py --resume-tournament {save_path}")
            return
            
        finally:
            # Final update and save if not interrupted
            print("\nTournament complete!")
            self.update_leaderboard() 
            self.display_leaderboard()
            self.save_results()
            
            # Clean up any autosaves since we have completed successfully
            try:
                for f in os.listdir('.'):
                    if f.startswith('tournament_autosave_') and f.endswith('.json'):
                        os.remove(f)
                        print(f"Cleaned up autosave file: {f}")
            except Exception as e:
                print(f"Error cleaning up autosave files: {e}")
