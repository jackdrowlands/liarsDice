import random
import os
import threading
import time
import json
import sys
import csv
import signal
import logging
import traceback
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
import concurrent.futures
import asyncio
import httpx

from .liars_dice import LiarsDice
from .ai_player import (
    AIPlayer, REQUESTS_AVAILABLE,
    PROVIDER_OPENROUTER, PROVIDER_LOCAL
)
from .metrics import GameMetrics, MetricsVisualizer

# Set up logging directories
os.makedirs("logs", exist_ok=True)
os.makedirs("diagnostics/timing_reports", exist_ok=True)
os.makedirs("diagnostics/game_stats", exist_ok=True)

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s [%(levelname)s] [%(filename)s:%(lineno)d] - %(message)s',
    handlers=[
        logging.FileHandler("logs/timeout_diagnostics.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("liarsdice")

class AsyncGameRunner:
    """
    Runs multiple games concurrently using asyncio.
    Each game runs as a separate task, with its moves executed sequentially.
    """
    def __init__(self, batch_runner):
        self.batch_runner = batch_runner
        limits = httpx.Limits(max_connections=50, max_keepalive_connections=50)
        self.client = httpx.AsyncClient(timeout=30.0, limits=limits)  # Single client for all API calls
        # Track timing data for diagnostics
        self.timing_data = defaultdict(list)
        self.api_call_counts = defaultdict(int)
        self.game_timing_stats = {}
        # Rate limiting for API calls
        self.last_api_call_time = {}
        self.min_delay_between_calls = 0.5  # 500ms between calls to the same model
        
    async def close(self):
        """Close the HTTP client"""
        await self.client.aclose()
    
    def save_timing_report(self, game_num):
        """Save detailed timing data to a report file"""
        try:
            # Only save timing data if we have collected some
            if not self.timing_data:
                return
                
            report_path = f"diagnostics/timing_reports/game_{game_num}_timing.json"
            
            # Calculate statistics
            timing_stats = {}
            for step, timings in self.timing_data.items():
                if not timings:
                    continue
                if isinstance(timings, list) and all(isinstance(item, (int, float)) for item in timings):
                    timing_stats[step] = {
                        "avg": sum(timings) / len(timings),
                        "max": max(timings),
                        "min": min(timings),
                        "total": sum(timings),
                        "count": len(timings)
                    }
                else:
                    # For non-numeric data, just count it
                    timing_stats[step] = {
                        "count": len(timings) if isinstance(timings, list) else 1
                    }
            
            # Create report data
            report_data = {
                "game_number": game_num,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "timing_stats": timing_stats,
                "api_call_counts": dict(self.api_call_counts),
                "game_timing": self.game_timing_stats.get(game_num, {})
            }
            
            # Save to file
            with open(report_path, 'w') as f:
                json.dump(report_data, f, indent=2)
                
            logger.info(f"Saved timing report to {report_path}")
            
        except Exception as e:
            logger.error(f"Error saving timing report: {e}")
            logger.error(traceback.format_exc())
    
    async def make_api_request(self, request_params, player_name=None, model=None):
        """Make an async API request with rate limiting and backoff"""
        endpoint_url = request_params["endpoint_url"]
        headers = request_params["headers"]
        data = request_params["data"]
        
        # Apply rate limiting for each model
        if model in self.last_api_call_time:
            elapsed = time.time() - self.last_api_call_time[model]
            if elapsed < self.min_delay_between_calls:
                delay = self.min_delay_between_calls - elapsed
                logger.debug(f"Rate limiting: Waiting {delay:.2f}s before API call for {model}")
                await asyncio.sleep(delay)
        
        api_start_time = time.time()
        logger.debug(f"API request starting for {model} (Player: {player_name})")
        
        # Track API call for this model
        if model:
            self.api_call_counts[model] += 1
            self.last_api_call_time[model] = time.time()
        
        # Exponential backoff parameters
        max_retries = 3
        retry_delay = 1.0
        
        for retry in range(max_retries + 1):
            try:
                response = await self.client.post(
                    endpoint_url,
                    headers=headers,
                    json=data
                )
                response_time = time.time() - api_start_time
                
                # Record timing data
                if model:
                    self.timing_data[f"api_call_{model}"].append(response_time)
                    
                logger.debug(f"API response received from {model} - time: {response_time:.2f}s, status: {response.status_code}")
                
                if response.status_code != 200:
                    error_msg = f"API Error for {model}: {response.status_code} - {response.text[:200]}"
                    logger.error(error_msg)
                    
                    # If we have retries left and the error is potentially retryable
                    if retry < max_retries and response.status_code in [429, 500, 502, 503, 504]:
                        logger.info(f"Retrying API request for {model} after error {response.status_code} (Retry {retry+1}/{max_retries})")
                        await asyncio.sleep(retry_delay)
                        retry_delay *= 2  # Exponential backoff
                        continue
                    
                    raise RuntimeError(error_msg)
                
                return response, response_time
                
            except httpx.ReadTimeout:
                elapsed = time.time() - api_start_time
                logger.error(f"API Timeout for {model} after {elapsed:.2f}s")
                
                # If we have retries left
                if retry < max_retries:
                    logger.info(f"Retrying API request for {model} after timeout (Retry {retry+1}/{max_retries})")
                    await asyncio.sleep(retry_delay)
                    retry_delay *= 2  # Exponential backoff
                    continue
                
                raise
                
            except Exception as e:
                elapsed = time.time() - api_start_time
                logger.error(f"API Exception for {model} after {elapsed:.2f}s: {str(e)}")
                
                # If we have retries left and the error might be temporary
                if retry < max_retries:
                    logger.info(f"Retrying API request for {model} after error: {str(e)} (Retry {retry+1}/{max_retries})")
                    await asyncio.sleep(retry_delay)
                    retry_delay *= 2  # Exponential backoff
                    continue
                
                raise
    
    async def get_ai_decision_async(self, player, game_state):
        """Async version of get_ai_decision with error handling and timeouts"""
        model_id = player.model
        player_name = player.name
        
        # Record start time for this operation
        decision_start_time = time.time()
        logger.debug(f"Starting AI decision for {player_name} (Model: {model_id})")
        
        # Prepare request
        prep_start_time = time.time()
        request_params = player.get_prompt_and_params(game_state)
        game_state['provider'] = request_params["provider"]
        prep_time = time.time() - prep_start_time
        self.timing_data["prepare_request"].append(prep_time)
        
        try:
            # Call API
            logger.debug(f"Making API call for player {player_name} (Model: {model_id})")
            response, response_time = await self.make_api_request(
                request_params, 
                player_name=player_name,
                model=model_id
            )
            
            # Record that a response was received
            logger.debug(f"Processing API response for {player_name} (Model: {model_id})")
            
            # Process response timing
            proc_start_time = time.time()
            
            # Process response just like the original method
            result = response.json()
            content = result["choices"][0]["message"]["content"].strip()
            
            # Extract token usage if available
            prompt_tokens = completion_tokens = total_tokens = 0
            
            if "usage" in result:
                usage = result["usage"]
                prompt_tokens = usage.get("prompt_tokens", 0)
                completion_tokens = usage.get("completion_tokens", 0)
                total_tokens = usage.get("total_tokens", 0)
                
                game_state['token_usage'] = {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": total_tokens
                }
                
                # Log token usage
                logger.debug(f"Token usage for {model_id}: {prompt_tokens} prompt + {completion_tokens} completion = {total_tokens} total")
            
            # Create logs directory if it doesn't exist
            os.makedirs("logs", exist_ok=True)
            
            # Log the response with prompt information
            with open("logs/llm_responses.json", "a") as f:
                # Create response log object with truncated content
                response_content = content
                if len(response_content) > 1000:
                    response_content = response_content[:1000] + "... [truncated]"
                    
                response_log = {
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "response_text": response_content,
                    "model": player.model,
                    "provider": game_state['provider'],
                    "response_time": response_time,
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": total_tokens,
                    "system_prompt": result["choices"][0]["message"].get("system_fingerprint", "")
                }
                
                # Add truncated prompt information to save space
                if "data" in request_params and "messages" in request_params["data"] and len(request_params["data"]["messages"]) >= 2:
                    # Include system prompt (usually smaller)
                    response_log["prompt_system"] = request_params["data"]["messages"][0]["content"]
                    
                    # Truncate user prompt if large
                    user_prompt = request_params["data"]["messages"][1]["content"]
                    if len(user_prompt) > 500:
                        user_prompt = user_prompt[:500] + "... [truncated]"
                    response_log["prompt_user"] = user_prompt 
                # Alternatively, try to get it from the player
                elif hasattr(player, 'get_prompt_for_game'):
                    try:
                        prompt = player.get_prompt_for_game(game_state)
                        if isinstance(prompt, dict) and "system" in prompt:
                            response_log["prompt_system"] = prompt["system"]
                    except Exception:
                        pass
                        
                json.dump(response_log, f)
                f.write("\n")
            
            # Process AI response
            game_state['response_time'] = response_time
            ai_decision = player.process_api_response(response, response_time, game_state)
            
            # Record processing time
            proc_time = time.time() - proc_start_time
            self.timing_data["process_response"].append(proc_time)
            
            # Record overall decision time
            total_decision_time = time.time() - decision_start_time
            self.timing_data[f"total_decision_{model_id}"].append(total_decision_time)
            logger.debug(f"Completed AI decision for {player_name} in {total_decision_time:.2f}s")
            
            # Log type of decision (bid or liar call)
            if ai_decision and "action" in ai_decision:
                logger.debug(f"Decision type: {ai_decision['action']} from {player_name}")
            
            return ai_decision
            
        except asyncio.TimeoutError as e:
            elapsed = time.time() - decision_start_time
            logger.error(f"TIMEOUT for {player_name} (Model: {model_id}) after {elapsed:.2f}s")
            self.timing_data["timeout_incidents"].append({
                "player": player_name,
                "model": model_id,
                "elapsed_time": elapsed
            })
            raise
            
        except Exception as e:
            elapsed = time.time() - decision_start_time
            logger.error(f"ERROR for {player_name} (Model: {model_id}) after {elapsed:.2f}s: {str(e)}")
            logger.error(traceback.format_exc())
            
            # Record error information
            error_info = {
                "model": player.model,
                "provider": game_state.get('provider', player.provider),
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "error": str(e),
                "elapsed_time": elapsed
            }
            
            # Record error in timing data
            self.timing_data["error_incidents"].append({
                "player": player_name,
                "model": model_id,
                "elapsed_time": elapsed,
                "error": str(e)
            })
            
            # Add response if available
            if 'response' in locals():
                try:
                    error_info["response"] = response.json()
                except:
                    error_info["response"] = "Could not parse response as JSON"
            
            # Add prompt information
            if 'request_params' in locals() and "data" in request_params and "messages" in request_params["data"] and len(request_params["data"]["messages"]) >= 2:
                error_info["prompt"] = {
                    "system": request_params["data"]["messages"][0]["content"],
                    "user": request_params["data"]["messages"][1]["content"]
                }
            # Alternatively, try to get it from the player
            elif hasattr(player, 'get_prompt_for_game'):
                try:
                    error_info["prompt"] = player.get_prompt_for_game(game_state)
                except Exception:
                    pass
            
            # Create logs directory if it doesn't exist
            os.makedirs("logs", exist_ok=True)
            
            # Log the error
            with open("logs/invalid_responses.json", "a") as f:
                json.dump(error_info, f)
                f.write("\n")
            
            raise RuntimeError(f"Error getting AI decision: {e}")
    
    async def get_player_bid_async(self, game, player_idx):
        """Asynchronous version of get_player_bid"""
        player = game.players[player_idx]
        
        # If the player is an AI
        if isinstance(player, AIPlayer):
            print(f"\n{player.name} (AI) is thinking...")
            
            # Record timing for this specific player's turn
            turn_start_time = time.time()
            logger.debug(f"Starting turn for player {player.name} (Model: {player.model})")
            
            # Create game state for AI
            game_state = game.create_game_state_for_ai(player_idx)
            
            # Set a per-turn timeout (independent of the game timeout)
            # This will prevent a single turn from hanging the entire game
            turn_timeout = 300  # 5 minutes per turn is still generous
            
            try:
                # Get AI decision asynchronously with turn timeout
                decision_task = asyncio.create_task(self.get_ai_decision_async(player, game_state))
                try:
                    # Apply per-turn timeout
                    decision = await asyncio.wait_for(decision_task, timeout=turn_timeout)
                    
                    # Record API response time if available
                    if 'response_time' in game_state:
                        game.metrics.record_api_response_time(player.model, game_state['response_time'])
                        # Also record in our timing data
                        self.timing_data[f"api_time_{player.model}"].append(game_state['response_time'])
                        
                    # Record token usage if available
                    if 'token_usage' in game_state:
                        usage = game_state['token_usage']
                        game.metrics.record_token_usage(
                            player.model,
                            usage.get('prompt_tokens', 0),
                            usage.get('completion_tokens', 0),
                            usage.get('total_tokens', 0)
                        )
                    
                    # Process decision
                    if decision["action"] == "liar":
                        is_liar_call = game._process_ai_liar_call(player, decision)
                        action_type = "liar call" if is_liar_call else "bid (from liar)"
                    else:
                        is_liar_call = game._process_ai_bid(player, decision)
                        action_type = "bid"
                    
                    # Record total turn time
                    turn_time = time.time() - turn_start_time
                    self.timing_data[f"player_{player.name}_times"].append(turn_time)
                    self.timing_data[f"model_{player.model}_times"].append(turn_time)
                    
                    logger.debug(f"Player {player.name} completed turn in {turn_time:.2f}s with action: {action_type}")
                    
                    return is_liar_call
                    
                except asyncio.TimeoutError:
                    # Per-turn timeout occurred
                    elapsed = time.time() - turn_start_time
                    logger.error(f"TURN TIMEOUT: Player {player.name} (Model: {player.model}) exceeded {turn_timeout}s turn limit")
                    
                    # Record timeout in statistics
                    self.timing_data["turn_timeouts"].append({
                        "player": player.name,
                        "model": player.model,
                        "elapsed_time": elapsed
                    })
                    
                    # Cancel the task if it's still running
                    if not decision_task.done():
                        decision_task.cancel()
                        
                    # Let this fall through to the fallback logic
                    raise asyncio.TimeoutError(f"Player turn timeout after {elapsed:.2f}s")
                    
            except Exception as e:
                # Record the error timing information
                error_time = time.time() - turn_start_time
                logger.error(f"Error during {player.name}'s turn after {error_time:.2f}s: {str(e)}")
                
                # Fallback to a simple bid with detailed logging
                logger.warning(f"Using fallback logic for {player.name} due to error: {str(e)}")
                
                if game.last_bid:
                    last_quantity, last_value = game.last_bid
                    if last_quantity > game.total_dice_in_game:
                        # Change decision to call liar due to invalid bid
                        logger.info(f"Fallback: {player.name} calls liar (automatic) due to impossible bid")
                        return True
                    if last_value < 6:
                        game.last_bid = (last_quantity, last_value + 1)
                    else:
                        game.last_bid = (last_quantity + 1, 1)
                else:
                    game.last_bid = (1, random.randint(3, 6))
                
                # Record in metrics that there was an error
                if isinstance(player, AIPlayer):
                    game.metrics.record_rule_adherence(player.model, False)
                
                # Record move in history
                move_data = {
                    "round": len(game.move_history) + 1,
                    "player": player.name,
                    "action": "bid",
                    "quantity": game.last_bid[0],
                    "value": game.last_bid[1],
                    "error_fallback": True,
                    "error_message": str(e)
                }
                game.move_history.append(move_data)
                
                # Track in player history
                game.player_history[player.name].append(move_data)
                
                # Log the fallback bid
                logger.info(f"Fallback bid for {player.name}: {game.last_bid[0]} {game.last_bid[1]}'s")
                
                print(f"{player.name} bids {game.last_bid[0]} {game.last_bid[1]}'s")
                
                # Record total turn time including error handling
                turn_time = time.time() - turn_start_time
                self.timing_data[f"player_{player.name}_error_times"].append(turn_time)
                
                return False
        
        # Human player logic (not used in AsyncGameRunner)
        raise ValueError("AsyncGameRunner only supports AI players")
    
    async def play_round_async(self, game, auto_mode=True, game_num=0):
        """Asynchronous version of play_round"""
        # Record time for round start
        round_start = time.time()
        
        # Start by rolling all dice and display round number
        game.roll_all_dice()
        print(f"\n===== GAME {game_num} | ROUND {game.round_number} =====")
        logger.debug(f"Game {game_num} Round {game.round_number} starting")
        
        # Track turns in this round
        turn_count = 0
        round_turns = []
        
        while not game.game_over:
            # Track turn start time
            turn_start = time.time()
            
            # Increment turn counter
            turn_count += 1
            current_player = game.players[game.current_player_idx]
            player_model = current_player.model if isinstance(current_player, AIPlayer) else "human"
            
            logger.debug(f"Game {game_num} Round {game.round_number} Turn {turn_count}: {current_player.name} ({player_model})")
            
            # Show dice to the player
            game.show_dice_to_player(game.current_player_idx)
            
            # For AI players, get decision asynchronously
            turn_timer_start = time.time()
            is_calling_liar = await self.get_player_bid_async(game, game.current_player_idx)
            turn_duration = time.time() - turn_timer_start
            
            # Record turn information
            turn_info = {
                "player": current_player.name,
                "model": player_model,
                "turn_number": turn_count,
                "action": "liar" if is_calling_liar else "bid",
                "duration": turn_duration
            }
            round_turns.append(turn_info)
            
            # Log turn duration
            if turn_duration > 60:  # Log turns taking more than 1 minute
                logger.warning(f"LONG TURN: Game {game_num} Round {game.round_number} - {current_player.name} took {turn_duration:.2f}s")
            
            # Handle liar call
            if is_calling_liar:
                logger.debug(f"Game {game_num} Round {game.round_number}: {current_player.name} called liar")
                
                # Handle the liar call
                game.handle_liar_call(auto_continue=auto_mode)
                
                # Check if game is over
                if game.check_game_over():
                    logger.info(f"Game {game_num} ended on round {game.round_number} after liar call")
                    break
                
                # Start new round
                game.round_number += 1
                
                # Reset for next round
                print(f"\n===== GAME {game_num} | ROUND {game.round_number} =====")
                round_start = time.time()  # Reset for next round
                turn_count = 0
                round_turns = []
                game.roll_all_dice()
                logger.debug(f"Game {game_num} Round {game.round_number} starting")
                continue
            
            # Advance to next player
            logger.debug(f"Game {game_num} Round {game.round_number}: Advancing to next player after {current_player.name}")
            game.next_player()
    
    async def play_single_game_async(self, game_num):
        """Run a single game with asynchronous API calls"""
        game, game_models = self.batch_runner.setup_game(game_num)
        
        # Reset timing data for this game
        self.timing_data = defaultdict(list)
        self.api_call_counts = defaultdict(int)
        self.timing_data["round_times"] = []
        
        # Initialize model-specific timing data
        for player in game.players:
            if isinstance(player, AIPlayer):
                self.timing_data[f"model_{player.model}_times"] = []
                self.timing_data[f"player_{player.name}_times"] = []
                
        # Create a timeout report entry
        self.game_timing_stats[game_num] = {
            "start_time": time.time(),
            "game_models": [model.get("id") for model in game_models],
            "player_models": {p.name: p.model for p in game.players if isinstance(p, AIPlayer)},
            "round_completion": {}
        }
        
        # Hide output if not verbose
        original_stdout = sys.stdout
        if not self.batch_runner.verbose_output:
            sys.stdout = open(os.devnull, 'w')
        
        # Set start time for timeout tracking
        start_time = time.time()
        max_duration = 3600  # 60 minutes in seconds
        
        logger.info(f"Starting game {game_num} with models: " + 
                   ", ".join([f"{p.name}: {p.model}" for p in game.players if isinstance(p, AIPlayer)]))
        
        try:
            # Play the game in auto mode
            round_number = 0
            while not game.game_over:
                round_number += 1
                round_start = time.time()
                
                # Check if we've exceeded the time limit
                elapsed = time.time() - start_time
                if elapsed > max_duration:
                    logger.error(f"Game {game_num} timed out after {elapsed:.2f}s ({round_number-1} rounds completed)")
                    self.game_timing_stats[game_num]["timeout"] = {
                        "elapsed_time": elapsed,
                        "rounds_completed": round_number - 1
                    }
                    raise asyncio.TimeoutError(f"Game {game_num} exceeded the time limit")
                
                # Play a round and track time
                logger.debug(f"Game {game_num}: Starting round {round_number}")
                await self.play_round_async(game, auto_mode=True, game_num=game_num)
                
                # Record round completion
                round_time = time.time() - round_start
                self.timing_data["round_times"].append(round_time)
                self.game_timing_stats[game_num]["round_completion"][round_number] = {
                    "time": round_time,
                    "elapsed": time.time() - start_time,
                    "total_moves": len(game.move_history)
                }
                
                # Log round metrics
                logger.debug(f"Game {game_num}: Completed round {round_number} in {round_time:.2f}s")
                
                # Check for extremely long rounds
                if round_time > 300:  # 5 minutes per round
                    logger.warning(f"Game {game_num}: LONG ROUND DETECTED - Round {round_number} took {round_time:.2f}s")
            
            # Record total game time
            game_time = time.time() - start_time
            self.timing_data["total_game_time"] = game_time
            self.game_timing_stats[game_num]["completion_time"] = game_time
            self.game_timing_stats[game_num]["rounds_completed"] = round_number
            
            # Log game completion
            logger.info(f"Game {game_num} completed in {game_time:.2f}s with {round_number} rounds")
            
            # Process results
            winner_model = self.batch_runner._process_game_results(game, game_num)
            
            # Restore output
            if not self.batch_runner.verbose_output:
                sys.stdout = original_stdout
            
            # Print game result
            print(f"Game {game_num}: Winner is {game.winner.name} ({winner_model}) after {game.round_number} rounds")
            
            # Return both winner model and game object to ensure metrics are preserved
            return {
                "winner": winner_model,
                "game": game,
                "game_num": game_num
            }
        
        except asyncio.TimeoutError:
            # Restore output in case of timeout
            if not self.batch_runner.verbose_output:
                sys.stdout = original_stdout
            print(f"Game {game_num} timed out and was terminated")
            return {
                "winner": None,
                "game": None,
                "game_num": game_num,
                "error": "timeout"
            }
        except Exception as e:
            # Restore output in case of error
            if not self.batch_runner.verbose_output:
                sys.stdout = original_stdout
            print(f"Error in game {game_num}: {e}")
            return {
                "winner": None,
                "game": None,
                "game_num": game_num,
                "error": str(e)
            }
    
    async def run_game_with_semaphore(self, game_num, semaphore):
        """Run a game with a semaphore to limit concurrent executions"""
        async with semaphore:
            return await self.play_single_game_async(game_num)
    
    async def run_games_async(self, game_nums):
        """Run multiple games concurrently using asyncio with rate limiting"""
        # Use a semaphore to limit concurrent API-heavy operations
        concurrency_limit = 32  # Match the ThreadPoolExecutor limit
        semaphore = asyncio.Semaphore(concurrency_limit)
        
        logger.info(f"Starting async game batch with {len(game_nums)} games (max concurrent: {concurrency_limit})")
        
        # Create tasks for all games
        tasks = [self.run_game_with_semaphore(game_num, semaphore) for game_num in game_nums]
        
        # Process games as they complete (gather doesn't preserve order)
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle any exceptions
        processed_results = []
        for i, result in enumerate(results):
            game_num = game_nums[i]
            if isinstance(result, Exception):
                logger.error(f"Game {game_num} error: {str(result)}")
                # Create the error result in the same format we return elsewhere
                processed_results.append({
                    "winner": None,
                    "game": None,
                    "game_num": game_num,
                    "error": str(result)
                })
                self.save_timing_report(game_num)
            else:
                # Process the expanded result that now includes both winner and game object
                processed_results.append(result)
                self.save_timing_report(game_num)
        
        logger.info(f"Completed {len(processed_results)} games in async mode")
        return processed_results

class GameBatchRunner:
    def __init__(self):
        self.leaderboard = {}
        self.game_results = []
        self.total_games = 0
        self.model_stats = {}
        self.use_local_endpoint = False
        self.local_endpoint_url = None
        self.local_models = []
        self.enable_autosaves = True  # Default to enabled
        self.save_individual_games = True  # Save individual games by default
        self.create_visualizations = True  # Create enhanced visualizations by default
    
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
        
        # Ask about execution mode
        self.use_async = input("\nUse asynchronous game execution? (y/n): ").lower().strip() == 'y'
        if self.use_async:
            print("Using asynchronous execution for improved performance.")
            try:
                import httpx
            except ImportError:
                print("httpx package not installed. Installing it is required for async execution.")
                print("Install with: pip install httpx")
                self.use_async = False
                print("Falling back to threaded execution.")
        
        # Ask about enabling autosaves
        self.enable_autosaves = input("Enable periodic autosaves? (y/n): ").lower().strip() == 'y'
        if self.enable_autosaves:
            print("Autosaves enabled - tournament state will be saved periodically.")
        else:
            print("Autosaves disabled - tournament state will not be saved until completion.")
            
        # Ask about saving individual games
        self.save_individual_games = input("Save individual game data after each game? (y/n): ").lower().strip() == 'y'
        if self.save_individual_games:
            print("Individual game saving enabled - each game will be saved to a separate file with complete move history.")
        else:
            print("Individual game saving disabled - only tournament summary will be saved.")
            
        # Ask about generating enhanced visualizations
        self.create_visualizations = input("Generate enhanced visualizations after tournament? (y/n): ").lower().strip() == 'y'
        if self.create_visualizations:
            print("Enhanced visualizations enabled - publication-quality charts and graphs will be generated.")
        else:
            print("Enhanced visualizations disabled - only basic visualizations will be included.")
        
        return True
    
    def setup_game(self, game_num):
        """Set up a single game with selected models but don't run it yet"""
        game = LiarsDice()
        
        # List of distinct human names for AI players
        human_names = [
        "John", "Sarah", "Michael", "Emily", "David", "Jessica", "James", "Amanda", "Daniel", "Ashley", "Matthew", "Jennifer", "Andrew", "Megan", "Brian", "Laura", "Kevin", "Nicole", "Thomas", "Rachel"
        ]
        
        # Randomly select models for this game
        if len(self.selected_models) <= self.models_per_game:
            game_models = self.selected_models.copy()
        else:
            game_models = random.sample(self.selected_models, self.models_per_game)
        
        # Add AI players with selected models
        for i, model_info in enumerate(game_models):
            model_id = model_info["id"]
            provider = model_info["provider"]
            
            # Assign a random human name, but keep consistency
            name_index = hash(model_id) % len(human_names)
            player_name = human_names[name_index]
            
            # Make names unique if needed
            if i > 0 and player_name in [game.players[j].name for j in range(len(game.players))]:
                player_name = f"{player_name}-{i+1}"
            
            # Add the AI player
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
        
        # Set start time for timeout tracking
        start_time = time.time()
        max_duration = 3600  # 60 minutes in seconds
        
        try:
            # Play the game in auto mode
            while not game.game_over:
                # Check if we've exceeded the time limit
                if time.time() - start_time > max_duration:
                    raise TimeoutError(f"Game {game_num} exceeded the time limit")
                
                game.play_round(auto_mode=True, game_num=game_num)
            
            # Process results
            winner_model = self._process_game_results(game, game_num)
            
            # Restore output
            if not self.verbose_output:
                sys.stdout = original_stdout
            
            # Print game result
            print(f"Game {game_num}: Winner is {game.winner.name} ({winner_model}) after {game.round_number} rounds")
            
            return winner_model
        
        except TimeoutError as e:
            # Restore output in case of timeout
            if not self.verbose_output:
                sys.stdout = original_stdout
            print(f"Game {game_num} timed out and was terminated")
            return None
        except Exception as e:
            # Restore output in case of error
            if not self.verbose_output:
                sys.stdout = original_stdout
            print(f"Error in game {game_num}: {e}")
            return None
    
    def save_individual_game(self, game, game_num):
        """Save an individual game's data to a file"""
        if not self.save_individual_games:
            return None
        
        # Create directory if it doesn't exist
        games_dir = "games"
        os.makedirs(games_dir, exist_ok=True)
        
        # Generate filename with timestamp and game number
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"{games_dir}/game_{game_num}_{timestamp}.json"
        
        # Get the winner model
        winner_model = None
        if game.winner:
            for player in game.players:
                if isinstance(player, AIPlayer) and player.name == game.winner.name:
                    winner_model = player.model
                    break
        
        # Prepare serializable game state
        game_state = {
            "game_number": game_num,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "players": [game._serialize_player(p) for p in game.players],
            "current_player_idx": game.current_player_idx,
            "last_bid": game.last_bid,
            "total_dice_in_game": game.total_dice_in_game,
            "game_over": game.game_over,
            "winner": game._serialize_player(game.winner) if game.winner else None,
            "winner_model": winner_model,
            "move_history": game.move_history,
            "round_number": game.round_number,
            "player_history": game.player_history,
            # Save metrics data
            "metrics": game.get_metrics()
        }
        
        # Check if there were any invalid responses
        invalid_responses = []
        for player in game.players:
            if isinstance(player, AIPlayer):
                for move in game.move_history:
                    if move["player"] == player.name and move.get("error_fallback", False):
                        # Find corresponding error info in logs/invalid_responses.json
                        try:
                            with open("logs/invalid_responses.json", "r") as f:
                                for line in f:
                                    try:
                                        error_info = json.loads(line)
                                        if error_info.get("model") == player.model:
                                            invalid_responses.append(error_info)
                                    except json.JSONDecodeError:
                                        continue
                        except (IOError, FileNotFoundError):
                            pass
        
        # Add invalid responses to game data if any
        if invalid_responses:
            game_state["invalid_responses"] = invalid_responses
            
        # Check if there were any API responses
        responses = []
        try:
            with open("logs/llm_responses.json", "r") as f:
                for line in f:
                    try:
                        response_data = json.loads(line)
                        # Add all responses that match any of the players' models to record
                        for player in game.players:
                            if isinstance(player, AIPlayer) and response_data.get("model") == player.model:
                                responses.append(response_data)
                    except json.JSONDecodeError:
                        continue
        except (IOError, FileNotFoundError):
            pass
            
        # Add responses to game data if any
        if responses:
            game_state["llm_responses"] = responses
        
        # Write to file
        with open(filename, 'w') as f:
            json.dump(game_state, f, indent=2)
            
        print(f"Saved game {game_num} data to {filename}")
        return filename

    def _process_game_results(self, game, game_num):
        """Process and store game results"""
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
        
        # Save the individual game data
        self.save_individual_game(game, game_num)
        
        return winner_model
        
    def run_multiple_games_batch(self, game_nums):
        """Run multiple games in parallel using ThreadPoolExecutor with reduced concurrency"""
        # Use a fixed, smaller number of workers
        max_workers = 32  # Reduced from original max value
        
        print(f"Running with {max_workers} parallel workers")
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
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
                    # Enforce a timeout per thread
                    result = future.result(timeout=3600)  # 60 minutes timeout
                    results.append(result)
                    completed += 1

                    # Print progress as a percentage
                    print(f"Completed game {game_num} ({completed}/{total}, {completed/total*100:.1f}%)")

                    # Autosave tournament state periodically if enabled
                    if self.enable_autosaves and completed % max(50, total // 5) == 0:
                        print(f"Starting autosave at {completed}/{total}")
                        threading.Thread(target=self.save_tournament_state).start()

                except concurrent.futures.TimeoutError:
                    print(f"Game {game_num} timed out and was terminated")
                    results.append(None)
                except Exception as e:
                    print(f"Error in game {game_num}: {e}")
                    results.append(None)
            
        return results
    
    def run_games_with_asyncio(self, game_nums):
        """Run games concurrently using asyncio"""
        # Create the async game runner
        async_runner = AsyncGameRunner(self)
        
        # Set up asyncio event loop
        if sys.platform == 'win32':
            asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
        
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            # Run games and collect results
            results = loop.run_until_complete(async_runner.run_games_async(game_nums))
            loop.run_until_complete(async_runner.close())
            
            # Process results and track progress
            processed_results = []
            completed = 0
            total = len(game_nums)
            
            for i, result in enumerate(results):
                game_num = game_nums[i]
                
                # Handle the expanded result that includes game object
                if result and isinstance(result, dict) and 'winner' in result and 'game' in result:
                    # Extract the winner model
                    winner_model = result['winner']
                    game = result['game']
                    game_num = result['game_num']
                    
                    # We don't need to call _process_game_results again, since it was already called
                    # in play_single_game_async, but we do need to ensure the game object is stored
                    # It's possible the game object might have been overwritten in game_results
                    # Find the corresponding result and update it with this game object
                    for existing_result in self.game_results:
                        if existing_result.get('game_number') == game_num:
                            # Update the existing result with the game object
                            existing_result['game_obj'] = game
                            break
                    
                    processed_results.append(winner_model)
                else:
                    # Handle the old format or None results
                    processed_results.append(result)
                
                completed += 1
                
                # Print progress
                print(f"Processed game {game_num} ({completed}/{total}, {completed/total*100:.1f}%)")
                
                # Autosave tournament state periodically
                if self.enable_autosaves and completed % max(50, total // 5) == 0:
                    print(f"Starting autosave at {completed}/{total}")
                    threading.Thread(target=self.save_tournament_state).start()
            
            # Make sure we processed all games
            print(f"Completed all {len(processed_results)} games in async mode")
            return processed_results
        
        finally:
            # Clean up event loop
            loop.close()
    
    def save_tournament_state(self, filepath=None):
        """Save the current tournament state to a file"""
        if filepath is None:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filepath = f"liars_dice_save_{timestamp}.json"
            
        print(f"Saving tournament state to {filepath}...")
        
        # Create a minimal version of tournament state
        minimal_game_results = []
        for result in self.game_results:
            # Only keep essential fields for each game
            minimal_result = {
                "game_number": result["game_number"],
                "winner": result["winner"],
                "winner_model": result["winner_model"],
                "rounds": result["rounds"]
            }
            minimal_game_results.append(minimal_result)
        
        # Create tournament state
        tournament_state = {
            "leaderboard": self.leaderboard,
            "game_results": minimal_game_results,
            "total_games": self.total_games,
            "model_stats": self.model_stats,
            "completed_games": len(self.game_results),
            "models_per_game": self.models_per_game,
            "selected_models": self.selected_models,
            "use_local_endpoint": self.use_local_endpoint,
            "enable_autosaves": self.enable_autosaves,
            "save_individual_games": self.save_individual_games,
            "use_async": self.use_async
        }
        
        # Write to file
        with open(filepath, 'w') as f:
            json.dump(tournament_state, f)
            
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
            self.use_async = tournament_state.get("use_async", False)
            
            # Preserve the autosave setting or ask for it when resuming
            autosave_choice = input("Enable autosaves for this resumed tournament? (y/n): ").lower().strip()
            self.enable_autosaves = autosave_choice == 'y'
            
            # Preserve the individual game saving setting or ask for it when resuming
            save_games_choice = input("Save individual game data for this resumed tournament? (y/n): ").lower().strip()
            self.save_individual_games = save_games_choice == 'y'
            if self.save_individual_games:
                print("Individual game saving enabled - each game will be saved to a separate file with complete move history.")
            else:
                print("Individual game saving disabled - only tournament summary will be saved.")
            
            # Restore the minimal game results format we saved
            for result in self.game_results:
                # Add empty fields for any data that wasn't saved but might be accessed
                if 'move_history' not in result:
                    result['move_history'] = []
                if 'players' not in result:
                    result['players'] = []
            
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
        
        # Get mapping of model IDs to the human names used
        model_to_human_name = self._get_model_to_human_name_mapping()
        
        # Basic leaderboard
        print(f"{'Rank':<6}{'Model':<42}{'Human Name':<20}{'Win Rate':<15}{'Wins':<10}{'Games':<10}")
        print("-" * 100)
        
        for i, (model, stats) in enumerate(sorted_models):
            # Shorten model name if too long
            model_name = model
            if len(model_name) > 40:
                model_name = model_name[:37] + "..."
            
            # Get human name used for this model
            human_name = model_to_human_name.get(model, "Unknown")
                
            print(f"{i+1:<6}{model_name:<42}{human_name:<20}{stats['win_rate']:.1f}%{' ':<10}{stats['wins']:<10}{stats['games_played']:<10}")
            
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
            
            # Get human name used for this model
            human_name = model_to_human_name.get(model, "Unknown")
            
            print(f"\n{i+1}. {model} (as '{human_name}')")
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
    
    def _get_model_to_human_name_mapping(self):
        """Get a mapping from model IDs to the human names used in games"""
        model_to_human_name = {}
        
        # List of human names used for mapping
        human_names = [
            "Alex Morgan", "Blake Taylor", "Cameron Reed", "Devon Parker", 
            "Ellis Jordan", "Finley Quinn", "Gray Wilson", "Harper Lee", 
            "Indigo Carter", "Jordan Smith", "Kennedy Ross", "Logan Bailey", 
            "Morgan Casey", "Nico Riley", "Parker Quinn", "Reese Johnson", 
            "Sidney Shaw", "Taylor Wright", "Vaughn Miller", "Winter Stone"
        ]
        
        # Go through game results to find player names
        for game in self.game_results:
            if "players" in game:
                for player in game["players"]:
                    if "name" in player and "model" in player:
                        player_name = player["name"]
                        model_id = player["model"]
                        
                        # If this is a human name (not containing model-specific patterns)
                        if not any(x in player_name for x in ["-", "GPT", "Claude", "Llama", "Mistral"]):
                            model_to_human_name[model_id] = player_name
            
        # For any models without a mapping, create one based on the hash function
        for model in self.leaderboard:
            if model not in model_to_human_name:
                name_index = hash(model) % len(human_names)
                model_to_human_name[model] = human_names[name_index]
                
        return model_to_human_name
    
    def save_results(self, create_visualizations=True):
        """
        Save the tournament results to a file
        
        Args:
            create_visualizations: Whether to generate visualization charts (default: True)
        """
        base_filename = f"liars_dice_tournament_{time.strftime('%Y%m%d_%H%M%S')}"
        txt_filename = f"{base_filename}.txt"
        csv_filename = f"{base_filename}.csv"
        visualization_dir = f"{base_filename}_visualizations"
        
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
            
            # Get mapping of model IDs to the human names used
            model_to_human_name = self._get_model_to_human_name_mapping()
                
            # Basic leaderboard
            f.write("LEADERBOARD\n")
            f.write("=" * 100 + "\n")
            f.write(f"{'Rank':<6}{'Model':<30}{'Human Name':<20}{'Elo':<10}{'Win Rate':<15}{'Wins':<10}{'Games':<10}\n")
            f.write("-" * 100 + "\n")
            
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
                
                # Get human name for this model
                human_name = model_to_human_name.get(model, "Unknown")
                
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
                
                f.write(f"{i+1:<6}{model_name:<30}{human_name:<20}{elo:<10}{stats['win_rate']:.1f}%{' ':<10}{stats['wins']:<10}{stats['games_played']:<10}\n")
            
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
                
                # Get human name for this model
                human_name = model_to_human_name.get(model, "Unknown")
                
                f.write(f"\n{i+1}. {model} (as '{human_name}')\n")
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
                "Model", "Human Name", "Provider", "Elo", "Win Rate", "Wins", "Games", 
                "Bluff Success Rate", "Lie Detection Precision", "Lie Detection Recall", "Lie Detection F1",
                "Average Final Bid", "Bid Optimality", "Adaptation Score", "Rule Adherence Rate",
                "Avg API Response Time", "Avg Rounds per Game", "Early Game Wins", "Mid Game Wins", "Long Game Wins"
            ]
            writer.writerow(header)
            
            # Write data rows
            for model, stats in sorted_models:
                provider = model.split('/')[0] if '/' in model else "unknown"
                model_stats = self.model_stats[model]
                
                # Get human name used for this model
                human_name = model_to_human_name.get(model, "Unknown")
                
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
                    model, human_name, provider, elo, stats["win_rate"], stats["wins"], stats["games_played"],
                    bluff_success, lie_precision, lie_recall, lie_f1,
                    avg_final_bid, bid_optimality, adaptation, rule_adherence,
                    avg_api_response_time, model_stats["avg_rounds_per_game"], model_stats["early_game_wins"], 
                    model_stats["mid_game_wins"], model_stats["long_game_wins"]
                ]
                writer.writerow(row)
        
        # Create visualizations of the results
        try:
            self.generate_visualizations(base_filename, all_advanced_metrics)
        except Exception as e:
            print(f"Could not generate visualizations: {e}")
        
        print(f"\nResults saved to {txt_filename} and {csv_filename}")
    
    def run_tournament(self, resume_from=None, create_visualizations=True):
        """
        Run a tournament of multiple games
        
        Args:
            resume_from: Optional filename to resume a tournament from
            create_visualizations: Whether to generate advanced visualizations (default: True)
        """
        if resume_from and os.path.exists(resume_from):
            print(f"Resuming tournament from {resume_from}")
            completed_games = self.load_tournament_state(resume_from)
            start_game = completed_games + 1
        else:
            if not self.setup_batch():
                return
            start_game = 1
        
        print(f"\nStarting tournament from game {start_game}...")
        
        # Create list of game numbers to run
        game_nums = list(range(start_game, self.total_games + 1))
        
        if not game_nums:
            print("No games to run.")
            return

        # Run the games
        if self.use_async:
            self.run_games_with_asyncio(game_nums)
        else:
            self.run_multiple_games_batch(game_nums)
        
        # Update statistics
        self.update_leaderboard()
        
        # Display final results
        print("\nTournament complete!")
        self.display_leaderboard()
        
        # Save results and create visualizations if requested
        self.save_results(create_visualizations=create_visualizations if create_visualizations is not None else self.create_visualizations)
        
    def generate_visualizations(self, base_filename, all_advanced_metrics=None):
        """Generate visualizations of the tournament results"""
        # Create directory for visualizations
        vis_dir = f"{base_filename}_visualizations"
        os.makedirs(vis_dir, exist_ok=True)
        
        # Plot Elo ratings
        plt.figure(figsize=(12, 6))
        models = []
        elos = []
        
        # Collect Elo ratings from game results
        for game in self.game_results:
            game_obj = game.get("game_obj")
            if game_obj:
                metrics = game_obj.get_metrics()
                for model, data in metrics.items():
                    if "elo_rating" in data:
                        models.append(model)
                        elos.append(data["elo_rating"])
        
        if not models:
            return
            
        # Sort by Elo (descending)
        sorted_indices = np.argsort(elos)[::-1]
        sorted_models = [models[i].split('/')[-1] if '/' in models[i] else models[i] for i in sorted_indices]
        sorted_elos = [elos[i] for i in sorted_indices]
        
        plt.bar(sorted_models, sorted_elos, color='skyblue')
        plt.title('Model Comparison: Elo Ratings')
        plt.xlabel('Model')
        plt.ylabel('Elo Rating')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(f"{vis_dir}/elo_ratings.png")
        plt.close()
        
        # Create a radar chart for top 5 models comparing all metrics
        if all_advanced_metrics:
            top_models = sorted_indices[:5]
            top_model_names = [models[i] for i in top_models]
            
            # Get metrics for radar chart
            metrics_to_plot = {
                "Bluff Success": [all_advanced_metrics[model]["bluff_success_rate"] / 100 for model in top_model_names],
                "Lie Detection": [all_advanced_metrics[model]["lie_detection"]["f1_score"] / 100 for model in top_model_names],
                "Bid Optimality": [all_advanced_metrics[model]["bid_optimality"] / 100 for model in top_model_names],
                "Adaptation": [all_advanced_metrics[model]["adaptation_score"] / 100 for model in top_model_names],
                "Rule Adherence": [all_advanced_metrics[model]["rule_adherence_rate"] / 100 for model in top_model_names]
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
                ax.plot(angles, values, linewidth=1, label=model.split('/')[-1] if '/' in model else model)
                ax.fill(angles, values, alpha=0.1)
            
            # Set category labels
            plt.xticks(angles[:-1], categories)
            
            # Add legend
            plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
            plt.title('Advanced Metrics Comparison for Top Models')
            plt.tight_layout()
            plt.savefig(f"{vis_dir}/radar_metrics.png")
            plt.close()
        
        # Generate enhanced visualizations using the MetricsVisualizer
        if create_visualizations:
            print(f"\nGenerating enhanced visualizations in {visualization_dir}...")
            
            # Create a GameMetrics instance from our tournament data
            metrics = GameMetrics()
            
            # Initialize metrics with ELO ratings from our game data
            for model in self.leaderboard:
                metrics.initialize_model(model)
            
            # Load model metrics from game results
            for game in self.game_results:
                game_obj = game.get("game_obj")
                if game_obj:
                    # Get metrics from this game
                    game_metrics = game_obj.get_metrics()
                    
                    # Transfer metrics to our combined metrics object
                    if game_metrics:
                        # If there were only 2 players, record the match result for ELO
                        if len(game["players"]) == 2:
                            p1_model = game["players"][0]["model"]
                            p2_model = game["players"][1]["model"]
                            winner_model = game.get("winner_model")
                            
                            if winner_model and (winner_model == p1_model or winner_model == p2_model):
                                loser_model = p2_model if winner_model == p1_model else p1_model
                                metrics.update_elo(winner_model, loser_model)
                        
                        # Transfer other metrics
                        for model, model_metrics in game_metrics.items():
                            # Transfer all the game-specific metrics
                            if "bluff_success_rate" in model_metrics:
                                bluff_rate = model_metrics["bluff_success_rate"]
                                if isinstance(bluff_rate, dict):
                                    success = bluff_rate.get("successful", 0)
                                    total = bluff_rate.get("total", 0)
                                    if total > 0:
                                        metrics.bluff_data[model]["successful"] += success
                                        metrics.bluff_data[model]["total"] += total
                                
                            if "lie_detection" in model_metrics:
                                lie_det = model_metrics["lie_detection"]
                                if isinstance(lie_det, dict):
                                    tp = lie_det.get("true_positives", 0)
                                    fp = lie_det.get("false_positives", 0)
                                    fn = lie_det.get("false_negatives", 0)
                                    
                                    metrics.lie_detection_data[model]["true_positive"] += tp
                                    metrics.lie_detection_data[model]["false_positive"] += fp
                                    metrics.lie_detection_data[model]["false_negative"] += fn
                            
                            if "bid_optimality" in model_metrics:
                                bid_opt = model_metrics["bid_optimality"]
                                if isinstance(bid_opt, dict):
                                    optimal = bid_opt.get("optimal", 0)
                                    total = bid_opt.get("total", 0)
                                    if total > 0:
                                        metrics.bid_optimality[model]["optimal"] += optimal
                                        metrics.bid_optimality[model]["total"] += total
                                        
                            if "adaptation_score" in model_metrics:
                                adapt = model_metrics["adaptation_score"]
                                if isinstance(adapt, dict):
                                    adapted = adapt.get("adapted", 0)
                                    opportunities = adapt.get("opportunities", 0)
                                    if opportunities > 0:
                                        metrics.adaptation_scores[model]["adapted"] += adapted
                                        metrics.adaptation_scores[model]["opportunities"] += opportunities
                                        
                            if "rule_adherence_rate" in model_metrics:
                                adhere = model_metrics["rule_adherence_rate"]
                                if isinstance(adhere, dict):
                                    valid = adhere.get("valid_actions", 0)
                                    total = adhere.get("total_actions", 0)
                                    if total > 0:
                                        metrics.rule_adherence[model]["valid_actions"] += valid
                                        metrics.rule_adherence[model]["total_actions"] += total
                                        
                            if "avg_api_response_time" in model_metrics:
                                time_val = model_metrics["avg_api_response_time"]
                                if isinstance(time_val, (int, float)):
                                    metrics.api_response_times[model].append(time_val)
                                elif isinstance(time_val, dict):
                                    avg_time = time_val.get("average", 0)
                                    if avg_time > 0:
                                        metrics.api_response_times[model].append(avg_time)
                        
            # Generate comprehensive visualizations
            generated_files = metrics.create_visualizations(output_dir=visualization_dir, prefix=base_filename)
            
            # Add a note to the text report about visualizations
            with open(txt_filename, 'a') as f:
                f.write("\n\nENHANCED VISUALIZATIONS\n")
                f.write("=" * 80 + "\n")
                f.write(f"Enhanced visualizations have been generated in {visualization_dir}/\n\n")
                
                if generated_files:
                    f.write("Generated visualization files:\n")
                    for key, filepath in generated_files.items():
                        filename = os.path.basename(filepath)
                        f.write(f"- {key}: {filename}\n")
            
            print(f"Enhanced visualizations saved to {visualization_dir}/")
        
        print(f"\nTournament results have been saved to {txt_filename} and {csv_filename}")
        return base_filename