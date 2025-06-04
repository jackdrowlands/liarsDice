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
from .event_logger import EventLoggerFactory, log_tournament_start, log_tournament_end, log_game_start, log_game_end, log_round_start, log_round_end

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

# Custom Exception for Rate Limiting
class RateLimitError(Exception):
    def __init__(self, model_id, message="Rate limit exceeded"):
        self.model_id = model_id
        self.message = f"{message} for model: {model_id}"
        super().__init__(self.message)

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
                
                if response.status_code == 429: # Rate limit exceeded
                    error_msg = f"RATE LIMIT EXCEEDED for {model}: {response.status_code} - {response.text[:200]}"
                    logger.error(error_msg)
                    print(f"STOPPING GAME: {error_msg}") # Clear output to console
                    raise RateLimitError(model_id=model)

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
        logger.debug(f"Request params: {request_params}")
        
        # Capture raw prompt for logging
        prompt_dict = player.get_prompt_for_game(game_state)
        raw_prompt = f"System: {prompt_dict['system']}\n\nUser: {prompt_dict['user']}"
        game_state['raw_prompt'] = raw_prompt
        logger.debug(f"Raw prompt: {raw_prompt}")
        
        prep_time = time.time() - prep_start_time
        self.timing_data["prepare_request"].append(prep_time)
        logger.debug(f"Prep time: {prep_time}")
        
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
            content = ""
            # Standard OpenAI-like response structure
            if "choices" in result and result["choices"] and \
               isinstance(result["choices"][0], dict) and \
               "message" in result["choices"][0] and \
               isinstance(result["choices"][0]["message"], dict) and \
               "content" in result["choices"][0]["message"]:
                content = result["choices"][0]["message"]["content"].strip()
            # Handle Google's Gemini specific response structure (common alternative)
            elif "candidates" in result and result.get("candidates") and \
                 isinstance(result["candidates"][0], dict) and \
                 "content" in result["candidates"][0] and \
                 isinstance(result["candidates"][0]["content"], dict) and \
                 "parts" in result["candidates"][0]["content"] and \
                 result["candidates"][0]["content"].get("parts") and \
                 isinstance(result["candidates"][0]["content"]["parts"][0], dict) and \
                 "text" in result["candidates"][0]["content"]["parts"][0]:
                content = result["candidates"][0]["content"]["parts"][0]["text"].strip()
            else:
                logger.error(f"Unrecognized API response structure for {model_id}: {str(result)[:500]}")
                # Fallback or raise error if content cannot be extracted
                raise ValueError(f"Could not extract content from API response for {model_id}")
            
            # Store raw response in game_state for logging
            game_state['raw_response'] = content
            
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
                    if isinstance(request_params["data"]["messages"][0].get("content"), str): # Check if content is a string
                        response_log["prompt_system"] = request_params["data"]["messages"][0]["content"]
                    elif isinstance(request_params["data"]["messages"][0].get("content"), list): # Handle list of content parts (e.g. for some multimodal models)
                         # Try to concatenate text parts if they exist
                        system_prompt_parts = []
                        for part in request_params["data"]["messages"][0]["content"]:
                            if isinstance(part, dict) and "text" in part:
                                system_prompt_parts.append(part["text"])
                        if system_prompt_parts:
                            response_log["prompt_system"] = " ".join(system_prompt_parts)
                        else:
                            response_log["prompt_system"] = "[Non-text system prompt content]"
                    else:
                        response_log["prompt_system"] = "[Unknown system prompt format]"

                    # Truncate user prompt if large
                    user_prompt_content = request_params["data"]["messages"][1]["content"]
                    user_prompt_str = ""
                    if isinstance(user_prompt_content, str):
                        user_prompt_str = user_prompt_content
                    elif isinstance(user_prompt_content, list):
                        user_prompt_parts = []
                        for part in user_prompt_content:
                            if isinstance(part, dict) and "text" in part:
                                user_prompt_parts.append(part["text"])
                        if user_prompt_parts:
                            user_prompt_str = " ".join(user_prompt_parts)
                        else:
                            user_prompt_str = "[Non-text user prompt content]"
                    else:
                        user_prompt_str = "[Unknown user prompt format]"

                    if len(user_prompt_str) > 500:
                        user_prompt_str = user_prompt_str[:500] + "... [truncated]"
                    response_log["prompt_user"] = user_prompt_str 
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
        
        if isinstance(player, AIPlayer):
            print(f"\n{player.name} (AI) is thinking...")
            turn_start_time = time.time()
            logger.debug(f"Starting turn for player {player.name} (Model: {player.model})")
            game_state = game.create_game_state_for_ai(player_idx)
            turn_timeout = 300
            
            try:
                decision_task = asyncio.create_task(self.get_ai_decision_async(player, game_state))
                try:
                    decision = await asyncio.wait_for(decision_task, timeout=turn_timeout)
                    
                    if 'response_time' in game_state:
                        game.metrics.record_api_response_time(player.model, game_state['response_time'])
                        self.timing_data[f"api_time_{player.model}"].append(game_state['response_time'])
                    if 'token_usage' in game_state:
                        usage = game_state['token_usage']
                        game.metrics.record_token_usage(
                            player.model,
                            usage.get('prompt_tokens', 0),
                            usage.get('completion_tokens', 0),
                            usage.get('total_tokens', 0)
                        )
                    
                    # Process the AI decision and capture any corrections
                    invalid_bid_corrected = False
                    original_quantity = None
                    original_face = None
                    
                    if decision["action"] == "liar":
                        is_liar_call = game._process_ai_liar_call(player, decision)
                        action_type = "liar call" if is_liar_call else "bid (from liar)"
                    else:
                        # For bid actions, check if we need to correct an invalid bid
                        original_quantity = decision.get("quantity")
                        original_face = decision.get("face")
                        
                        # Check if this would be an invalid bid
                        valid_bid = True
                        if original_quantity < 1 or original_face < 1 or original_face > 6:
                            valid_bid = False
                        
                        # Check if bid is higher than the last bid
                        if game.last_bid and valid_bid:
                            last_quantity, last_value = game.last_bid
                            if original_quantity < last_quantity or (original_quantity == last_quantity and original_face <= last_value):
                                valid_bid = False
                        
                        if not valid_bid:
                            invalid_bid_corrected = True
                        
                        is_liar_call = game._process_ai_bid(player, decision)
                        action_type = "bid"
                    
                    # Log the move event after processing, with correction information
                    game_num = getattr(game, '_current_game_num', 0)
                    game._log_move_if_enabled(game_num, player, game_state, decision,
                                            invalid_bid_corrected, original_quantity, original_face)
                    
                    turn_time = time.time() - turn_start_time
                    self.timing_data[f"player_{player.name}_times"].append(turn_time)
                    self.timing_data[f"model_{player.model}_times"].append(turn_time)
                    logger.debug(f"Player {player.name} completed turn in {turn_time:.2f}s with action: {action_type}")
                    return is_liar_call
                except asyncio.TimeoutError:
                    elapsed = time.time() - turn_start_time
                    logger.error(f"TURN TIMEOUT: Player {player.name} (Model: {player.model}) exceeded {turn_timeout}s turn limit")
                    self.timing_data["turn_timeouts"].append({"player": player.name, "model": player.model, "elapsed_time": elapsed})
                    if not decision_task.done(): decision_task.cancel()
                    raise asyncio.TimeoutError(f"Player turn timeout after {elapsed:.2f}s")
            except Exception as e:
                error_time = time.time() - turn_start_time
                logger.error(f"Error during {player.name}'s turn after {error_time:.2f}s: {str(e)}")
                logger.warning(f"Using fallback logic for {player.name} due to error: {str(e)}")
                if game.last_bid:
                    last_quantity, last_value = game.last_bid
                    if last_quantity > game.total_dice_in_game: return True
                    game.last_bid = (last_quantity, last_value + 1) if last_value < 6 else (last_quantity + 1, 1)
                else: game.last_bid = (1, random.randint(3, 6))
                if isinstance(player, AIPlayer): game.metrics.record_rule_adherence(player.model, False)
                move_data = {"round": len(game.move_history) + 1, "player": player.name, "action": "bid", "quantity": game.last_bid[0], "value": game.last_bid[1], "error_fallback": True, "error_message": str(e)}
                game.move_history.append(move_data)
                game.player_history[player.name].append(move_data)
                
                # Log the async error fallback move to raw logging
                if EventLoggerFactory.is_enabled():
                    fallback_decision = {
                        "action": "bid",
                        "quantity": game.last_bid[0],
                        "face": game.last_bid[1],
                        "reasoning": "Async error processing response, using default bid.",
                        "utterance": "I'll make this bid."
                    }
                    fallback_game_state = {
                        'raw_prompt': 'Async error during processing',
                        'raw_response': f'Error: {str(e)}',
                        'response_time': error_time,
                        'token_usage': {'prompt_tokens': 0, 'completion_tokens': 0, 'total_tokens': 0}
                    }
                    game_num = getattr(game, '_current_game_num', 0)
                    game._log_move_if_enabled(game_num, player, fallback_game_state, fallback_decision)
                
                logger.info(f"Fallback bid for {player.name}: {game.last_bid[0]} {game.last_bid[1]}'s")
                print(f"{player.name} bids {game.last_bid[0]} {game.last_bid[1]}'s")
                self.timing_data[f"player_{player.name}_error_times"].append(time.time() - turn_start_time)
                return False
        raise ValueError("AsyncGameRunner only supports AI players")

    async def play_round_async(self, game, auto_mode=True, game_num=0):
        game.roll_all_dice()
        print(f"\n===== GAME {game_num} | ROUND {game.round_number} =====")
        logger.debug(f"Game {game_num} Round {game.round_number} starting")
        
        # Log round start event
        if EventLoggerFactory.is_enabled():
            log_round_start(game_num, game.round_number, game.total_dice_in_game)
        
        turn_count = 0
        while not game.game_over:
            turn_count += 1
            current_player = game.players[game.current_player_idx]
            player_model = current_player.model if isinstance(current_player, AIPlayer) else "human"
            logger.debug(f"Game {game_num} Round {game.round_number} Turn {turn_count}: {current_player.name} ({player_model})")
            game.show_dice_to_player(game.current_player_idx)
            turn_timer_start = time.time()
            is_calling_liar = await self.get_player_bid_async(game, game.current_player_idx)
            turn_duration = time.time() - turn_timer_start
            if turn_duration > 60: logger.warning(f"LONG TURN: Game {game_num} Round {game.round_number} - {current_player.name} took {turn_duration:.2f}s")
            if is_calling_liar:
                logger.debug(f"Game {game_num} Round {game.round_number}: {current_player.name} called liar")
                game.handle_liar_call(auto_continue=auto_mode, game_num=game_num)
                if game.check_game_over():
                    logger.info(f"Game {game_num} ended on round {game.round_number} after liar call")
                    break
                
                # Log round end before starting new round
                if EventLoggerFactory.is_enabled():
                    surviving_players = [p.name for p in game.players if p.get_dice_count() > 0]
                    log_round_end(game_num, game.round_number, surviving_players)
                
                game.round_number += 1
                print(f"\n===== GAME {game_num} | ROUND {game.round_number} =====")
                game.roll_all_dice()
                logger.debug(f"Game {game_num} Round {game.round_number} starting")
                
                # Log new round start
                if EventLoggerFactory.is_enabled():
                    log_round_start(game_num, game.round_number, game.total_dice_in_game)
                
                continue
            logger.debug(f"Game {game_num} Round {game.round_number}: Advancing to next player after {current_player.name}")
            game.next_player()

    async def play_single_game_async(self, game_num):
        game, game_models = self.batch_runner.setup_game(game_num)
        # Set game number for logging
        game._current_game_num = game_num
        self.timing_data = defaultdict(list); self.api_call_counts = defaultdict(int); self.timing_data["round_times"] = []
        for player in game.players: 
            if isinstance(player, AIPlayer): self.timing_data[f"model_{player.model}_times"] = []; self.timing_data[f"player_{player.name}_times"] = []
        self.game_timing_stats[game_num] = {"start_time": time.time(), "game_models": [m.get("id") for m in game_models], "player_models": {p.name: p.model for p in game.players if isinstance(p, AIPlayer)}, "round_completion": {}}
        original_stdout = sys.stdout
        if not self.batch_runner.verbose_output: sys.stdout = open(os.devnull, 'w')
        start_time = time.time(); max_duration = 3600
        logger.info(f"Starting game {game_num} with models: {', '.join([f'{p.name}: {p.model}' for p in game.players if isinstance(p, AIPlayer)])}")
        try:
            round_number = 0
            while not game.game_over:
                round_number += 1; round_start_time = time.time()
                elapsed = time.time() - start_time
                if elapsed > max_duration: self.game_timing_stats[game_num]["timeout"] = {"elapsed_time": elapsed, "rounds_completed": round_number -1}; raise asyncio.TimeoutError(f"Game {game_num} exceeded time limit")
                logger.debug(f"Game {game_num}: Starting round {round_number}")
                await self.play_round_async(game, auto_mode=True, game_num=game_num)
                round_time = time.time() - round_start_time; self.timing_data["round_times"].append(round_time)
                self.game_timing_stats[game_num]["round_completion"][round_number] = {"time": round_time, "elapsed": time.time() - start_time, "total_moves": len(game.move_history)}
                logger.debug(f"Game {game_num}: Completed round {round_number} in {round_time:.2f}s")
                if round_time > 300: logger.warning(f"Game {game_num}: LONG ROUND DETECTED - Round {round_number} took {round_time:.2f}s")
            game_time = time.time() - start_time; self.timing_data["total_game_time"] = game_time
            self.game_timing_stats[game_num]["completion_time"] = game_time; self.game_timing_stats[game_num]["rounds_completed"] = round_number
            logger.info(f"Game {game_num} completed in {game_time:.2f}s with {round_number} rounds")
            winner_model = self.batch_runner._process_game_results(game, game_num)
            if not self.batch_runner.verbose_output: sys.stdout = original_stdout
            print(f"Game {game_num}: Winner is {game.winner.name} ({winner_model}) after {game.round_number} rounds")
            return {"winner": winner_model, "game": game, "game_num": game_num}
        except RateLimitError as rle:
            if not self.batch_runner.verbose_output: sys.stdout = original_stdout
            print(f"Game {game_num} ABORTED: {rle.message}"); logger.error(f"Game {game_num} ABORTED due to RateLimitError: {rle.message}")
            self.game_timing_stats[game_num]["error"] = f"RateLimitError: {rle.message}"; self.game_timing_stats[game_num]["aborted_due_to_rate_limit"] = True
            return {"winner": None, "game": game, "game_num": game_num, "error": f"RateLimitError: {rle.message}", "aborted_due_to_rate_limit": True}
        except asyncio.TimeoutError:
            if not self.batch_runner.verbose_output: sys.stdout = original_stdout
            print(f"Game {game_num} timed out and was terminated")
            return {"winner": None, "game": None, "game_num": game_num, "error": "timeout"}
        except Exception as e:
            if not self.batch_runner.verbose_output: sys.stdout = original_stdout
            print(f"Error in game {game_num}: {e}")
            return {"winner": None, "game": None, "game_num": game_num, "error": str(e)}

    async def run_game_with_semaphore(self, game_num, semaphore):
        async with semaphore: return await self.play_single_game_async(game_num)

    async def run_games_async(self, game_nums):
        concurrency_limit = 32; semaphore = asyncio.Semaphore(concurrency_limit)
        logger.info(f"Starting async game batch with {len(game_nums)} games (max concurrent: {concurrency_limit})")
        tasks = [self.run_game_with_semaphore(game_num, semaphore) for game_num in game_nums]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        processed_results = []
        for i, result in enumerate(results):
            game_num = game_nums[i]
            if isinstance(result, Exception): logger.error(f"Game {game_num} error: {str(result)}"); processed_results.append({"winner": None, "game": None, "game_num": game_num, "error": str(result)})
            else: processed_results.append(result)
            self.save_timing_report(game_num)
        logger.info(f"Completed {len(processed_results)} games in async mode")
        return processed_results

class GameBatchRunner:
    def __init__(self):
        self.leaderboard = {}
        self.game_results = []
        self.total_games = 0
        self.models_per_game = 0
        self.selected_models = []
        self.auto_mode = True
        self.verbose_output = False
        self.use_async = True
        self.enable_autosaves = False
        self.save_individual_games = False
        self.create_visualizations = True
        self.use_local_endpoint = False
        self.openrouter_api_key = None
        self.available_models = []
        self.model_stats = {}
        self.raw_logging_enabled = True  # Enable raw tournament data capture by default

    def setup_batch(self):
        """Setup a batch of games to run interactively"""
        if not REQUESTS_AVAILABLE:
            print("API requests are not available.")
            print("Install the required package with: pip install requests")
            return False
        
        print("\n" + "=" * 70)
        print("LIARS DICE MODEL TOURNAMENT SETUP")
        print("=" * 70)
        
        # 1. Determine LLM source (local vs remote) & API Key
        # These will only be asked if not already set in the current session/instance
        if not hasattr(self, '_use_local_endpoint_set_in_session'):
            self.use_local_endpoint = input("Do you want to use a local LLM server (e.g., at http://127.0.0.1:1234)? (y/n): ").lower().strip() == 'y'
            self._use_local_endpoint_set_in_session = True # Mark as set for this session
        else:
            print(f"(Keeping existing LLM source: {'Local' if self.use_local_endpoint else 'Remote/OpenRouter'})")

        if not self.use_local_endpoint:
            if not os.environ.get("OPENROUTER_API_KEY") and not getattr(self, 'openrouter_api_key', None):
                current_key_to_set = input("Enter your OpenRouter API key: ")
                if current_key_to_set: # Only set if user provides one
                    self.openrouter_api_key = current_key_to_set
                    os.environ["OPENROUTER_API_KEY"] = self.openrouter_api_key
            elif os.environ.get("OPENROUTER_API_KEY"):
                 print("Using OpenRouter API key from environment variable.")
            elif getattr(self, 'openrouter_api_key', None):
                 print("(Keeping existing OpenRouter API key for this session.)")
        else:
            print("Using local LLM server. Ensure it's running and configured.")

        # 2. Get available models (early check)
        # Create a temporary LiarsDice instance just to call get_available_models
        game_for_models = LiarsDice() 
        self.available_models = game_for_models.get_available_models(self.use_local_endpoint, False) # Pass the determined setting
        
        if not self.available_models:
            print("No models available. Check your API key and connection, or local server setup.")
            print("Setup cannot continue without available models.")
            # Reset session flags so they are asked again if setup is retried
            if hasattr(self, '_use_local_endpoint_set_in_session'): delattr(self, '_use_local_endpoint_set_in_session')
            if hasattr(self, 'openrouter_api_key'): delattr(self, 'openrouter_api_key')
            return False
        
        print("\nAvailable models:")
        for i, model in enumerate(self.available_models):
            provider = model.get("provider", "Unknown")
            model_id = model.get("id", "Unknown ID")
            provider_label = ""
            if provider == PROVIDER_LOCAL: # Assuming PROVIDER_LOCAL is defined
                provider_label = " (Local)"
            print(f"{i+1}. {model_id}{provider_label}")

        # 3. Get the number of games to run
        if not hasattr(self, 'total_games') or self.total_games < 1:
            while True:
                try:
                    self.total_games = int(input("How many games to run in the tournament? (e.g., 10): "))
                    if self.total_games < 1:
                        print("Please enter a positive number.")
                        continue
                    break
                except ValueError:
                    print("Please enter a valid number.")
        else:
            print(f"(Keeping existing setting for total games: {self.total_games})")


        # 4. Get the number of models per game
        max_allowable_models_per_game = min(6, len(self.available_models))
        # Ensure models_per_game is at least 2 and not more than available models or 6
        min_required_models = 2

        if not hasattr(self, 'models_per_game') or not (min_required_models <= self.models_per_game <= max_allowable_models_per_game):
            while True:
                try:
                    prompt_message = f"How many models per game? ({min_required_models}-{max_allowable_models_per_game} recommended, {len(self.available_models)} available): "
                    self.models_per_game = int(input(prompt_message))
                    if not (min_required_models <= self.models_per_game <= max_allowable_models_per_game):
                        print(f"Please enter a number between {min_required_models} and {max_allowable_models_per_game}.")
                        if self.models_per_game > len(self.available_models):
                             print(f"Note: You only have {len(self.available_models)} models available in total.")
                    elif self.models_per_game > len(self.available_models): # Specifically if they ask for more than available
                        print(f"You requested {self.models_per_game} models per game, but only {len(self.available_models)} are available.")
                        print("Please select a number less than or equal to the available models.")
                    else:
                        break # Valid number entered
                except ValueError:
                    print("Please enter a valid number.")
        else:
             print(f"(Keeping existing setting for models per game: {self.models_per_game})")


        # 5. Select models (with improved loop to handle count errors)
        # This loop will continue until a valid set of models meeting requirements is chosen.
        self.selected_models = [] # Clear previous selections if setup is re-entered on same instance.
        while True: 
            print("\nSelect models to include (enter model numbers separated by spaces, or 'all'):")
            selection_str = input("> ").strip()
            
            current_selection_attempt = []
            if selection_str.lower() == "all":
                current_selection_attempt = self.available_models.copy()
            else:
                try:
                    selected_indices = [int(idx_str) - 1 for idx_str in selection_str.split()]
                    valid_indices = True
                    for idx in selected_indices:
                        if 0 <= idx < len(self.available_models):
                            if self.available_models[idx] not in current_selection_attempt:
                                current_selection_attempt.append(self.available_models[idx])
                        else:
                            print(f"Invalid model number: {idx+1}. Please use numbers from the list (1 to {len(self.available_models)}).")
                            valid_indices = False
                            current_selection_attempt = [] # Clear partial selection on error
                            break 
                    if not valid_indices:
                        continue # Re-prompt for selection string
                    if not current_selection_attempt and selection_str: # Valid numbers but empty list (e.g. re-selected already selected)
                        print("No new models selected or selection was empty. Please try again or select 'all'.")
                        continue
                    if not current_selection_attempt and not selection_str: # Empty input
                        print("Please enter model numbers or 'all'.")
                        continue
                        
                except ValueError:
                    print("Invalid input. Please enter numbers separated by spaces or 'all'.")
                    continue # Re-prompt for selection string
            
            # Check if the selection meets requirements
            if len(current_selection_attempt) < 2:
                print("Error: At least 2 models must be selected for the tournament.")
                # Loop continues for new model selection input
            elif len(current_selection_attempt) < self.models_per_game:
                print(f"Error: You selected {len(current_selection_attempt)} model(s), but your 'models per game' setting is {self.models_per_game}.")
                print(f"Please select at least {self.models_per_game} models, or select 'all'.")
                # Loop continues for new model selection input
            else:
                self.selected_models = current_selection_attempt # Valid selection that meets requirements
                break # Exit model selection loop
        
        print("\nSelected models for tournament:")
        for model_info in self.selected_models:
            model_id = model_info.get("id", "Unknown ID")
            provider = model_info.get("provider", "Unknown") # Assuming PROVIDER_LOCAL is defined
            print(f"- {model_id}{' (Local)' if provider == PROVIDER_LOCAL else ''}")
        
        # Initialize leaderboard and stats for all selected models
        self.leaderboard = {}
        self.model_stats = {}
        for model_info in self.selected_models:
            model_id_str = str(model_info["id"]) # Ensure model_id is a string
            self.leaderboard[model_id_str] = {
                "wins": 0, "games_played": 0, "win_rate": 0.0, "model": model_id_str
            }
            self.model_stats[model_id_str] = {
                "total_rounds_played": 0, "avg_rounds_survived": 0, "total_bids": 0,
                "total_liar_calls": 0, "successful_liar_calls": 0, "unsuccessful_liar_calls": 0,
                "liar_success_rate": 0.0, "avg_rounds_per_game": 0.0,
                "early_game_wins": 0, "mid_game_wins": 0, "long_game_wins": 0
            }
        
        # 6. Boolean settings (auto_mode, verbose, async, autosaves, individual saves, visualizations)
        # These will only be asked if not already set in the current instance/session
        settings_prompts = [
            ('_auto_mode_set_in_session', 'auto_mode', True, "\nRun all games in automatic mode (AI vs AI, no human interaction during games)? (y/n): "),
            ('_verbose_output_set_in_session', 'verbose_output', False, "Show detailed game-by-game output in the console? (y/n): "),
            ('_use_async_set_in_session', 'use_async', True, "\nUse asynchronous game execution (recommended for speed with API calls)? (y/n): "),
            ('_enable_autosaves_set_in_session', 'enable_autosaves', False, "Enable periodic autosaves of tournament progress? (y/n): "),
            ('_save_individual_games_set_in_session', 'save_individual_games', False, "Save detailed data for each individual game? (y/n): "),
            ('_create_visualizations_set_in_session', 'create_visualizations', True, "Generate enhanced visualizations after the tournament? (y/n): "),
            ('_raw_logging_set_in_session', 'raw_logging_enabled', True, "Enable raw tournament data capture for replay analysis? (y/n): ")
        ]

        for flag_name, attr_name, default_val, prompt_text in settings_prompts:
            if not hasattr(self, flag_name):
                user_input = input(prompt_text).lower().strip()
                setattr(self, attr_name, user_input == 'y')
                setattr(self, flag_name, True) # Mark as asked in this session
                
                if attr_name == 'use_async' and getattr(self, attr_name):
                    try:
                        import httpx # Check for import success here
                        print("Using asynchronous execution.")
                    except ImportError:
                        print("httpx package not installed. It is required for async execution.")
                        print("Install with: pip install httpx")
                        setattr(self, attr_name, False) # Fallback
                        print("Falling back to threaded (synchronous) execution.")
                elif attr_name == 'use_async' and not getattr(self, attr_name):
                     print("Using threaded (synchronous) execution.")

            # Always print current status of the setting
            current_status = "Enabled" if getattr(self, attr_name) else "Disabled"
            readable_name = attr_name.replace('_', ' ').capitalize()
            # Special handling for async to show the actual state post-httpx check
            if attr_name == 'use_async': 
                print(f"({readable_name}: {current_status})")
            # elif attr_name == 'enable_autosaves' and getattr(self, attr_name):
            elif attr_name == 'enable_autosaves': # Temporarily removed getattr for linting test
                 print("Autosaves enabled - tournament state will be saved periodically.")
            # elif attr_name == 'save_individual_games' and getattr(self, attr_name):
            elif attr_name == 'save_individual_games': # Temporarily removed getattr for linting test
                 print("Individual game saving enabled.")
            # elif attr_name == 'create_visualizations' and getattr(self, attr_name):
            elif attr_name == 'create_visualizations': # Temporarily removed getattr for linting test
                 print("Enhanced visualizations will be generated.")
            # For auto_mode and verbose_output, the (Setting: Status) is good enough implicitly from loop
            elif attr_name in ['auto_mode', 'verbose_output']:
                 print(f"({readable_name}: {current_status})")


        return True

    def setup_game(self, game_num):
        game = LiarsDice()
        human_names = [ "John", "Sarah", "Michael", "Emily", "David", "Jessica", "James", "Amanda", "Daniel", "Ashley", "Matthew", "Jennifer", "Andrew", "Megan", "Brian", "Laura", "Kevin", "Nicole", "Thomas", "Rachel"]
        game_models = random.sample(self.selected_models, self.models_per_game) if len(self.selected_models) > self.models_per_game else self.selected_models.copy()
        for i, model_info in enumerate(game_models):
            model_id = model_info["id"]
            name_idx = hash(model_id) % len(human_names)
            player_name = human_names[name_idx]
            if i > 0 and player_name in [p.name for p in game.players]: player_name = f"{player_name}-{i+1}"
            game.add_ai_player(name=player_name, model=model_id, provider=model_info["provider"], api_key=model_info.get("api_key"), api_url=model_info.get("api_url"), project_id=model_info.get("project_id"), region=model_info.get("region"))
            self.leaderboard[model_id]["games_played"] += 1
        game.current_player_idx = random.randint(0, len(game_models) - 1)
        
        # Log game start event
        if self.raw_logging_enabled:
            player_roster = []
            starting_elos = {}
            for player in game.players:
                if isinstance(player, AIPlayer):
                    player_info = {
                        "name": player.name,
                        "model_id": player.model,
                        "provider": player.provider,
                        "initial_dice": 5  # Standard starting dice count
                    }
                    player_roster.append(player_info)
                    # Get current Elo rating for this model
                    if hasattr(game.metrics, 'elo_ratings') and player.model in game.metrics.elo_ratings:
                        starting_elos[player.name] = game.metrics.elo_ratings[player.model]
                    else:
                        starting_elos[player.name] = 1000.0  # Default Elo
            
            log_game_start(game_num, player_roster, starting_elos)
        
        return game, game_models

    def run_single_game(self, game_num):
        game, game_models = self.setup_game(game_num)
        original_stdout = sys.stdout
        if not self.verbose_output: sys.stdout = open(os.devnull, 'w')
        start_time = time.time(); max_duration = 3600
        try:
            while not game.game_over:
                if time.time() - start_time > max_duration: raise TimeoutError(f"Game {game_num} timed out")
                game.play_round(auto_mode=True, game_num=game_num)
            winner_model = self._process_game_results(game, game_num)
            if not self.verbose_output: sys.stdout = original_stdout
            print(f"Game {game_num}: Winner {game.winner.name} ({winner_model}) in {game.round_number} rounds")
            return winner_model
        except Exception as e:
            if not self.verbose_output: sys.stdout = original_stdout
            print(f"Error in game {game_num}: {e}"); return None

    def save_individual_game(self, game, game_num):
        if not self.save_individual_games: return None
        os.makedirs("games", exist_ok=True)
        filename = f"games/game_{game_num}_{time.strftime('%Y%m%d_%H%M%S')}.json"
        winner_model = next((p.model for p in game.players if isinstance(p, AIPlayer) and p.name == game.winner.name), None) if game.winner else None
        game_state = {"game_number": game_num, "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"), "players": [game._serialize_player(p) for p in game.players], "current_player_idx": game.current_player_idx, "last_bid": game.last_bid, "total_dice_in_game": game.total_dice_in_game, "game_over": game.game_over, "winner": game._serialize_player(game.winner) if game.winner else None, "winner_model": winner_model, "move_history": game.move_history, "round_number": game.round_number, "player_history": game.player_history, "metrics": game.get_metrics()}
        # Simplified logging for brevity in restoration
        with open(filename, 'w') as f: json.dump(game_state, f, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o))
        print(f"Saved game {game_num} to {filename}"); return filename

    def _process_game_results(self, game, game_num):
        winner_model = None
        if game.winner: 
            winner_player = next((p for p in game.players if p.name == game.winner.name and isinstance(p, AIPlayer)), None)
            if winner_player: 
                winner_model = winner_player.model
                self.leaderboard[winner_model]["wins"] +=1
                if game.round_number < 10: self.model_stats[winner_model]["early_game_wins"] +=1
                elif game.round_number < 20: self.model_stats[winner_model]["mid_game_wins"] +=1
                else: self.model_stats[winner_model]["long_game_wins"] +=1
        for player in game.players: 
            if isinstance(player, AIPlayer): 
                model_id = player.model
                bids=0; liar_calls=0; succ_liar=0; unsucc_liar=0
                for move in game.move_history: 
                    if move["player"] == player.name: 
                        if move["action"] == "bid": bids+=1
                        elif move["action"] == "liar": 
                            liar_calls+=1
                            if move.get("outcome")=="success": succ_liar+=1
                            elif move.get("outcome")=="failure": unsucc_liar+=1
                self.model_stats[model_id]["total_bids"]+=bids; self.model_stats[model_id]["total_liar_calls"]+=liar_calls; self.model_stats[model_id]["successful_liar_calls"]+=succ_liar; self.model_stats[model_id]["unsuccessful_liar_calls"]+=unsucc_liar; self.model_stats[model_id]["total_rounds_played"]+=game.round_number
        self.game_results.append({"game_number": game_num, "players": [{"name":p.name, "model":p.model} for p in game.players if isinstance(p,AIPlayer)], "winner": game.winner.name if game.winner else "N/A", "winner_model":winner_model, "rounds":game.round_number, "move_history": [move.copy() for move in game.move_history], "game_obj":game})
        
        # Log game end event
        if self.raw_logging_enabled:
            final_winner = game.winner.name if game.winner else "N/A"
            log_game_end(game_num, final_winner, game.round_number)
        
        self.save_individual_game(game, game_num); return winner_model

    def run_multiple_games_batch(self, game_nums):
        max_workers = 32; print(f"Running with {max_workers} parallel workers")
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(self.run_single_game, gn): gn for gn in game_nums}
            results = []; completed = 0; total = len(game_nums)
            for future in concurrent.futures.as_completed(futures):
                gn = futures[future]; completed+=1
                try: results.append(future.result(timeout=3600)); print(f"Completed game {gn} ({completed}/{total}, {completed/total*100:.1f}%)")
                except Exception as e: print(f"Error game {gn}: {e}"); results.append(None)
                if self.enable_autosaves and completed % max(1, total//5) == 0: print(f"Autosaving at {completed}/{total}"); threading.Thread(target=self.save_tournament_state).start()
            self.update_leaderboard(); self.save_tournament_state()
        return results

    def run_games_with_asyncio(self, game_nums):
        async_runner = AsyncGameRunner(self)
        if sys.platform == 'win32': asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
        loop = asyncio.new_event_loop(); asyncio.set_event_loop(loop)
        try:
            results = loop.run_until_complete(async_runner.run_games_async(game_nums))
            loop.run_until_complete(async_runner.close())
            processed_results = []; completed = 0; total = len(game_nums)
            for i, result_item in enumerate(results):
                game_num_res = game_nums[i]
                if result_item and isinstance(result_item, dict) and 'game' in result_item:
                    for gr in self.game_results: 
                        if gr.get('game_number') == result_item.get('game_num'): gr['game_obj'] = result_item['game']; break
                    processed_results.append(result_item.get('winner'))
                else: processed_results.append(result_item)
                completed += 1; print(f"Processed game {game_num_res} ({completed}/{total}, {completed/total*100:.1f}%)")
                if self.enable_autosaves and completed % max(1, total // 5) == 0: print(f"Autosaving at {completed}/{total}"); threading.Thread(target=self.save_tournament_state).start()
            self.update_leaderboard(); self.save_tournament_state()
            print(f"Completed all {len(processed_results)} games in async mode")
            return processed_results
        finally: loop.close()

    def save_tournament_state(self, filepath=None):
        self.update_leaderboard()
        if filepath is None: filepath = f"liars_dice_save_{time.strftime('%Y%m%d_%H%M%S')}.json"
        print(f"Saving tournament state to {filepath}...")
        minimal_game_results = [{"game_number": r["game_number"], "winner": r["winner"], "winner_model": r["winner_model"], "rounds": r["rounds"]} for r in self.game_results]
        state = {"leaderboard": self.leaderboard, "game_results": minimal_game_results, "total_games": self.total_games, "model_stats": self.model_stats, "completed_games": len(self.game_results), "models_per_game": self.models_per_game, "selected_models": self.selected_models, "use_local_endpoint": self.use_local_endpoint, "enable_autosaves": self.enable_autosaves, "save_individual_games": self.save_individual_games, "use_async": self.use_async, "auto_mode": self.auto_mode, "verbose_output": self.verbose_output, "create_visualizations": self.create_visualizations, "raw_logging_enabled": self.raw_logging_enabled}
        with open(filepath, 'w') as f: json.dump(state, f, indent=2) # Added indent for readability
        print(f"Tournament state saved to {filepath}"); return filepath

    def load_tournament_state(self, filepath):
        print(f"Loading tournament state from {filepath}...")
        try:
            with open(filepath, 'r') as f: state = json.load(f)
            self.leaderboard = state.get("leaderboard", {})
            self.game_results = state.get("game_results", []) # Game objects will be missing, handle this
            self.total_games = state.get("total_games", 0)
            self.model_stats = state.get("model_stats", {})
            self.models_per_game = state.get("models_per_game", 2)
            self.selected_models = state.get("selected_models", [])
            self.use_local_endpoint = state.get("use_local_endpoint", self.use_local_endpoint)
            self.use_async = state.get("use_async", self.use_async)
            self.auto_mode = state.get("auto_mode", self.auto_mode)
            self.verbose_output = state.get("verbose_output", self.verbose_output)
            self.create_visualizations = state.get("create_visualizations", self.create_visualizations)
            
            # Handle boolean settings not in older save files by prompting
            if "enable_autosaves" in state: self.enable_autosaves = state["enable_autosaves"]
            else: self.enable_autosaves = input("Enable autosaves? (y/n): ").lower().strip() == 'y'
            print(f"Autosaves: {'Enabled' if self.enable_autosaves else 'Disabled'}")
            if "save_individual_games" in state: self.save_individual_games = state["save_individual_games"]
            else: self.save_individual_games = input("Save individual games? (y/n): ").lower().strip() == 'y'
            print(f"Save Individual Games: {'Enabled' if self.save_individual_games else 'Disabled'}")
            if "raw_logging_enabled" in state: self.raw_logging_enabled = state["raw_logging_enabled"]
            else: self.raw_logging_enabled = input("Enable raw tournament data capture? (y/n): ").lower().strip() == 'y'
            print(f"Raw Logging: {'Enabled' if self.raw_logging_enabled else 'Disabled'}")
            
            for r in self.game_results: r.setdefault('move_history', []); r.setdefault('players', []) # Ensure keys exist
            print(f"Successfully loaded {state.get('completed_games',0)} completed games.")
            return state.get("completed_games", 0)
        except Exception as e: print(f"Error loading state: {e}"); return 0

    def update_leaderboard(self):
        for model_id in self.leaderboard:
            stats = self.leaderboard[model_id]; model_detail_stats = self.model_stats[model_id]
            stats["win_rate"] = (stats["wins"] / stats["games_played"] * 100) if stats["games_played"] > 0 else 0
            model_detail_stats["liar_success_rate"] = (model_detail_stats["successful_liar_calls"] / model_detail_stats["total_liar_calls"] * 100) if model_detail_stats["total_liar_calls"] > 0 else 0
            model_detail_stats["avg_rounds_per_game"] = (model_detail_stats["total_rounds_played"] / stats["games_played"]) if stats["games_played"] > 0 else 0

    def display_leaderboard(self):
        print("\n" + "="*80 + "\nLIARS DICE MODEL LEADERBOARD\n" + "="*80)
        sorted_models = sorted(self.leaderboard.items(), key=lambda x: (x[1]["win_rate"], x[1]["wins"]), reverse=True)
        name_map = self._get_model_to_human_name_mapping()
        print(f"{'Rank':<6}{'Model':<42}{'Human Name':<20}{'Win Rate':<15}{'Wins':<10}{'Games':<10}")
        print("-"*100)
        for i, (model_id, stats) in enumerate(sorted_models): print(f"{i+1:<6}{(model_id[:37] + '...') if len(model_id)>40 else model_id:<42}{name_map.get(model_id,'N/A'):<20}{stats['win_rate']:.1f}%{'':<10}{stats['wins']:<10}{stats['games_played']:<10}")
        print("\n" + "="*80 + "\nDETAILED MODEL STATISTICS\n" + "="*80)
        for i, (model_id, stats) in enumerate(sorted_models): # Elo added in save_results display
            model_s = self.model_stats[model_id]; provider = model_id.split('/')[0] if '/' in model_id else 'N/A'
            print(f"\n{i+1}. {model_id} (as '{name_map.get(model_id,'N/A')}')")
            print(f"   Provider: {provider}")
            print(f"   Win Rate: {stats['win_rate']:.1f}% ({stats['wins']}/{stats['games_played']})")
            print(f"   Avg. Rounds/Game: {model_s['avg_rounds_per_game']:.1f}")
            print(f"   Liar Call Success: {model_s['liar_success_rate']:.1f}% ({model_s['successful_liar_calls']}/{model_s['total_liar_calls']})")
            total_moves = model_s['total_bids'] + model_s['total_liar_calls']
            print(f"   Playing Style: {model_s['total_bids']} bids, {model_s['total_liar_calls']} liar calls ({(model_s['total_liar_calls']/total_moves*100) if total_moves>0 else 0:.1f}%)")
            print(f"   Wins by Length: {model_s['early_game_wins']} E, {model_s['mid_game_wins']} M, {model_s['long_game_wins']} L")

    def _get_model_to_human_name_mapping(self):
        # Simplified for restoration
        name_map = {}
        human_names = [ "Alex Morgan", "Blake Taylor", "Cameron Reed", "Devon Parker", "Ellis Jordan", "Finley Quinn", "Gray Wilson", "Harper Lee", "Indigo Carter", "Jordan Smith", "Kennedy Ross", "Logan Bailey", "Morgan Casey", "Nico Riley", "Parker Quinn", "Reese Johnson", "Sidney Shaw", "Taylor Wright", "Vaughn Miller", "Winter Stone"]
        for game in self.game_results: 
            for p_info in game.get("players",[]): 
                if p_info.get("name") and p_info.get("model") and not any(x in p_info["name"] for x in ['-', 'GPT', 'Claude']): name_map[p_info["model"]] = p_info["name"]
        for model_id in self.leaderboard: 
            if model_id not in name_map: name_map[model_id] = human_names[hash(model_id) % len(human_names)]
        return name_map

    def save_results(self):
        base_fn = f"liars_dice_tournament_{time.strftime('%Y%m%d_%H%M%S')}"
        txt_fn = f"{base_fn}.txt"; csv_fn = f"{base_fn}.csv"
        with open(txt_fn, 'w') as f:
            f.write("LIARS DICE TOURNAMENT RESULTS\n" + "="*80 + "\n")
            f.write(f"Date: {time.strftime('%Y-%m-%d %H:%M:%S')}\nTotal Games: {self.total_games}\nModels/Game: {self.models_per_game}\nProviders: {', '.join(set(m.split('/')[0] for m in self.leaderboard if '/' in m))}\n\n")
            name_map = self._get_model_to_human_name_mapping()
            f.write("LEADERBOARD\n" + "="*100 + "\n")
            f.write(f"{'Rank':<6}{'Model':<30}{'Human Name':<20}{'Elo':<10}{'Win Rate':<15}{'Wins':<10}{'Games':<10}\n" + "-"*100 + "\n")
            sorted_models = sorted(self.leaderboard.items(),key=lambda x: (x[1]["win_rate"], x[1]["wins"]), reverse=True)
            
            all_adv_metrics = {} # Simplified aggregation for restoration
            for gr in self.game_results: 
                g_obj = gr.get("game_obj")
                if g_obj: 
                    metrics = g_obj.get_metrics()
                    for m_id, m_metrics in metrics.items(): 
                        if m_id not in all_adv_metrics: all_adv_metrics[m_id] = m_metrics
                        else: # Basic averaging for restoration
                            for k,v in m_metrics.items(): 
                                if isinstance(v, (int,float)) and k != "elo_rating": all_adv_metrics[m_id][k] = all_adv_metrics[m_id].get(k,0) + v
                                elif isinstance(v, dict): 
                                    if k not in all_adv_metrics[m_id]: all_adv_metrics[m_id][k] = {}
                                    for sk,sv in v.items(): 
                                        if isinstance(sv, (int,float)): all_adv_metrics[m_id][k][sk] = all_adv_metrics[m_id][k].get(sk,0) + sv
            for m_id, metrics_data in all_adv_metrics.items():
                 games_p = self.leaderboard.get(m_id, {}).get("games_played", 1)
                 if games_p > 0:
                    for k,v in metrics_data.items():
                        if isinstance(v, (int,float)) and k != "elo_rating": metrics_data[k] = v / games_p
                        elif isinstance(v, dict): 
                            for sk,sv in v.items(): 
                                if isinstance(sv, (int,float)): metrics_data[k][sk] = sv / games_p
            
            for i, (model_id, stats) in enumerate(sorted_models):
                elo_val = all_adv_metrics.get(model_id,{}).get('elo_rating', self.leaderboard.get(model_id,{}).get('elo_rating',1000.0))
                f.write(f"{i+1:<6}{(model_id[:25]+'...') if len(model_id)>28 else model_id:<30}{name_map.get(model_id,'N/A'):<20}{elo_val:.1f}{'':<5}{stats['win_rate']:.1f}%{'':<10}{stats['wins']:<10}{stats['games_played']:<10}\n")
            f.write("\nDETAILED MODEL STATISTICS\n" + "="*80 + "\n")
            for i, (model_id, stats) in enumerate(sorted_models):
                # Simplified for restoration - uses all_adv_metrics where possible
                m_s = self.model_stats.get(model_id, {}); adv_m = all_adv_metrics.get(model_id, {})
                f.write(f"\n{i+1}. {model_id} (as '{name_map.get(model_id,'N/A')}')\n")
                elo_rating_val = adv_m.get('elo_rating', 'N/A')
                elo_rating_str = f"{elo_rating_val:.1f}" if isinstance(elo_rating_val, (int, float)) else str(elo_rating_val)
                f.write(f"   Provider: {model_id.split('/')[0] if '/' in model_id else 'N/A'}\n   Win Rate: {stats['win_rate']:.1f}% ({stats['wins']}/{stats['games_played']})\n   Elo: {elo_rating_str}\n")
                # Add more detailed stats using adv_m, example:
                bluff_success_val = adv_m.get('bluff_success_rate',0)
                bluff_success_str = f"{bluff_success_val:.1f}%" if isinstance(bluff_success_val, (int, float)) else str(bluff_success_val)
                lie_detection_f1_val = adv_m.get('lie_detection',{}).get('f1_score',0)
                lie_detection_f1_str = f"{lie_detection_f1_val:.1f}%" if isinstance(lie_detection_f1_val, (int, float)) else str(lie_detection_f1_val)
                rule_adherence_val = adv_m.get('rule_adherence_rate',0)
                rule_adherence_str = f"{rule_adherence_val:.1f}%" if isinstance(rule_adherence_val, (int, float)) else str(rule_adherence_val)
                f.write(f"   Bluff Success: {bluff_success_str}\n   Lie Detection (F1): {lie_detection_f1_str}\n")
                f.write(f"   Rule Adherence: {rule_adherence_str}\n")
                f.write(f"   Avg. Rounds/Game: {m_s.get('avg_rounds_per_game',0):.1f}\n   Liar Call Success: {m_s.get('liar_success_rate',0):.1f}%\n")
            # Game Results Section
            f.write("\nGAME-BY-GAME RESULTS\n" + "="*80 + "\n")
            for game_result in self.game_results:
                game_num = game_result.get("game_number", "N/A")
                winner = game_result.get("winner", "N/A")
                winner_model = game_result.get("winner_model", "N/A")
                players = game_result.get("players", [])
                
                # Find the loser(s)
                losers = [p for p in players if p.get("name") != winner]
                if losers:
                    loser_info = ", ".join([f"{p.get('name', 'Unknown')} ({p.get('model', 'Unknown')})" for p in losers])
                else:
                    loser_info = "Unknown"
                
                f.write(f"Game {game_num}:  {winner} ({winner_model}) defeated {loser_info}\n")
            
            # Performance Analysis Section
            f.write("\nKEY PERFORMANCE INSIGHTS\n" + "="*80 + "\n")
            
            # Find top performer
            top_model = max(sorted_models, key=lambda x: x[1]["win_rate"]) if sorted_models else ("N/A", {"win_rate": 0})
            top_model_id, top_stats = top_model
            top_adv_metrics = all_adv_metrics.get(top_model_id, {})
            
            f.write("\nModel Dominance:\n")
            f.write(f"- {top_model_id} achieved {top_stats['win_rate']:.1f}% win rate ({top_stats['wins']}/{top_stats['games_played']} games)\n")
            
            # Add performance insights from advanced metrics
            if top_adv_metrics:
                lie_detection = top_adv_metrics.get('lie_detection', {})
                f1_score = lie_detection.get('f1_score', 0)
                bluff_rate = top_adv_metrics.get('bluff_success_rate', 0)
                rule_adherence = top_adv_metrics.get('rule_adherence_rate', 0)
                
                f.write(f"- Exceptional lie detection with {f1_score:.1f}% F1 score\n")
                if 'liar_call_success_rate' in top_adv_metrics:
                    f.write(f"- {top_adv_metrics['liar_call_success_rate']:.1f}% liar call accuracy\n")
                f.write(f"- Strategic bluffing with {bluff_rate:.1f}% success rate\n")
                f.write(f"- Perfect rule adherence at {rule_adherence:.1f}%\n")
            
            # Strategic Analysis
            f.write("\nStrategic Analysis:\n")
            avg_rounds = sum(gr.get("rounds", 0) for gr in self.game_results) / len(self.game_results) if self.game_results else 0
            f.write(f"- Average game length: {avg_rounds:.1f} rounds\n")
            
            # Find performance patterns
            if len(sorted_models) >= 2:
                second_model_id, second_stats = sorted_models[1]
                second_adv_metrics = all_adv_metrics.get(second_model_id, {})
                
                f.write(f"- {top_model_id}: Superior strategy with {top_stats['win_rate']:.1f}% win rate\n")
                f.write(f"- {second_model_id}: {second_stats['win_rate']:.1f}% win rate\n")
                
                # Compare lie detection
                if top_adv_metrics and second_adv_metrics:
                    top_f1 = top_adv_metrics.get('lie_detection', {}).get('f1_score', 0)
                    second_f1 = second_adv_metrics.get('lie_detection', {}).get('f1_score', 0)
                    if top_f1 > second_f1:
                        f.write(f"- Clear advantage in lie detection: {top_f1:.1f}% vs {second_f1:.1f}% F1 score\n")
            
            # Elo progression
            f.write("\nElo Progression:\n")
            for model_id, stats in sorted_models:
                elo_val = all_adv_metrics.get(model_id, {}).get('elo_rating', 1000)
                elo_change = elo_val - 1000  # Assuming 1000 starting Elo
                sign = "+" if elo_change >= 0 else ""
                f.write(f"- {name_map.get(model_id, model_id)}: 1000 → {elo_val:.0f} ({sign}{elo_change:.0f} points)\n")
            
            # Tournament highlights
            f.write("\nTournament Highlights:\n")
            total_games = len(self.game_results)
            if top_stats['wins'] == total_games:
                f.write("- Perfect tournament with no losses\n")
            elif top_stats['win_rate'] >= 80:
                f.write("- Dominant performance with minimal losses\n")
            elif top_stats['win_rate'] >= 60:
                f.write("- Strong performance with good consistency\n")
            else:
                f.write("- Competitive tournament with close results\n")
            
            f.write(f"- {total_games} games completed successfully\n")
            if avg_rounds < 5:
                f.write("- Quick decisive games with efficient play\n")
            elif avg_rounds > 10:
                f.write("- Extended strategic battles\n")
            else:
                f.write("- Balanced gameplay with strategic depth\n")

            generated_files_dict = None
            if self.create_visualizations:
                print("\nAttempting to generate visualizations...")
                try: generated_files_dict = self.generate_visualizations(base_fn, all_adv_metrics)
                except Exception: print(f"Viz error: {traceback.format_exc()}")
            if self.create_visualizations:
                vis_dir_report_path = f"{base_fn}_visualizations"
                f.write("\n\nENHANCED VISUALIZATIONS\n" + "="*80 + "\n")
                if generated_files_dict and os.path.exists(vis_dir_report_path):
                    f.write(f"Visualizations in {vis_dir_report_path}/\nGenerated files:\n")
                    for k, v_fn in generated_files_dict.items(): f.write(f"- {k}: {v_fn}\n")
                else: f.write("Viz requested, but generation failed or no files produced.\n")
        with open(csv_fn, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            header = ["Model", "Human Name", "Provider", "Elo", "Win Rate", "Wins", "Games", "Bluff Success Rate", "Lie Detection F1", "Rule Adherence"]
            writer.writerow(header)
            for model_id, stats in sorted_models:
                adv_m = all_adv_metrics.get(model_id, {})
                writer.writerow([model_id, name_map.get(model_id,'N/A'), model_id.split('/')[0] if '/' in model_id else 'N/A', adv_m.get('elo_rating',1000.0), stats["win_rate"], stats["wins"], stats["games_played"], adv_m.get('bluff_success_rate',0), adv_m.get('lie_detection',{}).get('f1_score',0), adv_m.get('rule_adherence_rate',0)])
        print(f"\nResults saved to {txt_fn} and {csv_fn}")
        if self.create_visualizations: print(f"Visualizations also saved to {base_fn}_visualizations/" if generated_files_dict and os.path.exists(f"{base_fn}_visualizations") else "Viz requested, but failed.")

    def run_tournament(self, resume_from=None):
        if resume_from and os.path.exists(resume_from):
            print(f"Resuming from {resume_from}"); start_game = self.load_tournament_state(resume_from) + 1
        else:
            if not self.setup_batch(): return
            start_game = 1
        
        # Initialize event logger if enabled
        if self.raw_logging_enabled:
            EventLoggerFactory.set_enabled(True)
            event_logger = EventLoggerFactory.create_logger()
            cli_settings = {
                "total_games": self.total_games,
                "models_per_game": self.models_per_game,
                "selected_models": [m["id"] for m in self.selected_models],
                "use_async": self.use_async,
                "auto_mode": self.auto_mode,
                "verbose_output": self.verbose_output,
                "enable_autosaves": self.enable_autosaves,
                "save_individual_games": self.save_individual_games,
                "create_visualizations": self.create_visualizations,
                "use_local_endpoint": self.use_local_endpoint
            }
            log_tournament_start(cli_settings, random_seed=random.getstate()[1][0])
            print(f"Raw tournament data logging enabled. Data will be saved to: {event_logger.tournament_dir}")
        else:
            EventLoggerFactory.set_enabled(False)
            print("Raw tournament data logging disabled.")
        
        print(f"\nStarting tournament from game {start_game}...")
        game_nums = list(range(start_game, self.total_games + 1))
        if not game_nums: print("No games to run."); self.save_results(); return
        
        try:
            if self.use_async: self.run_games_with_asyncio(game_nums)
            else: self.run_multiple_games_batch(game_nums)
            self.update_leaderboard(); print("\nTournament complete!"); self.display_leaderboard(); self.save_results()
        finally:
            # Log tournament end and close logger
            if self.raw_logging_enabled:
                log_tournament_end({"leaderboard": self.leaderboard, "total_games": len(self.game_results)})
                EventLoggerFactory.close_current()

    def generate_visualizations(self, base_filename, all_advanced_metrics=None):
        vis_dir = f"{base_filename}_visualizations"; os.makedirs(vis_dir, exist_ok=True)
        generated_files = {}
        try: # Elo bar chart
            plt.figure(figsize=(12,6)); models_elo = []; elos_val = []
            for gr in self.game_results: 
                g_obj = gr.get("game_obj")
                if g_obj: 
                    m_data = g_obj.get_metrics()
                    for m,d in m_data.items(): 
                        if "elo_rating" in d: models_elo.append(m); elos_val.append(d["elo_rating"])
            if models_elo: 
                s_idx = np.argsort(elos_val)[::-1]; s_models = [(models_elo[i].split('/')[-1] if '/' in models_elo[i] else models_elo[i]) for i in s_idx]; s_elos = [elos_val[i] for i in s_idx]
                plt.bar(s_models, s_elos, color='skyblue'); plt.title('Elo Ratings'); plt.xlabel('Model'); plt.ylabel('Elo'); plt.xticks(rotation=45,ha='right'); plt.tight_layout()
                fn = "elo_ratings_comparison.png"; plt.savefig(os.path.join(vis_dir,fn)); generated_files["Elo Bar Chart"]=fn
            plt.close()
        except Exception as e: print(f"Elo bar chart error: {e}")
        # Simplified radar chart & MetricsVisualizer calls for restoration
        if all_advanced_metrics: print("Radar chart generation simplified in restore...") # Placeholder
        print("Preparing for MetricsVisualizer...")
        metrics_agg = GameMetrics()
        for m_id in self.leaderboard: metrics_agg.initialize_model(m_id)
        # ... (Simplified metric aggregation for MetricsVisualizer) ...

        # Populate metrics_agg with detailed data before initializing MetricsVisualizer
        # Temporary structures to hold aggregated data
        temp_all_elo_data = defaultdict(list)
        temp_win_matrix_data = defaultdict(lambda: defaultdict(int))

        for gr in self.game_results:
            game_obj = gr.get("game_obj")
            game_num = gr.get("game_number")

            if game_obj and game_num is not None:
                # 1. Aggregate ELO data
                # Prefer using game_obj.metrics.elo_ratings_over_time if it contains (overall_game_num, elo)
                # For this fix, we'll use the robust method of taking final ELO after each game via game_obj.get_metrics()
                # as the exact structure of game_obj.metrics.elo_ratings_over_time isn't confirmed for historical global game numbers.
                game_metrics_dict = game_obj.get_metrics() # Dict: {model_id: {stats}}
                for model_id_in_game, model_stats_dict in game_metrics_dict.items():
                    if 'elo_rating' in model_stats_dict:
                        temp_all_elo_data[model_id_in_game].append((game_num, model_stats_dict['elo_rating']))
                
                # 2. Aggregate Win Matrix data
                winner_model_in_game = gr.get("winner_model")
                player_infos = gr.get("players", []) 
                player_models_in_game = [p['model'] for p in player_infos if 'model' in p]

                if winner_model_in_game and player_models_in_game:
                    for p_model in player_models_in_game:
                        if p_model != winner_model_in_game:
                            temp_win_matrix_data[winner_model_in_game][p_model] += 1
        
        # Process and store aggregated ELO data in metrics_agg
        for model_id, data_points in temp_all_elo_data.items():
            # Assumes metrics_agg.initialize_model might set an initial Elo point e.g. (0, 1000)
            # We combine, sort, and unique-ify
            current_elo_history = metrics_agg.elo_ratings_over_time.get(model_id, [])
            combined_points = current_elo_history + data_points
            
            if combined_points:
                # Sort by game number to ensure chronological order
                # Deduplicate by taking the last entry for a given game_num
                unique_points_dict = {}
                for game_n, elo_val in sorted(combined_points, key=lambda x: x[0]):
                    unique_points_dict[game_n] = elo_val
                
                metrics_agg.elo_ratings_over_time[model_id] = sorted(unique_points_dict.items())


        # Populate win_matrix in metrics_agg
        # This assumes GameMetrics class has a 'win_matrix' attribute of type defaultdict(lambda: defaultdict(int))
        # If not, GameMetrics needs to be adapted, or MetricsVisualizer needs to accept this data differently.
        if hasattr(metrics_agg, 'win_matrix'):
            for winner_model, loser_map in temp_win_matrix_data.items():
                for loser_model, count in loser_map.items():
                    metrics_agg.win_matrix[winner_model][loser_model] = count
        else:
            # If GameMetrics doesn't have a win_matrix, we might need to log a warning or pass it to visualizer if possible.
            # For now, we assume it exists or the visualizer method can handle its absence / get it differently.
            # Alternatively, store it on metrics_agg dynamically if Python allows and visualizer expects it.
            setattr(metrics_agg, 'computed_win_matrix', temp_win_matrix_data) # Example: dynamically add if not predefined
            logger.info("GameMetrics does not have a 'win_matrix' attribute. Storing win data as 'computed_win_matrix'. Heatmap might need adjustment.")


        visualizer = MetricsVisualizer(metrics_agg); mv_prefix = base_filename
        
        # Ensure the directory for MetricsVisualizer outputs exists
        # These methods seem to prepend "metrics_visualizations/" to the path constructed from vis_dir
        metrics_visualizer_output_base_dir = os.path.join("metrics_visualizations", vis_dir)
        os.makedirs(metrics_visualizer_output_base_dir, exist_ok=True)
        
        try:
            for chart_type, method_name_str, f_suffix, desc in [
                ("elo_prog", "generate_elo_rating_chart", "elo_progression.png", "Elo Progression (Enhanced)"),
                ("radar", "generate_metric_comparison_radar", "metrics_radar_enhanced.png", "Metrics Radar (Enhanced)"),
                ("heatmap", "generate_win_matrix_heatmap", "win_matrix_heatmap.png", "Win Matrix (Enhanced)")
            ]:
                method_to_call = getattr(visualizer, method_name_str)
                # The output_file arg is relative to where metrics.py will prepend "metrics_visualizations/"
                # So, we pass "vis_dir/filename_suffix.png"
                # metrics.py is expected to save to "metrics_visualizations/vis_dir/filename_suffix.png"
                output_file_argument = os.path.join(vis_dir, f"{mv_prefix}_{f_suffix}")
                
                path_returned = ""
                if chart_type == "radar":
                    path_returned = method_to_call(output_file=output_file_argument, direct_model_metrics=all_advanced_metrics)
                else:
                    path_returned = method_to_call(output_file=output_file_argument)
                
                if path_returned and os.path.exists(path_returned):
                    generated_files[desc] = os.path.basename(path_returned)
                # else:
                #    logger.warning(f"Visualization file not found at expected path: {path_returned} for {desc}")

        except Exception as e: print(f"MetricsVisualizer error: {traceback.format_exc()}")
        return generated_files