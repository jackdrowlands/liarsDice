# src

## __init__.py
```python
# Liar's Dice package initialization
# This file is required to make the directory a Python package
```

## ai_player.py
```python
import os
import time
import json
import random
import re
from typing import Dict, List, Optional, Any, Union, Tuple, TypedDict, cast
from .player import Player

# Check for required libraries
try:
    import requests
    from requests import Response
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False

# Define API provider constants
PROVIDER_OPENROUTER = "openrouter"
PROVIDER_LOCAL = "local"
GOOGLE_SDK_AVAILABLE = False

# Type definitions for better type checking
class GameState(TypedDict, total=False):
    total_dice: int
    player_dice_counts: Dict[str, int]
    last_bid: Optional[Tuple[int, int]]
    current_player: str
    move_history: List[Dict[str, Any]]
    round_number: int
    provider: str
    response_time: float
    token_usage: Dict[str, int]

class PromptDict(TypedDict):
    system: str
    user: str

class RequestParams(TypedDict):
    endpoint_url: str
    headers: Dict[str, str]
    data: Dict[str, Any]
    game_state: GameState
    provider: str

class AIPlayer(Player):
    def __init__(self, name: str, model: str, provider: str = PROVIDER_OPENROUTER, 
                 api_key: Optional[str] = None, api_url: Optional[str] = None, 
                 project_id: Optional[str] = None, region: Optional[str] = None) -> None:
        super().__init__(name)
        self.model: str = model
        self.game_history: List[GameState] = []
        self.provider: str = provider
        self.api_url: Optional[str] = api_url
        self.project_id: Optional[str] = project_id
        self.region: Optional[str] = region
        
        # Set API key based on provider
        if api_key:
            self.api_key: str = api_key
        elif provider == PROVIDER_OPENROUTER:
            self.api_key = os.environ.get("OPENROUTER_API_KEY", "")
        else:
            self.api_key = ""
        
        # Check if using local endpoint
        self.is_local_endpoint: bool = bool(self.api_url and "127.0.0.1" in self.api_url)

    def record_game_state(self, game_state: GameState) -> None:
        self.game_history.append(game_state)
    
    def get_prompt_for_game(self, game_state: GameState) -> PromptDict:
        """Create a prompt for the AI model using the new format with exact JSON"""
        # Use the player's name as player_id (for the system message)
        player_id = self.name
        
        # Format the system prompt with strong emphasis on JSON output format
        system_prompt = f"""**You are Player {player_id}** in an n-player game of Liar's Dice. The goal is to win by being the last player with dice remaining.

### Game Rules
1. Each player begins with 5 standard six-sided dice (1-6). Dice are private and rerolled at the start of each round.
2. On your turn, you may **make a higher bid** or **call** the previous bid:
   - A bid is a claim like "four 3s" (quantity and face value).
   - Each new bid must either increase the quantity or keep the quantity the same and increase the face.
3. If you **call**, all players reveal their dice.
   - If the total count of dice showing the bid face is **less than the bid**, the last bidder loses one die.
   - Otherwise, the caller loses one die.
4. A player with no dice is eliminated. The game continues until one player remains.
5. Ones (1s) are **not wild**.
6. The dice are re-rolled each round; the player after the last round's loser starts the next round.

### Output Format
You MUST respond with ONLY a single JSON object with these keys, in this exact order:
1. **"reasoning"**: A brief explanation (max ~100 tokens) of your current thought process.
2. **"action"**: Either "bid" or "call".
3. **"quantity"**: An integer >= 0. If calling, set to 0.
4. **"face"**: An integer from 1 to 6. Ignored if calling.
5. **"utterance"**: Up to 50 tokens of what you'd say in-character (e.g., bluff, trash talk, etc.)

Example output:
```json
{{
  "reasoning": "There are still many dice in play, and it's likely someone has at least three 5s.",
  "action": "bid",
  "quantity": 3,
  "face": 5,
  "utterance": "I'm seeing triple fives — how about you all?"
}}
```

CRITICAL: Your entire response MUST be ONLY valid JSON. No text before or after the JSON object. No markdown formatting. No backticks. Just the raw JSON object."""
        
        # Format move history for prompt in a format that shows previous JSON actions and utterances
        move_history_text = ""
        for i, move in enumerate(game_state.get('move_history', [])):
                player_name = move['player']
                
                if move["action"] == "bid":
                    action_summary = f"{player_name}: {{\"action\": \"bid\", \"quantity\": {move['quantity']}, \"face\": {move['face']}}}"
                    # Add utterance if available in the future
                    if "utterance" in move:
                        action_summary += f", \"utterance\": \"{move['utterance']}\""
                    action_summary += "}\n"
                    move_history_text += action_summary
                    
                elif move["action"] == "liar":
                    action_summary = f"{player_name}: {{\"action\": \"call\""
                    # Add utterance if available
                    if "utterance" in move:
                        action_summary += f", \"utterance\": \"{move['utterance']}\""
                    action_summary += "}\n"
                    move_history_text += action_summary
                    
                    # Add outcome information
                    if "outcome" in move:
                        target_player = move.get('target_player', 'previous player')
                        if move["outcome"] == "success":
                            move_history_text += f"Outcome: {player_name} was right! {target_player} lost a die.\n"
                        else:
                            move_history_text += f"Outcome: {player_name} was wrong! {player_name} lost a die.\n"
        
        # Calculate player and dice information
        player_names = list(game_state['player_dice_counts'].keys())
        dice_counts = list(game_state['player_dice_counts'].values())
        
        # Get eliminated players
        eliminated_players = [name for name, count in game_state['player_dice_counts'].items() if count == 0]
        eliminated_text = ", ".join(eliminated_players) if eliminated_players else "None"
        
        # Current bid info
        current_bid = "0 × 0"
        if game_state.get('last_bid') is not None:
            last_bid = game_state['last_bid']
            if last_bid is not None:  # Extra check for type checker
                current_bid = f"{last_bid[0]} × {last_bid[1]}"
        
        # Create the user prompt in the exact format from the request
        user_prompt = f"""### Your private dice (keep secret)
{sorted(self.dice)}

### Game state
Players: {player_names}
Dice counts: {dice_counts}
Current bid: {current_bid}
Eliminated players: {eliminated_text}
Turns so far this round (latest last):
{move_history_text}

### Your turn
It is now your move. Return exactly one JSON object following the format described above."""

        # Return the prompts to be used in the API request
        return {"system": system_prompt, "user": user_prompt}
    
    def get_prompt_and_params(self, game_state: GameState) -> RequestParams:
        """Prepare the API request parameters but don't send yet"""
        # Require Requests
        if not REQUESTS_AVAILABLE:
            raise ImportError("Requests package is not installed. Install with 'pip install requests'")
        
        # No API key needed for local endpoints
        if not self.api_key and not self.is_local_endpoint and self.provider != PROVIDER_LOCAL:
            raise ValueError("API key is required for cloud endpoints")
        
        prompt = self.get_prompt_for_game(game_state)
        
        # OpenRouter and local providers use OpenAI-compatible format
        return self._get_openai_compatible_params(prompt, game_state)
            
    def _get_openai_compatible_params(self, prompt: PromptDict, game_state: GameState) -> RequestParams:
        """Get parameters for OpenAI-compatible APIs (OpenRouter and local)"""
        # Prepare headers
        headers: Dict[str, str] = {
            "Content-Type": "application/json"
        }
        
        # Add authorization for cloud endpoints
        if not self.is_local_endpoint and self.provider == PROVIDER_OPENROUTER:
            headers["Authorization"] = f"Bearer {self.api_key}"
            headers["HTTP-Referer"] = ""
        
        # Prepare request data with the new JSON format for messages
        system_message: Dict[str, str] = {
            "role": "system",
            "content": prompt["system"]
        }
        
        user_message: Dict[str, str] = {
            "role": "user",
            "content": prompt["user"]
        }
        
        # Define the schema for structured output - used for all endpoints
        schema: Dict[str, Any] = {
            "type": "object",
            "properties": {
                "reasoning": {
                    "type": "string",
                    "description": "A brief explanation of your current thought process"
                },
                "action": {
                    "type": "string",
                    "enum": ["bid", "call"],
                    "description": "Whether to make a bid or call the previous bid"
                },
                "quantity": {
                    "type": "integer",
                    "description": "The quantity of dice in your bid (set to 0 if calling)"
                },
                "face": {
                    "type": "integer",
                    "enum": [1, 2, 3, 4, 5, 6],
                    "description": "The face value for your bid (1-6)"
                },
                "utterance": {
                    "type": "string",
                    "description": "What you'd say in-character (e.g., bluff, trash talk, etc.)"
                }
            },
            "required": ["reasoning", "action", "quantity", "face", "utterance"]
        }
        
        # Prepare the base request data
        data: Dict[str, Any] = {
            "model": self.model,
            "messages": [system_message, user_message],
            "temperature": 0.7,  # Balanced temperature for creativity in utterances
        }
        
        # For OpenRouter, include structured output format 
        if self.is_local_endpoint or self.provider == PROVIDER_OPENROUTER:
            # OpenRouter structured output configuration
            data["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": "liars_dice_move",
                    "strict": True,
                    "schema": {
                        "type": "object",
                        "properties": {
                            "reasoning": {
                                "type": "string",
                                "description": "A brief explanation of your current thought process"
                            },
                            "action": {
                                "type": "string",
                                "enum": ["bid", "call"],
                                "description": "Whether to make a bid or call the previous bid"
                            },
                            "quantity": {
                                "type": "integer",
                                "description": "The quantity of dice in your bid (set to 0 if calling)"
                            },
                            "face": {
                                "type": "integer",
                                "description": "The face value for your bid (1-6)"
                            },
                            "utterance": {
                                "type": "string",
                                "description": "What you'd say in-character (e.g., bluff, trash talk, etc.)"
                            }
                        },
                        "required": ["reasoning", "action", "quantity", "face", "utterance"],
                        "additionalProperties": False
                    }
                }
            }
            data["provider"] = {"require_parameters": True}
        
        # For local endpoints, let's add a fallback by not using structured output
        # Some local endpoints might not support this feature yet
        # Instead, we'll rely on our prompt to get a proper JSON response
        
        # Determine the correct endpoint URL
        endpoint_url: str = self.api_url or "https://openrouter.ai/api/v1/chat/completions"
            
        return {
            "endpoint_url": endpoint_url,
            "headers": headers,
            "data": data,
            "game_state": game_state,
            "provider": self.provider
        }
        
    
    def get_ai_decision(self, game_state: GameState) -> Dict[str, Any]:
        """Get a decision from the AI model by calling the API"""
        request_params = self.get_prompt_and_params(game_state)
        provider = request_params["provider"]
        
        # Store provider in game_state for use in process_api_response
        game_state['provider'] = provider
        
        # Capture raw prompt for logging
        prompt_dict = self.get_prompt_for_game(game_state)
        raw_prompt = f"System: {prompt_dict['system']}\n\nUser: {prompt_dict['user']}"
        game_state['raw_prompt'] = raw_prompt
        
        # Handle REST API based calls
        endpoint_url = request_params["endpoint_url"]
        headers = request_params["headers"]
        data = request_params["data"]
        
        try:
            # Make the API request with timing
            start_time = time.time()
            response = requests.post(
                endpoint_url,
                headers=headers,
                json=data,
            )
            response_time = time.time() - start_time
            
            # OpenRouter and local follow OpenAI format
            if response.status_code != 200:
                raise RuntimeError(f"API Error: {response.status_code} - {response.text}")
            
            # Process the response
            return self.process_api_response(response, response_time, game_state)
            
        except Exception as e:
            error_info: Dict[str, Any] = {
                "model": self.model,
                "provider": provider,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "error": str(e),
            }
            
            # Add response if available
            if 'response' in locals():
                try:
                    error_info["response"] = response.json()
                except:
                    error_info["response"] = "Could not parse response as JSON"
            
            # Add prompt information
            try:
                # Get prompt from game state
                prompt = self.get_prompt_for_game(game_state)
                error_info["prompt"] = prompt
            except Exception:
                # If we can't get the prompt, continue without it
                pass
                    
            # Create logs directory if it doesn't exist
            os.makedirs("logs", exist_ok=True)
            
            # Log the error
            with open("logs/invalid_responses.json", "a") as f:
                json.dump(error_info, f)
                f.write("\n")
                
            if 'response' in locals():
                try:
                    print(response.json())
                except:
                    print(f"Response status: {response.status_code}")
                    print(f"Response text: {response.text[:500]}...")
                    
            raise RuntimeError(f"Error getting AI decision from {provider}: {e}")
            
            
    def process_api_response(self, response: 'Response', response_time: float, game_state: GameState) -> Dict[str, Any]:
        """Process the API response with structured output format"""
        # Get the provider from the request params
        provider = game_state.get('provider', self.provider)
        
        # OpenAI-compatible response format (OpenRouter/local)
        result: Dict[str, Any] = response.json()
        
        # Extract token usage if available
        prompt_tokens = completion_tokens = total_tokens = 0
        
        # Check for token usage in response (OpenAI/OpenRouter format)
        if "usage" in result:
            usage = result["usage"]
            prompt_tokens = usage.get("prompt_tokens", 0)
            completion_tokens = usage.get("completion_tokens", 0)
            total_tokens = usage.get("total_tokens", 0)
            
            # Store token usage in game state for metrics collection
            game_state['token_usage'] = {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens
            }
        
        # Get the content from the response which should be a JSON object
        content: str = result["choices"][0]["message"]["content"].strip()
        
        # Store raw response in game_state for logging
        game_state['raw_response'] = content
            
        # Log the response with token information and prompt
        with open("llm_responses.json", "a") as f:
                # Create response log object
                response_log: Dict[str, Any] = {
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "response_text": content,
                    "model": self.model,
                    "provider": provider,
                    "response_time": response_time,
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": total_tokens,
                    "system_prompt": result["choices"][0]["message"].get("system_fingerprint", "")
                }
                
                # Try to add prompt information if available in the request
                try:
                    # Get prompt from game state
                    prompt = self.get_prompt_for_game(game_state)
                    response_log["prompt"] = prompt
                except Exception:
                    # If we can't get the prompt, continue without it
                    pass
                    
                json.dump(response_log, f)
                f.write("\n")
        
        # Store response time in game state for metrics collection
        game_state['response_time'] = response_time
        
        try:
            # Enhanced JSON extraction for both structured and non-structured responses
            
            # First try: direct parse of the whole content (works with structured output)
            try:
                response_json: Dict[str, Any] = json.loads(content)
            except json.JSONDecodeError:
                # Second try: Remove any markdown backticks and try again
                try:
                    # Remove markdown code block syntax if present
                    clean_content = re.sub(r'```(?:json)?\s*|\s*```', '', content)
                    response_json = json.loads(clean_content.strip())
                except json.JSONDecodeError:
                    # Third try: Extract the most promising JSON object with regex
                    # This improved regex handles nested objects better
                    json_objects = re.findall(r'\{(?:[^{}]|(?:\{[^{}]*\}))*\}', content)
                    if not json_objects:
                        raise ValueError("No JSON objects found in the response")
                    
                    # Try each JSON object, starting with the largest one
                    json_objects.sort(key=len, reverse=True)
                    for obj in json_objects:
                        try:
                            response_json = json.loads(obj)
                            # If we get here, parsing succeeded
                            break
                        except json.JSONDecodeError:
                            continue
                    else:
                        # If we get here, no objects parsed successfully
                        raise ValueError("Could not parse any JSON objects in the response")
            
            # Convert action from "call" to "liar" for backward compatibility
            if response_json.get("action") == "call":
                response_json["action"] = "liar"
            
            # Make sure we have all required fields with proper values
            # This should be less necessary with the schema validation, but we'll keep it as a safety check
            
            # Ensure action is valid
            if "action" not in response_json or response_json["action"] not in ["bid", "liar"]:
                response_json["action"] = "bid"  # Default to bid if missing or invalid
                
            # Handle the "liar" action case
            if response_json["action"] == "liar":
                response_json["quantity"] = 0
                if "face" not in response_json or not isinstance(response_json["face"], int):
                    response_json["face"] = 0
            
            # Handle the "bid" action case
            else:
                # Ensure quantity is an integer
                if "quantity" not in response_json or not isinstance(response_json["quantity"], int):
                    response_json["quantity"] = 1
                elif response_json["quantity"] < 1:
                    response_json["quantity"] = 1
                
                # Ensure face is an integer within bounds
                if "face" not in response_json or not isinstance(response_json["face"], int):
                    response_json["face"] = 4
                elif response_json["face"] < 1 or response_json["face"] > 6:
                    # Clamp to valid range
                    response_json["face"] = max(1, min(response_json["face"], 6))
            
            # Ensure reasoning and utterance are present
            if "reasoning" not in response_json or not response_json["reasoning"]:
                response_json["reasoning"] = "Strategic decision based on the game state."
                
            if "utterance" not in response_json or not response_json["utterance"]:
                if response_json["action"] == "bid":
                    response_json["utterance"] = "I'll make this bid."
                else:
                    response_json["utterance"] = "I call liar!"
            
            return response_json
            
        except Exception as e:
            print(f"Error processing response: {e}")
            print(content)
            
            # Log the error with prompt information
            invalid_response: Dict[str, Any] = {
                "model": self.model,
                "provider": provider,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "response_text": content,
                "error": str(e)
            }
            
            # Try to add prompt information
            try:
                # Get prompt from game state
                prompt = self.get_prompt_for_game(game_state)
                invalid_response["prompt"] = prompt
            except Exception:
                # If we can't get the prompt, continue without it
                pass
                
            with open("invalid_llm_responses.json", "a") as f:
                json.dump(invalid_response, f)
                f.write("\n")
                
            # pull out last_bid and narrow its type
            last_bid = game_state.get('last_bid')
            total_dice = game_state.get('total_dice', 1) # Default to 1 if somehow missing, though it should be there.

            if last_bid is None:
                # First bid in the round - make a safe default bid
                fallback_qty = 1
                if total_dice == 0: # No dice in game, technically no bid possible.
                    # This state should ideally be prevented by game logic ending the round/game.
                    # If forced to bid, a (0,0) or (1,1) subject to main game correction might be needed.
                    # For now, make it 1, and LiarsDice class will handle clamping if total_dice is 0.
                    print(f"AI Fallback Warning (first bid): total_dice is 0. Defaulting to 1x4. Game logic should catch this.")
                    fallback_qty = 1 # Or perhaps 0, but action 'bid' with qty 0 is often problematic.
                elif fallback_qty > total_dice and total_dice > 0 : # total_dice must be > 0 for this
                     fallback_qty = total_dice # This case (1 > total_dice > 0) implies total_dice must be 0, already handled.

                return {
                    "action": "bid", 
                    "quantity": fallback_qty, 
                    "face": 4, # Default face
                    "reasoning": "Error processing response, using default bid.",
                    "utterance": "I'll make a simple bid."
                }
            else:
                # Get the last bid
                last_quantity, last_value = last_bid
                
                # If the last bid is implausible (higher than total dice), call liar
                # This check is on the *previous* player's bid.
                if last_quantity > total_dice and total_dice > 0:
                    return {
                        "action": "liar",
                        "quantity": 0,
                        "face": 0,
                        "reasoning": "Error processing response. Last bid exceeds total dice, calling liar.",
                        "utterance": "That's impossible! I call."
                    }
                # Otherwise make a valid higher bid
                else:
                    fallback_reasoning = "Error processing response, making minimal valid higher bid."
                    fallback_utterance_bid = "Let me increase that bid."
                    fallback_utterance_raise = "I'll raise the quantity."

                    if last_value < 6:
                        # Try to increase face value
                        new_quantity = last_quantity
                        
                        # Clamp quantity against total_dice, even if only face is changing.
                        if new_quantity > total_dice and total_dice > 0:
                            print(f"AI Fallback Warning (face increase): Clamping bid quantity from {new_quantity} to {total_dice} (total dice).")
                            new_quantity = total_dice
                        elif new_quantity <= 0 and total_dice > 0: # Ensure quantity is at least 1 if dice exist
                            print(f"AI Fallback Warning (face increase): Corrected quantity from {new_quantity} to 1.")
                            new_quantity = 1
                        elif total_dice == 0: # No dice left, cannot make a bid
                             print(f"AI Fallback Info (face increase): total_dice is 0. Cannot make a bid. Calling liar.")
                             return {
                                "action": "liar", "quantity": 0, "face": 0,
                                "reasoning": "Error processing response. No dice left to make a higher bid. Calling liar.",
                                "utterance": "No dice left! I have to call."
                            }


                        return {
                            "action": "bid", 
                            "quantity": new_quantity, 
                            "face": last_value + 1,
                            "reasoning": fallback_reasoning,
                            "utterance": fallback_utterance_bid
                        }
                    else: # last_value == 6, try to increase quantity
                        new_quantity = last_quantity + 1
                        
                        if new_quantity > total_dice and total_dice > 0:
                            print(f"AI Fallback Warning (quantity increase): Clamping bid quantity from {new_quantity} to {total_dice} (total dice).")
                            new_quantity = total_dice
                        elif new_quantity <= 0 and total_dice > 0: # Ensure quantity is at least 1
                            print(f"AI Fallback Warning (quantity increase): Corrected quantity from {new_quantity} to 1.")
                            new_quantity = 1
                        elif total_dice == 0: # No dice left
                            print(f"AI Fallback Info (quantity increase): total_dice is 0. Cannot make a bid. Calling liar.")
                            return {
                                "action": "liar", "quantity": 0, "face": 0,
                                "reasoning": "Error processing response. No dice left to make a higher bid. Calling liar.",
                                "utterance": "Game's over for bids, I call."
                            }

                        # If, after potential clamping, new_quantity is not strictly greater than last_quantity
                        # (this implies last_quantity was already == total_dice),
                        # then a bid of (new_quantity, 1) is not higher than (last_quantity, 6).
                        # In this situation, the AI cannot make a valid higher bid.
                        if new_quantity <= last_quantity : # True if last_quantity == total_dice and new_quantity got clamped to total_dice
                            print(f"AI Fallback Info: Cannot make a higher bid than ({last_quantity}x{last_value}) with total_dice {total_dice}. Fallback to liar call.")
                            return {
                                "action": "liar",
                                "quantity": 0,
                                "face": 0,
                                "reasoning": "Error processing response. Cannot make a valid higher bid with current dice. Calling liar.",
                                "utterance": "I can't beat that bid, so I'll call!"
                            }

                        return {
                            "action": "bid", 
                            "quantity": new_quantity, 
                            "face": 1, # Reset face to 1
                            "reasoning": fallback_reasoning,
                            "utterance": fallback_utterance_raise
                        }
```

## batch_runner.py
```python
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
logging.getLogger('matplotlib').setLevel(logging.WARNING)

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
        return generated_files```

## event_logger.py
```python
import json
import os
import time
import threading
import queue
import hashlib
import subprocess
from collections import defaultdict
from datetime import datetime
from typing import Dict, Any, Optional, List
import gzip


class EventLogger:
    """
    Non-blocking event logger for Liar's Dice tournaments.
    
    Captures all raw tournament events to enable later replay and analysis
    without requiring re-runs of the benchmark.
    """
    
    def __init__(self, tournament_id: Optional[str] = None, base_path: str = ".", 
                 compression: bool = False, queue_maxsize: int = 10000):
        """
        Initialize the EventLogger.
        
        Args:
            tournament_id: Unique identifier for this tournament
            base_path: Base directory for tournament files
            compression: Whether to use gzip compression for files
            queue_maxsize: Maximum queue size before blocking/fallback
        """
        self.tournament_id = tournament_id or self._generate_tournament_id()
        self.tournament_dir = os.path.join(base_path, f"tournament_{self.tournament_id}")
        self.games_dir = os.path.join(self.tournament_dir, "games")
        self.indices_dir = os.path.join(self.tournament_dir, "indices")
        self.compression = compression
        self.file_extension = ".jsonl.gz" if compression else ".jsonl"
        
        # Thread-safe queue for events
        self.event_queue = queue.Queue(maxsize=queue_maxsize)
        self.queue_maxsize = queue_maxsize
        
        # Writer thread
        self.writer_thread = None
        self.stop_event = threading.Event()
        self.closed = False
        
        # File handles for each game
        self.game_files: Dict[int, Any] = {}
        
        # Tournament metadata
        self.tournament_start_time = None
        self.cli_settings = {}
        self.repo_hash = self._get_repo_hash()
        
        # Thread lock for file operations
        self.file_lock = threading.Lock()
        
        # Initialize directories
        self._create_directories()
        
        # Start writer thread
        self._start_writer_thread()
    
    def _generate_tournament_id(self) -> str:
        """Generate a unique tournament ID."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        random_suffix = hashlib.md5(str(time.time()).encode()).hexdigest()[:8]
        return f"{timestamp}_{random_suffix}"
    
    def _get_repo_hash(self) -> str:
        """Get the current repository hash for reproducibility."""
        try:
            result = subprocess.run(
                ["git", "rev-parse", "HEAD"], 
                capture_output=True, 
                text=True, 
                cwd=os.path.dirname(__file__)
            )
            if result.returncode == 0:
                return result.stdout.strip()
        except Exception:
            pass
        return "unknown"
    
    def _create_directories(self):
        """Create necessary directory structure."""
        os.makedirs(self.tournament_dir, exist_ok=True)
        os.makedirs(self.games_dir, exist_ok=True)
        os.makedirs(self.indices_dir, exist_ok=True)
    
    def _start_writer_thread(self):
        """Start the background writer thread."""
        self.writer_thread = threading.Thread(target=self._writer_loop, daemon=True)
        self.writer_thread.start()
    
    def _writer_loop(self):
        """Main loop for the background writer thread."""
        while not self.stop_event.is_set():
            try:
                # Wait for events with timeout to check stop_event periodically
                event = self.event_queue.get(timeout=1.0)
                if event is None:  # Poison pill to stop
                    break
                    
                self._write_event(event)
                self.event_queue.task_done()
                
            except queue.Empty:
                continue
            except Exception as e:
                # Log error but continue processing
                print(f"Event logger error: {e}")
                continue
        
        # Flush remaining events
        while not self.event_queue.empty():
            try:
                event = self.event_queue.get_nowait()
                if event is not None:
                    self._write_event(event)
                    self.event_queue.task_done()
            except queue.Empty:
                break
            except Exception as e:
                print(f"Event logger flush error: {e}")
                break
    
    def _write_event(self, event: Dict[str, Any]):
        """Write a single event to the appropriate file."""
        event_type = event.get("type")
        
        if event_type in ["tournament_start", "tournament_end"]:
            self._write_meta_event(event)
        elif event_type in ["game_start", "game_end", "round_start", "round_end", "move", "liar_resolution"]:
            game_id = event.get("game_id")
            if game_id is not None:
                self._write_game_event(game_id, event)
    
    def _write_meta_event(self, event: Dict[str, Any]):
        """Write tournament-level events to meta.json."""
        meta_path = os.path.join(self.tournament_dir, "meta.json")
        
        with self.file_lock:
            # Read existing meta data
            meta_data = {}
            if os.path.exists(meta_path):
                try:
                    with open(meta_path, 'r') as f:
                        meta_data = json.load(f)
                except Exception:
                    pass
            
            # Add this event
            if "events" not in meta_data:
                meta_data["events"] = []
            meta_data["events"].append(event)
            
            # Write back to file
            with open(meta_path, 'w') as f:
                json.dump(meta_data, f, indent=2)
    
    def _write_game_event(self, game_id: int, event: Dict[str, Any]):
        """Write game-level events to the appropriate game file."""
        with self.file_lock:
            if game_id not in self.game_files:
                game_filename = f"game_{game_id:04d}{self.file_extension}"
                game_path = os.path.join(self.games_dir, game_filename)
                
                if self.compression:
                    self.game_files[game_id] = gzip.open(game_path, 'wt', encoding='utf-8')
                else:
                    self.game_files[game_id] = open(game_path, 'w', encoding='utf-8')
            
            # Write event as JSON line
            json.dump(event, self.game_files[game_id], separators=(',', ':'))
            self.game_files[game_id].write('\n')
            self.game_files[game_id].flush()
    
    def log_event(self, event: Dict[str, Any]):
        """
        Log an event asynchronously.
        
        Args:
            event: Event dictionary to log
        """
        if self.closed:
            return
        
        # Add timestamp if not present
        if "timestamp" not in event:
            event["timestamp"] = time.time()
        
        try:
            # Try to put event in queue without blocking
            self.event_queue.put_nowait(event)
        except queue.Full:
            # Queue is full - fall back to synchronous write with warning
            print(f"Event logger queue full, falling back to synchronous write")
            try:
                self._write_event(event)
            except Exception as e:
                print(f"Failed to write event synchronously: {e}")
    
    def log_tournament_start(self, cli_settings: Dict[str, Any], random_seed: Optional[int] = None):
        """Log tournament start event."""
        self.tournament_start_time = time.time()
        self.cli_settings = cli_settings.copy()
        
        event = {
            "type": "tournament_start",
            "timestamp": self.tournament_start_time,
            "tournament_id": self.tournament_id,
            "cli_settings": self.cli_settings,
            "random_seed": random_seed,
            "repo_hash": self.repo_hash,
            "compression": self.compression
        }
        self.log_event(event)
    
    def log_tournament_end(self, leaderboard_snapshot: Optional[Dict[str, Any]] = None):
        """Log tournament end event."""
        event = {
            "type": "tournament_end",
            "timestamp": time.time(),
            "tournament_id": self.tournament_id,
            "duration": time.time() - (self.tournament_start_time or 0),
            "leaderboard_snapshot": leaderboard_snapshot or {}
        }
        self.log_event(event)
    
    def log_game_start(self, game_id: int, player_roster: List[Dict[str, Any]], starting_elos: Dict[str, float]):
        """Log game start event."""
        event = {
            "type": "game_start",
            "timestamp": time.time(),
            "game_id": game_id,
            "player_roster": player_roster,
            "starting_elos": starting_elos
        }
        self.log_event(event)
    
    def log_game_end(self, game_id: int, final_winner: str, rounds_played: int):
        """Log game end event."""
        event = {
            "type": "game_end",
            "timestamp": time.time(),
            "game_id": game_id,
            "final_winner": final_winner,
            "rounds_played": rounds_played
        }
        self.log_event(event)
    
    def log_round_start(self, game_id: int, round_number: int, total_dice_count: int):
        """Log round start event."""
        event = {
            "type": "round_start",
            "timestamp": time.time(),
            "game_id": game_id,
            "round_number": round_number,
            "total_dice_count": total_dice_count
        }
        self.log_event(event)
    
    def log_round_end(self, game_id: int, round_number: int, surviving_players: List[str]):
        """Log round end event."""
        event = {
            "type": "round_end",
            "timestamp": time.time(),
            "game_id": game_id,
            "round_number": round_number,
            "surviving_players": surviving_players
        }
        self.log_event(event)
    
    def log_move(self, game_id: int, round_number: int, player_snapshot: Dict[str, Any], 
                 actor_name: str, actor_model: str, raw_prompt: str, raw_model_response: str,
                 parsed_action: str, parsed_quantity: Optional[int], parsed_face: Optional[int],
                 utterance: str, response_time: float, token_usage: Dict[str, int],
                 invalid_bid_corrected: bool = False, original_action: Optional[str] = None,
                 original_quantity: Optional[int] = None, original_face: Optional[int] = None):
        """Logs a player's move, including AI's raw thought process and response."""
        if not self.is_enabled():
            return

        event_data: Dict[str, Any] = {
            "type": "move",
            "timestamp": time.time(),
            "game_id": game_id,
            "round_number": round_number,
            "player_snapshot": player_snapshot,
            "actor_name": actor_name,
            "actor_model": actor_model,
            "raw_prompt": raw_prompt,
            "raw_model_response": raw_model_response,
            "parsed_action": parsed_action,
            "parsed_quantity": parsed_quantity,
            "parsed_face": parsed_face,
            "utterance": utterance,
            "response_time": response_time,
            "token_usage": token_usage,
            "invalid_bid_corrected": invalid_bid_corrected
        }
        if invalid_bid_corrected:
            # Only add original fields if a correction actually occurred
            if original_action is not None:
                event_data["original_action"] = original_action
            if original_quantity is not None:
                event_data["original_quantity"] = original_quantity
            if original_face is not None:
                event_data["original_face"] = original_face
        
        self.log_event(event_data)
    
    def log_liar_resolution(self, game_id: int, round_number: int, calling_player: str,
                           target_player: str, last_bid: tuple, actual_face_count: int,
                           outcome: str, all_players_dice: Dict[str, List[int]]):
        """Log liar call resolution event."""
        event = {
            "type": "liar_resolution",
            "timestamp": time.time(),
            "game_id": game_id,
            "round_number": round_number,
            "calling_player": calling_player,
            "target_player": target_player,
            "last_bid": {"quantity": last_bid[0], "face": last_bid[1]},
            "actual_face_count": actual_face_count,
            "outcome": outcome,
            "all_players_dice": all_players_dice
        }
        self.log_event(event)
    
    def close(self, timeout: float = 2.0):
        """
        Close the event logger and flush all pending events.
        
        Args:
            timeout: Maximum time to wait for queue to flush
        """
        if self.closed:
            return
        
        self.closed = True
        
        # Signal writer thread to stop
        self.stop_event.set()
        
        # Add poison pill to queue
        try:
            self.event_queue.put_nowait(None)
        except queue.Full:
            pass
        
        # Wait for writer thread to finish
        if self.writer_thread and self.writer_thread.is_alive():
            self.writer_thread.join(timeout=timeout)
        
        # Close all game files
        with self.file_lock:
            for game_file in self.game_files.values():
                try:
                    game_file.close()
                except Exception:
                    pass
            self.game_files.clear()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


class EventLoggerFactory:
    """Factory for creating and managing EventLogger instances."""
    
    _instance: Optional[EventLogger] = None
    _enabled: bool = True
    
    @classmethod
    def create_logger(cls, tournament_id: Optional[str] = None, **kwargs) -> EventLogger:
        """Create a new EventLogger instance."""
        if cls._instance:
            cls._instance.close()
        
        cls._instance = EventLogger(tournament_id=tournament_id, **kwargs)
        return cls._instance
    
    @classmethod
    def get_logger(cls) -> Optional[EventLogger]:
        """Get the current EventLogger instance."""
        return cls._instance if cls._enabled else None
    
    @classmethod
    def set_enabled(cls, enabled: bool):
        """Enable or disable event logging globally."""
        cls._enabled = enabled
        if not enabled and cls._instance:
            cls._instance.close()
            cls._instance = None
    
    @classmethod
    def is_enabled(cls) -> bool:
        """Check if event logging is enabled."""
        return cls._enabled
    
    @classmethod
    def close_current(cls):
        """Close the current logger instance."""
        if cls._instance:
            cls._instance.close()
            cls._instance = None


def log_if_enabled(func):
    """Decorator to only execute logging if enabled."""
    def wrapper(*args, **kwargs):
        logger = EventLoggerFactory.get_logger()
        if logger:
            return func(logger, *args, **kwargs)
        return None
    return wrapper


# Convenience functions for logging events
@log_if_enabled
def log_tournament_start(logger: EventLogger, cli_settings: Dict[str, Any], random_seed: Optional[int] = None):
    logger.log_tournament_start(cli_settings, random_seed)

@log_if_enabled
def log_tournament_end(logger: EventLogger, leaderboard_snapshot: Optional[Dict[str, Any]] = None):
    logger.log_tournament_end(leaderboard_snapshot)

@log_if_enabled
def log_game_start(logger: EventLogger, game_id: int, player_roster: List[Dict[str, Any]], starting_elos: Dict[str, float]):
    logger.log_game_start(game_id, player_roster, starting_elos)

@log_if_enabled
def log_game_end(logger: EventLogger, game_id: int, final_winner: str, rounds_played: int):
    logger.log_game_end(game_id, final_winner, rounds_played)

@log_if_enabled
def log_round_start(logger: EventLogger, game_id: int, round_number: int, total_dice_count: int):
    logger.log_round_start(game_id, round_number, total_dice_count)

@log_if_enabled
def log_round_end(logger: EventLogger, game_id: int, round_number: int, surviving_players: List[str]):
    logger.log_round_end(game_id, round_number, surviving_players)

@log_if_enabled
def log_move(logger: EventLogger, game_id: int, round_number: int, player_snapshot: Dict[str, Any], 
             actor_name: str, actor_model: str, raw_prompt: str, raw_model_response: str,
             parsed_action: str, parsed_quantity: Optional[int], parsed_face: Optional[int],
             utterance: str, response_time: float, token_usage: Dict[str, int],
             invalid_bid_corrected: bool = False, original_action: Optional[str] = None,
             original_quantity: Optional[int] = None, original_face: Optional[int] = None):
    """Wrapper for EventLogger.log_move for easier procedural calls."""
    logger.log_move(game_id, round_number, player_snapshot, actor_name, actor_model,
                    raw_prompt, raw_model_response, parsed_action, parsed_quantity,
                    parsed_face, utterance, response_time, token_usage,
                    invalid_bid_corrected, original_action, original_quantity, original_face)

@log_if_enabled
def log_liar_resolution(logger: EventLogger, game_id: int, round_number: int, calling_player: str,
                       target_player: str, last_bid: tuple, actual_face_count: int,
                       outcome: str, all_players_dice: Dict[str, List[int]]):
    logger.log_liar_resolution(game_id, round_number, calling_player, target_player, 
                              last_bid, actual_face_count, outcome, all_players_dice)```

## event_schema.py
```python
"""
Event schema definitions and validation for the Liar's Dice tournament raw-data capture layer.

This module defines the exact structure of all events that can be logged during tournaments
and provides validation functions to ensure data integrity.
"""

from typing import Dict, Any, List, Optional, Union
import time
import json


class EventValidationError(Exception):
    """Raised when an event fails validation."""
    pass


# Base event schema - all events must have these fields
BASE_EVENT_SCHEMA = {
    "type": str,
    "timestamp": (int, float),
}

# Tournament-level events
TOURNAMENT_START_SCHEMA = {
    **BASE_EVENT_SCHEMA,
    "tournament_id": str,
    "cli_settings": dict,
    "random_seed": (int, type(None)),
    "repo_hash": str,
    "compression": bool,
}

TOURNAMENT_END_SCHEMA = {
    **BASE_EVENT_SCHEMA,
    "tournament_id": str,
    "duration": (int, float),
    "leaderboard_snapshot": dict,
}

# Game-level events
GAME_START_SCHEMA = {
    **BASE_EVENT_SCHEMA,
    "game_id": int,
    "player_roster": list,
    "starting_elos": dict,
}

GAME_END_SCHEMA = {
    **BASE_EVENT_SCHEMA,
    "game_id": int,
    "final_winner": str,
    "rounds_played": int,
}

# Round-level events
ROUND_START_SCHEMA = {
    **BASE_EVENT_SCHEMA,
    "game_id": int,
    "round_number": int,
    "total_dice_count": int,
}

ROUND_END_SCHEMA = {
    **BASE_EVENT_SCHEMA,
    "game_id": int,
    "round_number": int,
    "surviving_players": list,
}

# Move-level events
MOVE_SCHEMA = {
    **BASE_EVENT_SCHEMA,
    "game_id": int,
    "round_number": int,
    "player_snapshot": dict,
    "actor_name": str,
    "actor_model": str,
    "raw_prompt": str,
    "raw_model_response": str,
    "parsed_action": str,
    "parsed_quantity": (int, type(None)),
    "parsed_face": (int, type(None)),
    "utterance": str,
    "response_time": (int, float),
    "token_usage": dict,
    # Optional fields for invalid bid correction tracking
    "invalid_bid_corrected": bool,
    "original_quantity": (int, type(None)),
    "original_face": (int, type(None)),
}

# Liar resolution events
LIAR_RESOLUTION_SCHEMA = {
    **BASE_EVENT_SCHEMA,
    "game_id": int,
    "round_number": int,
    "calling_player": str,
    "target_player": str,
    "last_bid": dict,
    "actual_face_count": int,
    "outcome": str,
    "all_players_dice": dict,
}

# Map event types to their schemas
EVENT_SCHEMAS = {
    "tournament_start": TOURNAMENT_START_SCHEMA,
    "tournament_end": TOURNAMENT_END_SCHEMA,
    "game_start": GAME_START_SCHEMA,
    "game_end": GAME_END_SCHEMA,
    "round_start": ROUND_START_SCHEMA,
    "round_end": ROUND_END_SCHEMA,
    "move": MOVE_SCHEMA,
    "liar_resolution": LIAR_RESOLUTION_SCHEMA,
}


def validate_event_type(event_type: str) -> bool:
    """Validate that the event type is recognized."""
    return event_type in EVENT_SCHEMAS


def validate_field_type(value: Any, expected_type: Union[type, tuple]) -> bool:
    """Validate that a field value matches the expected type(s)."""
    if isinstance(expected_type, tuple):
        return isinstance(value, expected_type)
    return isinstance(value, expected_type)


def validate_player_roster(roster: List[Dict[str, Any]]) -> bool:
    """Validate player roster structure."""
    if not isinstance(roster, list):
        return False
    
    for player in roster:
        if not isinstance(player, dict):
            return False
        
        required_fields = ["name", "model_id", "provider", "initial_dice"]
        for field in required_fields:
            if field not in player:
                return False
        
        # Validate initial_dice is a non-negative integer
        if not isinstance(player["initial_dice"], int) or player["initial_dice"] < 0:
            return False
    
    return True


def validate_player_snapshot(snapshot: Dict[str, Any]) -> bool:
    """Validate player snapshot structure for moves."""
    if not isinstance(snapshot, dict):
        return False
    
    required_fields = [
        "player_order", "dice_by_player", "counts", "total_dice",
        "last_bid", "current_player", "round", "game_id"
    ]
    
    for field in required_fields:
        if field not in snapshot:
            return False
    
    # Validate specific field types
    if not isinstance(snapshot["player_order"], list):
        return False
    
    if not isinstance(snapshot["dice_by_player"], dict):
        return False
    
    if not isinstance(snapshot["counts"], dict):
        return False
    
    if not isinstance(snapshot["total_dice"], int) or snapshot["total_dice"] < 0:
        return False
    
    # last_bid can be None or a tuple/list of length 2
    last_bid = snapshot["last_bid"]
    if last_bid is not None:
        if not isinstance(last_bid, (list, tuple)) or len(last_bid) != 2:
            return False
        if not all(isinstance(x, int) for x in last_bid):
            return False
    
    return True


def validate_token_usage(token_usage: Dict[str, int]) -> bool:
    """Validate token usage structure."""
    if not isinstance(token_usage, dict):
        return False
    
    expected_fields = ["prompt_tokens", "completion_tokens", "total_tokens"]
    for field in expected_fields:
        if field not in token_usage:
            return False
        if not isinstance(token_usage[field], int) or token_usage[field] < 0:
            return False
    
    # Validate that total_tokens equals prompt + completion
    if token_usage["total_tokens"] != (token_usage["prompt_tokens"] + token_usage["completion_tokens"]):
        return False
    
    return True


def validate_last_bid_dict(last_bid: Dict[str, int]) -> bool:
    """Validate last_bid dictionary structure in liar resolution."""
    if not isinstance(last_bid, dict):
        return False
    
    required_fields = ["quantity", "face"]
    for field in required_fields:
        if field not in last_bid:
            return False
        if not isinstance(last_bid[field], int):
            return False
    
    # Validate ranges
    if last_bid["quantity"] <= 0:
        return False
    if not (1 <= last_bid["face"] <= 6):
        return False
    
    return True


def validate_all_players_dice(all_dice: Dict[str, List[int]]) -> bool:
    """Validate all_players_dice structure in liar resolution."""
    if not isinstance(all_dice, dict):
        return False
    
    for player_name, dice in all_dice.items():
        if not isinstance(player_name, str):
            return False
        if not isinstance(dice, list):
            return False
        for die in dice:
            if not isinstance(die, int) or not (1 <= die <= 6):
                return False
    
    return True


def validate_event(event: Dict[str, Any]) -> bool:
    """
    Validate a complete event against its schema.
    
    Args:
        event: Event dictionary to validate
    
    Returns:
        True if valid, False otherwise
        
    Raises:
        EventValidationError: If validation fails with details
    """
    # Check if event has required type field
    if "type" not in event:
        raise EventValidationError("Event missing required 'type' field")
    
    event_type = event["type"]
    
    # Check if event type is recognized
    if not validate_event_type(event_type):
        raise EventValidationError(f"Unknown event type: {event_type}")
    
    schema = EVENT_SCHEMAS[event_type]
    
    # For move events, some fields are optional
    optional_fields = set()
    if event_type == "move":
        optional_fields = {"invalid_bid_corrected", "original_quantity", "original_face"}
    
    # Check all required fields are present and have correct types
    for field, expected_type in schema.items():
        if field in optional_fields and field not in event:
            continue  # Skip optional fields that are not present
            
        if field not in event:
            raise EventValidationError(f"Event '{event_type}' missing required field: {field}")
        
        if not validate_field_type(event[field], expected_type):
            raise EventValidationError(
                f"Event '{event_type}' field '{field}' has incorrect type. "
                f"Expected {expected_type}, got {type(event[field])}"
            )
    
    # Perform additional validation based on event type
    if event_type == "game_start":
        if not validate_player_roster(event["player_roster"]):
            raise EventValidationError("Invalid player_roster structure")
    
    elif event_type == "move":
        if not validate_player_snapshot(event["player_snapshot"]):
            raise EventValidationError("Invalid player_snapshot structure")
        
        if not validate_token_usage(event["token_usage"]):
            raise EventValidationError("Invalid token_usage structure")
        
        # Validate parsed_action values
        valid_actions = ["bid", "liar"]
        if event["parsed_action"] not in valid_actions:
            raise EventValidationError(f"Invalid parsed_action: {event['parsed_action']}")
        
        # For bid actions, quantity and face should be present
        if event["parsed_action"] == "bid":
            if event["parsed_quantity"] is None or event["parsed_face"] is None:
                raise EventValidationError("Bid action requires parsed_quantity and parsed_face")
            if event["parsed_quantity"] <= 0:
                raise EventValidationError("Bid quantity must be positive")
            if not (1 <= event["parsed_face"] <= 6):
                raise EventValidationError("Bid face must be between 1 and 6")
        
        # Validate invalid bid correction fields if present
        if "invalid_bid_corrected" in event:
            if event["invalid_bid_corrected"]:
                # If bid was corrected, original fields should be present
                if "original_quantity" not in event or "original_face" not in event:
                    raise EventValidationError("Invalid bid correction requires original_quantity and original_face")
                # Note: We don't validate original_quantity and original_face ranges here because
                # they are allowed to be invalid values - that's the whole point of tracking corrections!
    
    elif event_type == "liar_resolution":
        if not validate_last_bid_dict(event["last_bid"]):
            raise EventValidationError("Invalid last_bid structure")
        
        if not validate_all_players_dice(event["all_players_dice"]):
            raise EventValidationError("Invalid all_players_dice structure")
        
        # Validate outcome values
        valid_outcomes = ["success", "failure"]
        if event["outcome"] not in valid_outcomes:
            raise EventValidationError(f"Invalid outcome: {event['outcome']}")
        
        if event["actual_face_count"] < 0:
            raise EventValidationError("actual_face_count must be non-negative")
    
    elif event_type == "tournament_start":
        # Validate timestamp is reasonable (not too far in past/future)
        current_time = time.time()
        if abs(event["timestamp"] - current_time) > 86400:  # More than 1 day difference
            raise EventValidationError("tournament_start timestamp seems unreasonable")
    
    return True


def validate_jsonl_file(file_path: str) -> List[str]:
    """
    Validate a JSONL file containing events.
    
    Args:
        file_path: Path to the JSONL file
    
    Returns:
        List of validation errors (empty if all valid)
    """
    errors = []
    
    try:
        with open(file_path, 'r') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                
                try:
                    event = json.loads(line)
                    validate_event(event)
                except json.JSONDecodeError as e:
                    errors.append(f"Line {line_num}: Invalid JSON: {e}")
                except EventValidationError as e:
                    errors.append(f"Line {line_num}: {e}")
                except Exception as e:
                    errors.append(f"Line {line_num}: Unexpected error: {e}")
    
    except IOError as e:
        errors.append(f"Cannot read file: {e}")
    
    return errors


def validate_tournament_directory(tournament_dir: str) -> Dict[str, List[str]]:
    """
    Validate all files in a tournament directory.
    
    Args:
        tournament_dir: Path to the tournament directory
    
    Returns:
        Dictionary mapping file paths to lists of validation errors
    """
    import os
    import glob
    
    results = {}
    
    # Validate meta.json if it exists
    meta_path = os.path.join(tournament_dir, "meta.json")
    if os.path.exists(meta_path):
        try:
            with open(meta_path, 'r') as f:
                meta_data = json.load(f)
            
            # Validate events in meta.json
            errors = []
            for event in meta_data.get("events", []):
                try:
                    validate_event(event)
                except EventValidationError as e:
                    errors.append(str(e))
            
            results[meta_path] = errors
        except Exception as e:
            results[meta_path] = [f"Error reading meta.json: {e}"]
    
    # Validate all game files
    games_dir = os.path.join(tournament_dir, "games")
    if os.path.exists(games_dir):
        for game_file in glob.glob(os.path.join(games_dir, "game_*.jsonl*")):
            errors = validate_jsonl_file(game_file)
            results[game_file] = errors
    
    return results


# Convenience functions for creating valid events
def create_tournament_start_event(tournament_id: str, cli_settings: Dict[str, Any], 
                                 random_seed: Optional[int] = None, repo_hash: str = "unknown",
                                 compression: bool = False) -> Dict[str, Any]:
    """Create a valid tournament_start event."""
    return {
        "type": "tournament_start",
        "timestamp": time.time(),
        "tournament_id": tournament_id,
        "cli_settings": cli_settings,
        "random_seed": random_seed,
        "repo_hash": repo_hash,
        "compression": compression,
    }


def create_game_start_event(game_id: int, player_roster: List[Dict[str, Any]], 
                           starting_elos: Dict[str, float]) -> Dict[str, Any]:
    """Create a valid game_start event."""
    return {
        "type": "game_start",
        "timestamp": time.time(),
        "game_id": game_id,
        "player_roster": player_roster,
        "starting_elos": starting_elos,
    }


def create_move_event(game_id: int, round_number: int, player_snapshot: Dict[str, Any],
                     actor_name: str, actor_model: str, raw_prompt: str, 
                     raw_model_response: str, parsed_action: str, 
                     parsed_quantity: Optional[int], parsed_face: Optional[int],
                     utterance: str, response_time: float, 
                     token_usage: Dict[str, int]) -> Dict[str, Any]:
    """Create a valid move event."""
    return {
        "type": "move",
        "timestamp": time.time(),
        "game_id": game_id,
        "round_number": round_number,
        "player_snapshot": player_snapshot,
        "actor_name": actor_name,
        "actor_model": actor_model,
        "raw_prompt": raw_prompt,
        "raw_model_response": raw_model_response,
        "parsed_action": parsed_action,
        "parsed_quantity": parsed_quantity,
        "parsed_face": parsed_face,
        "utterance": utterance,
        "response_time": response_time,
        "token_usage": token_usage,
    }```

## liars_dice.py
```python
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
    
    def _log_move_if_enabled(self, game_num, player, game_state, decision, 
                            invalid_bid_corrected=False, original_action=None, 
                            original_quantity=None, original_face=None):
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
            utterance, response_time, token_usage,
            invalid_bid_corrected, original_action, original_quantity, original_face
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
    
    def _process_ai_liar_call(self, player, decision_from_ai):
        original_ai_reasoning = decision_from_ai.get("reasoning", "No reasoning provided")
        original_ai_utterance = decision_from_ai.get("utterance", "I call liar!")
        
        if not self.last_bid:
            # AI shouldn't call liar on first turn, make a bid instead.
            # This is an invalid action that needs correction.
            
            corrected_qty = 1
            corrected_face = random.randint(1, 6) if player.get_dice_count() > 0 else 4
            
            self.last_bid = (corrected_qty, corrected_face)
            
            self.metrics.record_rule_adherence(player.model, False) # Rule violation
            
            # Update the decision object that will be used by _log_move_if_enabled
            # Use neutral phrasing for what goes into the JSONL log, as if AI intended this opening bid.
            decision_from_ai['action'] = "bid" 
            decision_from_ai['quantity'] = corrected_qty
            decision_from_ai['face'] = corrected_face
            decision_from_ai['reasoning'] = "Making a standard opening bid."
            decision_from_ai['utterance'] = f"I'll start us off with {corrected_qty} {corrected_face}'s."
            decision_from_ai['correction_made_internally_from_liar_to_bid'] = True

            # For console display and internal history, be explicit about the correction.
            reason_for_correction_display = f"AI attempted to call liar on first turn (invalid action). Corrected to a bid of {corrected_qty} {corrected_face}'s."
            print(f"{player.name} tried to call liar on first turn. Corrected to bid: {corrected_qty} {corrected_face}'s.")
            print(f"Reasoning for correction: {reason_for_correction_display}")

            # Construct move_data for self.move_history (internal game log)
            # This retains the original AI thought process and the actual correction reason.
            move_data_for_history = {
                "round": self.round_number,
                "player": player.name,
                "action": "bid", # The action that actually occurred
                "quantity": corrected_qty,
                "face": corrected_face, 
                "reasoning": reason_for_correction_display, # Internal log shows why correction happened
                "utterance": decision_from_ai['utterance'], # Use the neutral utterance for consistency here too
                "invalid_action_corrected": True, 
                "original_intended_action": "liar",
                "original_ai_reasoning": original_ai_reasoning,
                "original_ai_utterance": original_ai_utterance
            }
            self.move_history.append(move_data_for_history)
            if player.name in self.player_history:
                 self.player_history[player.name].append(move_data_for_history)
            else: # Should ideally not happen if player_history is initialized properly
                 self.player_history[player.name] = [move_data_for_history]

            return False # Signifies that a liar call did not proceed as planned.
        
        # If self.last_bid exists, proceed with normal liar call logic
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
            "round": self.round_number,
            "player": player.name,
            "action": "liar",
            "target_player": self.players[(self.current_player_idx - 1) % len(self.players)].name,
            "is_optimal": is_optimal_call,
            "reasoning": original_ai_reasoning,
            "utterance": original_ai_utterance
        }
        self.move_history.append(move_data)
        
        # Track in player history
        self.player_history[player.name].append(move_data)
        
        print(f"{player.name} calls 'Liar!' on the previous bid.")
        print(f"Reasoning: {original_ai_reasoning}")
        print(f"{player.name} says: \"{original_ai_utterance}\"")
        
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
        if not (isinstance(quantity, int) and quantity >= 1 and
                isinstance(value, int) and 1 <= value <= 6):
            valid_bid = False
        
        if valid_bid and quantity > self.total_dice_in_game: # Check quantity against total dice
            print(f"Debug: AI bid quantity {quantity} exceeds total dice {self.total_dice_in_game}. Marking invalid.")
            valid_bid = False
            # Optionally, store a more specific reason if your logging needs it
            # decision['invalid_reason'] = "Bid quantity exceeds total dice in game."
        
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
        
        # Record rule adherence based on initial validation (including total_dice check)
        if isinstance(player, AIPlayer):
            self.metrics.record_rule_adherence(player.model, valid_bid)
        
        # Is this likely a bluff?
        player_dice_count = player.dice.count(value)
        is_bluff = player_dice_count < quantity / 2 if valid_bid else False # Only assess bluff if bid could be valid
        
        if valid_bid:
            self.last_bid = (quantity, value)
            
            # Record bid optimality
            if isinstance(player, AIPlayer):
                self.metrics.record_bid_optimality(player.model, is_optimal_bid)
                
            # Record move in history with bluff info and new fields
            move_data = {
                "round": self.round_number,
                "player": player.name,
                "action": "bid",
                "quantity": quantity,
                "face": value, # Changed from "value" to "face"
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
            # AI made an invalid bid (could be too low, bad format, or qty > total_dice)
            # Attempt to make a safe, valid, corrected bid.
            
            corrected_quantity_for_state = 0
            corrected_face_for_state = 0

            if self.last_bid: # If there was a previous bid to base correction on
                prev_q_state, prev_f_state = self.last_bid
                
                if prev_f_state < 6:
                    corrected_quantity_for_state = prev_q_state
                    corrected_face_for_state = prev_f_state + 1
                else: # prev_f_state == 6
                    corrected_quantity_for_state = prev_q_state + 1
                    corrected_face_for_state = 1
            else: # No previous bid, AI's first bid was invalid (e.g. qty=0 or qty > total_dice initially)
                corrected_quantity_for_state = 1
                # Use a common face like 1 or a random one for a corrected first bid
                corrected_face_for_state = random.randint(1,6) 

            # CRITICAL: Clamp the corrected quantity against total_dice_in_game
            if corrected_quantity_for_state > self.total_dice_in_game:
                print(f"  Correction Clamping: Auto-corrected bid quantity from {corrected_quantity_for_state} to {self.total_dice_in_game} (total dice).")
                corrected_quantity_for_state = self.total_dice_in_game
            
            # If clamping made quantity 0 (e.g., total_dice_in_game was 0, though game logic should prevent this state usually)
            # or if the corrected bid is not actually higher than the last bid (e.g. prev was (4,6), total 4, corrected (4,1) not higher)
            # this correction logic might still be imperfect for game flow, but it ensures self.last_bid is sane.
            if corrected_quantity_for_state == 0 and self.total_dice_in_game > 0:
                print(f"  Correction Warning: Auto-corrected bid quantity became 0 despite total_dice={self.total_dice_in_game}. Forcing to minimal (1,1). This may not be a valid higher bid.")
                corrected_quantity_for_state = 1 
                corrected_face_for_state = 1 # Needs to be higher than self.last_bid if possible

            # Ensure the corrected bid is actually higher than the original self.last_bid if self.last_bid existed.
            # If not, this "correction" might lead to a stuck game state or invalid progression.
            # This is a deeper issue with "simple" correction. For now, we prioritize a sane self.last_bid state.
            if self.last_bid:
                prev_q_state, prev_f_state = self.last_bid
                if not (corrected_quantity_for_state > prev_q_state or \
                       (corrected_quantity_for_state == prev_q_state and corrected_face_for_state > prev_f_state)):
                    print(f"  Correction Warning: Final corrected bid ({corrected_quantity_for_state}x{corrected_face_for_state}) is not higher than previous actual bid ({prev_q_state}x{prev_f_state}). AI may be stuck.")
                    # To avoid game getting stuck, if correction is not higher, what to do?
                    # The original code just set it. We are at least clamping quantity.
                    # The current primary goal is to ensure self.last_bid has quantity <= total_dice.


            if corrected_quantity_for_state > 0 : # Only set if a positive quantity bid could be formed
                self.last_bid = (corrected_quantity_for_state, corrected_face_for_state)
            else:
                # This case means total_dice_in_game is likely 0, or correction failed badly.
                # A bid of 0 is not allowed. The game should likely have ended.
                # If forced to make *some* bid, and previous was (say) (1,1) and total_dice became 0,
                # it's an impossible situation for _process_ai_bid.
                # For robustness, if self.last_bid wasn't updated, ensure it reflects something,
                # or accept that the AI turn might effectively be skipped if no valid state can be set.
                # However, the function expects to set self.last_bid and return False for a bid.
                # If total_dice_in_game > 0, we must set a bid.
                if self.total_dice_in_game > 0:
                    print(f"  Correction Critical Warning: Corrected quantity is {corrected_quantity_for_state} with total dice {self.total_dice_in_game}. Defaulting to (1,1) as last resort.")
                    self.last_bid = (1,1) # This is a last resort and might not be a valid *higher* bid.
                else:
                    # If total_dice_in_game is 0, no bid can be made.
                    # This state should ideally be caught before calling _process_ai_bid.
                    # For now, leave self.last_bid as is or handle as an error state.
                    # Let's assume if total_dice_in_game is 0, _process_ai_bid shouldn't be called
                    # or game should end. If it IS called, and we must set self.last_bid:
                    pass # Rely on self.last_bid being (1,1) if this block is reached from above logic.


            # Record in metrics that AI made an invalid bid (already done above by valid_bid check)
            # if isinstance(player, AIPlayer):
            #     self.metrics.record_rule_adherence(player.model, False) # This was already done
            
            # The `quantity` and `value` below are the AI's *original* (invalid) bid.
            # The `self.last_bid[0]` and `self.last_bid[1]` are the *corrected* values for the log.
            corrected_utterance = "I make this bid." # Generic utterance for a corrected bid
            corrected_reasoning = f"AI original bid ({quantity}x{value}) was invalid. Game corrected to {self.last_bid[0]}x{self.last_bid[1]}."
            
            # Record move in history with the invalid bid info
            move_data = {
                "round": self.round_number,
                "player": player.name,
                "action": "bid",
                "quantity": self.last_bid[0], # Log the corrected quantity
                "face": self.last_bid[1],    # Log the corrected value
                "invalid_bid_corrected": True,
                "original_quantity": quantity, # Log the AI's original attempted quantity
                "original_face": value,       # Log the AI's original attempted value
                "reasoning": corrected_reasoning, # Explain the correction
                "utterance": corrected_utterance
            }
            self.move_history.append(move_data)
            
            # Track in player history
            self.player_history[player.name].append(move_data)
            
            print(f"{player.name} attempted invalid bid ({quantity} {value}'s), corrected to {self.last_bid[0]} {self.last_bid[1]}'s")
            print(f"Reasoning: {corrected_reasoning}")
            print(f"{player.name} says: \"{corrected_utterance}\"")
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
                decision = player.get_ai_decision(game_state) # This is the raw AI decision
                
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
                
                # Capture initial AI output for logging originals if a correction occurs
                initial_ai_action = decision.get("action")
                initial_ai_quantity = decision.get("quantity") # Can be None if action is 'liar'
                initial_ai_face = decision.get("face")       # Can be None if action is 'liar'

                # Variables to store original values if a correction occurs
                final_original_action_for_log = None
                final_original_quantity_for_log = None
                final_original_face_for_log = None
                
                invalid_bid_corrected_for_logger = False 
                
                # This 'processed_decision' object starts as the AI's raw decision
                # and can be modified by _process_ai_liar_call or _process_ai_bid if corrections occur.
                # It represents the decision that the game engine will ultimately use.
                processed_decision = decision 

                if initial_ai_action == "liar":
                    # _process_ai_liar_call might modify processed_decision if it's an invalid liar call (e.g. first turn)
                    result = self._process_ai_liar_call(player, processed_decision) 
                    
                    if processed_decision.get('correction_made_internally_from_liar_to_bid'):
                        invalid_bid_corrected_for_logger = True
                        final_original_action_for_log = "liar" 
                        final_original_quantity_for_log = 0    # Standard for original 'liar' action
                        final_original_face_for_log = 0

                elif initial_ai_action == "bid":
                    # Validate the original bid from the AI before _process_ai_bid makes corrections
                    is_valid_structural_bid = (initial_ai_quantity is not None and isinstance(initial_ai_quantity, int) and initial_ai_quantity >= 1 and
                                               initial_ai_face is not None and isinstance(initial_ai_face, int) and 1 <= initial_ai_face <= 6)
                    
                    is_logically_higher_bid = True # Assume true if no prior bid or if structurally invalid
                    if self.last_bid and is_valid_structural_bid:
                        last_q, last_v = self.last_bid
                        if initial_ai_quantity < last_q or \
                           (initial_ai_quantity == last_q and initial_ai_face <= last_v):
                            is_logically_higher_bid = False
                    
                    if not (is_valid_structural_bid and is_logically_higher_bid):
                        invalid_bid_corrected_for_logger = True
                        final_original_action_for_log = "bid"
                        final_original_quantity_for_log = initial_ai_quantity
                        final_original_face_for_log = initial_ai_face
                        # _process_ai_bid will now be responsible for correcting 'processed_decision'
                    
                    result = self._process_ai_bid(player, processed_decision)
                
                else: # AI returned an unknown action
                    print(f"Error: AI {player.name} returned an unknown action: '{initial_ai_action}'. Fallback will be attempted.")
                    result = False # Indicate failure to get a valid bid, will trigger fallback logic later
                    invalid_bid_corrected_for_logger = True 
                    final_original_action_for_log = initial_ai_action
                    final_original_quantity_for_log = initial_ai_quantity
                    final_original_face_for_log = initial_ai_face
                    # Fallback logic in the except block or later in play_round will handle setting a valid move.
                    # For logging, processed_decision still holds the AI's problematic decision.

                # Log the move event after processing.
                # 'processed_decision' contains the action/qty/face that the game engine used.
                # 'final_original_action_for_log', etc., contain the AI's initial output if a correction occurred.
                self._log_move_if_enabled(game_num, player, game_state, processed_decision,
                                        invalid_bid_corrected_for_logger, 
                                        final_original_action_for_log, 
                                        final_original_quantity_for_log, 
                                        final_original_face_for_log)
                
                return result
                    
            except Exception as e:
                print(f"Error with AI decision: {e}")
                # Fallback to a simple bid
                if self.last_bid:
                    last_quantity, last_value = self.last_bid
                    if last_quantity > self.total_dice_in_game:
                        # Change decision to call liar due to invalid bid
                        print(f"{player.name} calls 'Liar!' (automatic fallback)")
                        move_data = {
                            "round": self.round_number,
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
                    "round": self.round_number,
                    "player": player.name,
                    "action": "bid",
                    "quantity": self.last_bid[0],
                    "face": self.last_bid[1],
                    "error_fallback": True,
                    "reasoning": "Error processing response, using default bid.",
                    "utterance": "I'll make this bid."
                }
                self.move_history.append(move_data)
                
                # Track in player history
                self.player_history[player.name].append(move_data)
                
                # Log the error fallback move to raw logging
                if EventLoggerFactory.is_enabled():
                    fallback_decision = {
                        "action": "bid",
                        "quantity": self.last_bid[0],
                        "face": self.last_bid[1],
                        "reasoning": "Error processing response, using default bid.",
                        "utterance": "I'll make this bid."
                    }
                    fallback_game_state = {
                        'raw_prompt': 'Error during processing',
                        'raw_response': f'Error: {str(e)}',
                        'response_time': 0.0,
                        'token_usage': {'prompt_tokens': 0, 'completion_tokens': 0, 'total_tokens': 0}
                    }
                    self._log_move_if_enabled(game_num, player, fallback_game_state, fallback_decision)
                
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
                    "round": self.round_number,
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
                
                # Log human player liar call to raw logging
                if EventLoggerFactory.is_enabled():
                    human_decision = {
                        "action": "liar",
                        "quantity": None,
                        "face": None,
                        "reasoning": reasoning if reasoning else "Human player called liar.",
                        "utterance": utterance if utterance else "I call liar!"
                    }
                    human_game_state = {
                        'raw_prompt': 'Human player input',
                        'raw_response': 'Human chose to call liar',
                        'response_time': 0.0,
                        'token_usage': {'prompt_tokens': 0, 'completion_tokens': 0, 'total_tokens': 0}
                    }
                    self._log_move_if_enabled(game_num, player, human_game_state, human_decision)
                
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
                
                if quantity > self.total_dice_in_game: # <<< KEY ADDITION for human player
                    print(f"Invalid bid. Quantity ({quantity}) cannot exceed total dice in game ({self.total_dice_in_game}).")
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
                    "round": self.round_number,
                    "player": player.name,
                    "action": "bid",
                    "quantity": quantity,
                    "face": value,
                    "reasoning": reasoning if reasoning else "Human player bid.",
                    "utterance": utterance if utterance else "I make this bid."
                }
                self.move_history.append(move_data)
                
                # Add to player history
                if player.name not in self.player_history:
                    self.player_history[player.name] = []
                self.player_history[player.name].append(move_data)
                
                # Log human player bid to raw logging
                if EventLoggerFactory.is_enabled():
                    human_decision = {
                        "action": "bid",
                        "quantity": quantity,
                        "face": value,
                        "reasoning": reasoning if reasoning else "Human player bid.",
                        "utterance": utterance if utterance else "I make this bid."
                    }
                    human_game_state = {
                        'raw_prompt': 'Human player input',
                        'raw_response': f'Human chose to bid {quantity} {value}s',
                        'response_time': 0.0,
                        'token_usage': {'prompt_tokens': 0, 'completion_tokens': 0, 'total_tokens': 0}
                    }
                    self._log_move_if_enabled(game_num, player, human_game_state, human_decision)
                
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
                print(f"{i+1}. {move['player']} bid {move['quantity']} {move['face']}'s")
                
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
                        "round": self.round_number,
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
                
                self.move_history = [] # Reset move history for the new round
                
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
                
        return "completed"```

## metrics.py
```python
import math
import numpy as np
import time
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
import logging
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
from matplotlib.colors import LinearSegmentedColormap
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Set, Optional, Union, Any, DefaultDict, Callable, TypeVar, Generic, Deque, cast

logger = logging.getLogger(__name__) # Initialized module-level logger

# Type aliases for better readability
ModelID = str
GameID = int
JsonDict = Dict[str, Any]
BluffData = DefaultDict[ModelID, JsonDict]
LieDetectionData = DefaultDict[ModelID, JsonDict]
BidData = List[Tuple[int, int]]
OptimalityData = Dict[str, int]
AdaptationData = Dict[str, int]
TokenUsageData = Dict[str, Union[int, float]]
MatchupStats = Dict[str, Dict[str, Dict[str, Union[int, float]]]]
StrategyPatterns = Dict[str, List[Any]]
TemporalMetrics = Dict[str, Union[List[Any], Dict[str, Any]]]

class GameMetrics:
    """
    Enhanced metrics tracking system for Liar's Dice.
    Provides comprehensive analytics on model performance, strategic patterns,
    and statistical analysis of gameplay data.
    """
    
    def __init__(self) -> None:
        # Core metrics tracking
        self.elo_ratings: Dict[ModelID, float] = {}        # Track Elo ratings for each model
        self.game_results: List[Dict[str, Any]] = []       # Store game results
        self.bluff_data: BluffData = defaultdict(
            lambda: cast(JsonDict, {"successful": 0, "total": 0})
        )
        # allow nested opponent‐maps and reasoning patterns
        self.lie_detection_data: LieDetectionData = defaultdict(
            lambda: cast(JsonDict, {"true_positive": 0, "false_positive": 0, "false_negative": 0})
        )
        self.final_bids: DefaultDict[ModelID, List[Tuple[int, int]]] = defaultdict(list)  # Track final bids in rounds
        self.bid_optimality: DefaultDict[ModelID, Dict[str, int]] = defaultdict(lambda: {"optimal": 0, "total": 0})
        self.adaptation_scores: DefaultDict[ModelID, Dict[str, int]] = defaultdict(lambda: {"adapted": 0, "opportunities": 0})
        self.rule_adherence: DefaultDict[ModelID, Dict[str, Any]] = defaultdict(lambda: {"valid_actions": 0, "total_actions": 0})
        self.api_response_times: DefaultDict[ModelID, List[float]] = defaultdict(list)  # Track API response times
        
        # Token usage tracking
        self.token_usage: DefaultDict[ModelID, Dict[str, Union[int, float]]] = defaultdict(
            lambda: {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "actions": 0}
        )
        
        # For tracking Elo
        self.k_factor: int = 32  # Standard K-factor for Elo calculation
        self.default_elo: int = 1000  # Starting Elo rating
        
        # Attributes for batch runner aggregated data
        self.elo_ratings_over_time: DefaultDict[ModelID, List[Tuple[int, float]]] = defaultdict(list) # (game_num, elo_score)
        self.win_matrix: DefaultDict[ModelID, DefaultDict[ModelID, int]] = defaultdict(lambda: defaultdict(int)) # winner -> loser -> count
        
        # Advanced metrics tracking
        self.historical_metrics: DefaultDict[ModelID, DefaultDict[str, List[Dict[str, Any]]]] = defaultdict(
            lambda: defaultdict(list)
        )  # Time-series metrics data
        
        self.matchup_stats: DefaultDict[ModelID, DefaultDict[ModelID, Dict[str, Union[int, float]]]] = defaultdict(
            lambda: defaultdict(lambda: {"wins": 0, "losses": 0, "games": 0})
        )  # Model vs model stats
        
        self.opponent_adaptation: DefaultDict[ModelID, DefaultDict[ModelID, Dict[str, int]]] = defaultdict(
            lambda: defaultdict(lambda: {"adapted": 0, "opportunities": 0})
        )  # Per-opponent adaptation
        
        self.strategy_patterns: DefaultDict[ModelID, Dict[str, List[Any]]] = defaultdict(
            lambda: {"early_game": [], "mid_game": [], "late_game": []}
        )  # Strategic patterns by game phase
        
        self.confidence_metrics: DefaultDict[ModelID, List[float]] = defaultdict(list)  # For confidence interval calculations
        
        # Game state and temporal analysis
        self.temporal_metrics: DefaultDict[ModelID, Dict[str, Union[List[Any], Dict[str, Any]]]] = defaultdict(
            lambda: {"round_history": [], "evolution": {}}
        )
        
        self.model_consistency: DefaultDict[ModelID, List[float]] = defaultdict(list)  # Track consistency of metrics over time
        
        # Settings for analysis
        self.historical_window_size: int = 10  # Number of data points to keep for rolling statistics
        self.statistical_confidence: float = 0.95  # Default confidence level for statistical tests
        
        # Game context
        self.timestamp: float = time.time()  # When metrics tracking began
    
    def initialize_model(self, model_id: ModelID) -> None:
        """Initialize a new model's metrics if it doesn't exist"""
        if model_id not in self.elo_ratings:
            self.elo_ratings[model_id] = self.default_elo
    
    def update_elo(self, winner_id: ModelID, loser_id: ModelID, game_data: Optional[Dict[str, Any]] = None) -> None:
        """
        Update Elo ratings after a game and track comprehensive matchup statistics
        
        Args:
            winner_id: ID of the winning model
            loser_id: ID of the losing model
            game_data: Optional dictionary with additional game data (round count, etc.)
        """
        # Initialize if needed
        self.initialize_model(winner_id)
        self.initialize_model(loser_id)
        
        # Save pre-update ratings for historical tracking
        old_winner_elo: float = self.elo_ratings[winner_id]
        old_loser_elo: float = self.elo_ratings[loser_id]
        
        # Calculate expected win probabilities
        r1: float = self.elo_ratings[winner_id]
        r2: float = self.elo_ratings[loser_id]
        
        # Expected score for winner
        expected_winner: float = 1 / (1 + 10 ** ((r2 - r1) / 400))
        
        # Update ratings
        self.elo_ratings[winner_id] += self.k_factor * (1 - expected_winner)
        self.elo_ratings[loser_id] += self.k_factor * (0 - (1 - expected_winner))
        
        # Calculate the "surprise factor" - how unexpected was this outcome?
        surprise_factor: float = 1 - expected_winner  # Higher means more surprising
        
        # Record temporal Elo progression
        timestamp: float = time.time()
        self.historical_metrics[winner_id]["elo"].append({
            "timestamp": timestamp,
            "value": self.elo_ratings[winner_id],
            "change": self.elo_ratings[winner_id] - old_winner_elo,
            "opponent": loser_id
        })
        
        self.historical_metrics[loser_id]["elo"].append({
            "timestamp": timestamp,
            "value": self.elo_ratings[loser_id],
            "change": self.elo_ratings[loser_id] - old_loser_elo,
            "opponent": winner_id
        })
        
        # Update matchup statistics
        self.matchup_stats[winner_id][loser_id]["wins"] += 1
        self.matchup_stats[winner_id][loser_id]["games"] += 1
        self.matchup_stats[loser_id][winner_id]["losses"] += 1
        self.matchup_stats[loser_id][winner_id]["games"] += 1
        
        # Calculate win probability for future games between these models
        for model1, opponents in self.matchup_stats.items():
            for model2, opponent_stats in opponents.items():
                if opponent_stats["games"] > 0:
                    win_rate: float = opponent_stats["wins"] / opponent_stats["games"]
                    r1 = self.elo_ratings[model1]
                    r2 = self.elo_ratings[model2]
                    expected_win_rate: float = 1 / (1 + 10 ** ((r2 - r1) / 400))
                    opponent_stats["win_rate"] = win_rate
                    opponent_stats["expected_win_rate"] = expected_win_rate
                    opponent_stats["performance_index"] = win_rate / expected_win_rate if expected_win_rate > 0 else float('inf')
        
        # Record the game result
        game_result: Dict[str, Any] = {
            "timestamp": timestamp,
            "winner": winner_id,
            "loser": loser_id,
            "winner_old_elo": old_winner_elo,
            "loser_old_elo": old_loser_elo,
            "winner_new_elo": self.elo_ratings[winner_id],
            "loser_new_elo": self.elo_ratings[loser_id],
            "expected_outcome": expected_winner,
            "surprise_factor": surprise_factor,
        }
        
        # Add additional game data if provided
        if game_data:
            game_result.update(game_data)
            
        self.game_results.append(game_result)
        
        # Update model consistency metric - how predictable are outcomes for this model
        self.model_consistency[winner_id].append(expected_winner)  # Higher means more consistent with expectations
        self.model_consistency[loser_id].append(1 - expected_winner)
    
    def record_bluff(self, model_id: ModelID, bluff_type: str, success: bool, 
                     game_context: Optional[Dict[str, Any]] = None) -> None:
        """
        Record bluffing success/failure with enhanced context tracking
        
        Args:
            model_id: ID of the model performing the bluff
            bluff_type: 'bluff' (false bid) or 'truth' (truthful bid challenged)
            success: Whether the bluff succeeded or truthful bid was incorrectly challenged
            game_context: Optional dict with additional context (round number, opponent, dice values, etc.)
        """
        # Initialize if needed
        self.initialize_model(model_id)
        
        # Basic tracking
        self.bluff_data[model_id]["total"] += 1
        if success:
            self.bluff_data[model_id]["successful"] += 1
            
        # Track by bluff type
        bluff_type_key: str = f"{bluff_type}_total"
        success_type_key: str = f"{bluff_type}_successful"
        
        if bluff_type_key not in self.bluff_data[model_id]:
            self.bluff_data[model_id][bluff_type_key] = 0
            self.bluff_data[model_id][success_type_key] = 0
            
        self.bluff_data[model_id][bluff_type_key] += 1
        if success:
            self.bluff_data[model_id][success_type_key] += 1
            
        # Record historical data
        bluff_record: Dict[str, Any] = {
            "timestamp": time.time(),
            "bluff_type": bluff_type,
            "success": success,
            "success_rate": self.calculate_bluff_success_rate(model_id)
        }
        
        # Add game context if provided
        if game_context:
            # Extract round info for game phase analysis
            if "round_number" in game_context:
                round_num: int = game_context["round_number"]
                total_rounds: int = game_context.get("total_rounds", 30)  # Default estimate
                
                # Categorize game phase
                if round_num <= total_rounds * 0.3:
                    phase = "early_game"
                elif round_num <= total_rounds * 0.7:
                    phase = "mid_game"
                else:
                    phase = "late_game"
                    
                # Add to strategy patterns by game phase
                self.strategy_patterns[model_id][phase].append({
                    "round": round_num,
                    "bluff_type": bluff_type,
                    "success": success,
                    **{k: v for k, v in game_context.items() if k != "round_number"}
                })
                
            # Track opponent-specific bluffing if opponent is provided
            if "opponent" in game_context:
                opponent = game_context["opponent"]
                opponents = cast(JsonDict, self.bluff_data[model_id].setdefault("opponents", {}))
                opponent_data: Dict[str, Dict[str, int]] = opponents
                if opponent not in opponent_data:
                    opponent_data[opponent] = {"total": 0, "successful": 0}
                opponent_data[opponent]["total"] += 1
                if success:
                    opponent_data[opponent]["successful"] += 1
                    
            # Add full context to the record
            bluff_record.update(game_context)
            
        # Add to historical metrics
        self.historical_metrics[model_id]["bluffs"].append(bluff_record)
        
        # Update confidence metrics list for statistical analysis
        self.confidence_metrics[f"{model_id}_bluff"].append(1 if success else 0)
        
    def calculate_bluff_success_rate(self, model_id: ModelID) -> float:
        """Calculate the bluff success rate for a model"""
        self.initialize_model(model_id)
        
        # Get bluff data for this model
        model_data = self.bluff_data.get(model_id, {"successful": 0, "total": 0})
        
        # Calculate success rate (handle division by zero)
        if model_data["total"] > 0:
            return (model_data["successful"] / model_data["total"]) * 100
        else:
            return 0.0
    
    def record_lie_detection(self, model_id: ModelID, detection_type: str, 
                            game_context: Optional[Dict[str, Any]] = None) -> None:
        """
        Enhanced lie detection tracking with context analysis
        
        Args:
            model_id: ID of the model detecting lies
            detection_type: 'true_positive' (correctly called bluff), 
                           'false_positive' (incorrectly called bluff),
                           'false_negative' (missed opponent's bluff)
            game_context: Optional dict with context (opponent, round, reasoning, etc.)
        """
        # Initialize if needed
        self.initialize_model(model_id)
        
        # Basic tracking
        self.lie_detection_data[model_id][detection_type] += 1
        
        # Record with temporal data
        detection_record = {
            "timestamp": time.time(),
            "detection_type": detection_type,
            "current_metrics": self.calculate_lie_detection_rate(model_id)
        }
        
        # Add game context and analyze
        if game_context:
            # Track opponent-specific detection if opponent is provided
            if "opponent" in game_context:
                opponent = game_context["opponent"]
                opponent_key = f"opponent_{opponent}"
                
                # Create a type-safe dictionary to store opponent-specific detection data
                opponent_data: Dict[str, int] = {
                    "true_positive": 0, 
                    "false_positive": 0, 
                    "false_negative": 0
                }
                
                # Get existing data if available
                existing_data = self.lie_detection_data[model_id].get(opponent_key)
                if isinstance(existing_data, dict):
                    for key in opponent_data:
                        if key in existing_data:
                            opponent_data[key] = existing_data[key]
                
                # Update with the new detection
                opponent_data[detection_type] += 1
                
                # Store back in the main dictionary
                self.lie_detection_data[model_id][opponent_key] = opponent_data
                
            # Track reasoning if provided
            if "reasoning" in game_context:
                # Initialize reasoning_patterns as a list if it doesn't exist
                if "reasoning_patterns" not in self.lie_detection_data[model_id]:
                    self.lie_detection_data[model_id]["reasoning_patterns"] = []
                
                # Get the current reasoning patterns safely with proper type annotation
                reasoning_patterns: List[Dict[str, Any]] = cast(List[Dict[str, Any]], self.lie_detection_data[model_id].get("reasoning_patterns", []))
                
                if isinstance(reasoning_patterns, list):
                    # Create a new reasoning pattern entry
                    reasoning_entry = {
                        "detection_type": detection_type,
                        "reasoning": game_context["reasoning"]
                    }
                    
                    # Append the new entry to the list
                    reasoning_patterns.append(reasoning_entry)
                    
                    # Update the dictionary with the modified list
                    self.lie_detection_data[model_id]["reasoning_patterns"] = reasoning_patterns
                
            detection_record.update(game_context)
            
        # Add to historical tracking
        self.historical_metrics[model_id]["lie_detection"].append(detection_record)
        
        # Update confidence metrics based on detection type
        if detection_type == "true_positive":
            self.confidence_metrics[f"{model_id}_lie_detection_precision"].append(1)
        elif detection_type == "false_positive":
            self.confidence_metrics[f"{model_id}_lie_detection_precision"].append(0)
            
        if detection_type == "true_positive":
            self.confidence_metrics[f"{model_id}_lie_detection_recall"].append(1)
        elif detection_type == "false_negative":
            self.confidence_metrics[f"{model_id}_lie_detection_recall"].append(0)
    
    def calculate_lie_detection_rate(self, model_id: ModelID) -> Dict[str, float]:
        """Calculate precision and recall for lie detection"""
        self.initialize_model(model_id)
        
        data = self.lie_detection_data.get(model_id, {
            "true_positive": 0, "false_positive": 0, "false_negative": 0
        })
        
        # Calculate precision (when model calls liar, how often is it right?)
        precision = 0.0
        if data["true_positive"] + data["false_positive"] > 0:
            precision = data["true_positive"] / (data["true_positive"] + data["false_positive"]) * 100
            
        # Calculate recall (of all actual bluffs, how many did the model catch?)
        recall = 0.0
        if data["true_positive"] + data["false_negative"] > 0:
            recall = data["true_positive"] / (data["true_positive"] + data["false_negative"]) * 100
            
        # Calculate F1-score (harmonic mean of precision and recall)
        f1_score = 0.0
        if precision + recall > 0:
            f1_score = 2 * (precision * recall) / (precision + recall)
            
        return {
            "precision": precision,
            "recall": recall,
            "f1_score": f1_score
        }
    
    def record_final_bid(self, model_id: ModelID, bid: Tuple[int, int], round_number: int, 
                        game_context: Optional[Dict[str, Any]] = None) -> None:
        """Record final bids in rounds for analyzing bid progression"""
        self.initialize_model(model_id)
        
        # Add to final bids list
        self.final_bids[model_id].append(bid)
        
        # Calculate the average final bid for this model
        avg_quantity = sum(b[0] for b in self.final_bids[model_id]) / len(self.final_bids[model_id])
        
        # Record with context
        bid_record = {
            "timestamp": time.time(),
            "round": round_number,
            "bid": bid,
            "avg_quantity": avg_quantity
        }
        
        if game_context:
            bid_record.update(game_context)
            
        # Add to historical data
        self.historical_metrics[model_id]["final_bids"].append(bid_record)
    
    def record_bid_optimality(self, model_id: ModelID, is_optimal: bool, 
                             game_context: Optional[Dict[str, Any]] = None) -> None:
        """Track how often a model makes optimal bids"""
        self.initialize_model(model_id)
        
        # Update optimality counters
        self.bid_optimality[model_id]["total"] += 1
        if is_optimal:
            self.bid_optimality[model_id]["optimal"] += 1
            
        # Calculate current optimality rate
        optimality_rate = 0.0
        if self.bid_optimality[model_id]["total"] > 0:
            optimality_rate = (self.bid_optimality[model_id]["optimal"] / self.bid_optimality[model_id]["total"]) * 100
            
        # Record with context
        optimality_record = {
            "timestamp": time.time(),
            "is_optimal": is_optimal,
            "optimality_rate": optimality_rate
        }
        
        if game_context:
            optimality_record.update(game_context)
            
        # Add to historical data
        self.historical_metrics[model_id]["bid_optimality"].append(optimality_record)
    
    def record_adaptation(self, model_id: ModelID, adapted: bool, 
                         game_context: Optional[Dict[str, Any]] = None) -> None:
        """Track how well the model adapts to opponent strategies"""
        self.initialize_model(model_id)
        
        # Update adaptation counters
        self.adaptation_scores[model_id]["opportunities"] += 1
        if adapted:
            self.adaptation_scores[model_id]["adapted"] += 1
            
        # Calculate adaptation rate
        adaptation_rate = 0.0
        if self.adaptation_scores[model_id]["opportunities"] > 0:
            adaptation_rate = (self.adaptation_scores[model_id]["adapted"] / 
                              self.adaptation_scores[model_id]["opportunities"]) * 100
                              
        # Record with context
        adaptation_record = {
            "timestamp": time.time(),
            "adapted": adapted,
            "adaptation_rate": adaptation_rate
        }
        
        if game_context and "opponent" in game_context:
            # Track opponent-specific adaptation
            opponent = game_context["opponent"]
            self.opponent_adaptation[model_id][opponent]["opportunities"] += 1
            if adapted:
                self.opponent_adaptation[model_id][opponent]["adapted"] += 1
                
            adaptation_record["opponent"] = opponent
            
        if game_context:
            adaptation_record.update(game_context)
            
        # Add to historical data
        self.historical_metrics[model_id]["adaptation"].append(adaptation_record)
    
    def record_rule_adherence(self, model_id: ModelID, is_valid: bool, 
                             game_context: Optional[Dict[str, Any]] = None) -> None:
        """Track how well the model follows game rules"""
        self.initialize_model(model_id)
        
        # Update rule adherence counters
        self.rule_adherence[model_id]["total_actions"] += 1
        if is_valid:
            self.rule_adherence[model_id]["valid_actions"] += 1
            
        # Calculate adherence rate
        adherence_rate = 0.0
        if self.rule_adherence[model_id]["total_actions"] > 0:
            adherence_rate = (self.rule_adherence[model_id]["valid_actions"] / 
                             self.rule_adherence[model_id]["total_actions"]) * 100
                             
        # Record with context
        adherence_record = {
            "timestamp": time.time(),
            "is_valid": is_valid,
            "adherence_rate": adherence_rate
        }
        
        if game_context:
            adherence_record.update(game_context)
            
        # Add to historical data
        self.historical_metrics[model_id]["rule_adherence"].append(adherence_record)
    
    def record_api_response_time(self, model_id: ModelID, response_time: float) -> None:
        """Track API response times for performance analysis"""
        self.initialize_model(model_id)
        
        # Add to response times list
        self.api_response_times[model_id].append(response_time)
        
        # Calculate averages
        avg_response_time = sum(self.api_response_times[model_id]) / len(self.api_response_times[model_id])
        
        # Record in temporal metrics
        response_record = {
            "timestamp": time.time(),
            "response_time": response_time,
            "avg_response_time": avg_response_time
        }
        
        # Add to historical data
        self.historical_metrics[model_id]["response_times"].append(response_record)
    
    def record_token_usage(self, model_id: ModelID, prompt_tokens: int, 
                          completion_tokens: int, total_tokens: int) -> None:
        """Track token usage for cost estimation and efficiency analysis"""
        self.initialize_model(model_id)
        
        # Update token counts
        self.token_usage[model_id]["prompt_tokens"] += prompt_tokens
        self.token_usage[model_id]["completion_tokens"] += completion_tokens
        self.token_usage[model_id]["total_tokens"] += total_tokens
        self.token_usage[model_id]["actions"] += 1
        
        # Calculate averages
        actions = self.token_usage[model_id]["actions"]
        avg_tokens = self.token_usage[model_id]["total_tokens"] / actions if actions > 0 else 0
        
        # Record usage with timestamp
        token_record = {
            "timestamp": time.time(),
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "cumulative_tokens": self.token_usage[model_id]["total_tokens"],
            "avg_tokens_per_action": avg_tokens
        }
        
        # Add to historical data
        self.historical_metrics[model_id]["token_usage"].append(token_record)
    
    def get_model_metrics(self, model_id: ModelID) -> Dict[str, Any]:
        """Get a comprehensive metrics summary for a specific model"""
        self.initialize_model(model_id)
        
        # Basic metrics
        metrics: JsonDict = {
            "elo_rating": self.elo_ratings.get(model_id, self.default_elo),
            "games_played": len([r for r in self.game_results if r["winner"] == model_id or r["loser"] == model_id]),
            "wins": len([r for r in self.game_results if r["winner"] == model_id]),
            "losses": len([r for r in self.game_results if r["loser"] == model_id])
        }
        
        # Calculate win rate
        if metrics["games_played"] > 0:
            metrics["win_rate"] = (metrics["wins"] / metrics["games_played"]) * 100
        else:
            metrics["win_rate"] = 0.0
            
        # Bluff effectiveness
        bluff_data = self.bluff_data.get(model_id, {"successful": 0, "total": 0})
        if bluff_data["total"] > 0:
            metrics["bluff_success_rate"] = (bluff_data["successful"] / bluff_data["total"]) * 100
        else:
            metrics["bluff_success_rate"] = 0.0
            
        # Lie detection capability
        lie_detection_metrics = self.calculate_lie_detection_rate(model_id)
        metrics["lie_detection"] = lie_detection_metrics
        
        # Average final bid
        if model_id in self.final_bids and self.final_bids[model_id]:
            metrics["average_final_bid"] = sum(bid[0] for bid in self.final_bids[model_id]) / len(self.final_bids[model_id])
        else:
            metrics["average_final_bid"] = 0.0
            
        # Bid optimality
        optimality_data = self.bid_optimality.get(model_id, {"optimal": 0, "total": 0})
        if optimality_data["total"] > 0:
            metrics["bid_optimality"] = (optimality_data["optimal"] / optimality_data["total"]) * 100
        else:
            metrics["bid_optimality"] = 0.0
            
        # Adaptation score
        adaptation_data = self.adaptation_scores.get(model_id, {"adapted": 0, "opportunities": 0})
        if adaptation_data["opportunities"] > 0:
            metrics["adaptation_score"] = (adaptation_data["adapted"] / adaptation_data["opportunities"]) * 100
        else:
            metrics["adaptation_score"] = 0.0
            
        # Rule adherence
        adherence_data = self.rule_adherence.get(model_id, {"valid_actions": 0, "total_actions": 0})
        if adherence_data["total_actions"] > 0:
            metrics["rule_adherence_rate"] = (adherence_data["valid_actions"] / adherence_data["total_actions"]) * 100
        else:
            metrics["rule_adherence_rate"] = 0.0
            
        # API performance
        if model_id in self.api_response_times and self.api_response_times[model_id]:
            metrics["avg_api_response_time"] = sum(self.api_response_times[model_id]) / len(self.api_response_times[model_id])
            metrics["max_api_response_time"] = max(self.api_response_times[model_id])
            metrics["min_api_response_time"] = min(self.api_response_times[model_id])
        else:
            metrics["avg_api_response_time"] = 0.0
            metrics["max_api_response_time"] = 0.0
            metrics["min_api_response_time"] = 0.0
            
        # Token usage
        if model_id in self.token_usage:
            token_data = self.token_usage[model_id]
            actions = int(token_data.get("actions", 0))
            total_tokens = int(token_data.get("total_tokens", 0))
            prompt_tokens = int(token_data.get("prompt_tokens", 0))
            completion_tokens = int(token_data.get("completion_tokens", 0))
            
            metrics["token_usage"] = {
                "total_tokens": total_tokens,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "actions": actions,
                "avg_tokens_per_action": total_tokens / actions if actions > 0 else 0
            }
        else:
            metrics["token_usage"] = {
                "total_tokens": 0,
                "prompt_tokens": 0, 
                "completion_tokens": 0,
                "actions": 0,
                "avg_tokens_per_action": 0
            }
            
        # Model consistency (lower standard deviation means more consistent)
        if model_id in self.model_consistency and len(self.model_consistency[model_id]) > 1:
            # Convert numpy.float64 to Python float
            std_dev = float(np.std(self.model_consistency[model_id]))
            metrics["consistency"] = 1.0 - std_dev
        else:
            metrics["consistency"] = 0.0
            
        return metrics
    
    def get_all_metrics(self) -> Dict[str, Dict[str, Any]]:
        """Get metrics for all models"""
        return {model_id: self.get_model_metrics(model_id) for model_id in self.elo_ratings}


class BidAnalyzer:
    """
    Static class for analyzing bid optimality and strategic decisions.
    """
    
    @staticmethod
    def is_bid_optimal(
        player_dice: List[int], 
        player_dice_count: int,
        total_dice: int, 
        previous_bid: Optional[Tuple[int, int]],
        current_bid: Tuple[int, int]
    ) -> bool:
        """
        Determines if a bid is mathematically optimal based on known information
        
        Args:
            player_dice: The player's own dice values
            player_dice_count: Number of dice the player has
            total_dice: Total number of dice in the game
            previous_bid: The previous bid in the format (quantity, face)
            current_bid: The current bid to analyze in format (quantity, face)
            
        Returns:
            bool: True if the bid is optimal, False otherwise
        """
        # Unpack bids
        quantity, face = current_bid
        
        # Count how many of the bid face the player has
        player_matches = player_dice.count(face)
        
        # Calculate unknown dice (other players' dice)
        unknown_dice = total_dice - player_dice_count
        
        # Calculate probability thresholds based on game state
        if previous_bid is None:
            # First bid - just check if it's reasonable
            if quantity <= player_matches:
                # Safe bid based on own dice
                return True
            else:
                # Bid is higher than own dice - check probability
                return BidAnalyzer._calculate_probability(quantity - player_matches, unknown_dice, face) > 0.5
        else:
            prev_quantity, prev_face = previous_bid
            
            # Calculate expected values
            if prev_face == face:
                # Same face, higher quantity
                expected = BidAnalyzer._calculate_expected_count(unknown_dice, face) + player_matches
                return quantity <= expected + (total_dice * 0.1)  # Allow small bluff
            else:
                # Different face
                expected = BidAnalyzer._calculate_expected_count(unknown_dice, face) + player_matches
                # Compare with probability of previous bid
                prev_expected = BidAnalyzer._calculate_expected_count(unknown_dice, prev_face)
                prev_expected += player_dice.count(prev_face)
                
                # If previous bid was already unlikely, calling would be better
                if prev_expected < prev_quantity - (total_dice * 0.1):
                    return False  # Should have called instead
                
                # Otherwise check if current bid is reasonable
                return quantity <= expected + (total_dice * 0.15)  # Allow slightly larger bluff
    
    @staticmethod
    def should_call_liar(
        player_dice: List[int],
        player_dice_count: int,
        total_dice: int,
        bid: Tuple[int, int]
    ) -> bool:
        """
        Determines if calling 'liar' on a bid is mathematically optimal
        
        Args:
            player_dice: The player's own dice values
            player_dice_count: Number of dice the player has
            total_dice: Total number of dice in the game
            bid: The bid to challenge in format (quantity, face)
            
        Returns:
            bool: True if calling liar is optimal, False otherwise
        """
        quantity, face = bid
        
        # Count how many of the bid face the player has
        player_matches = player_dice.count(face)
        
        # Calculate unknown dice (other players' dice)
        unknown_dice = total_dice - player_dice_count
        
        # Calculate expected number of matching dice in the game
        expected_total = player_matches + BidAnalyzer._calculate_expected_count(unknown_dice, face)
        
        # If the bid is significantly higher than expected, call liar
        if quantity > expected_total * 1.3:  # Allow for 30% variance before calling
            return True
        
        # If the bid is impossible or highly unlikely
        probability = BidAnalyzer._calculate_probability(quantity - player_matches, unknown_dice, face)
        if probability < 0.2:  # Less than 20% chance is unlikely
            return True
            
        return False
    
    @staticmethod
    def _calculate_probability(needed: int, dice: int, face: int, faces: int = 6) -> float:
        """
        Calculate the probability of rolling at least 'needed' of a specific face with 'dice' dice
        
        Args:
            needed: Number of the specific face needed
            dice: Number of dice to roll
            face: The target face value
            faces: Number of faces on each die (default 6)
            
        Returns:
            float: Probability between 0 and 1
        """
        import scipy.stats as stats
        
        # Probability of rolling the target face on a single die
        p = 1.0 / faces
        
        # If we need more than available dice, probability is 0
        if needed > dice:
            return 0.0
            
        # Calculate probability of at least 'needed' successes
        # Use binomial distribution to calculate P(X >= needed)
        # P(X >= needed) = 1 - P(X < needed)
        probability = 1.0 - stats.binom.cdf(needed - 1, dice, p)
        
        return probability
    
    @staticmethod
    def _calculate_expected_count(dice: int, face: int, faces: int = 6) -> float:
        """
        Calculate the expected number of occurrences of a specific face when rolling 'dice' dice
        
        Args:
            dice: Number of dice to roll
            face: The target face value
            faces: Number of faces on each die (default 6)
            
        Returns:
            float: Expected count
        """
        # Probability of rolling the target face on a single die
        p = 1.0 / faces
        
        # Expected count = number of dice × probability
        return dice * p
        
    @staticmethod
    def check_adaptation(player_history: List[Dict[str, Any]], opponent_history: List[Dict[str, Any]]) -> bool:
        """
        Check if a player has adapted their strategy based on opponent's history
        
        Args:
            player_history: History of moves by the player 
            opponent_history: History of moves by the opponent
            
        Returns:
            bool: True if player shows adaptation, False otherwise
        """
        # Need enough history to analyze
        if len(player_history) < 3 or len(opponent_history) < 3:
            return False
            
        # Check if opponent has a clear pattern
        opponent_pattern = BidAnalyzer._detect_pattern(opponent_history)
        if not opponent_pattern:
            return False
            
        # Check if player behavior changed in response to opponent pattern
        early_player_actions = player_history[:len(player_history)//2]
        recent_player_actions = player_history[len(player_history)//2:]
        
        early_behavior = BidAnalyzer._summarize_behavior(early_player_actions)
        recent_behavior = BidAnalyzer._summarize_behavior(recent_player_actions)
        
        # Look for significant changes in behavior
        for key, value in recent_behavior.items():
            if key in early_behavior:
                # If behavior changed by more than 20%, consider it adaptation
                if abs(value - early_behavior[key]) > 0.2:
                    return True
                    
        return False
        
    @staticmethod
    def _detect_pattern(history: List[Dict[str, Any]]) -> Optional[Dict[str, float]]:
        """
        Detect patterns in a player's history
        
        Args:
            history: List of move dictionaries
            
        Returns:
            Optional[Dict[str, float]]: Pattern metrics if detected, None otherwise
        """
        if len(history) < 3:
            return None
            
        # Count different types of actions
        bid_count = sum(1 for move in history if move.get("action") == "bid")
        liar_count = sum(1 for move in history if move.get("action") == "liar")
        bluff_count = sum(1 for move in history if move.get("action") == "bid" and move.get("bluff", False))
        
        # Calculate proportions
        total_moves = len(history)
        pattern = {
            "bid_rate": bid_count / total_moves if total_moves > 0 else 0,
            "liar_rate": liar_count / total_moves if total_moves > 0 else 0,
            "bluff_rate": bluff_count / bid_count if bid_count > 0 else 0
        }
        
        # Only return if there's a strong pattern
        if any(value > 0.7 for value in pattern.values()):
            return pattern
            
        return None
        
    @staticmethod
    def _summarize_behavior(history: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        Summarize player behavior from history
        
        Args:
            history: List of move dictionaries
            
        Returns:
            Dict[str, float]: Behavior metrics
        """
        if not history:
            return {}
            
        # Count different types of actions
        bid_count = sum(1 for move in history if move.get("action") == "bid")
        liar_count = sum(1 for move in history if move.get("action") == "liar")
        bluff_count = sum(1 for move in history if move.get("action") == "bid" and move.get("bluff", False))
        
        # Calculate proportions
        total_moves = len(history)
        return {
            "bid_rate": bid_count / total_moves if total_moves > 0 else 0,
            "liar_rate": liar_count / total_moves if total_moves > 0 else 0,
            "bluff_rate": bluff_count / bid_count if bid_count > 0 else 0
        }


class MetricsVisualizer:
    """
    Generates visualizations of game metrics and performance data.
    """
    
    def __init__(self, metrics: GameMetrics) -> None:
        self.metrics = metrics
        self.output_dir = "metrics_visualizations"
        os.makedirs(self.output_dir, exist_ok=True)
        
    def generate_elo_rating_chart(self, output_file: str = "elo_ratings.png") -> str:
        """Generate a chart showing Elo rating progression for all models over games."""
        plt.style.use('seaborn-v0_8-darkgrid') # Using a seaborn style for better aesthetics
        fig, ax = plt.subplots(figsize=(14, 8))
        
        all_game_numbers = set()
        model_elo_trajectories = defaultdict(dict)

        # Use the new elo_ratings_over_time attribute populated by BatchRunner
        for model_id, elo_data_points in self.metrics.elo_ratings_over_time.items():
            if not elo_data_points:
                continue
            # elo_data_points is expected to be List[Tuple[int, float]] -> [(game_num, elo)]
            sorted_points = sorted(elo_data_points, key=lambda x: x[0]) # Sort by game_number
            game_nums = [point[0] for point in sorted_points]
            elos = [point[1] for point in sorted_points]
            
            for gn, elo in zip(game_nums, elos):
                all_game_numbers.add(gn)
                model_elo_trajectories[model_id][gn] = elo
            
            ax.plot(game_nums, elos, marker='o', linestyle='-', label=model_id, markersize=5, linewidth=2)

        if not all_game_numbers:
            logger.warning("No Elo data available to generate chart.")
            plt.close(fig)
            return ""

        # Determine a sensible set of x-ticks
        sorted_game_numbers = sorted(list(all_game_numbers))
        if len(sorted_game_numbers) > 20: # If too many game numbers, sample or use major ticks
            step = max(1, len(sorted_game_numbers) // 15) # Show around 15 ticks
            x_ticks = sorted_game_numbers[::step]
            if sorted_game_numbers[-1] not in x_ticks: # Ensure last game number is a tick
                x_ticks.append(sorted_game_numbers[-1])
        elif sorted_game_numbers:
            x_ticks = sorted_game_numbers
        else:
            x_ticks = []

        ax.set_xlabel("Game Number", fontsize=14)
        ax.set_ylabel("Elo Rating", fontsize=14)
        ax.set_title("Elo Rating Progression Over Games", fontsize=18, fontweight='bold')
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=10)
        ax.tick_params(axis='both', which='major', labelsize=12)
        if x_ticks:
            ax.set_xticks(x_ticks)
            ax.set_xticklabels(x_ticks, rotation=45, ha="right")
        
        plt.grid(True, which='major', linestyle='--', linewidth=0.5)
        plt.tight_layout(rect=[0, 0, 0.85, 1]) # Adjust layout to make space for legend
        
        # Ensure the output directory exists
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        plt.savefig(output_file)
        plt.close(fig)
        logger.info(f"Elo rating chart saved to {output_file}")
        return output_file
        
    def generate_metric_comparison_radar(self, output_file: str = "radar_metrics.png", direct_model_metrics: Optional[Dict[ModelID, Dict[str, Any]]] = None) -> str:
        """
        Generate a radar chart comparing all models across key metrics
        
        Args:
            output_file: The path to save the radar chart image.
            direct_model_metrics: Optional pre-computed dictionary of model metrics. 
                                  If None, self.metrics.get_all_metrics() is used.

        Returns:
            Path to the generated chart image or an error message string.
        """
        # Get all model metrics
        if direct_model_metrics is not None:
            all_metrics = direct_model_metrics
        else:
            all_metrics = self.metrics.get_all_metrics()
        
        # Select metrics for radar chart
        metrics_to_plot = [
            "win_rate", 
            "bluff_success_rate", 
            "lie_detection", 
            "bid_optimality", 
            "adaptation_score", 
            "rule_adherence_rate"
        ]
        
        # Category labels for the radar chart
        categories = [
            "Win Rate", 
            "Bluff Success", 
            "Lie Detection", 
            "Bid Optimality", 
            "Adaptation", 
            "Rule Adherence"
        ]
        
        # Get model data
        model_ids = list(all_metrics.keys())
        
        if not model_ids:
            return "No models to visualize"
            
        # Prepare data for plotting
        num_vars = len(metrics_to_plot)
        angles = cast(List[float], np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist())
        angles.append(angles[0])
        
        # Set up the figure
        fig, ax = plt.subplots(figsize=(12, 10), subplot_kw=dict(polar=True))
        
        # For each model, plot on the radar chart
        for i, model_id in enumerate(model_ids):
            metrics = all_metrics[model_id]
            
            # Extract values, handling lie_detection specially
            values: List[float] = []
            for metric in metrics_to_plot:
                if metric == "lie_detection":
                    # Use F1 score for lie detection
                    lie_data = metrics.get(metric, {})
                    if isinstance(lie_data, dict):
                        values.append(float(lie_data.get("f1_score", 0)))
                    else:
                        values.append(0.0)
                else:
                    metric_value = metrics.get(metric, 0)
                    if isinstance(metric_value, (int, float)):
                        values.append(float(metric_value))
                    else:
                        values.append(0.0)
            
            # Normalize values to 0-1 scale
            values = [v / 100 for v in values]
            values.append(values[0])
            # Plot the model metrics
            ax.plot(angles, values, linewidth=2, label=model_id)
            ax.fill(angles, values, alpha=0.1)
        
        # Set category labels
        plt.xticks(angles[:-1], categories, fontsize=12)
        
        # Set y-ticks
        ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_yticklabels(['20%', '40%', '60%', '80%', '100%'], fontsize=10)
        
        # Add legend
        plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1), fontsize=10)
        
        plt.title('Model Performance Comparison', fontsize=16)
        
        output_path = os.path.join(self.output_dir, output_file)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return output_path
        
    def generate_win_matrix_heatmap(self, output_file: str = "win_matrix.png") -> str:
        """Generate a heatmap showing win/loss records between models."""
        # Use the win_matrix attribute populated by BatchRunner
        win_matrix_data = self.metrics.win_matrix
        
        if not win_matrix_data:
            logger.warning("No win matrix data available to generate heatmap.")
            return ""

        # Convert win_matrix_data (DefaultDict[ModelID, DefaultDict[ModelID, int]]) to a DataFrame
        # Rows: Winners, Columns: Losers, Values: Number of wins
        # We need to get all unique model IDs that participated in any game recorded in the win_matrix
        all_models_in_matrix = set()
        for winner_model, loser_map in win_matrix_data.items():
            all_models_in_matrix.add(winner_model)
            for loser_model in loser_map.keys():
                all_models_in_matrix.add(loser_model)
        
        sorted_model_ids = sorted(list(all_models_in_matrix))
        df_data = []
        for winner_id in sorted_model_ids:
            row = []
            for loser_id in sorted_model_ids:
                if winner_id == loser_id:
                    row.append(0) # Or np.nan if you prefer to show diagonals differently
                else:
                    row.append(win_matrix_data.get(winner_id, {}).get(loser_id, 0))
            df_data.append(row)
            
        if not df_data:
            logger.warning("Could not form DataFrame for win matrix heatmap.")
            return ""

        df = pd.DataFrame(df_data, index=sorted_model_ids, columns=sorted_model_ids)

        plt.style.use('seaborn-v0_8-whitegrid')
        fig, ax = plt.subplots(figsize=(max(10, len(sorted_model_ids) * 0.8), max(8, len(sorted_model_ids) * 0.7)))
        
        # Using a sequential colormap (e.g., Blues, Reds)
        cmap = sns.color_palette("Blues", as_cmap=True)
        
        sns.heatmap(df, annot=True, fmt="d", cmap=cmap, linewidths=.5, ax=ax, cbar_kws={'label': 'Number of Wins'})
        
        ax.set_title('Model vs. Model Win Matrix Heatmap', fontsize=18, fontweight='bold', pad=20)
        ax.set_xlabel('Losing Model', fontsize=14, labelpad=15)
        ax.set_ylabel('Winning Model', fontsize=14, labelpad=15)
        
        # Set x-axis tick parameters
        ax.tick_params(axis='x', labelsize=12, labelrotation=45)
        for label in ax.get_xticklabels():
            label.set_ha('right') # Set horizontal alignment for rotated labels
            
        ax.tick_params(axis='y', rotation=0, labelsize=12) # rotation=0 is default but kept for clarity
        
        plt.tight_layout()
        
        # Ensure the output directory exists
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        plt.savefig(output_file)
        plt.close(fig)
        logger.info(f"Win matrix heatmap saved to {output_file}")
        return output_file

    def generate_performance_distribution_plot(self, metric_name: str, output_file: str = "performance_dist.png") -> str:
        # Implementation of generate_performance_distribution_plot method
        # This method is not provided in the original file or the code block
        # It's assumed to exist as it's called in the generate_win_matrix_heatmap method
        # Placeholder return, actual implementation needed
        return ""```

## player.py
```python
import random
from typing import List, Optional, Union, Any

class Player:
    def __init__(self, name: str) -> None:
        self.name: str = name
        self.dice: List[int] = []
        self.num_dice: int = 5
    
    def roll_dice(self) -> None:
        self.dice = [random.randint(1, 6) for _ in range(self.num_dice)]
    
    def remove_die(self) -> None:
        if self.num_dice > 0:
            self.num_dice -= 1
            if self.num_dice > 0:
                self.dice = self.dice[:self.num_dice]
            else:
                self.dice = []
    
    def get_dice_count(self) -> int:
        return self.num_dice

class HumanPlayer(Player):
    def __init__(self, name: str) -> None:
        super().__init__(name)

    def get_bid_input(self, game: Any, max_dice_qty: int) -> Optional[tuple[str, Optional[tuple[int, int]]]]:
        """Get bid or liar call from human player."""
        # This method would contain the logic to prompt the human user for their move.
        # For now, it's a placeholder. In a full implementation, you'd handle
        # input parsing, validation against game rules, etc.
        print(f"{self.name}, it's your turn. Your dice: {self.dice}")
        action = input("Enter 'bid' or 'liar': ").lower().strip()
        if action == "bid":
            while True:
                try:
                    qty_str = input(f"Enter quantity (1-{max_dice_qty}): ")
                    if not qty_str: return None # Allow empty input to re-prompt main action
                    quantity = int(qty_str)

                    val_str = input("Enter value (1-6): ")
                    if not val_str: return None # Allow empty input to re-prompt main action
                    value = int(val_str)
                    
                    # Basic validation (more can be added from LiarsDice.is_valid_bid)
                    if not (1 <= quantity <= max_dice_qty and 1 <= value <= 6):
                        print("Invalid quantity or value. Try again.")
                        continue
                    return "bid", (quantity, value)
                except ValueError:
                    print("Invalid input. Please enter numbers for quantity and value.")
                except Exception as e:
                    print(f"An error occurred: {e}")
                    return None # Or handle more gracefully
        elif action == "liar":
            return "liar", None
        else:
            print("Invalid action. Type 'bid' or 'liar'.")
            return None # Re-prompt
```

## utils.py
```python
import os
from typing import List, Optional, Union, Dict, Any, Tuple

def print_intro() -> None:
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


def print_main_menu() -> str:
    """Print the main menu options and get user selection"""
    print("\n" + "=" * 70)
    print("LIARS DICE MAIN MENU")
    print("=" * 70)
    print("1. Play a single game")
    print("2. Run a model tournament")
    print("3. Load a saved game")
    print("4. Exit")
    return input("\nSelect an option (1-4): ").strip()
    
def find_saved_games() -> List[str]:
    """Find all saved game files in the current directory"""
    import glob
    saves: List[str] = glob.glob("liars_dice_save_*.json")
    saves.sort(reverse=True)  # Most recent first
    return saves
    
def select_saved_game() -> Optional[str]:
    """Let user select a saved game from available saves"""
    saves: List[str] = find_saved_games()
    
    if not saves:
        print("No saved games found.")
        return None
    
    print("\nAvailable saved games:")
    for i, save in enumerate(saves):
        print(f"{i+1}. {save}")
    
    try:
        choice: int = int(input("\nSelect a game to load (number) or 0 to cancel: "))
        if choice == 0:
            return None
        elif 1 <= choice <= len(saves):
            return saves[choice-1]
        else:
            print("Invalid selection.")
            return None
    except ValueError:
        print("Invalid input.")
        return None
        
# Add any other utility functions here with type hints

def validate_bid(bid: Tuple[int, int], last_bid: Optional[Tuple[int, int]] = None) -> bool:
    """
    Validate if a bid is legal according to game rules
    
    Args:
        bid: A tuple of (quantity, face value)
        last_bid: The previous bid to compare against (if any)
        
    Returns:
        True if the bid is valid, False otherwise
    """
    quantity, face = bid
    
    # Basic validation
    if quantity < 1 or face < 1 or face > 6:
        return False
        
    # If there's no previous bid, any valid bid is acceptable
    if not last_bid:
        return True
        
    # Check against previous bid
    last_quantity, last_face = last_bid
    
    # Higher quantity always wins
    if quantity > last_quantity:
        return True
        
    # Same quantity but higher face
    if quantity == last_quantity and face > last_face:
        return True
        
    # Otherwise, bid is not higher
    return False

def format_time(seconds: float) -> str:
    """Format seconds into a readable time string"""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        secs = seconds % 60
        return f"{minutes}m {secs:.1f}s"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        return f"{hours}h {minutes}m"
```

