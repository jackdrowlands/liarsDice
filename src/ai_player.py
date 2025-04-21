import os
import time
import json
import random
import re
from .player import Player

# Check for required libraries
try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False

# Define API provider constants
PROVIDER_OPENROUTER = "openrouter"
PROVIDER_LOCAL = "local"
GOOGLE_SDK_AVAILABLE = False

class AIPlayer(Player):
    def __init__(self, name, model, provider=PROVIDER_OPENROUTER, api_key=None, api_url=None, project_id=None, region=None):
        super().__init__(name)
        self.model = model
        self.game_history = []
        self.provider = provider
        self.api_url = api_url
        self.project_id = None
        self.region = None
        
        # Set API key based on provider
        if api_key:
            self.api_key = api_key
        elif provider == PROVIDER_OPENROUTER:
            self.api_key = os.environ.get("OPENROUTER_API_KEY", "")
        else:
            self.api_key = ""
        
        # Check if using local endpoint
        self.is_local_endpoint = self.api_url and "127.0.0.1" in self.api_url

    def record_game_state(self, game_state):
        self.game_history.append(game_state)
    
    def get_prompt_for_game(self, game_state):
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
        if game_state['move_history']:
            for i, move in enumerate(game_state['move_history']):
                player_name = move['player']
                
                if move["action"] == "bid":
                    action_summary = f"{player_name}: {{\"action\": \"bid\", \"quantity\": {move['quantity']}, \"face\": {move['value']}"
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
        if game_state['last_bid']:
            current_bid = f"{game_state['last_bid'][0]} × {game_state['last_bid'][1]}"
        
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
    
    def get_prompt_and_params(self, game_state):
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
            
    def _get_openai_compatible_params(self, prompt, game_state):
        """Get parameters for OpenAI-compatible APIs (OpenRouter and local)"""
        # Prepare headers
        headers = {
            "Content-Type": "application/json"
        }
        
        # Add authorization for cloud endpoints
        if not self.is_local_endpoint and self.provider == PROVIDER_OPENROUTER:
            headers["Authorization"] = f"Bearer {self.api_key}"
            headers["HTTP-Referer"] = ""
        
        # Prepare request data with the new JSON format for messages
        system_message = {
            "role": "system",
            "content": prompt["system"]
        }
        
        user_message = {
            "role": "user",
            "content": prompt["user"]
        }
        
        # Define the schema for structured output - used for all endpoints
        schema = {
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
        data = {
            "model": self.model,
            "messages": [system_message, user_message],
            "temperature": 0.7,  # Balanced temperature for creativity in utterances
        }
        
        # For OpenRouter, include structured output format 
        if not self.is_local_endpoint and self.provider == PROVIDER_OPENROUTER:
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
        if self.api_url:
            endpoint_url = self.api_url
        else:
            endpoint_url = "https://openrouter.ai/api/v1/chat/completions"
            
        return {
            "endpoint_url": endpoint_url,
            "headers": headers,
            "data": data,
            "game_state": game_state,
            "provider": self.provider
        }
        
    
    def get_ai_decision(self, game_state):
        """Get a decision from the AI model by calling the API"""
        request_params = self.get_prompt_and_params(game_state)
        provider = request_params["provider"]
        
        # Store provider in game_state for use in process_api_response
        game_state['provider'] = provider
        
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
            error_info = {
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
            
            
    def process_api_response(self, response, response_time, game_state):
        """Process the API response with structured output format"""
        # Get the provider from the request params
        provider = game_state.get('provider', self.provider)
        
        # OpenAI-compatible response format (OpenRouter/local)
        result = response.json()
        
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
        content = result["choices"][0]["message"]["content"].strip()
            
        # Log the response with token information and prompt
        with open("llm_responses.json", "a") as f:
                # Create response log object
                response_log = {
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
                response_json = json.loads(content)
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
            invalid_response = {
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
                
            # Return a valid fallback bid based on current game state
            if not game_state.get('last_bid'):
                # First bid in the round - make a safe default bid
                return {
                    "action": "bid", 
                    "quantity": 1, 
                    "face": 4,
                    "reasoning": "Error processing response, using default bid.",
                    "utterance": "I'll make a simple bid."
                }
            else:
                # Get the last bid
                last_quantity, last_value = game_state['last_bid']
                total_dice = game_state.get('total_dice', 0)
                
                # If the last bid is implausible (higher than total dice), call liar
                if last_quantity > total_dice:
                    return {
                        "action": "liar",
                        "quantity": 0,
                        "face": 0,
                        "reasoning": "Error processing response. Last bid exceeds total dice, calling liar.",
                        "utterance": "That's impossible! I call."
                    }
                # Otherwise make a valid higher bid
                else:
                    if last_value < 6:
                        # Increase face value
                        return {
                            "action": "bid", 
                            "quantity": last_quantity, 
                            "face": last_value + 1,
                            "reasoning": "Error processing response, making minimal valid higher bid.",
                            "utterance": "Let me increase that bid."
                        }
                    else:
                        # Increase quantity, reset face to 1
                        return {
                            "action": "bid", 
                            "quantity": last_quantity + 1, 
                            "face": 1,
                            "reasoning": "Error processing response, making minimal valid higher bid.",
                            "utterance": "I'll raise the quantity."
                        }
