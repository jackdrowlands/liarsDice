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
        """Create a prompt for the AI model"""            
        # Format the prompt with game information - same for all models
        system_prompt = f"""
        You are playing Liar's Dice. In this game, each player has dice that only they can see.
        Players take turns making bids about how many dice of a certain value exist among all players.
        Each bid must be higher than the previous one (either more dice, or same number but higher value).
        When a player thinks the previous bid is a lie, they can call "Liar".
        
        Rules for making decisions:
        1. You can either make a higher bid or call the previous player a liar
        2. A bid consists of a quantity and a value (e.g., "3 4's" means "three dice with value 4")
        3. A bid must increase either the quantity or the value of the previous bid
        4. Be strategic - consider probability and bluffing
        5. Return your decision in JSON format as specified
        
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
        
        # Prepare request data
        data = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": prompt["system"]},
                {"role": "user", "content": prompt["user"]},
            ],
        }
        
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
                    
            # Log the error
            with open("invalid_responses.json", "a") as f:
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
        """Process the API response and extract the decision"""
        # Get the provider from the request params
        provider = game_state.get('provider', self.provider)
        
        # OpenAI-compatible response format (OpenRouter/local)
        result = response.json()
        content = result["choices"][0]["message"]["content"].strip()
        
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
            
        # Log the response with token information
        with open("llm_responses.json", "a") as f:
                json.dump({
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "response_text": content,
                    "model": self.model,
                    "provider": provider,
                    "response_time": response_time,
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": total_tokens
                }, f)
                f.write("\n")
        
        # Store response time in game state for metrics collection
        game_state['response_time'] = response_time
        
        try:
            json_objects = re.findall(r'\{[^{}]*\}', content)
            number_words = {
                "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4,
                "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9,
                "ten": 10
            }

            for obj in json_objects:
                try:
                    parsed = json.loads(obj)
                    if isinstance(parsed, dict) and "action" in parsed:
                        if parsed["action"] == "bid":
                            # Convert number words to integers if needed
                            for key in ["quantity", "value"]:
                                if isinstance(parsed.get(key), str):
                                    word = parsed[key].lower()
                                    if word in number_words:
                                        parsed[key] = number_words[word]
                        return parsed
                except json.JSONDecodeError:
                    continue
            raise ValueError("No valid JSON with 'action' found.")
        except Exception:
            print(content)
            with open("invalid_llm_responses.json", "a") as f:
                json.dump({
                    "model": self.model,
                    "provider": provider,
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "response_text": content
                }, f)
                f.write("\n")
            raise ValueError("No valid JSON found in LLM response.")
            
