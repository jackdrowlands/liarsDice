import os
import time
import json
import random
import re
from .player import Player

# Check for OpenRouter support (required)
try:
    import requests
    OPENROUTER_AVAILABLE = True
except ImportError:
    OPENROUTER_AVAILABLE = False

class AIPlayer(Player):
    def __init__(self, name, model, api_key=None, api_url=None):
        super().__init__(name)
        self.model = model
        self.game_history = []
        self.api_key = api_key or os.environ.get("OPENROUTER_API_KEY", "")
        self.api_url = api_url
        
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
        2. A bid consists of a quantity and a value (e.g., "three 4's" means "three dice with value 4")
        3. A bid must increase either the quantity or the value of the previous bid
        4. Be strategic - consider probability and bluffing
        5. Return your decision in JSON format as specified
        
        Strategy tips:
        1. Calculate probability distributions for each value based on visible dice
        2. Track each player's behavior patterns over time
        3. Use strategic bluffing when appropriate
        4. Consider the risk/reward of calling "Liar" vs making a higher bid
        5. Use the move history to understand each player's tendencies
        6. Consider how many dice are left in the game when calculating probabilities
        
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
        
        # No API key needed for local endpoints
        if not self.api_key and not self.is_local_endpoint:
            raise ValueError("API key is required for cloud endpoints")
        
        prompt = self.get_prompt_for_game(game_state)
        
        try:
            # Prepare headers based on endpoint type
            headers = {
                "Content-Type": "application/json"
            }
            
            # Add authorization for cloud endpoints
            if not self.is_local_endpoint:
                headers["Authorization"] = f"Bearer {self.api_key}"
                headers["HTTP-Referer"] = ""
            
            # Prepare request data with common structure
            data = {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": prompt["system"]},
                    {"role": "user", "content": prompt["user"]},
                ],
            }
            
            # Determine the correct endpoint URL
            endpoint_url = "https://openrouter.ai/api/v1/chat/completions"
            if self.api_url:
                endpoint_url = self.api_url
            
            # Make the API request
            response = requests.post(
                endpoint_url,
                headers=headers,
                json=data,
            )
            
            if response.status_code != 200:
                raise RuntimeError(f"API Error: {response.status_code} - {response.text}")
            
            result = response.json()
            content = result["choices"][0]["message"]["content"].strip()
            with open("llm_responses.json", "a") as f:
                    json.dump({
                        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                        "response_text": content,
                        "model": self.model,
                    }, f)
                    f.write("\n")
            
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
                        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                        "response_text": content
                    }, f)
                    f.write("\n")
                raise ValueError("No valid JSON found in LLM response.")
        except Exception as e:
            with open("invalid_responses.json", "a") as f:
                    json.dump({
                        "model": self.model,
                        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                        "response": response.json(),
                        "error": str(e),
                    }, f)
                    f.write("\n")
            print(response.json())
            raise RuntimeError(f"Error getting AI decision: {e}")
