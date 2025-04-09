import os
import time
import json
import random
from .player import Player

# Check for OpenRouter support (required)
try:
    import requests
    OPENROUTER_AVAILABLE = True
except ImportError:
    OPENROUTER_AVAILABLE = False

class AIPlayer(Player):
    def __init__(self, name, model, api_key=None):
        super().__init__(name)
        self.model = model
        self.game_history = []
        self.api_key = api_key or os.environ.get("OPENROUTER_API_KEY", "")

    
    def record_game_state(self, game_state):
        self.game_history.append(game_state)
    
    def get_prompt_for_game(self, game_state):
        """Create a prompt for the AI model"""
        # Use the same strategy prompt for all models to ensure fair comparison
        strategy_style = """
        Strategy tips:
        1. Calculate probability distributions for each value based on visible dice
        2. Track each player's behavior patterns over time
        3. Use strategic bluffing when appropriate
        4. Consider the risk/reward of calling "Liar" vs making a higher bid
        5. Use the move history to understand each player's tendencies
        6. Consider how many dice are left in the game when calculating probabilities
        """
            
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
