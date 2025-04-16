import math
import numpy as np
from collections import defaultdict

class GameMetrics:
    """
    Handles computation and tracking of game metrics as specified in Section 3.5
    of the evaluation metrics document.
    """
    
    def __init__(self):
        self.elo_ratings = {}        # Track Elo ratings for each model
        self.game_results = []       # Store game results
        self.bluff_data = defaultdict(lambda: {"successful": 0, "total": 0})
        self.lie_detection_data = defaultdict(lambda: {"true_positive": 0, "false_positive": 0, "false_negative": 0})
        self.final_bids = defaultdict(list)  # Track final bids in rounds
        self.bid_optimality = defaultdict(lambda: {"optimal": 0, "total": 0})
        self.adaptation_scores = defaultdict(lambda: {"adapted": 0, "opportunities": 0})
        self.rule_adherence = defaultdict(lambda: {"valid_actions": 0, "total_actions": 0})
        self.api_response_times = defaultdict(list)  # Track API response times
        
        # Token usage tracking
        self.token_usage = defaultdict(lambda: {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "actions": 0})
        
        # For tracking Elo
        self.k_factor = 32  # Standard K-factor for Elo calculation
        self.default_elo = 1000  # Starting Elo rating
    
    def initialize_model(self, model_id):
        """Initialize a new model's metrics if it doesn't exist"""
        if model_id not in self.elo_ratings:
            self.elo_ratings[model_id] = self.default_elo
    
    def update_elo(self, winner_id, loser_id):
        """Update Elo ratings after a game"""
        self.initialize_model(winner_id)
        self.initialize_model(loser_id)
        
        # Calculate expected win probabilities
        r1 = self.elo_ratings[winner_id]
        r2 = self.elo_ratings[loser_id]
        
        # Expected score for winner
        expected_winner = 1 / (1 + 10 ** ((r2 - r1) / 400))
        
        # Update ratings
        self.elo_ratings[winner_id] += self.k_factor * (1 - expected_winner)
        self.elo_ratings[loser_id] += self.k_factor * (0 - (1 - expected_winner))
        
        # Record the game result
        self.game_results.append({
            "winner": winner_id,
            "loser": loser_id,
            "winner_new_elo": self.elo_ratings[winner_id],
            "loser_new_elo": self.elo_ratings[loser_id]
        })
    
    def record_bluff(self, model_id, bluff_type, success):
        """
        Record bluffing success/failure
        bluff_type: 'bluff' (false bid) or 'truth' (truthful bid challenged)
        success: Whether the bluff succeeded or truthful bid was incorrectly challenged
        """
        self.bluff_data[model_id]["total"] += 1
        if success:
            self.bluff_data[model_id]["successful"] += 1
    
    def record_lie_detection(self, model_id, detection_type):
        """
        Record lie detection outcome
        detection_type: 'true_positive' (correctly called bluff), 
                        'false_positive' (incorrectly called bluff),
                        'false_negative' (missed opponent's bluff)
        """
        self.lie_detection_data[model_id][detection_type] += 1
    
    def record_final_bid(self, model_id, bid, round_number):
        """Record the final bid in a round where the model participated"""
        # Store the final bid value and the round number
        self.final_bids[model_id].append((bid, round_number))
    
    def record_bid_optimality(self, model_id, is_optimal):
        """Record whether a bid was mathematically optimal"""
        self.bid_optimality[model_id]["total"] += 1
        if is_optimal:
            self.bid_optimality[model_id]["optimal"] += 1
    
    def record_adaptation(self, model_id, adapted_to_opponent):
        """Record whether the model adapted its strategy"""
        self.adaptation_scores[model_id]["opportunities"] += 1
        if adapted_to_opponent:
            self.adaptation_scores[model_id]["adapted"] += 1
    
    def record_rule_adherence(self, model_id, valid_action):
        """Record whether the model followed game rules"""
        self.rule_adherence[model_id]["total_actions"] += 1
        if valid_action:
            self.rule_adherence[model_id]["valid_actions"] += 1
            
    def record_api_response_time(self, model_id, response_time):
        """Record the API response time for a model"""
        if response_time is not None and response_time > 0:
            self.api_response_times[model_id].append(response_time)
    
    def calculate_bluff_success_rate(self, model_id):
        """Calculate the bluff success rate for a model"""
        data = self.bluff_data[model_id]
        if data["total"] == 0:
            return 0
        return (data["successful"] / data["total"]) * 100
    
    def calculate_lie_detection_rate(self, model_id):
        """Calculate the lie detection rate for a model"""
        data = self.lie_detection_data[model_id]
        tp = data["true_positive"]
        fp = data["false_positive"]
        fn = data["false_negative"]
        
        # Calculate precision and recall if possible
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        
        # Calculate F1 score (harmonic mean of precision and recall)
        if precision + recall > 0:
            f1_score = 2 * (precision * recall) / (precision + recall)
        else:
            f1_score = 0
            
        return {
            "precision": precision * 100,
            "recall": recall * 100,
            "f1_score": f1_score * 100
        }
    
    def calculate_average_final_bid(self, model_id):
        """Calculate the average final bid for a model"""
        bids = [bid[0] for bid in self.final_bids[model_id]]
        if not bids:
            return 0
        return sum(bids) / len(bids)
    
    def calculate_bid_optimality(self, model_id):
        """Calculate the bid optimality percentage for a model"""
        data = self.bid_optimality[model_id]
        if data["total"] == 0:
            return 0
        return (data["optimal"] / data["total"]) * 100
    
    def calculate_adaptation_score(self, model_id):
        """Calculate the adaptation score for a model"""
        data = self.adaptation_scores[model_id]
        if data["opportunities"] == 0:
            return 0
        return (data["adapted"] / data["opportunities"]) * 100
    
    def calculate_rule_adherence_rate(self, model_id):
        """Calculate the rule adherence rate for a model"""
        data = self.rule_adherence[model_id]
        if data["total_actions"] == 0:
            return 100  # If no actions yet, assume perfect adherence
        return (data["valid_actions"] / data["total_actions"]) * 100
        
    def calculate_avg_api_response_time(self, model_id):
        """Calculate the average API response time for a model"""
        times = self.api_response_times[model_id]
        if not times:
            return 0
        return sum(times) / len(times)
    
    def record_token_usage(self, model_id, prompt_tokens, completion_tokens, total_tokens=None):
        """Record token usage for an API call"""
        if total_tokens is None:
            total_tokens = prompt_tokens + completion_tokens
            
        self.token_usage[model_id]["prompt_tokens"] += prompt_tokens
        self.token_usage[model_id]["completion_tokens"] += completion_tokens
        self.token_usage[model_id]["total_tokens"] += total_tokens
        self.token_usage[model_id]["actions"] += 1
    
    def calculate_token_metrics(self, model_id):
        """Calculate token usage metrics for a model"""
        data = self.token_usage[model_id]
        actions = data["actions"]
        
        if actions == 0:
            return {
                "total_prompt_tokens": 0,
                "total_completion_tokens": 0,
                "total_tokens": 0,
                "avg_prompt_tokens_per_action": 0,
                "avg_completion_tokens_per_action": 0, 
                "avg_tokens_per_action": 0
            }
            
        return {
            "total_prompt_tokens": data["prompt_tokens"],
            "total_completion_tokens": data["completion_tokens"],
            "total_tokens": data["total_tokens"],
            "avg_prompt_tokens_per_action": data["prompt_tokens"] / actions,
            "avg_completion_tokens_per_action": data["completion_tokens"] / actions,
            "avg_tokens_per_action": data["total_tokens"] / actions
        }
        
    def get_model_metrics(self, model_id):
        """Get comprehensive metrics for a specific model"""
        return {
            "elo_rating": self.elo_ratings.get(model_id, self.default_elo),
            "bluff_success_rate": self.calculate_bluff_success_rate(model_id),
            "lie_detection": self.calculate_lie_detection_rate(model_id),
            "average_final_bid": self.calculate_average_final_bid(model_id),
            "bid_optimality": self.calculate_bid_optimality(model_id),
            "adaptation_score": self.calculate_adaptation_score(model_id),
            "rule_adherence_rate": self.calculate_rule_adherence_rate(model_id),
            "avg_api_response_time": self.calculate_avg_api_response_time(model_id),
            "token_usage": self.calculate_token_metrics(model_id)
        }
    
    def get_all_metrics(self):
        """Get metrics for all models"""
        metrics = {}
        for model_id in self.elo_ratings:
            metrics[model_id] = self.get_model_metrics(model_id)
        return metrics

class BidAnalyzer:
    """
    Helper class to analyze bid optimality and probabilistic reasoning
    in Liar's Dice.
    """
    
    @staticmethod
    def is_bid_optimal(dice_values, player_dice_count, total_dice, last_bid, current_bid):
        """
        Determine if a bid is mathematically optimal based on known information
        This is a simplified approximation - a full game theory solution would be much more complex
        
        Args:
            dice_values: Current player's dice values
            player_dice_count: Number of dice the player has
            total_dice: Total number of dice in the game
            last_bid: Last bid as (quantity, value)
            current_bid: Current bid as (quantity, value)
            
        Returns:
            Boolean indicating if the bid appears optimal
        """
        if not last_bid:  # First bid in the round
            # Check if player is bidding on values they have
            player_has = dice_values.count(current_bid[1])
            unknown_dice = total_dice - player_dice_count
            
            # A bid is likely reasonable if:
            # 1. The player has at least one of the value
            # 2. The quantity is not too high relative to total dice
            expected = (player_has + (unknown_dice * 1/6))  # Expected number with 1/6 probability
            return current_bid[0] <= expected * 1.5  # Allow some bluffing
        
        # Comparing to last bid
        last_quantity, last_value = last_bid
        curr_quantity, curr_value = current_bid
        
        # Determine player's known dice matching the bid value
        player_has = dice_values.count(curr_value)
        
        # Unknown dice that could match the bid value
        unknown_dice = total_dice - player_dice_count
        
        # Estimated probability of the current bid being achievable
        # Assuming uniform distribution for unknown dice (1/6 chance per die)
        prob_per_unknown = 1/6
        if curr_value == 1:  # If bidding on ones (which might be wild)
            prob_per_unknown = 1/6  # Adjust if ones are wild in your rules
            
        # Number of additional dice needed to meet the bid
        additional_needed = curr_quantity - player_has
        
        if additional_needed <= 0:
            # Player already has enough dice to satisfy the bid
            return True
            
        if additional_needed > unknown_dice:
            # Impossible bid - requires more dice than remain unknown
            return False
            
        # Calculate probability using binomial probability
        # P(at least k successes in n trials) = 1 - P(less than k successes)
        # Using cumulative binomial distribution
        prob = 1 - BidAnalyzer._binom_cdf(additional_needed - 1, unknown_dice, prob_per_unknown)
        
        # A bid is considered optimal if it has reasonable probability (adjustable threshold)
        return prob >= 0.3  # 30% chance of being true
    
    @staticmethod
    def _binom_cdf(k, n, p):
        """Calculate cumulative binomial probability P(X ≤ k) with n trials and probability p"""
        result = 0
        for i in range(k + 1):
            result += BidAnalyzer._binom_pmf(i, n, p)
        return result
    
    @staticmethod
    def _binom_pmf(k, n, p):
        """Calculate binomial probability mass function P(X = k) with n trials and probability p"""
        if k < 0 or k > n:
            return 0
        return math.comb(n, k) * (p ** k) * ((1 - p) ** (n - k))
    
    @staticmethod
    def should_call_liar(dice_values, player_dice_count, total_dice, last_bid):
        """
        Determine if calling "liar" is mathematically optimal
        
        Args:
            dice_values: Current player's dice values
            player_dice_count: Number of dice the player has
            total_dice: Total number of dice in the game
            last_bid: Last bid as (quantity, value)
            
        Returns:
            Boolean indicating if calling "liar" appears optimal
        """
        if not last_bid:
            return False  # Can't call liar on no bid
            
        quantity, value = last_bid
        
        # Count how many of the bid value the player already has
        player_has = dice_values.count(value)
        
        # Total number of dice with this value needed to exist
        needed = quantity - player_has
        
        # Number of unknown dice
        unknown_dice = total_dice - player_dice_count
        
        # If more are needed than exist, calling liar is optimal
        if needed > unknown_dice:
            return True
            
        # Probability that the remaining dice have enough of the value
        prob_per_die = 1/6
        
        # Calculate probability using binomial distribution
        prob = 1 - BidAnalyzer._binom_cdf(needed - 1, unknown_dice, prob_per_die)
        
        # Call liar if probability is low enough
        return prob < 0.25  # 25% or less chance the bid is true
    
    @staticmethod
    def check_adaptation(player_history, opponent_history):
        """
        Check if a player has adapted their strategy based on opponent behavior
        
        Args:
            player_history: List of player's actions across multiple games
            opponent_history: List of opponent's actions across multiple games
            
        Returns:
            Boolean indicating if adaptation is detected
        """
        # This is a simplified implementation
        # A real implementation would analyze patterns more thoroughly
        if len(player_history) < 3 or len(opponent_history) < 3:
            return False  # Not enough history to detect adaptation
            
        # Check if opponent frequently calls liar
        opponent_liar_calls = sum(1 for action in opponent_history[-10:] if action["action"] == "liar")
        opponent_liar_freq = opponent_liar_calls / len(opponent_history[-10:])
        
        # Check if player's bluffing frequency has adjusted to opponent behavior
        player_early_bluffs = sum(1 for action in player_history[:10] 
                              if action["action"] == "bid" and action.get("bluff", False))
        player_early_bluff_freq = player_early_bluffs / min(10, len(player_history[:10]))
        
        player_recent_bluffs = sum(1 for action in player_history[-10:] 
                               if action["action"] == "bid" and action.get("bluff", False))
        player_recent_bluff_freq = player_recent_bluffs / len(player_history[-10:])
        
        # If opponent frequently calls liar and player reduced bluffing, that's adaptation
        if opponent_liar_freq > 0.4 and player_recent_bluff_freq < player_early_bluff_freq * 0.7:
            return True
            
        # If opponent rarely calls liar and player increased bluffing, that's adaptation
        if opponent_liar_freq < 0.2 and player_recent_bluff_freq > player_early_bluff_freq * 1.3:
            return True
            
        return False