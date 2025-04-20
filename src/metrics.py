import math
import numpy as np
import time
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
from matplotlib.colors import LinearSegmentedColormap
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Set, Optional, Union, Any

class GameMetrics:
    """
    Enhanced metrics tracking system for Liar's Dice.
    Provides comprehensive analytics on model performance, strategic patterns,
    and statistical analysis of gameplay data.
    """
    
    def __init__(self):
        # Core metrics tracking
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
        
        # Advanced metrics tracking
        self.historical_metrics = defaultdict(lambda: defaultdict(list))  # Time-series metrics data
        self.matchup_stats = defaultdict(lambda: defaultdict(lambda: {"wins": 0, "losses": 0, "games": 0}))  # Model vs model stats
        self.opponent_adaptation = defaultdict(lambda: defaultdict(lambda: {"adapted": 0, "opportunities": 0}))  # Per-opponent adaptation
        self.strategy_patterns = defaultdict(lambda: {"early_game": [], "mid_game": [], "late_game": []})  # Strategic patterns by game phase
        self.confidence_metrics = defaultdict(list)  # For confidence interval calculations
        
        # Game state and temporal analysis
        self.temporal_metrics = defaultdict(lambda: {"round_history": [], "evolution": {}})
        self.model_consistency = defaultdict(list)  # Track consistency of metrics over time
        
        # Settings for analysis
        self.historical_window_size = 10  # Number of data points to keep for rolling statistics
        self.statistical_confidence = 0.95  # Default confidence level for statistical tests
        
        # Game context
        self.timestamp = time.time()  # When metrics tracking began
    
    def initialize_model(self, model_id):
        """Initialize a new model's metrics if it doesn't exist"""
        if model_id not in self.elo_ratings:
            self.elo_ratings[model_id] = self.default_elo
    
    def update_elo(self, winner_id, loser_id, game_data=None):
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
        old_winner_elo = self.elo_ratings[winner_id]
        old_loser_elo = self.elo_ratings[loser_id]
        
        # Calculate expected win probabilities
        r1 = self.elo_ratings[winner_id]
        r2 = self.elo_ratings[loser_id]
        
        # Expected score for winner
        expected_winner = 1 / (1 + 10 ** ((r2 - r1) / 400))
        
        # Update ratings
        self.elo_ratings[winner_id] += self.k_factor * (1 - expected_winner)
        self.elo_ratings[loser_id] += self.k_factor * (0 - (1 - expected_winner))
        
        # Calculate the "surprise factor" - how unexpected was this outcome?
        surprise_factor = 1 - expected_winner  # Higher means more surprising
        
        # Record temporal Elo progression
        timestamp = time.time()
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
            for model2, stats in opponents.items():
                if stats["games"] > 0:
                    win_rate = stats["wins"] / stats["games"]
                    r1 = self.elo_ratings[model1]
                    r2 = self.elo_ratings[model2]
                    expected_win_rate = 1 / (1 + 10 ** ((r2 - r1) / 400))
                    stats["win_rate"] = win_rate
                    stats["expected_win_rate"] = expected_win_rate
                    stats["performance_index"] = win_rate / expected_win_rate if expected_win_rate > 0 else float('inf')
        
        # Record the game result
        game_result = {
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
    
    def record_bluff(self, model_id, bluff_type, success, game_context=None):
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
        bluff_type_key = f"{bluff_type}_total"
        success_type_key = f"{bluff_type}_successful"
        
        if bluff_type_key not in self.bluff_data[model_id]:
            self.bluff_data[model_id][bluff_type_key] = 0
            self.bluff_data[model_id][success_type_key] = 0
            
        self.bluff_data[model_id][bluff_type_key] += 1
        if success:
            self.bluff_data[model_id][success_type_key] += 1
            
        # Record historical data
        bluff_record = {
            "timestamp": time.time(),
            "bluff_type": bluff_type,
            "success": success,
            "success_rate": self.calculate_bluff_success_rate(model_id)
        }
        
        # Add game context if provided
        if game_context:
            # Extract round info for game phase analysis
            if "round_number" in game_context:
                round_num = game_context["round_number"]
                total_rounds = game_context.get("total_rounds", 30)  # Default estimate
                
                # Categorize game phase
                if round_num <= total_rounds * 0.3:
                    phase = "early_game"
                elif round_num <= total_rounds * 0.7:
                    phase = "mid_game"
                else:
                    phase = "late_game"
                    
                # Record for phase-specific analysis
                self.strategy_patterns[model_id][phase].append(bluff_record)
                
            # Track opponent-specific bluffing if opponent is provided
            if "opponent" in game_context:
                opponent = game_context["opponent"]
                if "opponents" not in self.bluff_data[model_id]:
                    self.bluff_data[model_id]["opponents"] = defaultdict(lambda: {"total": 0, "successful": 0})
                    
                self.bluff_data[model_id]["opponents"][opponent]["total"] += 1
                if success:
                    self.bluff_data[model_id]["opponents"][opponent]["successful"] += 1
                    
            bluff_record.update(game_context)
            
        # Add to historical metrics
        self.historical_metrics[model_id]["bluffs"].append(bluff_record)
        
        # Update confidence metrics list for statistical analysis
        self.confidence_metrics[f"{model_id}_bluff"].append(1 if success else 0)
    
    def record_lie_detection(self, model_id, detection_type, game_context=None):
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
                
                if opponent_key not in self.lie_detection_data[model_id]:
                    self.lie_detection_data[model_id][opponent_key] = {
                        "true_positive": 0, "false_positive": 0, "false_negative": 0
                    }
                    
                self.lie_detection_data[model_id][opponent_key][detection_type] += 1
                
            # Track reasoning if provided
            if "reasoning" in game_context:
                if "reasoning_patterns" not in self.lie_detection_data[model_id]:
                    self.lie_detection_data[model_id]["reasoning_patterns"] = []
                    
                self.lie_detection_data[model_id]["reasoning_patterns"].append({
                    "detection_type": detection_type,
                    "reasoning": game_context["reasoning"]
                })
                
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
    
    def record_final_bid(self, model_id, bid, round_number, game_context=None):
        """
        Record the final bid in a round with enhanced tracking
        
        Args:
            model_id: ID of the model making the bid
            bid: The numeric value of the bid
            round_number: Current round number
            game_context: Optional dict with additional context
        """
        # Initialize if needed
        self.initialize_model(model_id)
        
        # Store the final bid value and the round number
        self.final_bids[model_id].append((bid, round_number))
        
        # Create bid record with temporal tracking
        bid_record = {
            "timestamp": time.time(),
            "bid_value": bid,
            "round_number": round_number,
            "current_avg": self.calculate_average_final_bid(model_id)
        }
        
        # Add game context if provided
        if game_context:
            bid_record.update(game_context)
            
            # Track opponent-specific bidding patterns
            if "opponent" in game_context:
                opponent = game_context["opponent"]
                bid_key = f"bids_vs_{opponent}"
                
                if bid_key not in self.historical_metrics[model_id]:
                    self.historical_metrics[model_id][bid_key] = []
                    
                self.historical_metrics[model_id][bid_key].append(bid_record)
                
            # Track dice information if provided
            if "dice_values" in game_context and "total_dice" in game_context:
                dice_values = game_context["dice_values"]
                total_dice = game_context["total_dice"]
                player_dice_count = len(dice_values)
                
                # Calculate bid aggressiveness - how much the bid exceeds expected dice count
                expected_count = dice_values.count(bid) + (total_dice - player_dice_count) / 6
                aggressiveness = bid / expected_count if expected_count > 0 else float('inf')
                bid_record["aggressiveness"] = aggressiveness
                
                # Track percentile rank of bid among all of this player's bids
                all_bids = [b[0] for b in self.final_bids[model_id]]
                if all_bids:
                    percentile = sum(1 for b in all_bids if b <= bid) / len(all_bids)
                    bid_record["percentile"] = percentile
        
        # Add to historical metrics
        self.historical_metrics[model_id]["final_bids"].append(bid_record)
        
        # Track bid progression through the game
        if "round_history" not in self.temporal_metrics[model_id]:
            self.temporal_metrics[model_id]["round_history"] = []
            
        self.temporal_metrics[model_id]["round_history"].append({
            "round": round_number,
            "bid": bid
        })
    
    def record_bid_optimality(self, model_id, is_optimal, game_context=None):
        """
        Record whether a bid was mathematically optimal with detailed analysis
        
        Args:
            model_id: ID of the model making the bid
            is_optimal: Whether the bid was mathematically optimal
            game_context: Optional dict with additional context
        """
        # Initialize if needed
        self.initialize_model(model_id)
        
        # Update core metrics
        self.bid_optimality[model_id]["total"] += 1
        if is_optimal:
            self.bid_optimality[model_id]["optimal"] += 1
            
        # Create optimality record
        opt_record = {
            "timestamp": time.time(),
            "is_optimal": is_optimal,
            "current_rate": self.calculate_bid_optimality(model_id)
        }
        
        # Add game context and analyze
        if game_context:
            opt_record.update(game_context)
            
            # Track by game phase if round information available
            if "round_number" in game_context and "total_rounds" in game_context:
                round_num = game_context["round_number"]
                total_rounds = game_context["total_rounds"]
                
                # Categorize game phase
                phase = None
                if round_num <= total_rounds * 0.3:
                    phase = "early_game"
                elif round_num <= total_rounds * 0.7:
                    phase = "mid_game"
                else:
                    phase = "late_game"
                    
                # Track phase-specific optimality
                phase_key = f"optimality_{phase}"
                if phase_key not in self.bid_optimality[model_id]:
                    self.bid_optimality[model_id][phase_key] = {"total": 0, "optimal": 0}
                    
                self.bid_optimality[model_id][phase_key]["total"] += 1
                if is_optimal:
                    self.bid_optimality[model_id][phase_key]["optimal"] += 1
                    
            # Track dice-specific optimality if dice info available
            if "dice_values" in game_context:
                dice_values = game_context["dice_values"]
                dice_key = "dice_combinations"
                
                if dice_key not in self.bid_optimality[model_id]:
                    self.bid_optimality[model_id][dice_key] = {}
                    
                # Create a simple hash of the dice combination
                dice_hash = ''.join(map(str, sorted(dice_values)))
                if dice_hash not in self.bid_optimality[model_id][dice_key]:
                    self.bid_optimality[model_id][dice_key][dice_hash] = {"total": 0, "optimal": 0}
                    
                self.bid_optimality[model_id][dice_key][dice_hash]["total"] += 1
                if is_optimal:
                    self.bid_optimality[model_id][dice_key][dice_hash]["optimal"] += 1
                
        # Add to historical metrics
        self.historical_metrics[model_id]["bid_optimality"].append(opt_record)
        
        # Update confidence metrics for statistical analysis
        self.confidence_metrics[f"{model_id}_bid_optimality"].append(1 if is_optimal else 0)
    
    def record_adaptation(self, model_id, adapted_to_opponent, opponent_id=None, game_context=None):
        """
        Record whether the model adapted its strategy with opponent tracking
        
        Args:
            model_id: ID of the model adapting
            adapted_to_opponent: Whether adaptation was detected
            opponent_id: Optional ID of the opponent
            game_context: Optional dict with additional context
        """
        # Initialize if needed
        self.initialize_model(model_id)
        
        # Update core metrics
        self.adaptation_scores[model_id]["opportunities"] += 1
        if adapted_to_opponent:
            self.adaptation_scores[model_id]["adapted"] += 1
            
        # Create adaptation record
        adapt_record = {
            "timestamp": time.time(),
            "adapted": adapted_to_opponent,
            "current_score": self.calculate_adaptation_score(model_id)
        }
        
        # Track opponent-specific adaptation if opponent provided
        if opponent_id:
            adapt_record["opponent"] = opponent_id
            
            # Update opponent-specific adaptation tracking
            if opponent_id not in self.opponent_adaptation[model_id]:
                self.opponent_adaptation[model_id][opponent_id] = {"opportunities": 0, "adapted": 0}
                
            self.opponent_adaptation[model_id][opponent_id]["opportunities"] += 1
            if adapted_to_opponent:
                self.opponent_adaptation[model_id][opponent_id]["adapted"] += 1
                
        # Add game context if provided
        if game_context:
            adapt_record.update(game_context)
            
        # Add to historical metrics
        self.historical_metrics[model_id]["adaptation"].append(adapt_record)
        
        # Update confidence metrics for statistical analysis
        self.confidence_metrics[f"{model_id}_adaptation"].append(1 if adapted_to_opponent else 0)
    
    def record_rule_adherence(self, model_id, valid_action, game_context=None):
        """
        Record whether the model followed game rules with detailed tracking
        
        Args:
            model_id: ID of the model taking the action
            valid_action: Whether the action follows game rules
            game_context: Optional dict with additional context
        """
        # Initialize if needed
        self.initialize_model(model_id)
        
        # Update core metrics
        self.rule_adherence[model_id]["total_actions"] += 1
        if valid_action:
            self.rule_adherence[model_id]["valid_actions"] += 1
            
        # Create adherence record
        adh_record = {
            "timestamp": time.time(),
            "valid": valid_action,
            "current_rate": self.calculate_rule_adherence_rate(model_id)
        }
        
        # Add game context and analyze
        if game_context:
            adh_record.update(game_context)
            
            # Track action-type adherence if provided
            if "action_type" in game_context:
                action_type = game_context["action_type"]
                action_key = f"action_{action_type}"
                
                if action_key not in self.rule_adherence[model_id]:
                    self.rule_adherence[model_id][action_key] = {"total": 0, "valid": 0}
                    
                self.rule_adherence[model_id][action_key]["total"] += 1
                if valid_action:
                    self.rule_adherence[model_id][action_key]["valid"] += 1
                    
            # Track round-specific adherence if provided
            if "round_number" in game_context:
                round_num = game_context["round_number"]
                
                if "round_adherence" not in self.rule_adherence[model_id]:
                    self.rule_adherence[model_id]["round_adherence"] = defaultdict(lambda: {"total": 0, "valid": 0})
                    
                self.rule_adherence[model_id]["round_adherence"][round_num]["total"] += 1
                if valid_action:
                    self.rule_adherence[model_id]["round_adherence"][round_num]["valid"] += 1
                
        # Add to historical metrics
        self.historical_metrics[model_id]["rule_adherence"].append(adh_record)
        
        # Update confidence metrics for statistical analysis
        self.confidence_metrics[f"{model_id}_rule_adherence"].append(1 if valid_action else 0)
            
    def record_api_response_time(self, model_id, response_time, game_context=None):
        """
        Record the API response time for a model with extended analysis
        
        Args:
            model_id: ID of the model making the API call
            response_time: Time in seconds for the API response
            game_context: Optional dict with additional context
        """
        # Skip invalid response times
        if response_time is None or response_time <= 0:
            return
            
        # Initialize if needed
        self.initialize_model(model_id)
        
        # Add to core metrics
        self.api_response_times[model_id].append(response_time)
        
        # Create response time record
        time_record = {
            "timestamp": time.time(),
            "response_time": response_time,
            "avg_time": self.calculate_avg_api_response_time(model_id)
        }
        
        # Add game context if provided
        if game_context:
            time_record.update(game_context)
            
            # Track by round number if provided
            if "round_number" in game_context:
                round_num = game_context["round_number"]
                
                if "round_times" not in self.temporal_metrics[model_id]:
                    self.temporal_metrics[model_id]["round_times"] = {}
                    
                self.temporal_metrics[model_id]["round_times"][round_num] = response_time
                
            # Track by complexity if provided
            if "prompt_complexity" in game_context:
                complexity = game_context["prompt_complexity"]
                
                if "complexity_times" not in self.temporal_metrics[model_id]:
                    self.temporal_metrics[model_id]["complexity_times"] = defaultdict(list)
                    
                self.temporal_metrics[model_id]["complexity_times"][complexity].append(response_time)
                
        # Add to historical metrics for time series analysis
        self.historical_metrics[model_id]["response_times"].append(time_record)
    
    def calculate_bluff_success_rate(self, model_id):
        """
        Calculate comprehensive bluff success metrics with statistical analysis
        
        Args:
            model_id: ID of the model to calculate metrics for
            
        Returns:
            Dict with success rates and statistical confidence measures
        """
        data = self.bluff_data[model_id]
        if data["total"] == 0:
            return 0
            
        # Calculate basic success rate
        success_rate = (data["successful"] / data["total"]) * 100
        
        # Calculate additional metrics if we have enough data
        if data["total"] >= 5:
            # Calculate by bluff type if available
            bluff_types = {}
            for key in data:
                if key.endswith("_total") and key != "total":
                    bluff_type = key[:-6]  # Remove "_total" suffix
                    total = data[key]
                    success_key = f"{bluff_type}_successful"
                    successful = data.get(success_key, 0)
                    
                    if total > 0:
                        bluff_types[bluff_type] = {
                            "rate": (successful / total) * 100,
                            "total": total,
                            "successful": successful
                        }
                        
            # Calculate opponent-specific rates if available
            opponent_rates = {}
            if "opponents" in data:
                for opponent, stats in data["opponents"].items():
                    if stats["total"] > 0:
                        opponent_rate = (stats["successful"] / stats["total"]) * 100
                        opponent_rates[opponent] = {
                            "rate": opponent_rate,
                            "total": stats["total"],
                            "successful": stats["successful"]
                        }
                        
            # Calculate confidence interval using binomial proportion confidence interval
            confidence_level = self.statistical_confidence
            z = stats.norm.ppf((1 + confidence_level) / 2)
            p = data["successful"] / data["total"]
            margin_of_error = z * math.sqrt((p * (1 - p)) / data["total"])
            confidence_interval = (max(0, (p * 100) - (margin_of_error * 100)), 
                                 min(100, (p * 100) + (margin_of_error * 100)))
            
            # Calculate trend if we have historical data
            trend = None
            if model_id in self.historical_metrics and "bluffs" in self.historical_metrics[model_id]:
                history = self.historical_metrics[model_id]["bluffs"]
                if len(history) >= 5:  # Need at least 5 data points for trend
                    # Get last 5 success rates
                    recent_rates = [h.get("success_rate", 0) for h in history[-5:]]
                    if recent_rates:
                        # Simple trend: positive if increasing, negative if decreasing
                        if recent_rates[-1] > recent_rates[0]:
                            trend = "increasing"
                        elif recent_rates[-1] < recent_rates[0]:
                            trend = "decreasing"
                        else:
                            trend = "stable"
            
            return {
                "rate": success_rate,
                "total": data["total"],
                "successful": data["successful"],
                "confidence_interval": confidence_interval,
                "confidence_level": confidence_level * 100,
                "by_type": bluff_types,
                "by_opponent": opponent_rates,
                "trend": trend
            }
            
        # Basic result if not enough data
        return success_rate
    
    def calculate_lie_detection_rate(self, model_id):
        """
        Enhanced lie detection metrics with statistical confidence
        
        Args:
            model_id: ID of the model to calculate metrics for
            
        Returns:
            Dict with precision, recall, F1 score, and statistical measures
        """
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
            
        # Basic metrics
        basic_metrics = {
            "precision": precision * 100,
            "recall": recall * 100,
            "f1_score": f1_score * 100,
            "true_positives": tp,
            "false_positives": fp,
            "false_negatives": fn
        }
        
        # Calculate additional metrics if we have enough data
        total_predictions = tp + fp + fn
        if total_predictions >= 5:
            # Calculate confidence intervals
            confidence_level = self.statistical_confidence
            z = stats.norm.ppf((1 + confidence_level) / 2)
            
            # Precision confidence interval
            if (tp + fp) > 0:
                p = precision
                margin_of_error = z * math.sqrt((p * (1 - p)) / (tp + fp))
                precision_ci = (max(0, p * 100 - margin_of_error * 100), 
                               min(100, p * 100 + margin_of_error * 100))
            else:
                precision_ci = (0, 0)
                
            # Recall confidence interval
            if (tp + fn) > 0:
                r = recall
                margin_of_error = z * math.sqrt((r * (1 - r)) / (tp + fn))
                recall_ci = (max(0, r * 100 - margin_of_error * 100), 
                            min(100, r * 100 + margin_of_error * 100))
            else:
                recall_ci = (0, 0)
                
            # Calculate opponent-specific metrics if available
            opponent_metrics = {}
            for key, opp_data in data.items():
                if key.startswith("opponent_"):
                    opponent = key[9:]  # Remove "opponent_" prefix
                    opp_tp = opp_data["true_positive"]
                    opp_fp = opp_data["false_positive"]
                    opp_fn = opp_data["false_negative"]
                    
                    opp_precision = opp_tp / (opp_tp + opp_fp) if (opp_tp + opp_fp) > 0 else 0
                    opp_recall = opp_tp / (opp_tp + opp_fn) if (opp_tp + opp_fn) > 0 else 0
                    
                    if opp_precision + opp_recall > 0:
                        opp_f1 = 2 * (opp_precision * opp_recall) / (opp_precision + opp_recall)
                    else:
                        opp_f1 = 0
                        
                    opponent_metrics[opponent] = {
                        "precision": opp_precision * 100,
                        "recall": opp_recall * 100,
                        "f1_score": opp_f1 * 100,
                        "true_positives": opp_tp,
                        "false_positives": opp_fp,
                        "false_negatives": opp_fn
                    }
            
            # Add pattern analysis from reasoning if available
            reasoning_patterns = None
            if "reasoning_patterns" in data and len(data["reasoning_patterns"]) >= 5:
                reasoning_patterns = {
                    "total_reasonings": len(data["reasoning_patterns"]),
                    "success_rate": (sum(1 for p in data["reasoning_patterns"] 
                                      if p["detection_type"] == "true_positive") / 
                                   len(data["reasoning_patterns"])) * 100
                }
                
            # Calculate trend if we have historical data
            trend = None
            if model_id in self.historical_metrics and "lie_detection" in self.historical_metrics[model_id]:
                history = self.historical_metrics[model_id]["lie_detection"]
                if len(history) >= 5:  # Need at least 5 data points for trend
                    # Get last 5 F1 scores
                    recent_f1 = [h.get("current_metrics", {}).get("f1_score", 0) for h in history[-5:]]
                    if recent_f1 and all(f1 is not None for f1 in recent_f1):
                        # Simple trend: positive if increasing, negative if decreasing
                        if recent_f1[-1] > recent_f1[0]:
                            trend = "improving"
                        elif recent_f1[-1] < recent_f1[0]:
                            trend = "declining"
                        else:
                            trend = "stable"
            
            # Add enhanced metrics
            return {
                **basic_metrics,
                "precision_ci": precision_ci,
                "recall_ci": recall_ci,
                "confidence_level": confidence_level * 100,
                "by_opponent": opponent_metrics,
                "reasoning_patterns": reasoning_patterns,
                "trend": trend
            }
            
        # Return basic metrics if not enough data
        return basic_metrics
    
    def calculate_average_final_bid(self, model_id):
        """
        Calculate comprehensive metrics for final bids with statistical analysis
        
        Args:
            model_id: ID of the model to analyze
            
        Returns:
            Dict with bid metrics, distribution analysis, and confidence intervals
        """
        bids = [bid[0] for bid in self.final_bids[model_id]]
        if not bids:
            return 0
            
        # Basic average calculation
        avg_bid = sum(bids) / len(bids)
        
        # Calculate additional metrics if we have enough data
        if len(bids) >= 5:
            # Basic statistics
            median_bid = sorted(bids)[len(bids) // 2]
            min_bid = min(bids)
            max_bid = max(bids)
            std_dev = np.std(bids)
            
            # Calculate percentiles
            percentiles = {
                "25th": np.percentile(bids, 25),
                "50th": np.percentile(bids, 50),
                "75th": np.percentile(bids, 75),
                "90th": np.percentile(bids, 90)
            }
            
            # Calculate confidence interval for the mean
            confidence_level = self.statistical_confidence
            ci = stats.t.interval(
                confidence_level, 
                len(bids) - 1, 
                loc=avg_bid, 
                scale=stats.sem(bids)
            )
            
            # Detect bid strategy patterns
            # Consistent: low standard deviation
            # Aggressive: higher average bid
            # Conservative: lower average bid
            if std_dev <= 1.0:
                strategy = "consistent"
            elif avg_bid >= 1.2 * median_bid:
                strategy = "aggressive"
            elif avg_bid <= 0.8 * median_bid:
                strategy = "conservative"
            else:
                strategy = "balanced"
                
            # Calculate temporal trend if we have history
            trend = None
            if model_id in self.temporal_metrics and "round_history" in self.temporal_metrics[model_id]:
                history = self.temporal_metrics[model_id]["round_history"]
                if len(history) >= 5:
                    # Extract bids from last 5 rounds
                    recent_bids = [h.get("bid", 0) for h in history[-5:]]
                    if all(bid is not None for bid in recent_bids):
                        # Calculate slope of trend line
                        rounds = list(range(len(recent_bids)))
                        slope, _, _, _, _ = stats.linregress(rounds, recent_bids)
                        if slope > 0.2:
                            trend = "increasing"
                        elif slope < -0.2:
                            trend = "decreasing"
                        else:
                            trend = "stable"
            
            # Return comprehensive metrics
            return {
                "average": avg_bid,
                "median": median_bid,
                "min": min_bid,
                "max": max_bid,
                "std_dev": std_dev,
                "percentiles": percentiles,
                "confidence_interval": ci,
                "confidence_level": confidence_level * 100,
                "bid_strategy": strategy,
                "sample_size": len(bids),
                "trend": trend
            }
            
        # Return basic average if not enough data
        return avg_bid
    
    def calculate_bid_optimality(self, model_id):
        """
        Calculate comprehensive bid optimality metrics with statistical rigor
        
        Args:
            model_id: ID of the model to analyze
            
        Returns:
            Dict with optimality rate, confidence intervals, and phase-specific analysis
        """
        data = self.bid_optimality[model_id]
        if data["total"] == 0:
            return 0
            
        # Basic optimality rate
        optimality_rate = (data["optimal"] / data["total"]) * 100
        
        # Calculate additional metrics if we have enough data
        if data["total"] >= 5:
            # Calculate confidence interval using binomial proportion confidence interval
            confidence_level = self.statistical_confidence
            z = stats.norm.ppf((1 + confidence_level) / 2)
            p = data["optimal"] / data["total"]
            margin_of_error = z * math.sqrt((p * (1 - p)) / data["total"])
            confidence_interval = (max(0, (p * 100) - (margin_of_error * 100)), 
                                   min(100, (p * 100) + (margin_of_error * 100)))
            
            # Calculate phase-specific optimality if available
            phase_analysis = {}
            for key, phase_data in data.items():
                if key.startswith("optimality_") and isinstance(phase_data, dict) and "total" in phase_data:
                    phase = key[11:]  # Remove "optimality_" prefix
                    if phase_data["total"] > 0:
                        phase_rate = (phase_data["optimal"] / phase_data["total"]) * 100
                        phase_analysis[phase] = {
                            "rate": phase_rate,
                            "total": phase_data["total"],
                            "optimal": phase_data["optimal"]
                        }
            
            # Calculate dice-specific optimality if available
            dice_analysis = None
            if "dice_combinations" in data:
                combinations = data["dice_combinations"]
                if combinations:
                    dice_analysis = {
                        "total_combinations": len(combinations),
                        "best_combination": max(combinations.items(), 
                                             key=lambda x: x[1]["optimal"] / x[1]["total"] if x[1]["total"] > 0 else 0),
                        "worst_combination": min(combinations.items(),
                                              key=lambda x: x[1]["optimal"] / x[1]["total"] if x[1]["total"] > 0 else 1)
                    }
            
            # Calculate trend using confidence metrics
            trend = None
            if f"{model_id}_bid_optimality" in self.confidence_metrics:
                history = self.confidence_metrics[f"{model_id}_bid_optimality"]
                if len(history) >= 10:  # Need sufficient data for trend
                    # Use last 10 data points
                    recent = history[-10:]
                    # Calculate moving average to smooth noise
                    window_size = min(5, len(recent))
                    moving_avg = [sum(recent[i:i+window_size]) / window_size 
                                 for i in range(len(recent) - window_size + 1)]
                    
                    if len(moving_avg) >= 2:
                        if moving_avg[-1] > moving_avg[0] * 1.1:
                            trend = "improving"
                        elif moving_avg[-1] < moving_avg[0] * 0.9:
                            trend = "declining"
                        else:
                            trend = "stable"
            
            # Return comprehensive metrics
            return {
                "rate": optimality_rate,
                "total": data["total"],
                "optimal": data["optimal"],
                "confidence_interval": confidence_interval,
                "confidence_level": confidence_level * 100,
                "phase_analysis": phase_analysis,
                "dice_analysis": dice_analysis,
                "trend": trend
            }
        
        # Return basic rate if not enough data
        return optimality_rate
    
    def calculate_adaptation_score(self, model_id):
        """
        Calculate comprehensive adaptation metrics with opponent-specific analysis
        
        Args:
            model_id: ID of the model to analyze
            
        Returns:
            Dict with adaptation score, opponent analysis, and confidence intervals
        """
        data = self.adaptation_scores[model_id]
        if data["opportunities"] == 0:
            return 0
            
        # Basic adaptation score
        adaptation_score = (data["adapted"] / data["opportunities"]) * 100
        
        # Calculate additional metrics if we have enough data
        if data["opportunities"] >= 5:
            # Calculate confidence interval using binomial proportion confidence interval
            confidence_level = self.statistical_confidence
            z = stats.norm.ppf((1 + confidence_level) / 2)
            p = data["adapted"] / data["opportunities"]
            margin_of_error = z * math.sqrt((p * (1 - p)) / data["opportunities"])
            confidence_interval = (max(0, (p * 100) - (margin_of_error * 100)), 
                                  min(100, (p * 100) + (margin_of_error * 100)))
            
            # Calculate opponent-specific adaptation if available
            opponent_analysis = {}
            for opponent, opp_data in self.opponent_adaptation[model_id].items():
                if opp_data["opportunities"] > 0:
                    opp_score = (opp_data["adapted"] / opp_data["opportunities"]) * 100
                    opponent_analysis[opponent] = {
                        "score": opp_score,
                        "opportunities": opp_data["opportunities"],
                        "adapted": opp_data["adapted"]
                    }
                    
                    # Add confidence interval if enough data
                    if opp_data["opportunities"] >= 3:
                        p_opp = opp_data["adapted"] / opp_data["opportunities"]
                        margin_error = z * math.sqrt((p_opp * (1 - p_opp)) / opp_data["opportunities"])
                        ci = (max(0, (p_opp * 100) - (margin_error * 100)), 
                             min(100, (p_opp * 100) + (margin_error * 100)))
                        opponent_analysis[opponent]["confidence_interval"] = ci
            
            # Calculate ranking among opponents
            if opponent_analysis:
                best_opponent = max(opponent_analysis.items(), key=lambda x: x[1]["score"])
                worst_opponent = min(opponent_analysis.items(), key=lambda x: x[1]["score"])
                opponent_ranking = {
                    "best": {
                        "opponent": best_opponent[0],
                        "score": best_opponent[1]["score"]
                    },
                    "worst": {
                        "opponent": worst_opponent[0],
                        "score": worst_opponent[1]["score"]
                    }
                }
            else:
                opponent_ranking = None
                
            # Perform significance testing if we have historical data
            significance_test = None
            if f"{model_id}_adaptation" in self.confidence_metrics:
                history = self.confidence_metrics[f"{model_id}_adaptation"]
                if len(history) >= 10:
                    # Split into first half and second half to test for improvement
                    midpoint = len(history) // 2
                    first_half = history[:midpoint]
                    second_half = history[midpoint:]
                    
                    # Calculate means
                    first_mean = sum(first_half) / len(first_half) if first_half else 0
                    second_mean = sum(second_half) / len(second_half) if second_half else 0
                    
                    # Perform t-test if sufficient data
                    if first_half and second_half:
                        t_stat, p_value = stats.ttest_ind(first_half, second_half)
                        significance_test = {
                            "t_statistic": t_stat,
                            "p_value": p_value,
                            "significant_improvement": p_value < 0.05 and second_mean > first_mean,
                            "significant_decline": p_value < 0.05 and second_mean < first_mean,
                            "first_half_mean": first_mean * 100,
                            "second_half_mean": second_mean * 100
                        }
            
            # Return comprehensive metrics
            return {
                "score": adaptation_score,
                "opportunities": data["opportunities"],
                "adapted": data["adapted"],
                "confidence_interval": confidence_interval,
                "confidence_level": confidence_level * 100,
                "opponent_analysis": opponent_analysis,
                "opponent_ranking": opponent_ranking,
                "significance_test": significance_test
            }
            
        # Return basic score if not enough data
        return adaptation_score
    
    def calculate_rule_adherence_rate(self, model_id):
        """
        Calculate comprehensive rule adherence metrics with statistical analysis
        
        Args:
            model_id: ID of the model to analyze
            
        Returns:
            Dict with adherence rate, action-specific analysis, and confidence intervals
        """
        data = self.rule_adherence[model_id]
        if data["total_actions"] == 0:
            return 100  # If no actions yet, assume perfect adherence
            
        # Basic adherence rate
        adherence_rate = (data["valid_actions"] / data["total_actions"]) * 100
        
        # Calculate additional metrics if we have enough data
        if data["total_actions"] >= 5:
            # Calculate confidence interval using binomial proportion confidence interval
            confidence_level = self.statistical_confidence
            z = stats.norm.ppf((1 + confidence_level) / 2)
            p = data["valid_actions"] / data["total_actions"]
            margin_of_error = z * math.sqrt((p * (1 - p)) / data["total_actions"])
            confidence_interval = (max(0, (p * 100) - (margin_of_error * 100)), 
                                  min(100, (p * 100) + (margin_of_error * 100)))
            
            # Calculate action-type specific adherence if available
            action_analysis = {}
            for key, action_data in data.items():
                if key.startswith("action_") and isinstance(action_data, dict) and "total" in action_data:
                    action_type = key[7:]  # Remove "action_" prefix
                    if action_data["total"] > 0:
                        action_rate = (action_data["valid"] / action_data["total"]) * 100
                        action_analysis[action_type] = {
                            "rate": action_rate,
                            "total": action_data["total"],
                            "valid": action_data["valid"]
                        }
                        
                        # Add confidence interval if enough data
                        if action_data["total"] >= 3:
                            p_action = action_data["valid"] / action_data["total"]
                            margin = z * math.sqrt((p_action * (1 - p_action)) / action_data["total"])
                            ci = (max(0, (p_action * 100) - (margin * 100)), 
                                 min(100, (p_action * 100) + (margin * 100)))
                            action_analysis[action_type]["confidence_interval"] = ci
            
            # Calculate round-specific adherence if available
            round_analysis = None
            if "round_adherence" in data:
                rounds = data["round_adherence"]
                if rounds:
                    # Calculate aggregated early, mid, and late game adherence
                    early_rounds = {r: rd for r, rd in rounds.items() if int(r) <= 3}
                    mid_rounds = {r: rd for r, rd in rounds.items() if 3 < int(r) <= 6}
                    late_rounds = {r: rd for r, rd in rounds.items() if int(r) > 6}
                    
                    # Calculate rates for each game phase
                    phases = {}
                    
                    # Early game
                    if early_rounds:
                        early_total = sum(rd["total"] for rd in early_rounds.values())
                        early_valid = sum(rd["valid"] for rd in early_rounds.values())
                        early_rate = (early_valid / early_total * 100) if early_total > 0 else 0
                        phases["early_game"] = {"rate": early_rate, "total": early_total}
                    
                    # Mid game
                    if mid_rounds:
                        mid_total = sum(rd["total"] for rd in mid_rounds.values())
                        mid_valid = sum(rd["valid"] for rd in mid_rounds.values())
                        mid_rate = (mid_valid / mid_total * 100) if mid_total > 0 else 0
                        phases["mid_game"] = {"rate": mid_rate, "total": mid_total}
                    
                    # Late game
                    if late_rounds:
                        late_total = sum(rd["total"] for rd in late_rounds.values())
                        late_valid = sum(rd["valid"] for rd in late_rounds.values())
                        late_rate = (late_valid / late_total * 100) if late_total > 0 else 0
                        phases["late_game"] = {"rate": late_rate, "total": late_total}
                    
                    round_analysis = {
                        "phases": phases,
                        "by_round": {r: {"rate": (rd["valid"] / rd["total"] * 100) if rd["total"] > 0 else 0,
                                        "total": rd["total"]} 
                                    for r, rd in rounds.items()}
                    }
                    
                    # Identify round with worst adherence
                    if rounds:
                        worst_round = min(rounds.items(), 
                                        key=lambda x: x[1]["valid"] / x[1]["total"] if x[1]["total"] > 0 else 1.0)
                        round_analysis["worst_round"] = {
                            "round": worst_round[0],
                            "rate": (worst_round[1]["valid"] / worst_round[1]["total"] * 100) 
                                    if worst_round[1]["total"] > 0 else 0
                        }
            
            # Calculate learning curve
            learning_curve = None
            if f"{model_id}_rule_adherence" in self.confidence_metrics:
                history = self.confidence_metrics[f"{model_id}_rule_adherence"]
                if len(history) >= 10:
                    # Calculate rolling average with window size 5
                    window_size = 5
                    rolling_avg = []
                    for i in range(len(history) - window_size + 1):
                        window = history[i:i+window_size]
                        rolling_avg.append(sum(window) / window_size)
                    
                    # Calculate linear regression to find learning rate
                    if rolling_avg:
                        x = np.array(range(len(rolling_avg)))
                        y = np.array(rolling_avg)
                        slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
                        
                        learning_curve = {
                            "slope": slope,
                            "r_squared": r_value ** 2,
                            "p_value": p_value,
                            "significant": p_value < 0.05,
                            "learning_rate": "improving" if slope > 0.01 else 
                                            "declining" if slope < -0.01 else "stable"
                        }
            
            # Return comprehensive metrics
            return {
                "rate": adherence_rate,
                "total_actions": data["total_actions"],
                "valid_actions": data["valid_actions"],
                "confidence_interval": confidence_interval,
                "confidence_level": confidence_level * 100,
                "action_analysis": action_analysis,
                "round_analysis": round_analysis,
                "learning_curve": learning_curve
            }
            
        # Return basic rate if not enough data
        return adherence_rate
        
    def calculate_avg_api_response_time(self, model_id):
        """
        Calculate comprehensive API response time metrics with statistical analysis
        
        Args:
            model_id: ID of the model to analyze
            
        Returns:
            Dict with timing statistics, distribution analysis, and round-based patterns
        """
        times = self.api_response_times[model_id]
        if not times:
            return 0
            
        # Basic average calculation
        avg_time = sum(times) / len(times)
        
        # Calculate additional metrics if we have enough data
        if len(times) >= 5:
            # Basic statistics
            median_time = sorted(times)[len(times) // 2]
            min_time = min(times)
            max_time = max(times)
            std_dev = np.std(times)
            
            # Calculate percentiles
            percentiles = {
                "25th": np.percentile(times, 25),
                "50th": np.percentile(times, 50),
                "75th": np.percentile(times, 75),
                "90th": np.percentile(times, 90)
            }
            
            # Calculate confidence interval for the mean
            confidence_level = self.statistical_confidence
            ci = stats.t.interval(
                confidence_level, 
                len(times) - 1, 
                loc=avg_time, 
                scale=stats.sem(times)
            )
            
            # Calculate latency stability (low std_dev indicates stable latency)
            stability = 1.0 - (std_dev / avg_time) if avg_time > 0 else 0
            stability = max(0, min(1, stability)) * 100  # Convert to percentage
            
            # Analyze round-specific timing if available
            round_analysis = None
            if model_id in self.temporal_metrics and "round_times" in self.temporal_metrics[model_id]:
                round_times = self.temporal_metrics[model_id]["round_times"]
                if round_times:
                    # Group by early, mid, and late game
                    rounds = sorted(int(r) for r in round_times.keys())
                    if rounds:
                        max_round = max(rounds)
                        early_cutoff = max(1, max_round // 3)
                        late_cutoff = max(early_cutoff + 1, max_round * 2 // 3)
                        
                        early_times = [round_times[str(r)] for r in rounds if r <= early_cutoff]
                        mid_times = [round_times[str(r)] for r in rounds if early_cutoff < r <= late_cutoff]
                        late_times = [round_times[str(r)] for r in rounds if r > late_cutoff]
                        
                        round_analysis = {
                            "early_game": {
                                "avg": sum(early_times) / len(early_times) if early_times else 0,
                                "count": len(early_times)
                            },
                            "mid_game": {
                                "avg": sum(mid_times) / len(mid_times) if mid_times else 0,
                                "count": len(mid_times)
                            },
                            "late_game": {
                                "avg": sum(late_times) / len(late_times) if late_times else 0,
                                "count": len(late_times)
                            }
                        }
            
            # Analyze complexity-based timing if available
            complexity_analysis = None
            if model_id in self.temporal_metrics and "complexity_times" in self.temporal_metrics[model_id]:
                complexity_times = self.temporal_metrics[model_id]["complexity_times"]
                if complexity_times:
                    complexity_analysis = {
                        complexity: {
                            "avg": sum(times_list) / len(times_list) if times_list else 0,
                            "count": len(times_list),
                            "min": min(times_list) if times_list else 0,
                            "max": max(times_list) if times_list else 0
                        }
                        for complexity, times_list in complexity_times.items()
                    }
                    
                    # Calculate correlation between complexity and response time
                    if len(complexity_times) >= 2:
                        # Create paired data of complexity level and average time
                        try:
                            # Try to convert complexity to numeric (if it's numeric)
                            complexities = [float(c) for c in complexity_times.keys()]
                            avg_times = [sum(times_list) / len(times_list) if times_list else 0 
                                      for times_list in complexity_times.values()]
                            
                            # Calculate correlation
                            correlation, p_value = stats.pearsonr(complexities, avg_times)
                            complexity_analysis["correlation"] = {
                                "coefficient": correlation,
                                "p_value": p_value,
                                "significant": p_value < 0.05
                            }
                        except (ValueError, TypeError):
                            # If complexity keys aren't numeric, skip correlation
                            pass
            
            # Detect timing trend over API calls
            timing_trend = None
            if len(times) >= 10:
                # Use exponential moving average to smooth out noise
                alpha = 0.3  # Smoothing factor
                ema = [times[0]]
                for t in times[1:]:
                    ema.append(alpha * t + (1 - alpha) * ema[-1])
                
                # Compare first and last third of EMA values
                first_third = ema[:len(ema)//3]
                last_third = ema[-len(ema)//3:]
                
                if first_third and last_third:
                    first_avg = sum(first_third) / len(first_third)
                    last_avg = sum(last_third) / len(last_third)
                    
                    if last_avg < first_avg * 0.9:
                        timing_trend = "improving"
                    elif last_avg > first_avg * 1.1:
                        timing_trend = "slowing"
                    else:
                        timing_trend = "stable"
            
            # Return comprehensive metrics
            return {
                "average": avg_time,
                "median": median_time,
                "min": min_time,
                "max": max_time,
                "std_dev": std_dev,
                "percentiles": percentiles,
                "confidence_interval": ci,
                "confidence_level": confidence_level * 100,
                "stability": stability,
                "sample_count": len(times),
                "round_analysis": round_analysis,
                "complexity_analysis": complexity_analysis,
                "trend": timing_trend
            }
            
        # Return basic average if not enough data
        return avg_time
    
    def record_token_usage(self, model_id, prompt_tokens, completion_tokens, total_tokens=None, game_context=None):
        """
        Record token usage for an API call with enhanced tracking
        
        Args:
            model_id: ID of the model making the API call
            prompt_tokens: Number of tokens in the prompt
            completion_tokens: Number of tokens in the completion
            total_tokens: Optional total token count (if not sum of prompt+completion)
            game_context: Optional dict with additional context (round, action type, etc.)
        """
        # Initialize if needed
        self.initialize_model(model_id)
        
        # Set total tokens if not provided
        if total_tokens is None:
            total_tokens = prompt_tokens + completion_tokens
            
        # Update core metrics
        self.token_usage[model_id]["prompt_tokens"] += prompt_tokens
        self.token_usage[model_id]["completion_tokens"] += completion_tokens
        self.token_usage[model_id]["total_tokens"] += total_tokens
        self.token_usage[model_id]["actions"] += 1
        
        # Create token usage record
        token_record = {
            "timestamp": time.time(),
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "action": self.token_usage[model_id]["actions"]
        }
        
        # Add additional tracking by type if context is provided
        if game_context:
            token_record.update(game_context)
            
            # Track by action type if provided
            if "action_type" in game_context:
                action_type = game_context["action_type"]
                key = f"tokens_by_action_{action_type}"
                
                if key not in self.token_usage[model_id]:
                    self.token_usage[model_id][key] = {
                        "prompt_tokens": 0,
                        "completion_tokens": 0,
                        "total_tokens": 0,
                        "actions": 0
                    }
                    
                self.token_usage[model_id][key]["prompt_tokens"] += prompt_tokens
                self.token_usage[model_id][key]["completion_tokens"] += completion_tokens
                self.token_usage[model_id][key]["total_tokens"] += total_tokens
                self.token_usage[model_id][key]["actions"] += 1
                
            # Track by round if provided
            if "round_number" in game_context:
                round_num = game_context["round_number"]
                
                if "tokens_by_round" not in self.token_usage[model_id]:
                    self.token_usage[model_id]["tokens_by_round"] = {}
                    
                if round_num not in self.token_usage[model_id]["tokens_by_round"]:
                    self.token_usage[model_id]["tokens_by_round"][round_num] = {
                        "prompt_tokens": 0,
                        "completion_tokens": 0,
                        "total_tokens": 0,
                        "actions": 0
                    }
                    
                self.token_usage[model_id]["tokens_by_round"][round_num]["prompt_tokens"] += prompt_tokens
                self.token_usage[model_id]["tokens_by_round"][round_num]["completion_tokens"] += completion_tokens
                self.token_usage[model_id]["tokens_by_round"][round_num]["total_tokens"] += total_tokens
                self.token_usage[model_id]["tokens_by_round"][round_num]["actions"] += 1
        
        # Add to historical metrics
        if "token_history" not in self.historical_metrics[model_id]:
            self.historical_metrics[model_id]["token_history"] = []
            
        self.historical_metrics[model_id]["token_history"].append(token_record)
    
    def calculate_token_metrics(self, model_id):
        """
        Calculate comprehensive token usage metrics with statistical analysis
        
        Args:
            model_id: ID of the model to analyze
            
        Returns:
            Dict with token usage statistics, efficiency analysis, and cost estimates
        """
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
        
        # Basic metrics
        basic_metrics = {
            "total_prompt_tokens": data["prompt_tokens"],
            "total_completion_tokens": data["completion_tokens"],
            "total_tokens": data["total_tokens"],
            "total_actions": actions,
            "avg_prompt_tokens_per_action": data["prompt_tokens"] / actions,
            "avg_completion_tokens_per_action": data["completion_tokens"] / actions,
            "avg_tokens_per_action": data["total_tokens"] / actions,
            "prompt_to_completion_ratio": data["prompt_tokens"] / data["completion_tokens"] if data["completion_tokens"] > 0 else float('inf')
        }
        
        # Calculate additional metrics if we have enough data
        if actions >= 5:
            # Calculate action-type specific metrics if available
            action_type_metrics = {}
            for key, action_data in data.items():
                if key.startswith("tokens_by_action_") and isinstance(action_data, dict) and "actions" in action_data:
                    action_type = key[16:]  # Remove "tokens_by_action_" prefix
                    if action_data["actions"] > 0:
                        action_type_metrics[action_type] = {
                            "total_tokens": action_data["total_tokens"],
                            "prompt_tokens": action_data["prompt_tokens"],
                            "completion_tokens": action_data["completion_tokens"],
                            "actions": action_data["actions"],
                            "avg_tokens_per_action": action_data["total_tokens"] / action_data["actions"],
                            "prompt_to_completion_ratio": action_data["prompt_tokens"] / action_data["completion_tokens"] 
                                                          if action_data["completion_tokens"] > 0 else float('inf')
                        }
            
            # Calculate round-specific metrics if available
            round_metrics = None
            if "tokens_by_round" in data:
                round_data = data["tokens_by_round"]
                if round_data:
                    rounds = sorted(int(r) for r in round_data.keys())
                    if rounds:
                        max_round = max(rounds)
                        early_cutoff = max(1, max_round // 3)
                        late_cutoff = max(early_cutoff + 1, max_round * 2 // 3)
                        
                        # Calculate aggregate values for each game phase
                        early_data = {k: v for k, v in round_data.items() if int(k) <= early_cutoff}
                        mid_data = {k: v for k, v in round_data.items() if early_cutoff < int(k) <= late_cutoff}
                        late_data = {k: v for k, v in round_data.items() if int(k) > late_cutoff}
                        
                        round_metrics = {
                            "early_game": self._aggregate_token_phase(early_data),
                            "mid_game": self._aggregate_token_phase(mid_data),
                            "late_game": self._aggregate_token_phase(late_data),
                            "by_round": {
                                r: {
                                    "total_tokens": rd["total_tokens"],
                                    "avg_tokens_per_action": rd["total_tokens"] / rd["actions"] if rd["actions"] > 0 else 0
                                } for r, rd in round_data.items()
                            }
                        }
                        
                        # Calculate efficiency trend over rounds
                        if len(rounds) >= 3:
                            token_per_action_by_round = [(int(r), rd["total_tokens"] / rd["actions"] if rd["actions"] > 0 else 0) 
                                                     for r, rd in round_data.items()]
                            token_per_action_by_round.sort()  # Sort by round number
                            
                            # Calculate linear regression
                            x = [tpa[0] for tpa in token_per_action_by_round]
                            y = [tpa[1] for tpa in token_per_action_by_round]
                            if len(x) >= 2 and len(y) >= 2:
                                slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
                                
                                round_metrics["token_efficiency_trend"] = {
                                    "slope": slope,
                                    "r_squared": r_value ** 2,
                                    "p_value": p_value,
                                    "significant": p_value < 0.05,
                                    "trend": "increasing" if slope > 0.01 else 
                                            "decreasing" if slope < -0.01 else "stable"
                                }
            
            # Calculate token distribution
            if "token_history" in self.historical_metrics[model_id]:
                history = self.historical_metrics[model_id]["token_history"]
                if history:
                    # Extract token counts by type
                    prompt_tokens = [h["prompt_tokens"] for h in history]
                    completion_tokens = [h["completion_tokens"] for h in history]
                    total_tokens = [h["total_tokens"] for h in history]
                    
                    token_distribution = {
                        "prompt_tokens": {
                            "min": min(prompt_tokens),
                            "max": max(prompt_tokens),
                            "median": np.median(prompt_tokens),
                            "std_dev": np.std(prompt_tokens),
                            "percentiles": {
                                "25th": np.percentile(prompt_tokens, 25),
                                "75th": np.percentile(prompt_tokens, 75),
                                "90th": np.percentile(prompt_tokens, 90)
                            }
                        },
                        "completion_tokens": {
                            "min": min(completion_tokens),
                            "max": max(completion_tokens),
                            "median": np.median(completion_tokens),
                            "std_dev": np.std(completion_tokens),
                            "percentiles": {
                                "25th": np.percentile(completion_tokens, 25),
                                "75th": np.percentile(completion_tokens, 75),
                                "90th": np.percentile(completion_tokens, 90)
                            }
                        },
                        "total_tokens": {
                            "min": min(total_tokens),
                            "max": max(total_tokens),
                            "median": np.median(total_tokens),
                            "std_dev": np.std(total_tokens),
                            "percentiles": {
                                "25th": np.percentile(total_tokens, 25),
                                "75th": np.percentile(total_tokens, 75),
                                "90th": np.percentile(total_tokens, 90)
                            }
                        }
                    }
                    
                    # Analyze token usage trend
                    if len(history) >= 10:
                        # Calculate moving average to smooth noise
                        window_size = 3
                        moving_avg = []
                        for i in range(len(total_tokens) - window_size + 1):
                            window = total_tokens[i:i+window_size]
                            moving_avg.append(sum(window) / window_size)
                        
                        # Compare first and last 3 values
                        first_avg = sum(moving_avg[:3]) / 3 if len(moving_avg) >= 3 else moving_avg[0]
                        last_avg = sum(moving_avg[-3:]) / 3 if len(moving_avg) >= 3 else moving_avg[-1]
                        
                        token_distribution["trend"] = {
                            "direction": "increasing" if last_avg > first_avg * 1.05 else
                                       "decreasing" if last_avg < first_avg * 0.95 else "stable",
                            "first_avg": first_avg,
                            "last_avg": last_avg,
                            "percent_change": ((last_avg - first_avg) / first_avg * 100) if first_avg > 0 else 0
                        }
                        
                    # Return enhanced metrics
                    return {
                        **basic_metrics,
                        "action_type_metrics": action_type_metrics,
                        "round_metrics": round_metrics,
                        "token_distribution": token_distribution
                    }
        
        # Return basic metrics if not enough data
        return basic_metrics
        
    def _aggregate_token_phase(self, phase_data):
        """Helper method to aggregate token data for a game phase"""
        if not phase_data:
            return {
                "total_tokens": 0,
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "actions": 0,
                "avg_tokens_per_action": 0
            }
            
        total_tokens = sum(rd["total_tokens"] for rd in phase_data.values())
        prompt_tokens = sum(rd["prompt_tokens"] for rd in phase_data.values())
        completion_tokens = sum(rd["completion_tokens"] for rd in phase_data.values())
        actions = sum(rd["actions"] for rd in phase_data.values())
        
        return {
            "total_tokens": total_tokens,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "actions": actions,
            "avg_tokens_per_action": total_tokens / actions if actions > 0 else 0
        }
        
    def get_model_metrics(self, model_id, include_temporal=False):
        """
        Get comprehensive metrics for a specific model
        
        Args:
            model_id: ID of the model to get metrics for
            include_temporal: Whether to include temporal data series
            
        Returns:
            Dict with all available metrics for the model
        """
        # Basic metrics with enhanced calculation methods
        metrics = {
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
        
        # Add temporal metrics if requested
        if include_temporal and model_id in self.historical_metrics:
            metrics["historical_data"] = self.historical_metrics[model_id]
            
        # Add matchup data
        if model_id in self.matchup_stats:
            # Sort opponents by win rate
            opponents = sorted(
                self.matchup_stats[model_id].items(),
                key=lambda x: x[1]["win_rate"] if "win_rate" in x[1] else 0,
                reverse=True
            )
            
            metrics["matchups"] = {
                "opponent_data": {opp: data for opp, data in opponents},
                "best_matchup": opponents[0][0] if opponents else None,
                "worst_matchup": opponents[-1][0] if opponents else None
            }
        
        # Add opponent adaptation data
        if model_id in self.opponent_adaptation and self.opponent_adaptation[model_id]:
            metrics["opponent_adaptation"] = self.opponent_adaptation[model_id]
            
        # Add game phase strategy analysis
        if model_id in self.strategy_patterns:
            # Analyze phase-specific strategies
            phases = {}
            for phase, data in self.strategy_patterns[model_id].items():
                if data:
                    bluff_count = sum(1 for move in data if move.get("bluff_type") == "bluff")
                    success_count = sum(1 for move in data if move.get("success", False))
                    
                    phases[phase] = {
                        "total_moves": len(data),
                        "bluff_rate": (bluff_count / len(data) * 100) if data else 0,
                        "success_rate": (success_count / len(data) * 100) if data else 0
                    }
            
            metrics["phase_strategy"] = phases
            
        # Add consistency measures
        if model_id in self.model_consistency and self.model_consistency[model_id]:
            consistency_data = self.model_consistency[model_id]
            if consistency_data:
                metrics["consistency"] = {
                    "predictability": np.mean(consistency_data) if consistency_data else 0,
                    "volatility": np.std(consistency_data) if len(consistency_data) > 1 else 0
                }
                
        # Add meta-analysis (correlations between metrics)
        metrics["meta_analysis"] = self.analyze_metric_correlations(model_id)
        
        return metrics
    
    def analyze_metric_correlations(self, model_id):
        """
        Analyze correlations between different metrics for a model
        
        Args:
            model_id: ID of the model to analyze
            
        Returns:
            Dict with correlation analysis between key metrics
        """
        # Only run correlation analysis if we have sufficient data
        if model_id not in self.historical_metrics:
            return None
            
        # Metrics to correlate (metric name, path to values in historical data)
        metrics_to_analyze = [
            ("bluff_success", ("bluffs", "success_rate")),
            ("lie_detection", ("lie_detection", "current_metrics", "f1_score")),
            ("bid_optimality", ("bid_optimality", "current_rate")),
            ("rule_adherence", ("rule_adherence", "current_rate")),
            ("response_time", ("response_times", "response_time")),
            ("token_usage", ("token_history", "total_tokens"))
        ]
        
        # Extract metrics data
        metric_data = {}
        for metric_name, path in metrics_to_analyze:
            # Extract values following the path
            if path[0] in self.historical_metrics[model_id]:
                series = self.historical_metrics[model_id][path[0]]
                values = []
                
                # Extract values based on path depth
                if len(path) == 2:
                    values = [entry.get(path[1]) for entry in series if path[1] in entry]
                elif len(path) == 3:
                    values = [entry.get(path[1], {}).get(path[2]) 
                             for entry in series if path[1] in entry and path[2] in entry.get(path[1], {})]
                
                # Filter out None values
                values = [v for v in values if v is not None]
                
                if values:
                    metric_data[metric_name] = values
        
        # Only proceed if we have at least 2 metrics with sufficient data
        metrics_with_data = {k: v for k, v in metric_data.items() if len(v) >= 5}
        if len(metrics_with_data) < 2:
            return None
            
        # Calculate correlations between all pairs of metrics
        correlations = {}
        for metric1 in metrics_with_data:
            for metric2 in metrics_with_data:
                if metric1 != metric2:
                    # Ensure we have the same number of data points
                    data1 = metrics_with_data[metric1]
                    data2 = metrics_with_data[metric2]
                    
                    # Use the smaller length
                    min_length = min(len(data1), len(data2))
                    
                    if min_length >= 5:  # Need at least 5 data points for meaningful correlation
                        # Calculate correlation
                        try:
                            corr, p_value = stats.pearsonr(data1[:min_length], data2[:min_length])
                            
                            # Store correlation data
                            key = f"{metric1}_vs_{metric2}"
                            correlations[key] = {
                                "correlation": corr,
                                "p_value": p_value,
                                "significant": p_value < 0.05,
                                "samples": min_length
                            }
                        except Exception:
                            # Skip if correlation calculation fails
                            pass
        
        # Find strongest positive and negative correlations
        if correlations:
            strongest_positive = max(correlations.items(), key=lambda x: x[1]["correlation"])
            strongest_negative = min(correlations.items(), key=lambda x: x[1]["correlation"])
            
            # Prepare summary
            correlation_summary = {
                "strongest_positive": {
                    "metrics": strongest_positive[0],
                    "correlation": strongest_positive[1]["correlation"],
                    "significant": strongest_positive[1]["significant"]
                },
                "strongest_negative": {
                    "metrics": strongest_negative[0],
                    "correlation": strongest_negative[1]["correlation"],
                    "significant": strongest_negative[1]["significant"]
                },
                "all_correlations": correlations
            }
            
            return correlation_summary
            
        return None
    
    def get_all_metrics(self, include_temporal=False):
        """
        Get metrics for all models
        
        Args:
            include_temporal: Whether to include time series data
            
        Returns:
            Dict with metrics for all models
        """
        metrics = {}
        for model_id in self.elo_ratings:
            metrics[model_id] = self.get_model_metrics(model_id, include_temporal)
            
        # Add comparative rankings across models
        self._add_comparative_rankings(metrics)
        
        # Add global statistics
        metrics["__global__"] = self._calculate_global_statistics()
            
        return metrics
        
    def _add_comparative_rankings(self, metrics_dict):
        """Add model rankings for each metric across all models"""
        if not metrics_dict:
            return
            
        # Metrics to rank
        ranking_metrics = [
            ("elo_rating", None, True),  # (metric_name, nested_path, higher_is_better)
            ("bluff_success_rate.rate" if isinstance(next(iter(metrics_dict.values()))["bluff_success_rate"], dict) 
             else "bluff_success_rate", None, True),
            ("lie_detection.f1_score", None, True),
            ("bid_optimality.rate" if isinstance(next(iter(metrics_dict.values()))["bid_optimality"], dict) 
             else "bid_optimality", None, True),
            ("adaptation_score.score" if isinstance(next(iter(metrics_dict.values()))["adaptation_score"], dict) 
             else "adaptation_score", None, True),
            ("rule_adherence_rate.rate" if isinstance(next(iter(metrics_dict.values()))["rule_adherence_rate"], dict) 
             else "rule_adherence_rate", None, True),
            ("avg_api_response_time.average" if isinstance(next(iter(metrics_dict.values()))["avg_api_response_time"], dict) 
             else "avg_api_response_time", None, False)  # Lower is better for response time
        ]
        
        # For each metric, calculate rankings
        for metric_name, nested_path, higher_is_better in ranking_metrics:
            # Extract metric values
            model_values = []
            for model_id, model_metrics in metrics_dict.items():
                if model_id == "__global__":
                    continue
                    
                # Get metric value based on path
                value = None
                if nested_path:
                    parts = nested_path.split('.')
                    current = model_metrics
                    for part in parts:
                        if part in current:
                            current = current[part]
                        else:
                            current = None
                            break
                    value = current
                else:
                    # Handle dot notation in metric_name
                    if '.' in metric_name:
                        parts = metric_name.split('.')
                        current = model_metrics
                        for part in parts:
                            if part in current:
                                current = current[part]
                            else:
                                current = None
                                break
                        value = current
                    else:
                        value = model_metrics.get(metric_name)
                
                if value is not None:
                    model_values.append((model_id, value))
            
            # Sort values
            sorted_values = sorted(model_values, key=lambda x: x[1], reverse=higher_is_better)
            
            # Add ranking to each model's metrics
            for rank, (model_id, _) in enumerate(sorted_values, 1):
                rank_metric = metric_name.replace('.', '_') + "_rank"
                if "__rankings__" not in metrics_dict[model_id]:
                    metrics_dict[model_id]["__rankings__"] = {}
                metrics_dict[model_id]["__rankings__"][rank_metric] = rank
                
                # Add percentile
                percentile = 100 - (rank / len(sorted_values) * 100) if higher_is_better else (rank / len(sorted_values) * 100)
                metrics_dict[model_id]["__rankings__"][metric_name.replace('.', '_') + "_percentile"] = percentile
    
    def _calculate_global_statistics(self):
        """Calculate global statistics across all models"""
        global_stats = {
            "total_models": len(self.elo_ratings),
            "total_games": len(self.game_results),
            "metrics_timestamp": time.time()
        }
        
        # Calculate average metrics across all models
        avg_metrics = {
            "avg_elo": np.mean(list(self.elo_ratings.values())) if self.elo_ratings else self.default_elo,
            "avg_rule_adherence": np.mean([
                self.calculate_rule_adherence_rate(model_id) 
                if isinstance(self.calculate_rule_adherence_rate(model_id), (int, float)) 
                else self.calculate_rule_adherence_rate(model_id)["rate"] 
                for model_id in self.elo_ratings
            ]) if self.elo_ratings else 0,
            "avg_bluff_success": np.mean([
                self.calculate_bluff_success_rate(model_id)
                if isinstance(self.calculate_bluff_success_rate(model_id), (int, float))
                else self.calculate_bluff_success_rate(model_id)["rate"]
                for model_id in self.elo_ratings
            ]) if self.elo_ratings else 0
        }
        
        global_stats["avg_metrics"] = avg_metrics
        
        # Add correlation between key metrics and winning
        if self.game_results and self.elo_ratings:
            # Create mapping of model ID to win count
            win_counts = defaultdict(int)
            for game in self.game_results:
                if "winner_model" in game and game["winner_model"]:
                    win_counts[game["winner_model"]] += 1
            
            # Correlate metrics with win counts
            metric_wins_corr = {}
            
            # Metrics to correlate with winning
            metrics_to_check = [
                "bluff_success_rate", "bid_optimality", "adaptation_score", 
                "rule_adherence_rate", "avg_api_response_time"
            ]
            
            for metric_name in metrics_to_check:
                metric_values = []
                win_values = []
                
                for model_id in self.elo_ratings:
                    if model_id in win_counts:
                        # Get metric value
                        metric_value = getattr(self, f"calculate_{metric_name}")(model_id)
                        
                        # Extract numeric value if metric returns a dict
                        if isinstance(metric_value, dict):
                            if "rate" in metric_value:
                                metric_value = metric_value["rate"]
                            elif "score" in metric_value:
                                metric_value = metric_value["score"]
                            elif "average" in metric_value:
                                metric_value = metric_value["average"]
                        
                        if isinstance(metric_value, (int, float)):
                            metric_values.append(metric_value)
                            win_values.append(win_counts[model_id])
                
                # Calculate correlation if we have enough data points
                if len(metric_values) >= 3:
                    try:
                        corr, p_value = stats.pearsonr(metric_values, win_values)
                        
                        metric_wins_corr[metric_name] = {
                            "correlation": corr,
                            "p_value": p_value,
                            "significant": p_value < 0.05
                        }
                    except Exception:
                        pass
            
            global_stats["winning_correlations"] = metric_wins_corr
            
        return global_stats
        
    def create_visualizations(self, output_dir="visualizations", prefix=None):
        """
        Create comprehensive visualizations of all metrics
        
        Args:
            output_dir: Directory where visualizations will be saved
            prefix: Optional prefix for the report directory
            
        Returns:
            Dict with paths to all generated visualizations
        """
        # Create visualizer and generate report
        visualizer = MetricsVisualizer(self, output_dir=output_dir)
        return visualizer.generate_comprehensive_report(output_prefix=prefix)

class BidAnalyzer:
    """
    Advanced bid analysis for Liar's Dice using probabilistic reasoning,
    game theory, and statistical analysis.
    """
    
    @staticmethod
    def is_bid_optimal(dice_values, player_dice_count, total_dice, last_bid, current_bid, risk_tolerance=0.3):
        """
        Determine if a bid is mathematically optimal based on known information
        with enhanced probabilistic analysis
        
        Args:
            dice_values: Current player's dice values
            player_dice_count: Number of dice the player has
            total_dice: Total number of dice in the game
            last_bid: Last bid as (quantity, value)
            current_bid: Current bid as (quantity, value)
            risk_tolerance: Probability threshold for "optimal" bids (0.0-1.0, default 0.3)
            
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
        return prob >= risk_tolerance
        
    @staticmethod
    def evaluate_bid_detailed(dice_values, player_dice_count, total_dice, last_bid, current_bid):
        """
        Provide detailed probabilistic analysis of a bid's optimality
        
        Args:
            dice_values: Current player's dice values
            player_dice_count: Number of dice the player has
            total_dice: Total number of dice in the game
            last_bid: Last bid as (quantity, value)
            current_bid: Current bid as (quantity, value)
            
        Returns:
            Dict with comprehensive bid analysis
        """
        curr_quantity, curr_value = current_bid
        
        # Calculate basic probability metrics
        player_has = dice_values.count(curr_value)
        unknown_dice = total_dice - player_dice_count
        expected_matches = unknown_dice / 6  # Expected number of matches in unknown dice
        total_expected = player_has + expected_matches
        
        # Probability of exactly the needed number of additional matches
        additional_needed = curr_quantity - player_has
        
        # Calculate possible outcomes
        if additional_needed <= 0:
            # Player already has enough dice
            probability = 1.0
            assessment = "safe"
            outcome_distribution = {"exact": 0, "exceeded": 1.0, "fell_short": 0}
        elif additional_needed > unknown_dice:
            # Impossible bid
            probability = 0.0
            assessment = "impossible"
            outcome_distribution = {"exact": 0, "exceeded": 0, "fell_short": 1.0}
        else:
            # Calculate probability using binomial
            prob_per_die = 1/6
            
            # Probability of at least the needed number of matches
            probability = 1 - BidAnalyzer._binom_cdf(additional_needed - 1, unknown_dice, prob_per_die)
            
            # Calculate probabilities of different outcomes
            exact_prob = BidAnalyzer._binom_pmf(additional_needed, unknown_dice, prob_per_die)
            exceed_prob = 1 - BidAnalyzer._binom_cdf(additional_needed, unknown_dice, prob_per_die)
            fall_short_prob = BidAnalyzer._binom_cdf(additional_needed - 1, unknown_dice, prob_per_die)
            
            outcome_distribution = {
                "exact": exact_prob,
                "exceeded": exceed_prob,
                "fell_short": fall_short_prob
            }
            
            # Assess the bid
            if probability >= 0.7:
                assessment = "conservative"
            elif probability >= 0.4:
                assessment = "balanced"
            elif probability >= 0.2:
                assessment = "risky"
            elif probability > 0:
                assessment = "bluff"
            else:
                assessment = "impossible"
        
        # Compare to previous bid if available
        comparative_analysis = None
        if last_bid:
            last_quantity, last_value = last_bid
            
            # Calculate minimum legal next bid
            if last_value < 6:
                min_next_quantity = last_quantity
                min_next_value = last_value + 1
            else:
                min_next_quantity = last_quantity + 1
                min_next_value = 1
                
            # Calculate aggressive factor (how much bid exceeds minimum)
            if curr_value == min_next_value:
                aggressiveness = (curr_quantity - min_next_quantity) / min_next_quantity if min_next_quantity > 0 else 0
            else:
                # Different face value, calculate equivalent bid
                effective_quantity = curr_quantity
                effective_min_quantity = min_next_quantity
                
                if curr_value > min_next_value:
                    # Converting to higher value (more aggressive)
                    aggressiveness = (effective_quantity - effective_min_quantity * 0.8) / effective_min_quantity if effective_min_quantity > 0 else 0
                else:
                    # Converting to lower value (potentially less aggressive)
                    aggressiveness = (effective_quantity - effective_min_quantity * 1.2) / effective_min_quantity if effective_min_quantity > 0 else 0
                    
            comparative_analysis = {
                "aggressiveness_factor": aggressiveness,
                "bid_type": "conservative" if aggressiveness <= 0 else
                           "balanced" if aggressiveness <= 0.3 else
                           "aggressive" if aggressiveness <= 0.7 else "very_aggressive",
                "relative_to_minimum": "minimum" if (curr_quantity == min_next_quantity and curr_value == min_next_value) else
                                      "above_minimum"
            }
        
        # Return comprehensive analysis
        return {
            "player_has": player_has,
            "unknown_dice": unknown_dice,
            "expected_matches": expected_matches,
            "total_expected": total_expected,
            "additional_needed": additional_needed,
            "probability": probability,
            "assessment": assessment,
            "outcome_distribution": outcome_distribution,
            "comparative_analysis": comparative_analysis,
            "optimal": probability >= 0.3  # Default threshold
        }
    
    @staticmethod
    def _binom_cdf(k, n, p):
        """
        Calculate cumulative binomial probability P(X ≤ k) with n trials and probability p
        Uses SciPy for numerical stability with large values
        """
        try:
            # Use SciPy for better numerical stability
            return stats.binom.cdf(k, n, p)
        except (ImportError, ValueError):
            # Fallback to direct calculation
            result = 0
            for i in range(k + 1):
                result += BidAnalyzer._binom_pmf(i, n, p)
            return result
    
    @staticmethod
    def _binom_pmf(k, n, p):
        """
        Calculate binomial probability mass function P(X = k) with n trials and probability p
        Uses SciPy for numerical stability with large values
        """
        try:
            # Use SciPy for better numerical stability
            return stats.binom.pmf(k, n, p)
        except (ImportError, ValueError):
            # Fallback to direct calculation
            if k < 0 or k > n:
                return 0
            return math.comb(n, k) * (p ** k) * ((1 - p) ** (n - k))
    
    @staticmethod
    def should_call_liar(dice_values, player_dice_count, total_dice, last_bid, risk_tolerance=0.25):
        """
        Determine if calling "liar" is mathematically optimal
        with enhanced probabilistic analysis
        
        Args:
            dice_values: Current player's dice values
            player_dice_count: Number of dice the player has
            total_dice: Total number of dice in the game
            last_bid: Last bid as (quantity, value)
            risk_tolerance: Probability threshold for calling liar (default 0.25)
            
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
        return prob < risk_tolerance
        
    @staticmethod
    def evaluate_call_liar_detailed(dice_values, player_dice_count, total_dice, last_bid):
        """
        Provide detailed probabilistic analysis of calling liar
        
        Args:
            dice_values: Current player's dice values
            player_dice_count: Number of dice the player has
            total_dice: Total number of dice in the game
            last_bid: Last bid as (quantity, value)
            
        Returns:
            Dict with comprehensive liar call analysis
        """
        if not last_bid:
            return {"valid": False, "reason": "no_previous_bid"}
            
        quantity, value = last_bid
        
        # Calculate basic probability metrics
        player_has = dice_values.count(value)
        unknown_dice = total_dice - player_dice_count
        expected_matches = unknown_dice / 6  # Expected number of matches in unknown dice
        total_expected = player_has + expected_matches
        
        # Number of additional matches needed for the bid to be true
        additional_needed = quantity - player_has
        
        # Calculate possible outcomes
        if additional_needed <= 0:
            # Bid is definitely true based on player's dice alone
            probability_false = 0.0
            assessment = "definitely_true"
        elif additional_needed > unknown_dice:
            # Bid is impossible (requires more dice than exist)
            probability_false = 1.0
            assessment = "definitely_false"
        else:
            # Calculate probability using binomial
            prob_per_die = 1/6
            
            # Probability that the bid is true
            probability_true = 1 - BidAnalyzer._binom_cdf(additional_needed - 1, unknown_dice, prob_per_die)
            probability_false = 1 - probability_true
            
            # Assess the liar call
            if probability_false >= 0.9:
                assessment = "almost_certainly_false"
            elif probability_false >= 0.75:
                assessment = "likely_false"
            elif probability_false >= 0.5:
                assessment = "possibly_false"
            elif probability_false >= 0.25:
                assessment = "possibly_true"
            elif probability_false >= 0.1:
                assessment = "likely_true"
            else:
                assessment = "almost_certainly_true"
        
        # Return comprehensive analysis
        return {
            "player_has": player_has,
            "unknown_dice": unknown_dice,
            "total_expected": total_expected,
            "probability_false": probability_false if 'probability_false' in locals() else None,
            "probability_true": 1 - probability_false if 'probability_false' in locals() else None,
            "assessment": assessment,
            "should_call_liar": probability_false >= 0.75 if 'probability_false' in locals() else (assessment == "definitely_false"),
            "confidence": "high" if (assessment in ["definitely_true", "definitely_false", "almost_certainly_false", "almost_certainly_true"]) else
                         "medium" if (assessment in ["likely_false", "likely_true"]) else "low"
        }
    
    @staticmethod
    def check_adaptation(player_history, opponent_history):
        """
        Check if a player has adapted their strategy based on opponent behavior
        with enhanced pattern recognition
        
        Args:
            player_history: List of player's actions across multiple games
            opponent_history: List of opponent's actions across multiple games
            
        Returns:
            Dict with adaptation analysis
        """
        # Need sufficient history to detect adaptation
        if len(player_history) < 5 or len(opponent_history) < 5:
            return {"adapted": False, "reason": "insufficient_history"}
            
        # Divide history into early and recent periods for comparison
        early_cutoff = max(5, len(player_history) // 3)
        player_early = player_history[:early_cutoff]
        player_recent = player_history[-early_cutoff:]
        
        opponent_early = opponent_history[:min(early_cutoff, len(opponent_history))]
        opponent_recent = opponent_history[-min(early_cutoff, len(opponent_history)):]
        
        # Analysis of opponent behavior
        opponent_traits = {
            "liar_caller": {
                "early": sum(1 for action in opponent_early if action["action"] == "liar") / len(opponent_early),
                "recent": sum(1 for action in opponent_recent if action["action"] == "liar") / len(opponent_recent)
            },
            "bluffer": {
                "early": sum(1 for action in opponent_early 
                          if action["action"] == "bid" and action.get("bluff", False)) / 
                       sum(1 for action in opponent_early if action["action"] == "bid") 
                       if sum(1 for action in opponent_early if action["action"] == "bid") > 0 else 0,
                "recent": sum(1 for action in opponent_recent 
                           if action["action"] == "bid" and action.get("bluff", False)) / 
                        sum(1 for action in opponent_recent if action["action"] == "bid")
                        if sum(1 for action in opponent_recent if action["action"] == "bid") > 0 else 0
            }
        }
        
        # Analysis of player adaptation
        player_traits = {
            "bluffing_rate": {
                "early": sum(1 for action in player_early 
                          if action["action"] == "bid" and action.get("bluff", False)) / 
                       sum(1 for action in player_early if action["action"] == "bid")
                       if sum(1 for action in player_early if action["action"] == "bid") > 0 else 0,
                "recent": sum(1 for action in player_recent 
                           if action["action"] == "bid" and action.get("bluff", False)) / 
                        sum(1 for action in player_recent if action["action"] == "bid")
                        if sum(1 for action in player_recent if action["action"] == "bid") > 0 else 0
            },
            "liar_calling_rate": {
                "early": sum(1 for action in player_early if action["action"] == "liar") / len(player_early),
                "recent": sum(1 for action in player_recent if action["action"] == "liar") / len(player_recent)
            }
        }
        
        # Detect adaptations based on opponent behavior
        adaptations = []
        adaptation_detected = False
        
        # Adaptation 1: If opponent calls liar frequently, reduce bluffing
        if opponent_traits["liar_caller"]["recent"] > 0.4:
            if player_traits["bluffing_rate"]["recent"] < player_traits["bluffing_rate"]["early"] * 0.7:
                adaptations.append({
                    "type": "reduced_bluffing_against_liar_caller",
                    "strength": "strong" if player_traits["bluffing_rate"]["recent"] < player_traits["bluffing_rate"]["early"] * 0.5 else "moderate",
                    "opponent_liar_call_rate": opponent_traits["liar_caller"]["recent"],
                    "player_bluff_rate_change": player_traits["bluffing_rate"]["recent"] - player_traits["bluffing_rate"]["early"]
                })
                adaptation_detected = True
                
        # Adaptation 2: If opponent rarely calls liar, increase bluffing
        if opponent_traits["liar_caller"]["recent"] < 0.2:
            if player_traits["bluffing_rate"]["recent"] > player_traits["bluffing_rate"]["early"] * 1.3:
                adaptations.append({
                    "type": "increased_bluffing_against_passive_opponent",
                    "strength": "strong" if player_traits["bluffing_rate"]["recent"] > player_traits["bluffing_rate"]["early"] * 1.5 else "moderate",
                    "opponent_liar_call_rate": opponent_traits["liar_caller"]["recent"],
                    "player_bluff_rate_change": player_traits["bluffing_rate"]["recent"] - player_traits["bluffing_rate"]["early"]
                })
                adaptation_detected = True
                
        # Adaptation 3: If opponent bluffs frequently, increase liar calls
        if opponent_traits["bluffer"]["recent"] > 0.4:
            if player_traits["liar_calling_rate"]["recent"] > player_traits["liar_calling_rate"]["early"] * 1.3:
                adaptations.append({
                    "type": "increased_liar_calls_against_bluffer",
                    "strength": "strong" if player_traits["liar_calling_rate"]["recent"] > player_traits["liar_calling_rate"]["early"] * 1.5 else "moderate",
                    "opponent_bluff_rate": opponent_traits["bluffer"]["recent"],
                    "player_liar_call_rate_change": player_traits["liar_calling_rate"]["recent"] - player_traits["liar_calling_rate"]["early"]
                })
                adaptation_detected = True
                
        # Return comprehensive adaptation analysis
        return {
            "adapted": adaptation_detected,
            "adaptations": adaptations,
            "opponent_traits": opponent_traits,
            "player_traits": player_traits,
            "analysis_confidence": "high" if len(player_history) >= 10 and len(opponent_history) >= 10 else "medium" if len(player_history) >= 5 else "low"
        }
        
    @staticmethod
    def analyze_bid_strategy(bid_history, dice_information=None):
        """
        Analyze a player's bidding strategy over multiple rounds
        
        Args:
            bid_history: List of player's bidding actions
            dice_information: Optional information about player's dice
            
        Returns:
            Dict with comprehensive strategy analysis
        """
        if not bid_history:
            return {"valid": False, "reason": "no_bid_history"}
            
        # Filter to just include bids (not liar calls)
        bids = [action for action in bid_history if action["action"] == "bid"]
        
        if not bids:
            return {"valid": False, "reason": "no_bids_found"}
            
        # Extract bid data
        quantities = [bid["quantity"] for bid in bids if "quantity" in bid]
        values = [bid["value"] for bid in bids if "value" in bid]
        bluffs = [bid.get("bluff", False) for bid in bids]
        
        # Calculate basic statistics
        avg_quantity = sum(quantities) / len(quantities) if quantities else 0
        avg_value = sum(values) / len(values) if values else 0
        bluff_rate = sum(1 for b in bluffs if b) / len(bluffs) if bluffs else 0
        
        # Find preferred bid values
        value_counts = defaultdict(int)
        for v in values:
            value_counts[v] += 1
            
        preferred_values = sorted(value_counts.items(), key=lambda x: x[1], reverse=True)
        
        # Analyze bidding patterns
        patterns = {
            "repeating_value": False,
            "incrementing_quantity": False,
            "alternating_values": False,
            "prefers_high_values": False,
            "prefers_low_values": False
        }
        
        # Check for repeating value pattern
        if len(preferred_values) > 0 and preferred_values[0][1] > len(values) * 0.6:
            patterns["repeating_value"] = True
            patterns["repeated_value"] = preferred_values[0][0]
            
        # Check for incrementing quantity pattern
        if len(quantities) >= 3:
            increments = [quantities[i+1] - quantities[i] for i in range(len(quantities)-1)]
            if all(inc > 0 for inc in increments):
                patterns["incrementing_quantity"] = True
                patterns["avg_increment"] = sum(increments) / len(increments)
                
        # Check for alternating values pattern
        if len(values) >= 4:
            odds_avg = sum(values[1::2]) / len(values[1::2]) if values[1::2] else 0
            evens_avg = sum(values[0::2]) / len(values[0::2]) if values[0::2] else 0
            if abs(odds_avg - evens_avg) > 2:
                patterns["alternating_values"] = True
                
        # Check for value preferences
        high_values = sum(1 for v in values if v > 3)
        low_values = sum(1 for v in values if v <= 3)
        
        if high_values > low_values * 2:
            patterns["prefers_high_values"] = True
        elif low_values > high_values * 2:
            patterns["prefers_low_values"] = True
            
        # Analyze optimal vs. actual bidding if dice information available
        optimality_analysis = None
        if dice_information:
            # dice_information should be a list of (dice_values, all_dice, bid) tuples
            optimal_bids = 0
            for dice_values, total_dice, bid in dice_information:
                # Calculate if the bid was optimal
                if BidAnalyzer.is_bid_optimal(dice_values, len(dice_values), total_dice, None, bid):
                    optimal_bids += 1
                    
            optimality_analysis = {
                "optimal_bids": optimal_bids,
                "total_bids": len(dice_information),
                "optimality_rate": optimal_bids / len(dice_information) if dice_information else 0
            }
            
        # Determine overall strategy based on patterns
        strategy_type = "unknown"
        if patterns["repeating_value"]:
            strategy_type = "value_focused"
        elif patterns["incrementing_quantity"]:
            strategy_type = "escalating"
        elif bluff_rate > 0.6:
            strategy_type = "bluffer"
        elif bluff_rate < 0.3:
            strategy_type = "conservative"
        else:
            strategy_type = "balanced"
            
        # Return comprehensive strategy analysis
        return {
            "avg_quantity": avg_quantity,
            "avg_value": avg_value,
            "bluff_rate": bluff_rate,
            "preferred_values": preferred_values[:3],  # Top 3 values
            "patterns": patterns,
            "strategy_type": strategy_type,
            "optimality_analysis": optimality_analysis,
            "sample_size": len(bids)
        }


class MetricsVisualizer:
    """
    Comprehensive visualization system for Liar's Dice metrics.
    Creates publication-quality visualizations for statistical analysis and insights.
    """
    
    def __init__(self, metrics_data, output_dir="visualizations"):
        """
        Initialize the visualizer with metrics data
        
        Args:
            metrics_data: GameMetrics instance or metrics dict from get_all_metrics()
            output_dir: Directory where visualizations will be saved
        """
        self.metrics = metrics_data
        self.output_dir = output_dir
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Configure visualization style
        self._setup_visualization_style()
    
    def _setup_visualization_style(self):
        """Configure matplotlib and seaborn for consistent, publication-quality visuals"""
        # Set style for professional, academic visualizations
        sns.set_style("whitegrid")
        plt.rcParams['font.family'] = 'sans-serif'
        plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Liberation Sans', 'Bitstream Vera Sans', 'sans-serif']
        
        # Increase font sizes for readability
        plt.rcParams['axes.labelsize'] = 12
        plt.rcParams['axes.titlesize'] = 14
        plt.rcParams['xtick.labelsize'] = 10
        plt.rcParams['ytick.labelsize'] = 10
        plt.rcParams['legend.fontsize'] = 10
        plt.rcParams['figure.titlesize'] = 16
        
        # Custom color palette for consistent branding
        self.color_palette = sns.color_palette("viridis", 8)
        self.categorical_palette = sns.color_palette("Set2", 8)
        self.sequential_palette = sns.light_palette("seagreen", as_cmap=True)
        
        # Default figure size for better readability
        plt.rcParams['figure.figsize'] = (10, 6)
        
        # Higher DPI for better quality
        plt.rcParams['figure.dpi'] = 120
        plt.rcParams['savefig.dpi'] = 300
        
    def visualize_elo_ratings(self, save_path=None):
        """
        Create an ELO rating visualization with confidence intervals
        
        Args:
            save_path: Optional path to save the visualization
            
        Returns:
            Figure handle
        """
        # Check if we're working with a GameMetrics instance or dict
        if hasattr(self.metrics, 'elo_ratings'):
            models = list(self.metrics.elo_ratings.keys())
            ratings = [self.metrics.elo_ratings[model] for model in models]
            # Get historical ELO data if available for uncertainty calculation
            uncertainties = []
            for model in models:
                if model in self.metrics.historical_metrics and 'elo' in self.metrics.historical_metrics[model]:
                    history = self.metrics.historical_metrics[model]['elo']
                    if len(history) >= 5:
                        # Use standard deviation of recent changes as uncertainty
                        recent_changes = [h['change'] for h in history[-5:]]
                        uncertainties.append(np.std(recent_changes))
                    else:
                        uncertainties.append(20)  # Default uncertainty
                else:
                    uncertainties.append(20)  # Default uncertainty
        else:
            # Assume it's a metrics dict from get_all_metrics()
            models = [model for model in self.metrics.keys() if model != '__global__']
            ratings = [self.metrics[model]['elo_rating'] for model in models]
            uncertainties = [20] * len(models)  # Default uncertainty
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 7))
        
        # Sort by ELO rating
        sorted_data = sorted(zip(models, ratings, uncertainties), key=lambda x: x[1], reverse=True)
        models = [x[0] for x in sorted_data]
        ratings = [x[1] for x in sorted_data]
        uncertainties = [x[2] for x in sorted_data]
        
        # Plot bars with error bars
        bars = ax.bar(range(len(models)), ratings, color=self.color_palette)
        ax.errorbar(range(len(models)), ratings, yerr=uncertainties, fmt='none', ecolor='black', capsize=5)
        
        # Add labels and title
        ax.set_xticks(range(len(models)))
        ax.set_xticklabels(models, rotation=45, ha='right')
        ax.set_ylabel('ELO Rating')
        ax.set_title('Model Performance: ELO Ratings with Uncertainty', fontweight='bold')
        
        # Add grid and improve aesthetics
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        # Add average ELO line
        avg_elo = np.mean(ratings)
        ax.axhline(y=avg_elo, color='r', linestyle='--', alpha=0.8, label=f'Average: {avg_elo:.1f}')
        
        # Add legend
        ax.legend()
        
        # Add value labels on top of bars
        for i, bar in enumerate(bars):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 5,
                   f'{ratings[i]:.1f}', ha='center', va='bottom')
        
        plt.tight_layout()
        
        # Save if path provided
        if save_path:
            plt.savefig(os.path.join(self.output_dir, save_path), bbox_inches='tight')
            
        return fig
        
    def visualize_win_rates(self, save_path=None):
        """
        Create a win rate heatmap showing matchup performance
        
        Args:
            save_path: Optional path to save the visualization
            
        Returns:
            Figure handle
        """
        # Check if we're working with a GameMetrics instance
        if hasattr(self.metrics, 'matchup_stats'):
            matchup_data = self.metrics.matchup_stats
        else:
            # Try to extract from metrics dict
            matchup_data = {}
            for model in self.metrics:
                if model != '__global__' and 'matchups' in self.metrics[model]:
                    matchup_data[model] = {
                        opp: data for opp, data in 
                        self.metrics[model]['matchups'].get('opponent_data', {}).items()
                    }
        
        # Create a DataFrame for the heatmap
        models = sorted(list(matchup_data.keys()))
        win_rate_matrix = np.zeros((len(models), len(models)))
        
        for i, model1 in enumerate(models):
            for j, model2 in enumerate(models):
                if model1 == model2:
                    win_rate_matrix[i, j] = 0.5  # Draw against self
                elif model2 in matchup_data.get(model1, {}):
                    win_rate = matchup_data[model1][model2].get('win_rate', 0)
                    win_rate_matrix[i, j] = win_rate
        
        # Create heatmap
        fig, ax = plt.subplots(figsize=(10, 8))
        heatmap = sns.heatmap(win_rate_matrix, annot=True, fmt='.2f', cmap='RdYlGn',
                            xticklabels=models, yticklabels=models, vmin=0, vmax=1,
                            cbar_kws={'label': 'Win Rate'})
        
        # Add labels and title
        ax.set_title('Matchup Win Rates', fontweight='bold')
        ax.set_xlabel('Opponent')
        ax.set_ylabel('Model')
        
        plt.tight_layout()
        
        # Save if path provided
        if save_path:
            plt.savefig(os.path.join(self.output_dir, save_path), bbox_inches='tight')
            
        return fig
    
    def visualize_metric_correlations(self, model_id=None, save_path=None):
        """
        Create a correlation heatmap of different metrics
        
        Args:
            model_id: Optional specific model to analyze
            save_path: Optional path to save the visualization
            
        Returns:
            Figure handle
        """
        # Get correlation data
        if hasattr(self.metrics, 'analyze_metric_correlations'):
            if model_id:
                correlation_data = self.metrics.analyze_metric_correlations(model_id)
            else:
                # If no model specified, use the first one with sufficient data
                correlation_data = None
                for model in self.metrics.elo_ratings:
                    corr = self.metrics.analyze_metric_correlations(model)
                    if corr:
                        correlation_data = corr
                        model_id = model
                        break
        else:
            # Try to extract from metrics dict
            if model_id and model_id in self.metrics:
                correlation_data = self.metrics[model_id].get('meta_analysis')
            else:
                # If no model specified, find one with meta_analysis
                correlation_data = None
                for model in self.metrics:
                    if model != '__global__' and 'meta_analysis' in self.metrics[model]:
                        correlation_data = self.metrics[model]['meta_analysis']
                        model_id = model
                        break
        
        if not correlation_data or 'all_correlations' not in correlation_data:
            return None  # Not enough data for visualization
            
        # Extract correlation values into a symmetric matrix
        all_correlations = correlation_data['all_correlations']
        metric_names = set()
        
        # Extract all unique metric names
        for key in all_correlations:
            metrics_pair = key.split('_vs_')
            metric_names.update(metrics_pair)
            
        metric_names = sorted(list(metric_names))
        corr_matrix = np.zeros((len(metric_names), len(metric_names)))
        
        # Fill the correlation matrix
        for i, metric1 in enumerate(metric_names):
            for j, metric2 in enumerate(metric_names):
                if i == j:
                    corr_matrix[i, j] = 1.0  # Perfect correlation with self
                else:
                    key = f"{metric1}_vs_{metric2}"
                    alt_key = f"{metric2}_vs_{metric1}"
                    
                    if key in all_correlations:
                        corr_matrix[i, j] = all_correlations[key]['correlation']
                    elif alt_key in all_correlations:
                        corr_matrix[i, j] = all_correlations[alt_key]['correlation']
        
        # Create heatmap
        fig, ax = plt.subplots(figsize=(10, 8))
        heatmap = sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm',
                            xticklabels=metric_names, yticklabels=metric_names, vmin=-1, vmax=1,
                            cbar_kws={'label': 'Correlation Coefficient'})
        
        # Add labels and title
        title = f"Metric Correlations for {model_id}" if model_id else "Metric Correlations"
        ax.set_title(title, fontweight='bold')
        
        plt.tight_layout()
        
        # Save if path provided
        if save_path:
            plt.savefig(os.path.join(self.output_dir, save_path), bbox_inches='tight')
            
        return fig
    
    def visualize_metric_trends(self, metric_name, models=None, save_path=None):
        """
        Create a time series visualization of a metric's evolution
        
        Args:
            metric_name: Name of the metric to visualize
            models: Optional list of models to include (default: all)
            save_path: Optional path to save the visualization
            
        Returns:
            Figure handle
        """
        # Check if we're working with a GameMetrics instance
        if hasattr(self.metrics, 'historical_metrics'):
            historical_data = self.metrics.historical_metrics
            if not models:
                models = list(self.metrics.elo_ratings.keys())
        else:
            # Try to extract from metrics dict
            historical_data = {}
            available_models = []
            
            for model in self.metrics:
                if model != '__global__' and 'historical_data' in self.metrics[model]:
                    historical_data[model] = self.metrics[model]['historical_data']
                    available_models.append(model)
            
            if not models:
                models = available_models
        
        # Map metric names to paths in historical data
        metric_paths = {
            'elo': 'elo',
            'bluff_success': 'bluffs',
            'lie_detection': 'lie_detection',
            'bid_optimality': 'bid_optimality',
            'adaptation': 'adaptation',
            'rule_adherence': 'rule_adherence',
            'response_time': 'response_times'
        }
        
        if metric_name not in metric_paths:
            return None  # Unknown metric
            
        path = metric_paths[metric_name]
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 7))
        
        # Plot each model's trend
        for model in models:
            if model in historical_data and path in historical_data[model]:
                data = historical_data[model][path]
                
                # Extract timestamps and values
                timestamps = []
                values = []
                
                for entry in data:
                    # Handle different data structures
                    if 'timestamp' in entry:
                        timestamps.append(entry['timestamp'])
                        
                        # Extract value based on metric type
                        if path == 'elo':
                            values.append(entry.get('value', 0))
                        elif path == 'bluffs':
                            values.append(entry.get('success_rate', 0))
                        elif path == 'lie_detection':
                            if 'current_metrics' in entry and 'f1_score' in entry['current_metrics']:
                                values.append(entry['current_metrics']['f1_score'])
                            else:
                                continue
                        elif path in ['bid_optimality', 'rule_adherence']:
                            values.append(entry.get('current_rate', 0))
                        elif path == 'adaptation':
                            values.append(entry.get('current_score', 0))
                        elif path == 'response_times':
                            values.append(entry.get('response_time', 0))
                        else:
                            continue
                
                # Sort by timestamp
                sorted_data = sorted(zip(timestamps, values))
                if sorted_data:
                    timestamps, values = zip(*sorted_data)
                    
                    # Convert timestamps to relative game numbers for clearer visualization
                    game_numbers = list(range(1, len(timestamps) + 1))
                    
                    # Plot trend line
                    ax.plot(game_numbers, values, marker='o', linestyle='-', alpha=0.7, 
                           label=f"{model}")
        
        # Add labels and title
        metric_labels = {
            'elo': 'ELO Rating',
            'bluff_success': 'Bluff Success Rate (%)',
            'lie_detection': 'Lie Detection F1 Score (%)',
            'bid_optimality': 'Bid Optimality Rate (%)',
            'adaptation': 'Adaptation Score (%)',
            'rule_adherence': 'Rule Adherence Rate (%)',
            'response_time': 'Response Time (seconds)'
        }
        
        ax.set_xlabel('Game Number')
        ax.set_ylabel(metric_labels.get(metric_name, metric_name.capitalize()))
        ax.set_title(f'{metric_labels.get(metric_name, metric_name.capitalize())} Over Time', fontweight='bold')
        
        # Add grid and improve aesthetics
        ax.grid(linestyle='--', alpha=0.7)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        # Add legend
        if len(models) > 1:
            ax.legend(loc='best')
        
        plt.tight_layout()
        
        # Save if path provided
        if save_path:
            plt.savefig(os.path.join(self.output_dir, save_path), bbox_inches='tight')
            
        return fig
    
    def visualize_radar_metrics(self, models=None, save_path=None):
        """
        Create a radar chart comparing models across multiple metrics
        
        Args:
            models: Optional list of models to include (default: top 5 by ELO)
            save_path: Optional path to save the visualization
            
        Returns:
            Figure handle
        """
        # Select metrics to include in radar
        radar_metrics = [
            'bluff_success_rate', 
            'lie_detection',
            'bid_optimality', 
            'adaptation_score',
            'rule_adherence_rate'
        ]
        
        # Extract data
        if hasattr(self.metrics, 'elo_ratings'):
            # If we have a GameMetrics instance
            if not models:
                # Get top 5 models by ELO
                models = sorted(self.metrics.elo_ratings.keys(), 
                               key=lambda m: self.metrics.elo_ratings[m], reverse=True)[:5]
            
            # Get metric values for each model
            model_data = {}
            for model in models:
                metrics_dict = {}
                for metric in radar_metrics:
                    # Call the appropriate calculation method
                    calc_method = getattr(self.metrics, f"calculate_{metric}", None)
                    if calc_method:
                        value = calc_method(model)
                        # Extract numeric value if result is a dict
                        if isinstance(value, dict):
                            if "rate" in value:
                                value = value["rate"]
                            elif "score" in value:
                                value = value["score"]
                            elif "f1_score" in value:
                                value = value["f1_score"]
                            else:
                                value = 0
                    else:
                        value = 0
                        
                    metrics_dict[metric] = value
                
                model_data[model] = metrics_dict
        else:
            # If we have a metrics dict
            if not models:
                # Get top 5 models by ELO
                models = sorted([m for m in self.metrics.keys() if m != '__global__'], 
                               key=lambda m: self.metrics[m]['elo_rating'], reverse=True)[:5]
            
            # Get metric values for each model
            model_data = {}
            for model in models:
                metrics_dict = {}
                for metric in radar_metrics:
                    value = self.metrics[model].get(metric, 0)
                    
                    # Extract numeric value if result is a dict
                    if isinstance(value, dict):
                        if "rate" in value:
                            value = value["rate"]
                        elif "score" in value:
                            value = value["score"]
                        elif "f1_score" in value:
                            value = value["f1_score"]
                        else:
                            value = 0
                            
                    metrics_dict[metric] = value
                
                model_data[model] = metrics_dict
        
        # Prepare data for radar chart
        metrics_labels = {
            'bluff_success_rate': 'Bluff Success',
            'lie_detection': 'Lie Detection',
            'bid_optimality': 'Bid Optimality',
            'adaptation_score': 'Adaptation',
            'rule_adherence_rate': 'Rule Adherence'
        }
        
        # Create radar chart
        num_metrics = len(radar_metrics)
        angles = np.linspace(0, 2*np.pi, num_metrics, endpoint=False).tolist()
        angles += angles[:1]  # Close the loop
        
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
        
        # Draw one line for each model
        for i, model in enumerate(models):
            values = [model_data[model].get(metric, 0) for metric in radar_metrics]
            
            # Normalize values to 0-100 if they aren't already
            values = [min(100, max(0, v)) for v in values]
            
            # Close the loop
            values += values[:1]
            
            # Plot the model's values
            ax.plot(angles, values, linewidth=2, linestyle='solid', label=model, 
                   color=self.categorical_palette[i % len(self.categorical_palette)])
            ax.fill(angles, values, alpha=0.1, 
                   color=self.categorical_palette[i % len(self.categorical_palette)])
        
        # Set category labels
        metric_labels = [metrics_labels.get(metric, metric) for metric in radar_metrics]
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metric_labels, fontsize=12)
        
        # Set radial limits
        ax.set_ylim(0, 100)
        ax.set_yticks([20, 40, 60, 80, 100])
        ax.set_yticklabels(['20%', '40%', '60%', '80%', '100%'])
        
        # Remove grid lines
        ax.grid(True, alpha=0.3)
        
        # Add legend
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
        
        # Add title
        plt.title('Model Performance Comparison Across Key Metrics', fontweight='bold', size=14, y=1.1)
        
        plt.tight_layout()
        
        # Save if path provided
        if save_path:
            plt.savefig(os.path.join(self.output_dir, save_path), bbox_inches='tight')
            
        return fig
    
    def visualize_confidence_intervals(self, metric_name, models=None, save_path=None):
        """
        Create a visualization of a metric with confidence intervals
        
        Args:
            metric_name: Name of the metric to visualize
            models: Optional list of models to include
            save_path: Optional path to save the visualization
            
        Returns:
            Figure handle
        """
        # Map metric names to calculation methods
        metric_methods = {
            'bluff_success_rate': 'calculate_bluff_success_rate',
            'lie_detection': 'calculate_lie_detection_rate',
            'bid_optimality': 'calculate_bid_optimality',
            'adaptation_score': 'calculate_adaptation_score',
            'rule_adherence_rate': 'calculate_rule_adherence_rate'
        }
        
        if metric_name not in metric_methods:
            return None  # Unknown metric
        
        # Get the data
        if hasattr(self.metrics, metric_methods[metric_name]):
            # If we have a GameMetrics instance
            if not models:
                models = list(self.metrics.elo_ratings.keys())
                
            # Get metric values with confidence intervals
            model_values = []
            ci_lows = []
            ci_highs = []
            
            for model in models:
                calc_method = getattr(self.metrics, metric_methods[metric_name])
                result = calc_method(model)
                
                if isinstance(result, dict) and 'confidence_interval' in result:
                    model_values.append(result.get('rate', result.get('score', 0)))
                    ci_lows.append(result['confidence_interval'][0])
                    ci_highs.append(result['confidence_interval'][1])
                elif isinstance(result, (int, float)):
                    model_values.append(result)
                    ci_lows.append(result * 0.9)  # Dummy CIs if not available
                    ci_highs.append(min(100, result * 1.1))
                else:
                    continue
        else:
            # If we have a metrics dict
            if not models:
                models = [m for m in self.metrics.keys() if m != '__global__']
                
            # Get metric values with confidence intervals
            model_values = []
            ci_lows = []
            ci_highs = []
            valid_models = []
            
            for model in models:
                if metric_name in self.metrics[model]:
                    result = self.metrics[model][metric_name]
                    
                    if isinstance(result, dict) and 'confidence_interval' in result:
                        valid_models.append(model)
                        model_values.append(result.get('rate', result.get('score', 0)))
                        ci_lows.append(result['confidence_interval'][0])
                        ci_highs.append(result['confidence_interval'][1])
                    elif isinstance(result, (int, float)):
                        valid_models.append(model)
                        model_values.append(result)
                        ci_lows.append(result * 0.9)  # Dummy CIs if not available
                        ci_highs.append(min(100, result * 1.1))
            
            models = valid_models
        
        if not model_values:
            return None  # No data
            
        # Calculate error bars
        errors_low = [v - l for v, l in zip(model_values, ci_lows)]
        errors_high = [h - v for v, h in zip(model_values, ci_highs)]
        
        # Create the figure
        fig, ax = plt.subplots(figsize=(12, 7))
        
        # Sort by metric value
        sorted_data = sorted(zip(models, model_values, errors_low, errors_high), 
                            key=lambda x: x[1], reverse=True)
        models = [x[0] for x in sorted_data]
        model_values = [x[1] for x in sorted_data]
        errors_low = [x[2] for x in sorted_data]
        errors_high = [x[3] for x in sorted_data]
        
        # Plot the bars with asymmetric error bars
        x_pos = np.arange(len(models))
        bars = ax.bar(x_pos, model_values, color=self.color_palette[:len(models)])
        
        # Add error bars
        asymmetric_errors = [errors_low, errors_high]
        ax.errorbar(x_pos, model_values, yerr=asymmetric_errors, fmt='none', ecolor='black', capsize=5)
        
        # Add labels and title
        ax.set_xticks(x_pos)
        ax.set_xticklabels(models, rotation=45, ha='right')
        
        metric_labels = {
            'bluff_success_rate': 'Bluff Success Rate (%)',
            'lie_detection': 'Lie Detection Rate (%)',
            'bid_optimality': 'Bid Optimality Rate (%)',
            'adaptation_score': 'Adaptation Score (%)',
            'rule_adherence_rate': 'Rule Adherence Rate (%)'
        }
        
        ax.set_ylabel(metric_labels.get(metric_name, metric_name))
        ax.set_title(f'{metric_labels.get(metric_name, metric_name)} with Confidence Intervals', fontweight='bold')
        
        # Add grid and improve aesthetics
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        # Add value labels
        for i, bar in enumerate(bars):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                   f'{model_values[i]:.1f}%', ha='center', va='bottom')
        
        plt.tight_layout()
        
        # Save if path provided
        if save_path:
            plt.savefig(os.path.join(self.output_dir, save_path), bbox_inches='tight')
            
        return fig
    
    def generate_comprehensive_report(self, output_prefix=None):
        """
        Generate a comprehensive visualization report with all metrics
        
        Args:
            output_prefix: Optional prefix for saved files
            
        Returns:
            Dict with paths to all generated visualizations
        """
        # Create a directory for this report
        if output_prefix:
            report_dir = os.path.join(self.output_dir, f"{output_prefix}_visualizations")
        else:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            report_dir = os.path.join(self.output_dir, f"metrics_report_{timestamp}")
            
        os.makedirs(report_dir, exist_ok=True)
        
        # Generate all visualizations
        generated_files = {}
        
        # ELO ratings
        try:
            fig = self.visualize_elo_ratings(save_path="elo_ratings.png")
            if fig:
                generated_files['elo_ratings'] = os.path.join(report_dir, "elo_ratings.png")
                plt.close(fig)
        except Exception as e:
            print(f"Error generating ELO visualization: {e}")
        
        # Win rates heatmap
        try:
            fig = self.visualize_win_rates(save_path="win_rates.png")
            if fig:
                generated_files['win_rates'] = os.path.join(report_dir, "win_rates.png")
                plt.close(fig)
        except Exception as e:
            print(f"Error generating win rates visualization: {e}")
        
        # Radar chart
        try:
            fig = self.visualize_radar_metrics(save_path="radar_metrics.png")
            if fig:
                generated_files['radar_metrics'] = os.path.join(report_dir, "radar_metrics.png")
                plt.close(fig)
        except Exception as e:
            print(f"Error generating radar visualization: {e}")
        
        # Metric correlations
        try:
            fig = self.visualize_metric_correlations(save_path="metric_correlations.png")
            if fig:
                generated_files['metric_correlations'] = os.path.join(report_dir, "metric_correlations.png")
                plt.close(fig)
        except Exception as e:
            print(f"Error generating correlations visualization: {e}")
        
        # Confidence intervals for each major metric
        metrics = ['bluff_success_rate', 'lie_detection', 'bid_optimality', 
                  'adaptation_score', 'rule_adherence_rate']
        
        for metric in metrics:
            try:
                fig = self.visualize_confidence_intervals(metric, save_path=f"{metric}_intervals.png")
                if fig:
                    generated_files[f'{metric}_intervals'] = os.path.join(report_dir, f"{metric}_intervals.png")
                    plt.close(fig)
            except Exception as e:
                print(f"Error generating {metric} intervals visualization: {e}")
        
        # Trends for selected metrics
        trend_metrics = ['elo', 'bluff_success', 'bid_optimality']
        
        for metric in trend_metrics:
            try:
                fig = self.visualize_metric_trends(metric, save_path=f"{metric}_trend.png")
                if fig:
                    generated_files[f'{metric}_trend'] = os.path.join(report_dir, f"{metric}_trend.png")
                    plt.close(fig)
            except Exception as e:
                print(f"Error generating {metric} trend visualization: {e}")
        
        return generated_files