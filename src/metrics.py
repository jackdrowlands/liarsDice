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
from typing import Dict, List, Tuple, Set, Optional, Union, Any, DefaultDict, Callable, TypeVar, Generic, Deque, cast

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
        """
        Generate a chart showing Elo rating progression over time
        
        Returns:
            Path to the generated chart image
        """
        plt.figure(figsize=(12, 8))
        
        for model_id, historical in self.metrics.historical_metrics.items():
            if "elo" in historical and historical["elo"]:
                # Extract timestamps and values
                timestamps = [entry["timestamp"] for entry in historical["elo"]]
                values = [entry["value"] for entry in historical["elo"]]
                
                # Normalize timestamps to start from 0
                if timestamps:
                    start_time = min(timestamps)
                    norm_timestamps = [(t - start_time) / 3600 for t in timestamps]  # Hours since start
                    
                    # Plot
                    plt.plot(norm_timestamps, values, marker='o', label=model_id, linewidth=2)
        
        plt.title("Model Elo Ratings Over Time", fontsize=16)
        plt.xlabel("Hours Since First Game", fontsize=12)
        plt.ylabel("Elo Rating", fontsize=12)
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        
        # Add horizontal line for the starting Elo
        plt.axhline(y=self.metrics.default_elo, color='gray', linestyle='--', alpha=0.5)
        
        output_path = os.path.join(self.output_dir, output_file)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return output_path
        
    def generate_metric_comparison_radar(self, output_file: str = "radar_metrics.png") -> str:
        """
        Generate a radar chart comparing all models across key metrics
        
        Returns:
            Path to the generated chart image
        """
        # Get all model metrics
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
        """
        Generate a heatmap showing win rates between different models
        
        Returns:
            Path to the generated chart image
        """
        model_ids = list(self.metrics.elo_ratings.keys())
        
        if len(model_ids) < 2:
            return "Not enough models to generate win matrix"
            
        # Create win rate matrix
        win_matrix = np.zeros((len(model_ids), len(model_ids)))
        
        for i, model1 in enumerate(model_ids):
            for j, model2 in enumerate(model_ids):
                if i == j:
                    # Diagonal is not applicable
                    win_matrix[i, j] = np.nan
                else:
                    # Get matchup stats safely with proper type annotation
                    model1_matchups: Dict[str, Dict[str, Union[int, float]]] = cast(Dict[str, Dict[str, Union[int, float]]], self.metrics.matchup_stats.get(model1, {}))
                    if isinstance(model1_matchups, dict):
                        model2_stats = model1_matchups.get(model2, {})
                        if isinstance(model2_stats, dict):
                            games = int(model2_stats.get("games", 0))
                            wins = int(model2_stats.get("wins", 0))
                        else:
                            games = 0
                            wins = 0
                    else:
                        games = 0
                        wins = 0
                    
                    if games > 0:
                        win_matrix[i, j] = wins / games * 100
                    else:
                        win_matrix[i, j] = np.nan
        
        # Create heatmap
        plt.figure(figsize=(12, 10))
        mask = np.isnan(win_matrix)
        
        # Use a colormap that goes from red (0%) to green (100%)
        cmap = LinearSegmentedColormap.from_list("win_rate", [(0.8, 0, 0), (1, 1, 0.7), (0, 0.7, 0)])
        
        sns.heatmap(win_matrix, 
                   annot=True, 
                   fmt=".1f", 
                   cmap=cmap, 
                   mask=mask,
                   linewidths=0.5, 
                   vmin=0, 
                   vmax=100,
                   xticklabels=model_ids, 
                   yticklabels=model_ids)
        
        plt.title("Win Rate Matrix (Row vs Column)", fontsize=16)
        plt.xlabel("Opponent Model", fontsize=12)
        plt.ylabel("Model", fontsize=12)
        
        # Rotate x-axis labels if they're too long
        plt.xticks(rotation=45, ha='right', fontsize=10)
        plt.yticks(fontsize=10)
        
        output_path = os.path.join(self.output_dir, output_file)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return output_path