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
    }