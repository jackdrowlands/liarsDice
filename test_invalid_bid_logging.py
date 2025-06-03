#!/usr/bin/env python3
"""
Test script to verify invalid bid correction logging works correctly.
"""

import sys
import os
import json
import tempfile
import shutil
import time

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from event_logger import EventLoggerFactory, EventLogger, log_move
from event_schema import validate_event

def test_invalid_bid_logging():
    """Test that invalid bid corrections are properly logged"""
    print("Testing invalid bid correction logging...")
    
    # Create a temporary directory for test data
    temp_dir = tempfile.mkdtemp()
    try:
        # Initialize event logger
        logger = EventLogger(
            tournament_id="test_invalid_bid",
            base_path=temp_dir,
            compression=False
        )
        EventLoggerFactory._instance = logger
        EventLoggerFactory._enabled = True
        
        # Create mock player snapshot
        player_snapshot = {
            "player_order": ["TestAI"],
            "dice_by_player": {"TestAI": [1, 2, 3, 4, 5]},
            "counts": {"1": 1, "2": 1, "3": 1, "4": 1, "5": 1, "6": 0},
            "total_dice": 5,
            "last_bid": None,
            "current_player": "TestAI",
            "round": 1,
            "game_id": 1
        }
        
        # Create mock token usage
        token_usage = {'prompt_tokens': 10, 'completion_tokens': 5, 'total_tokens': 15}
        
        # Test logging with invalid bid correction
        log_move(
            game_id=1,
            round_number=1,
            player_snapshot=player_snapshot,
            actor_name="TestAI",
            actor_model="test/model",
            raw_prompt="Test prompt",
            raw_model_response='{"action": "bid", "quantity": -1, "face": 7}',
            parsed_action="bid",
            parsed_quantity=2,  # Corrected quantity
            parsed_face=3,      # Corrected face
            utterance="Test utterance",
            response_time=1.0,
            token_usage=token_usage,
            invalid_bid_corrected=True,
            original_quantity=-1,
            original_face=7
        )
        
        # Close the logger to flush events
        logger.close()
        
        # Add a small delay to ensure all events are written
        time.sleep(0.1)
        
        # Read the logged data
        tournament_dir = os.path.join(temp_dir, "tournament_test_invalid_bid")
        game_file = os.path.join(tournament_dir, "games", "game_0001.jsonl")
        if os.path.exists(game_file):
            with open(game_file, 'r') as f:
                for line in f:
                    event = json.loads(line.strip())
                    if event['type'] == 'move':
                        print("✓ Move event logged successfully")
                        
                        # Validate the event structure
                        try:
                            validate_event(event)
                            print("✓ Event validation passed")
                        except Exception as e:
                            print(f"✗ Event validation failed: {e}")
                            return False
                        
                        # Check for invalid bid correction fields
                        if event.get('invalid_bid_corrected'):
                            print("✓ Invalid bid correction flag present")
                            if event.get('original_quantity') == -1:
                                print("✓ Original quantity logged correctly")
                            else:
                                print(f"✗ Original quantity incorrect: {event.get('original_quantity')}")
                                return False
                                
                            if event.get('original_face') == 7:
                                print("✓ Original face logged correctly")
                            else:
                                print(f"✗ Original face incorrect: {event.get('original_face')}")
                                return False
                        else:
                            print("✗ Invalid bid correction flag not set")
                            return False
                        
                        # Check corrected values
                        if event.get('parsed_quantity') == 2:
                            print("✓ Corrected quantity logged correctly")
                        else:
                            print(f"✗ Corrected quantity incorrect: {event.get('parsed_quantity')}")
                            return False
                            
                        if event.get('parsed_face') == 3:
                            print("✓ Corrected face logged correctly")
                        else:
                            print(f"✗ Corrected face incorrect: {event.get('parsed_face')}")
                            return False
                        
                        return True
        else:
            print(f"✗ Game file not found: {game_file}")
            return False
            
    finally:
        # Clean up
        shutil.rmtree(temp_dir)
        EventLoggerFactory._instance = None

def test_valid_bid_logging():
    """Test that valid bids don't have correction fields"""
    print("\nTesting valid bid logging...")
    
    # Create a temporary directory for test data
    temp_dir = tempfile.mkdtemp()
    try:
        # Initialize event logger
        logger = EventLogger(
            tournament_id="test_valid_bid",
            base_path=temp_dir,
            compression=False
        )
        EventLoggerFactory._instance = logger
        EventLoggerFactory._enabled = True
        
        # Create mock player snapshot
        player_snapshot = {
            "player_order": ["TestAI"],
            "dice_by_player": {"TestAI": [1, 2, 3, 4, 5]},
            "counts": {"1": 1, "2": 1, "3": 1, "4": 1, "5": 1, "6": 0},
            "total_dice": 5,
            "last_bid": None,
            "current_player": "TestAI",
            "round": 1,
            "game_id": 1
        }
        
        # Create mock token usage
        token_usage = {'prompt_tokens': 10, 'completion_tokens': 5, 'total_tokens': 15}
        
        # Test logging without invalid bid correction (default values)
        log_move(
            game_id=1,
            round_number=1,
            player_snapshot=player_snapshot,
            actor_name="TestAI",
            actor_model="test/model",
            raw_prompt="Test prompt",
            raw_model_response='{"action": "bid", "quantity": 2, "face": 3}',
            parsed_action="bid",
            parsed_quantity=2,
            parsed_face=3,
            utterance="Test utterance",
            response_time=1.0,
            token_usage=token_usage
            # invalid_bid_corrected, original_quantity, original_face are default (False, None, None)
        )
        
        # Close the logger to flush events
        logger.close()
        
        # Add a small delay to ensure all events are written
        time.sleep(0.1)
        
        # Read the logged data
        tournament_dir = os.path.join(temp_dir, "tournament_test_valid_bid")
        game_file = os.path.join(tournament_dir, "games", "game_0001.jsonl")
        if os.path.exists(game_file):
            with open(game_file, 'r') as f:
                for line in f:
                    event = json.loads(line.strip())
                    if event['type'] == 'move':
                        print("✓ Move event logged successfully")
                        
                        # Validate the event structure
                        try:
                            validate_event(event)
                            print("✓ Event validation passed")
                        except Exception as e:
                            print(f"✗ Event validation failed: {e}")
                            return False
                        
                        # Check that invalid bid correction fields are not present
                        if 'invalid_bid_corrected' not in event:
                            print("✓ No invalid bid correction fields present (as expected)")
                        elif not event.get('invalid_bid_corrected'):
                            print("✓ Invalid bid correction flag correctly set to false")
                        else:
                            print("✗ Invalid bid correction flag incorrectly set to true")
                            return False
                        
                        return True
        else:
            print(f"✗ Game file not found: {game_file}")
            return False
            
    finally:
        # Clean up
        shutil.rmtree(temp_dir)
        EventLoggerFactory._instance = None

if __name__ == "__main__":
    print("Testing invalid bid correction logging implementation...")
    
    success = True
    success = test_invalid_bid_logging() and success
    success = test_valid_bid_logging() and success
    
    if success:
        print("\n✓ All tests passed! Invalid bid correction logging is working correctly.")
    else:
        print("\n✗ Some tests failed.")
        sys.exit(1) 