#!/usr/bin/env python3
"""
Test script to verify async tournament logging works correctly.
"""

import sys
import os
import json
import tempfile
import shutil
import time

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from event_logger import EventLoggerFactory, EventLogger
from liars_dice import LiarsDice
from ai_player import AIPlayer, PROVIDER_LOCAL

def test_async_logging():
    """Test that async tournaments log moves correctly"""
    print("Testing async tournament move logging...")
    
    # Create a temporary directory for test data
    temp_dir = tempfile.mkdtemp()
    try:
        # Mock up a minimal tournament
        batch_runner = GameBatchRunner()
        batch_runner.total_games = 1
        batch_runner.models_per_game = 2
        batch_runner.use_async = True
        batch_runner.verbose_output = False
        batch_runner.raw_logging_enabled = True
        batch_runner.auto_mode = True
        
        # Create mock models (we'll use local models that don't exist, which will cause fallback behavior)
        batch_runner.selected_models = [
            {
                "id": "local/test-model-1",
                "provider": PROVIDER_LOCAL,
                "name": "test-model-1",
                "api_key": None,
                "api_url": "http://127.0.0.1:1234/v1/chat/completions",
                "project_id": None,
                "region": None
            },
            {
                "id": "local/test-model-2", 
                "provider": PROVIDER_LOCAL,
                "name": "test-model-2",
                "api_key": None,
                "api_url": "http://127.0.0.1:1234/v1/chat/completions",
                "project_id": None,
                "region": None
            }
        ]
        
        # Initialize event logger in the temp directory
        EventLoggerFactory.set_enabled(True)
        logger = EventLogger(
            tournament_id="test_async_logging",
            base_path=temp_dir,
            compression=False
        )
        
        print("Running async tournament with mock models (will use fallback logic)...")
        
        # The models don't exist so API calls will fail and use fallback logic
        # This will still test that moves are logged in async mode
        try:
            batch_runner.run_games_with_asyncio([1])
        except Exception as e:
            print(f"Expected error from non-existent models: {e}")
        
        # Close the logger to flush events
        logger.close()
        time.sleep(0.1)
        
        # Check for logged data
        tournament_dir = os.path.join(temp_dir, "tournament_test_async_logging")
        game_file = os.path.join(tournament_dir, "games", "game_0001.jsonl")
        
        if os.path.exists(game_file):
            print("✓ Game file created successfully")
            
            # Read and verify events
            events = []
            with open(game_file, 'r') as f:
                for line in f:
                    if line.strip():
                        events.append(json.loads(line.strip()))
            
            # Look for move events
            move_events = [e for e in events if e.get('type') == 'move']
            
            if move_events:
                print(f"✓ Found {len(move_events)} move events in async mode")
                
                # Check that moves have the expected structure
                sample_move = move_events[0]
                required_fields = ['actor_name', 'actor_model', 'parsed_action', 'raw_prompt', 'raw_response']
                missing_fields = [field for field in required_fields if field not in sample_move]
                
                if not missing_fields:
                    print("✓ Move events have correct structure")
                else:
                    print(f"✗ Missing fields in move events: {missing_fields}")
                    return False
                    
                print("✓ Async tournament move logging is working correctly!")
                return True
            else:
                print("✗ No move events found in async mode")
                return False
        else:
            print(f"✗ Game file not found: {game_file}")
            return False
            
    finally:
        # Clean up
        EventLoggerFactory.close_current()
        shutil.rmtree(temp_dir)

if __name__ == "__main__":
    success = test_async_logging()
    if success:
        print("\n✓ All async logging tests passed!")
        sys.exit(0)
    else:
        print("\n✗ Async logging tests failed!")
        sys.exit(1) 