"""
Unit tests for the EventLogger module.
"""

import unittest
import tempfile
import shutil
import os
import json
import time
import threading
from unittest.mock import patch, MagicMock

# Add src to path
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from event_logger import EventLogger, EventLoggerFactory, log_if_enabled
from event_schema import validate_event, EventValidationError


class TestEventLogger(unittest.TestCase):
    """Test the EventLogger class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.test_dir = tempfile.mkdtemp()
        self.tournament_id = "test_tournament_123"
    
    def tearDown(self):
        """Clean up test fixtures."""
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)
    
    def test_logger_initialization(self):
        """Test EventLogger initialization."""
        logger = EventLogger(
            tournament_id=self.tournament_id,
            base_path=self.test_dir,
            compression=False
        )
        
        # Check directory structure
        expected_tournament_dir = os.path.join(self.test_dir, f"tournament_{self.tournament_id}")
        self.assertTrue(os.path.exists(expected_tournament_dir))
        self.assertTrue(os.path.exists(os.path.join(expected_tournament_dir, "games")))
        self.assertTrue(os.path.exists(os.path.join(expected_tournament_dir, "indices")))
        
        # Check logger properties
        self.assertEqual(logger.tournament_id, self.tournament_id)
        self.assertFalse(logger.compression)
        self.assertEqual(logger.file_extension, ".jsonl")
        
        logger.close()
    
    def test_logger_with_compression(self):
        """Test EventLogger with compression enabled."""
        logger = EventLogger(
            tournament_id=self.tournament_id,
            base_path=self.test_dir,
            compression=True
        )
        
        self.assertTrue(logger.compression)
        self.assertEqual(logger.file_extension, ".jsonl.gz")
        
        logger.close()
    
    def test_tournament_start_event(self):
        """Test logging tournament start event."""
        logger = EventLogger(
            tournament_id=self.tournament_id,
            base_path=self.test_dir
        )
        
        cli_settings = {
            "total_games": 10,
            "models_per_game": 2,
            "use_async": True
        }
        
        logger.log_tournament_start(cli_settings, random_seed=12345)
        
        # Allow time for background thread to process
        time.sleep(0.1)
        logger.close(timeout=1.0)
        
        # Check meta.json was created
        meta_path = os.path.join(logger.tournament_dir, "meta.json")
        self.assertTrue(os.path.exists(meta_path))
        
        with open(meta_path, 'r') as f:
            meta_data = json.load(f)
        
        self.assertIn("events", meta_data)
        self.assertEqual(len(meta_data["events"]), 1)
        
        event = meta_data["events"][0]
        self.assertEqual(event["type"], "tournament_start")
        self.assertEqual(event["tournament_id"], self.tournament_id)
        self.assertEqual(event["cli_settings"], cli_settings)
        self.assertEqual(event["random_seed"], 12345)
    
    def test_game_events(self):
        """Test logging game start and end events."""
        logger = EventLogger(
            tournament_id=self.tournament_id,
            base_path=self.test_dir
        )
        
        # Log game start
        player_roster = [
            {"name": "Alice", "model_id": "gpt-4", "provider": "openai", "initial_dice": 5},
            {"name": "Bob", "model_id": "claude-3", "provider": "anthropic", "initial_dice": 5}
        ]
        starting_elos = {"Alice": 1200.0, "Bob": 1100.0}
        
        logger.log_game_start(1, player_roster, starting_elos)
        
        # Log game end
        logger.log_game_end(1, "Alice", 8)
        
        # Allow time for background thread to process
        time.sleep(0.1)
        logger.close(timeout=1.0)
        
        # Check game file was created
        game_file = os.path.join(logger.games_dir, "game_0001.jsonl")
        self.assertTrue(os.path.exists(game_file))
        
        # Read and validate events
        with open(game_file, 'r') as f:
            lines = f.readlines()
        
        self.assertEqual(len(lines), 2)
        
        # Validate game_start event
        game_start = json.loads(lines[0])
        self.assertEqual(game_start["type"], "game_start")
        self.assertEqual(game_start["game_id"], 1)
        self.assertEqual(game_start["player_roster"], player_roster)
        self.assertEqual(game_start["starting_elos"], starting_elos)
        
        # Validate game_end event
        game_end = json.loads(lines[1])
        self.assertEqual(game_end["type"], "game_end")
        self.assertEqual(game_end["game_id"], 1)
        self.assertEqual(game_end["final_winner"], "Alice")
        self.assertEqual(game_end["rounds_played"], 8)
    
    def test_move_event(self):
        """Test logging move events."""
        logger = EventLogger(
            tournament_id=self.tournament_id,
            base_path=self.test_dir
        )
        
        player_snapshot = {
            "player_order": ["Alice", "Bob"],
            "dice_by_player": {"Alice": [1, 2, 3, 4, 5], "Bob": [2, 3, 4, 5, 6]},
            "counts": {"1": 1, "2": 2, "3": 2, "4": 2, "5": 2, "6": 1},
            "total_dice": 10,
            "last_bid": [2, 3],
            "current_player": "Alice",
            "round": 1,
            "game_id": 1
        }
        
        token_usage = {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150}
        
        logger.log_move(
            game_id=1,
            round_number=1,
            player_snapshot=player_snapshot,
            actor_name="Alice",
            actor_model="gpt-4",
            raw_prompt="You are playing Liar's Dice...",
            raw_model_response='{"action": "bid", "quantity": 3, "face": 4}',
            parsed_action="bid",
            parsed_quantity=3,
            parsed_face=4,
            utterance="I'll bid 3 fours!",
            response_time=1.2,
            token_usage=token_usage
        )
        
        time.sleep(0.1)
        logger.close(timeout=1.0)
        
        # Read and validate event
        game_file = os.path.join(logger.games_dir, "game_0001.jsonl")
        with open(game_file, 'r') as f:
            event = json.loads(f.read().strip())
        
        self.assertEqual(event["type"], "move")
        self.assertEqual(event["game_id"], 1)
        self.assertEqual(event["actor_name"], "Alice")
        self.assertEqual(event["parsed_action"], "bid")
        self.assertEqual(event["parsed_quantity"], 3)
        self.assertEqual(event["parsed_face"], 4)
        self.assertEqual(event["token_usage"], token_usage)
    
    def test_liar_resolution_event(self):
        """Test logging liar resolution events."""
        logger = EventLogger(
            tournament_id=self.tournament_id,
            base_path=self.test_dir
        )
        
        all_players_dice = {
            "Alice": [1, 2, 3, 4, 5],
            "Bob": [2, 3, 4, 5, 6]
        }
        
        logger.log_liar_resolution(
            game_id=1,
            round_number=2,
            calling_player="Bob",
            target_player="Alice",
            last_bid=(3, 4),
            actual_face_count=2,
            outcome="success",
            all_players_dice=all_players_dice
        )
        
        time.sleep(0.1)
        logger.close(timeout=1.0)
        
        # Read and validate event
        game_file = os.path.join(logger.games_dir, "game_0001.jsonl")
        with open(game_file, 'r') as f:
            event = json.loads(f.read().strip())
        
        self.assertEqual(event["type"], "liar_resolution")
        self.assertEqual(event["calling_player"], "Bob")
        self.assertEqual(event["target_player"], "Alice")
        self.assertEqual(event["last_bid"]["quantity"], 3)
        self.assertEqual(event["last_bid"]["face"], 4)
        self.assertEqual(event["actual_face_count"], 2)
        self.assertEqual(event["outcome"], "success")
        self.assertEqual(event["all_players_dice"], all_players_dice)
    
    def test_concurrent_logging(self):
        """Test logging from multiple threads concurrently."""
        logger = EventLogger(
            tournament_id=self.tournament_id,
            base_path=self.test_dir
        )
        
        def log_events(game_id, num_events):
            for i in range(num_events):
                logger.log_round_start(game_id, i + 1, 10)
        
        # Create multiple threads
        threads = []
        for game_id in range(1, 4):
            thread = threading.Thread(target=log_events, args=(game_id, 5))
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        time.sleep(0.1)
        logger.close(timeout=2.0)
        
        # Check that files were created for each game
        for game_id in range(1, 4):
            game_file = os.path.join(logger.games_dir, f"game_{game_id:04d}.jsonl")
            self.assertTrue(os.path.exists(game_file))
            
            with open(game_file, 'r') as f:
                lines = f.readlines()
            
            # Should have 5 events per game
            self.assertEqual(len(lines), 5)
    
    def test_queue_overflow_fallback(self):
        """Test that full queue falls back to synchronous writing."""
        # Create logger with very small queue
        logger = EventLogger(
            tournament_id=self.tournament_id,
            base_path=self.test_dir,
            queue_maxsize=2
        )
        
        # Fill queue beyond capacity
        for i in range(10):
            logger.log_round_start(1, i + 1, 10)
        
        logger.close(timeout=1.0)
        
        # Events should still be logged (via fallback)
        game_file = os.path.join(logger.games_dir, "game_0001.jsonl")
        self.assertTrue(os.path.exists(game_file))
        
        with open(game_file, 'r') as f:
            lines = f.readlines()
        
        # All events should be logged
        self.assertEqual(len(lines), 10)


class TestEventLoggerFactory(unittest.TestCase):
    """Test the EventLoggerFactory class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.test_dir = tempfile.mkdtemp()
        # Reset factory state
        EventLoggerFactory._instance = None
        EventLoggerFactory._enabled = True
    
    def tearDown(self):
        """Clean up test fixtures."""
        EventLoggerFactory.close_current()
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)
    
    def test_factory_singleton_behavior(self):
        """Test that factory creates and manages singleton logger."""
        logger1 = EventLoggerFactory.create_logger(
            tournament_id="test1",
            base_path=self.test_dir
        )
        
        logger2 = EventLoggerFactory.get_logger()
        
        self.assertIs(logger1, logger2)
        
        # Creating new logger should close old one
        logger3 = EventLoggerFactory.create_logger(
            tournament_id="test2",
            base_path=self.test_dir
        )
        
        self.assertIsNot(logger1, logger3)
        self.assertTrue(logger1.closed)
    
    def test_factory_enabled_disabled(self):
        """Test factory enable/disable functionality."""
        # Create logger when enabled
        EventLoggerFactory.set_enabled(True)
        logger = EventLoggerFactory.create_logger(base_path=self.test_dir)
        self.assertIsNotNone(logger)
        
        # Disable should close logger
        EventLoggerFactory.set_enabled(False)
        self.assertIsNone(EventLoggerFactory.get_logger())
        self.assertTrue(logger.closed)
        
        # Re-enable should allow new logger creation
        EventLoggerFactory.set_enabled(True)
        new_logger = EventLoggerFactory.create_logger(base_path=self.test_dir)
        self.assertIsNotNone(new_logger)
        self.assertIsNot(logger, new_logger)
    
    def test_log_if_enabled_decorator(self):
        """Test the log_if_enabled decorator."""
        mock_func = MagicMock()
        decorated_func = log_if_enabled(mock_func)
        
        # Should not call when disabled
        EventLoggerFactory.set_enabled(False)
        decorated_func("arg1", "arg2", kwarg="value")
        mock_func.assert_not_called()
        
        # Should call when enabled
        EventLoggerFactory.set_enabled(True)
        logger = EventLoggerFactory.create_logger(base_path=self.test_dir)
        decorated_func("arg1", "arg2", kwarg="value")
        mock_func.assert_called_once_with(logger, "arg1", "arg2", kwarg="value")


class TestEventValidation(unittest.TestCase):
    """Test event validation functionality."""
    
    def test_valid_tournament_start_event(self):
        """Test validation of valid tournament start event."""
        event = {
            "type": "tournament_start",
            "timestamp": time.time(),
            "tournament_id": "test_123",
            "cli_settings": {"total_games": 10},
            "random_seed": 12345,
            "repo_hash": "abc123",
            "compression": False
        }
        
        # Should not raise exception
        self.assertTrue(validate_event(event))
    
    def test_invalid_event_missing_field(self):
        """Test validation failure for missing required field."""
        event = {
            "type": "tournament_start",
            "timestamp": time.time(),
            # Missing tournament_id
            "cli_settings": {"total_games": 10},
            "random_seed": 12345,
            "repo_hash": "abc123",
            "compression": False
        }
        
        with self.assertRaises(EventValidationError) as cm:
            validate_event(event)
        
        self.assertIn("missing required field: tournament_id", str(cm.exception))
    
    def test_invalid_event_wrong_type(self):
        """Test validation failure for wrong field type."""
        event = {
            "type": "tournament_start",
            "timestamp": "not_a_number",  # Should be number
            "tournament_id": "test_123",
            "cli_settings": {"total_games": 10},
            "random_seed": 12345,
            "repo_hash": "abc123",
            "compression": False
        }
        
        with self.assertRaises(EventValidationError) as cm:
            validate_event(event)
        
        self.assertIn("incorrect type", str(cm.exception))
    
    def test_valid_move_event(self):
        """Test validation of valid move event."""
        event = {
            "type": "move",
            "timestamp": time.time(),
            "game_id": 1,
            "round_number": 2,
            "player_snapshot": {
                "player_order": ["Alice", "Bob"],
                "dice_by_player": {"Alice": [1, 2, 3], "Bob": [4, 5, 6]},
                "counts": {"1": 1, "2": 1, "3": 1, "4": 1, "5": 1, "6": 1},
                "total_dice": 6,
                "last_bid": [2, 3],
                "current_player": "Alice",
                "round": 2,
                "game_id": 1
            },
            "actor_name": "Alice",
            "actor_model": "gpt-4",
            "raw_prompt": "You are playing...",
            "raw_model_response": '{"action": "bid"}',
            "parsed_action": "bid",
            "parsed_quantity": 3,
            "parsed_face": 4,
            "utterance": "I bid 3 fours",
            "response_time": 1.5,
            "token_usage": {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150}
        }
        
        # Should not raise exception
        self.assertTrue(validate_event(event))
    
    def test_invalid_move_event_bad_action(self):
        """Test validation failure for invalid parsed_action."""
        event = {
            "type": "move",
            "timestamp": time.time(),
            "game_id": 1,
            "round_number": 2,
            "player_snapshot": {
                "player_order": ["Alice"],
                "dice_by_player": {"Alice": [1, 2, 3]},
                "counts": {"1": 1, "2": 1, "3": 1},
                "total_dice": 3,
                "last_bid": None,
                "current_player": "Alice",
                "round": 2,
                "game_id": 1
            },
            "actor_name": "Alice",
            "actor_model": "gpt-4",
            "raw_prompt": "You are playing...",
            "raw_model_response": '{"action": "invalid"}',
            "parsed_action": "invalid_action",  # Should be "bid" or "liar"
            "parsed_quantity": None,
            "parsed_face": None,
            "utterance": "Invalid action",
            "response_time": 1.5,
            "token_usage": {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150}
        }
        
        with self.assertRaises(EventValidationError) as cm:
            validate_event(event)
        
        self.assertIn("Invalid parsed_action", str(cm.exception))


if __name__ == '__main__':
    unittest.main()