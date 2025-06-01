#!/usr/bin/env python3
"""
Smoke test script for the Tournament Raw-Data Capture Layer.

This script validates that the event logging system works correctly
without requiring actual API calls. It uses mocked events to simulate
a complete tournament and verifies the output structure.
"""

import os
import sys
import json
import time
import tempfile
import shutil
from typing import Dict, Any, List

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from event_logger import EventLogger, EventLoggerFactory
from event_schema import validate_event, validate_tournament_directory


class MockTournamentGenerator:
    """Generate mock tournament data for testing."""
    
    def __init__(self, tournament_id: str = None):
        self.tournament_id = tournament_id or f"smoke_test_{int(time.time())}"
        self.models = [
            "anthropic/claude-3-sonnet",
            "openai/gpt-4-turbo",
            "meta-llama/llama-2-70b",
            "google/gemini-pro"
        ]
        self.players = ["Alice", "Bob", "Charlie", "Diana"]
    
    def generate_cli_settings(self) -> Dict[str, Any]:
        """Generate mock CLI settings."""
        return {
            "total_games": 3,
            "models_per_game": 2,
            "use_async": True,
            "auto_mode": True,
            "verbose_output": False,
            "enable_autosaves": False,
            "save_individual_games": True,
            "create_visualizations": False,
            "use_local_endpoint": False,
            "raw_logging_enabled": True
        }
    
    def generate_player_roster(self, game_id: int) -> List[Dict[str, Any]]:
        """Generate mock player roster for a game."""
        # Select 2 models for this game
        selected_models = self.models[:2]
        roster = []
        
        for i, model in enumerate(selected_models):
            player_info = {
                "name": self.players[i],
                "model_id": model,
                "provider": model.split('/')[0],
                "initial_dice": 5
            }
            roster.append(player_info)
        
        return roster
    
    def generate_starting_elos(self, roster: List[Dict[str, Any]]) -> Dict[str, float]:
        """Generate mock starting Elo ratings."""
        return {player["name"]: 1000.0 + (i * 50) for i, player in enumerate(roster)}
    
    def generate_player_snapshot(self, game_id: int, round_num: int, 
                                current_player: str, total_dice: int) -> Dict[str, Any]:
        """Generate mock player snapshot."""
        # Mock dice for each player
        dice_by_player = {}
        player_order = self.players[:2]  # 2 players per game
        
        for player in player_order:
            if player == current_player:
                dice_by_player[player] = [1, 2, 3, 4, 5]  # Current player's dice
            else:
                dice_by_player[player] = [2, 3, 4, 5, 6]  # Other player's dice
        
        # Count dice faces
        all_dice = []
        for dice in dice_by_player.values():
            all_dice.extend(dice)
        
        counts = {}
        for face in range(1, 7):
            counts[str(face)] = all_dice.count(face)
        
        return {
            "player_order": player_order,
            "dice_by_player": dice_by_player,
            "counts": counts,
            "total_dice": total_dice,
            "last_bid": [2, 3] if round_num > 1 else None,
            "current_player": current_player,
            "round": round_num,
            "game_id": game_id
        }
    
    def generate_token_usage(self) -> Dict[str, int]:
        """Generate mock token usage."""
        prompt_tokens = 150
        completion_tokens = 25
        return {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens
        }


def run_smoke_test() -> bool:
    """
    Run the smoke test for the raw logging system.
    
    Returns:
        True if all tests pass, False otherwise
    """
    print("🚀 Starting Tournament Raw-Data Capture Layer Smoke Test")
    print("=" * 60)
    
    # Create temporary directory for test output
    test_dir = tempfile.mkdtemp(prefix="liars_dice_smoke_test_")
    print(f"Test output directory: {test_dir}")
    
    try:
        # Initialize mock generator
        generator = MockTournamentGenerator()
        
        # Test 1: Create and configure EventLogger
        print("\n📝 Test 1: EventLogger initialization")
        logger = EventLogger(
            tournament_id=generator.tournament_id,
            base_path=test_dir,
            compression=False
        )
        
        expected_tournament_dir = os.path.join(test_dir, f"tournament_{generator.tournament_id}")
        if not os.path.exists(expected_tournament_dir):
            print("❌ Tournament directory not created")
            return False
        print("✅ EventLogger initialized successfully")
        
        # Test 2: Log tournament start
        print("\n📝 Test 2: Tournament start event")
        cli_settings = generator.generate_cli_settings()
        logger.log_tournament_start(cli_settings, random_seed=12345)
        time.sleep(0.1)  # Allow background thread to process
        
        # Test 3: Simulate multiple games
        print("\n📝 Test 3: Game events simulation")
        num_games = 3
        
        for game_id in range(1, num_games + 1):
            print(f"  Simulating game {game_id}...")
            
            # Game start
            roster = generator.generate_player_roster(game_id)
            starting_elos = generator.generate_starting_elos(roster)
            logger.log_game_start(game_id, roster, starting_elos)
            
            # Simulate 2-3 rounds per game
            num_rounds = 2 + (game_id % 2)  # 2 or 3 rounds
            current_player_idx = 0
            total_dice = 10
            
            for round_num in range(1, num_rounds + 1):
                # Round start
                logger.log_round_start(game_id, round_num, total_dice)
                
                # Simulate 2-4 moves per round
                num_moves = 2 + (round_num % 3)
                
                for move_idx in range(num_moves):
                    current_player = roster[current_player_idx]["name"]
                    current_model = roster[current_player_idx]["model_id"]
                    
                    # Generate move event
                    player_snapshot = generator.generate_player_snapshot(
                        game_id, round_num, current_player, total_dice
                    )
                    
                    if move_idx < num_moves - 1:
                        # Regular bid
                        logger.log_move(
                            game_id=game_id,
                            round_number=round_num,
                            player_snapshot=player_snapshot,
                            actor_name=current_player,
                            actor_model=current_model,
                            raw_prompt=f"You are playing Liar's Dice. Make your move for round {round_num}.",
                            raw_model_response='{"action": "bid", "quantity": 3, "face": 4, "reasoning": "I think there are 3 fours", "utterance": "I bid 3 fours!"}',
                            parsed_action="bid",
                            parsed_quantity=3,
                            parsed_face=4,
                            utterance="I bid 3 fours!",
                            response_time=1.2,
                            token_usage=generator.generate_token_usage()
                        )
                    else:
                        # Liar call (end of round)
                        logger.log_move(
                            game_id=game_id,
                            round_number=round_num,
                            player_snapshot=player_snapshot,
                            actor_name=current_player,
                            actor_model=current_model,
                            raw_prompt=f"You are playing Liar's Dice. The last bid was 3 fours. Call liar or make a higher bid.",
                            raw_model_response='{"action": "liar", "reasoning": "I don\'t think there are 3 fours", "utterance": "I call liar!"}',
                            parsed_action="liar",
                            parsed_quantity=None,
                            parsed_face=None,
                            utterance="I call liar!",
                            response_time=0.8,
                            token_usage=generator.generate_token_usage()
                        )
                        
                        # Liar resolution
                        calling_player = current_player
                        target_player = roster[1 - current_player_idx]["name"]
                        outcome = "success" if move_idx % 2 == 0 else "failure"
                        
                        all_players_dice = {
                            roster[0]["name"]: [1, 2, 3, 4, 5],
                            roster[1]["name"]: [2, 3, 4, 5, 6]
                        }
                        
                        logger.log_liar_resolution(
                            game_id=game_id,
                            round_number=round_num,
                            calling_player=calling_player,
                            target_player=target_player,
                            last_bid=(3, 4),
                            actual_face_count=2,
                            outcome=outcome,
                            all_players_dice=all_players_dice
                        )
                    
                    # Switch to next player
                    current_player_idx = 1 - current_player_idx
                
                # Round end
                surviving_players = [p["name"] for p in roster]
                logger.log_round_end(game_id, round_num, surviving_players)
                
                # Reduce dice count for next round
                total_dice = max(2, total_dice - 1)
            
            # Game end
            winner = roster[game_id % 2]["name"]  # Alternate winners
            logger.log_game_end(game_id, winner, num_rounds)
        
        # Test 4: Log tournament end
        print("\n📝 Test 4: Tournament end event")
        leaderboard_snapshot = {
            "total_games": num_games,
            "completed_games": num_games
        }
        logger.log_tournament_end(leaderboard_snapshot)
        
        # Allow time for background processing
        time.sleep(0.2)
        
        # Test 5: Close logger and verify files
        print("\n📝 Test 5: File structure validation")
        logger.close(timeout=2.0)
        
        # Check meta.json
        meta_path = os.path.join(expected_tournament_dir, "meta.json")
        if not os.path.exists(meta_path):
            print("❌ meta.json not found")
            return False
        
        with open(meta_path, 'r') as f:
            meta_data = json.load(f)
        
        if len(meta_data.get("events", [])) != 2:  # tournament_start + tournament_end
            print(f"❌ Expected 2 events in meta.json, found {len(meta_data.get('events', []))}")
            return False
        
        print("✅ meta.json created successfully")
        
        # Check game files
        games_dir = os.path.join(expected_tournament_dir, "games")
        game_files = [f for f in os.listdir(games_dir) if f.startswith("game_") and f.endswith(".jsonl")]
        
        if len(game_files) != num_games:
            print(f"❌ Expected {num_games} game files, found {len(game_files)}")
            return False
        
        print(f"✅ {len(game_files)} game files created successfully")
        
        # Test 6: Validate event schemas
        print("\n📝 Test 6: Event schema validation")
        validation_results = validate_tournament_directory(expected_tournament_dir)
        
        total_errors = sum(len(errors) for errors in validation_results.values())
        if total_errors > 0:
            print(f"❌ Found {total_errors} validation errors:")
            for file_path, errors in validation_results.items():
                if errors:
                    print(f"  {file_path}: {errors}")
            return False
        
        print("✅ All events pass schema validation")
        
        # Test 7: Verify file contents sample
        print("\n📝 Test 7: Content verification")
        
        # Check a sample game file
        sample_game_file = os.path.join(games_dir, "game_0001.jsonl")
        with open(sample_game_file, 'r') as f:
            lines = f.readlines()
        
        if len(lines) < 5:  # Should have game_start + rounds + moves + game_end
            print(f"❌ Game file seems too short: {len(lines)} lines")
            return False
        
        # Verify first and last events
        first_event = json.loads(lines[0])
        last_event = json.loads(lines[-1])
        
        if first_event["type"] != "game_start":
            print(f"❌ First event should be game_start, got {first_event['type']}")
            return False
        
        if last_event["type"] != "game_end":
            print(f"❌ Last event should be game_end, got {last_event['type']}")
            return False
        
        print("✅ Game file structure is correct")
        
        # Test 8: Storage size calculation
        print("\n📝 Test 8: Storage size analysis")
        
        total_size = 0
        for root, dirs, files in os.walk(expected_tournament_dir):
            for file in files:
                total_size += os.path.getsize(os.path.join(root, file))
        
        print(f"💾 Total storage used: {total_size:,} bytes ({total_size / 1024:.1f} KB)")
        
        # Calculate events per file
        total_events = 0
        for file_path, errors in validation_results.items():
            if file_path.endswith('.jsonl'):
                with open(file_path, 'r') as f:
                    total_events += len(f.readlines())
        
        if total_events > 0:
            print(f"📊 Average bytes per event: {total_size / total_events:.1f}")
        
        print("\n🎉 All smoke tests passed successfully!")
        print("\n📁 Generated files:")
        print(f"  Tournament directory: {expected_tournament_dir}")
        print(f"  Meta file: meta.json")
        print(f"  Game files: {len(game_files)} files")
        print(f"  Total events logged: {total_events}")
        
        return True
    
    except Exception as e:
        print(f"\n❌ Smoke test failed with exception: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        # Clean up (comment out to keep files for inspection)
        # shutil.rmtree(test_dir)
        print(f"\n🧹 Test files preserved at: {test_dir}")
        print("   (Delete manually when done inspecting)")


def main():
    """Main function."""
    print("Tournament Raw-Data Capture Layer - Smoke Test")
    print("This test validates the event logging system without API calls.")
    print()
    
    success = run_smoke_test()
    
    if success:
        print("\n✅ SMOKE TEST PASSED")
        print("The raw-data capture layer is working correctly!")
        sys.exit(0)
    else:
        print("\n❌ SMOKE TEST FAILED")
        print("Please check the output above for error details.")
        sys.exit(1)


if __name__ == "__main__":
    main()