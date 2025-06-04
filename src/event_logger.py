import json
import os
import time
import threading
import queue
import hashlib
import subprocess
from collections import defaultdict
from datetime import datetime
from typing import Dict, Any, Optional, List
import gzip


class EventLogger:
    """
    Non-blocking event logger for Liar's Dice tournaments.
    
    Captures all raw tournament events to enable later replay and analysis
    without requiring re-runs of the benchmark.
    """
    
    def __init__(self, tournament_id: Optional[str] = None, base_path: str = ".", 
                 compression: bool = False, queue_maxsize: int = 10000):
        """
        Initialize the EventLogger.
        
        Args:
            tournament_id: Unique identifier for this tournament
            base_path: Base directory for tournament files
            compression: Whether to use gzip compression for files
            queue_maxsize: Maximum queue size before blocking/fallback
        """
        self.tournament_id = tournament_id or self._generate_tournament_id()
        self.tournament_dir = os.path.join(base_path, f"tournament_{self.tournament_id}")
        self.games_dir = os.path.join(self.tournament_dir, "games")
        self.indices_dir = os.path.join(self.tournament_dir, "indices")
        self.compression = compression
        self.file_extension = ".jsonl.gz" if compression else ".jsonl"
        
        # Thread-safe queue for events
        self.event_queue = queue.Queue(maxsize=queue_maxsize)
        self.queue_maxsize = queue_maxsize
        
        # Writer thread
        self.writer_thread = None
        self.stop_event = threading.Event()
        self.closed = False
        
        # File handles for each game
        self.game_files: Dict[int, Any] = {}
        
        # Tournament metadata
        self.tournament_start_time = None
        self.cli_settings = {}
        self.repo_hash = self._get_repo_hash()
        
        # Thread lock for file operations
        self.file_lock = threading.Lock()
        
        # Initialize directories
        self._create_directories()
        
        # Start writer thread
        self._start_writer_thread()
    
    def _generate_tournament_id(self) -> str:
        """Generate a unique tournament ID."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        random_suffix = hashlib.md5(str(time.time()).encode()).hexdigest()[:8]
        return f"{timestamp}_{random_suffix}"
    
    def _get_repo_hash(self) -> str:
        """Get the current repository hash for reproducibility."""
        try:
            result = subprocess.run(
                ["git", "rev-parse", "HEAD"], 
                capture_output=True, 
                text=True, 
                cwd=os.path.dirname(__file__)
            )
            if result.returncode == 0:
                return result.stdout.strip()
        except Exception:
            pass
        return "unknown"
    
    def _create_directories(self):
        """Create necessary directory structure."""
        os.makedirs(self.tournament_dir, exist_ok=True)
        os.makedirs(self.games_dir, exist_ok=True)
        os.makedirs(self.indices_dir, exist_ok=True)
    
    def _start_writer_thread(self):
        """Start the background writer thread."""
        self.writer_thread = threading.Thread(target=self._writer_loop, daemon=True)
        self.writer_thread.start()
    
    def _writer_loop(self):
        """Main loop for the background writer thread."""
        while not self.stop_event.is_set():
            try:
                # Wait for events with timeout to check stop_event periodically
                event = self.event_queue.get(timeout=1.0)
                if event is None:  # Poison pill to stop
                    break
                    
                self._write_event(event)
                self.event_queue.task_done()
                
            except queue.Empty:
                continue
            except Exception as e:
                # Log error but continue processing
                print(f"Event logger error: {e}")
                continue
        
        # Flush remaining events
        while not self.event_queue.empty():
            try:
                event = self.event_queue.get_nowait()
                if event is not None:
                    self._write_event(event)
                    self.event_queue.task_done()
            except queue.Empty:
                break
            except Exception as e:
                print(f"Event logger flush error: {e}")
                break
    
    def _write_event(self, event: Dict[str, Any]):
        """Write a single event to the appropriate file."""
        event_type = event.get("type")
        
        if event_type in ["tournament_start", "tournament_end"]:
            self._write_meta_event(event)
        elif event_type in ["game_start", "game_end", "round_start", "round_end", "move", "liar_resolution"]:
            game_id = event.get("game_id")
            if game_id is not None:
                self._write_game_event(game_id, event)
    
    def _write_meta_event(self, event: Dict[str, Any]):
        """Write tournament-level events to meta.json."""
        meta_path = os.path.join(self.tournament_dir, "meta.json")
        
        with self.file_lock:
            # Read existing meta data
            meta_data = {}
            if os.path.exists(meta_path):
                try:
                    with open(meta_path, 'r') as f:
                        meta_data = json.load(f)
                except Exception:
                    pass
            
            # Add this event
            if "events" not in meta_data:
                meta_data["events"] = []
            meta_data["events"].append(event)
            
            # Write back to file
            with open(meta_path, 'w') as f:
                json.dump(meta_data, f, indent=2)
    
    def _write_game_event(self, game_id: int, event: Dict[str, Any]):
        """Write game-level events to the appropriate game file."""
        with self.file_lock:
            if game_id not in self.game_files:
                game_filename = f"game_{game_id:04d}{self.file_extension}"
                game_path = os.path.join(self.games_dir, game_filename)
                
                if self.compression:
                    self.game_files[game_id] = gzip.open(game_path, 'wt', encoding='utf-8')
                else:
                    self.game_files[game_id] = open(game_path, 'w', encoding='utf-8')
            
            # Write event as JSON line
            json.dump(event, self.game_files[game_id], separators=(',', ':'))
            self.game_files[game_id].write('\n')
            self.game_files[game_id].flush()
    
    def log_event(self, event: Dict[str, Any]):
        """
        Log an event asynchronously.
        
        Args:
            event: Event dictionary to log
        """
        if self.closed:
            return
        
        # Add timestamp if not present
        if "timestamp" not in event:
            event["timestamp"] = time.time()
        
        try:
            # Try to put event in queue without blocking
            self.event_queue.put_nowait(event)
        except queue.Full:
            # Queue is full - fall back to synchronous write with warning
            print(f"Event logger queue full, falling back to synchronous write")
            try:
                self._write_event(event)
            except Exception as e:
                print(f"Failed to write event synchronously: {e}")
    
    def log_tournament_start(self, cli_settings: Dict[str, Any], random_seed: Optional[int] = None):
        """Log tournament start event."""
        self.tournament_start_time = time.time()
        self.cli_settings = cli_settings.copy()
        
        event = {
            "type": "tournament_start",
            "timestamp": self.tournament_start_time,
            "tournament_id": self.tournament_id,
            "cli_settings": self.cli_settings,
            "random_seed": random_seed,
            "repo_hash": self.repo_hash,
            "compression": self.compression
        }
        self.log_event(event)
    
    def log_tournament_end(self, leaderboard_snapshot: Optional[Dict[str, Any]] = None):
        """Log tournament end event."""
        event = {
            "type": "tournament_end",
            "timestamp": time.time(),
            "tournament_id": self.tournament_id,
            "duration": time.time() - (self.tournament_start_time or 0),
            "leaderboard_snapshot": leaderboard_snapshot or {}
        }
        self.log_event(event)
    
    def log_game_start(self, game_id: int, player_roster: List[Dict[str, Any]], starting_elos: Dict[str, float]):
        """Log game start event."""
        event = {
            "type": "game_start",
            "timestamp": time.time(),
            "game_id": game_id,
            "player_roster": player_roster,
            "starting_elos": starting_elos
        }
        self.log_event(event)
    
    def log_game_end(self, game_id: int, final_winner: str, rounds_played: int):
        """Log game end event."""
        event = {
            "type": "game_end",
            "timestamp": time.time(),
            "game_id": game_id,
            "final_winner": final_winner,
            "rounds_played": rounds_played
        }
        self.log_event(event)
    
    def log_round_start(self, game_id: int, round_number: int, total_dice_count: int):
        """Log round start event."""
        event = {
            "type": "round_start",
            "timestamp": time.time(),
            "game_id": game_id,
            "round_number": round_number,
            "total_dice_count": total_dice_count
        }
        self.log_event(event)
    
    def log_round_end(self, game_id: int, round_number: int, surviving_players: List[str]):
        """Log round end event."""
        event = {
            "type": "round_end",
            "timestamp": time.time(),
            "game_id": game_id,
            "round_number": round_number,
            "surviving_players": surviving_players
        }
        self.log_event(event)
    
    def log_move(self, game_id: int, round_number: int, player_snapshot: Dict[str, Any], 
                 actor_name: str, actor_model: str, raw_prompt: str, raw_model_response: str,
                 parsed_action: str, parsed_quantity: Optional[int], parsed_face: Optional[int],
                 utterance: str, response_time: float, token_usage: Dict[str, int],
                 invalid_bid_corrected: bool = False, original_action: Optional[str] = None,
                 original_quantity: Optional[int] = None, original_face: Optional[int] = None):
        """Logs a player's move, including AI's raw thought process and response."""
        event_data: Dict[str, Any] = {
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
            "invalid_bid_corrected": invalid_bid_corrected
        }
        if invalid_bid_corrected:
            # Only add original fields if a correction actually occurred
            if original_action is not None:
                event_data["original_action"] = original_action
            if original_quantity is not None:
                event_data["original_quantity"] = original_quantity
            if original_face is not None:
                event_data["original_face"] = original_face
        
        self.log_event(event_data)
    
    def log_liar_resolution(self, game_id: int, round_number: int, calling_player: str,
                           target_player: str, last_bid: tuple, actual_face_count: int,
                           outcome: str, all_players_dice: Dict[str, List[int]]):
        """Log liar call resolution event."""
        event = {
            "type": "liar_resolution",
            "timestamp": time.time(),
            "game_id": game_id,
            "round_number": round_number,
            "calling_player": calling_player,
            "target_player": target_player,
            "last_bid": {"quantity": last_bid[0], "face": last_bid[1]},
            "actual_face_count": actual_face_count,
            "outcome": outcome,
            "all_players_dice": all_players_dice
        }
        self.log_event(event)
    
    def close(self, timeout: float = 2.0):
        """
        Close the event logger and flush all pending events.
        
        Args:
            timeout: Maximum time to wait for queue to flush
        """
        if self.closed:
            return
        
        self.closed = True
        
        # Signal writer thread to stop
        self.stop_event.set()
        
        # Add poison pill to queue
        try:
            self.event_queue.put_nowait(None)
        except queue.Full:
            pass
        
        # Wait for writer thread to finish
        if self.writer_thread and self.writer_thread.is_alive():
            self.writer_thread.join(timeout=timeout)
        
        # Close all game files
        with self.file_lock:
            for game_file in self.game_files.values():
                try:
                    game_file.close()
                except Exception:
                    pass
            self.game_files.clear()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


class EventLoggerFactory:
    """Factory for creating and managing EventLogger instances."""
    
    _instance: Optional[EventLogger] = None
    _enabled: bool = True
    
    @classmethod
    def create_logger(cls, tournament_id: Optional[str] = None, **kwargs) -> EventLogger:
        """Create a new EventLogger instance."""
        if cls._instance:
            cls._instance.close()
        
        cls._instance = EventLogger(tournament_id=tournament_id, **kwargs)
        return cls._instance
    
    @classmethod
    def get_logger(cls) -> Optional[EventLogger]:
        """Get the current EventLogger instance."""
        return cls._instance if cls._enabled else None
    
    @classmethod
    def set_enabled(cls, enabled: bool):
        """Enable or disable event logging globally."""
        cls._enabled = enabled
        if not enabled and cls._instance:
            cls._instance.close()
            cls._instance = None
    
    @classmethod
    def is_enabled(cls) -> bool:
        """Check if event logging is enabled."""
        return cls._enabled
    
    @classmethod
    def close_current(cls):
        """Close the current logger instance."""
        if cls._instance:
            cls._instance.close()
            cls._instance = None


def log_if_enabled(func):
    """Decorator to only execute logging if enabled."""
    def wrapper(*args, **kwargs):
        logger = EventLoggerFactory.get_logger()
        if logger:
            return func(logger, *args, **kwargs)
        return None
    return wrapper


# Convenience functions for logging events
@log_if_enabled
def log_tournament_start(logger: EventLogger, cli_settings: Dict[str, Any], random_seed: Optional[int] = None):
    logger.log_tournament_start(cli_settings, random_seed)

@log_if_enabled
def log_tournament_end(logger: EventLogger, leaderboard_snapshot: Optional[Dict[str, Any]] = None):
    logger.log_tournament_end(leaderboard_snapshot)

@log_if_enabled
def log_game_start(logger: EventLogger, game_id: int, player_roster: List[Dict[str, Any]], starting_elos: Dict[str, float]):
    logger.log_game_start(game_id, player_roster, starting_elos)

@log_if_enabled
def log_game_end(logger: EventLogger, game_id: int, final_winner: str, rounds_played: int):
    logger.log_game_end(game_id, final_winner, rounds_played)

@log_if_enabled
def log_round_start(logger: EventLogger, game_id: int, round_number: int, total_dice_count: int):
    logger.log_round_start(game_id, round_number, total_dice_count)

@log_if_enabled
def log_round_end(logger: EventLogger, game_id: int, round_number: int, surviving_players: List[str]):
    logger.log_round_end(game_id, round_number, surviving_players)

@log_if_enabled
def log_move(logger: EventLogger, game_id: int, round_number: int, player_snapshot: Dict[str, Any], 
             actor_name: str, actor_model: str, raw_prompt: str, raw_model_response: str,
             parsed_action: str, parsed_quantity: Optional[int], parsed_face: Optional[int],
             utterance: str, response_time: float, token_usage: Dict[str, int],
             invalid_bid_corrected: bool = False, original_action: Optional[str] = None,
             original_quantity: Optional[int] = None, original_face: Optional[int] = None):
    """Wrapper for EventLogger.log_move for easier procedural calls."""
    logger.log_move(game_id, round_number, player_snapshot, actor_name, actor_model,
                    raw_prompt, raw_model_response, parsed_action, parsed_quantity,
                    parsed_face, utterance, response_time, token_usage,
                    invalid_bid_corrected, original_action, original_quantity, original_face)

@log_if_enabled
def log_liar_resolution(logger: EventLogger, game_id: int, round_number: int, calling_player: str,
                       target_player: str, last_bid: tuple, actual_face_count: int,
                       outcome: str, all_players_dice: Dict[str, List[int]]):
    logger.log_liar_resolution(game_id, round_number, calling_player, target_player, 
                              last_bid, actual_face_count, outcome, all_players_dice)