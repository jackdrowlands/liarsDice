# Tournament Raw-Data Capture Layer

This document describes the raw-data capture system for Liar's Dice tournaments, which persists every event during gameplay to enable later replay and analysis without re-running benchmarks.

## Overview

The raw-data capture layer is designed to log all primitive events that occur during a tournament:

- **Tournament-level events**: Start/end with configuration snapshots
- **Game-level events**: Individual game start/end with player rosters and outcomes  
- **Round-level events**: Round boundaries with surviving players and dice counts
- **Move-level events**: Every player action with full context, prompts, and responses
- **Resolution events**: Liar call outcomes with complete dice revelation

All events are logged to append-only JSONL files with strict schema validation, ensuring data integrity and enabling deterministic replay.

## File Format

### Directory Structure

```
tournament_<timestamp>/
├── meta.json                 # Tournament-level events and metadata
├── games/                    # Individual game files
│   ├── game_0001.jsonl      # Game 1 events
│   ├── game_0002.jsonl      # Game 2 events
│   └── ...
└── indices/                  # Optional parquet/index files
```

### Event Types

#### Tournament Events (meta.json)

**tournament_start**
```json
{
  "type": "tournament_start",
  "timestamp": 1640995200.123,
  "tournament_id": "20220101_120000_abc123",
  "cli_settings": {
    "total_games": 100,
    "models_per_game": 4,
    "use_async": true,
    "raw_logging_enabled": true
  },
  "random_seed": 12345,
  "repo_hash": "a1b2c3d4e5f6",
  "compression": false
}
```

**tournament_end**
```json
{
  "type": "tournament_end",
  "timestamp": 1640998800.456,
  "tournament_id": "20220101_120000_abc123",
  "duration": 3600.333,
  "leaderboard_snapshot": {
    "gpt-4": {"wins": 25, "games": 50},
    "claude-3": {"wins": 23, "games": 50}
  }
}
```

#### Game Events (game_XXXX.jsonl)

**game_start**
```json
{
  "type": "game_start",
  "timestamp": 1640995201.789,
  "game_id": 1,
  "player_roster": [
    {
      "name": "Alice",
      "model_id": "openai/gpt-4",
      "provider": "openai",
      "initial_dice": 5
    },
    {
      "name": "Bob", 
      "model_id": "anthropic/claude-3-sonnet",
      "provider": "anthropic",
      "initial_dice": 5
    }
  ],
  "starting_elos": {
    "Alice": 1200.0,
    "Bob": 1150.0
  }
}
```

**round_start**
```json
{
  "type": "round_start",
  "timestamp": 1640995202.100,
  "game_id": 1,
  "round_number": 1,
  "total_dice_count": 10
}
```

**move** (Player Action)
```json
{
  "type": "move",
  "timestamp": 1640995203.456,
  "game_id": 1,
  "round_number": 1,
  "player_snapshot": {
    "player_order": ["Alice", "Bob"],
    "dice_by_player": {
      "Alice": [1, 2, 3, 4, 5],
      "Bob": [2, 3, 4, 5, 6]
    },
    "counts": {"1": 1, "2": 2, "3": 2, "4": 2, "5": 2, "6": 1},
    "total_dice": 10,
    "last_bid": [2, 3],
    "current_player": "Alice",
    "round": 1,
    "game_id": 1
  },
  "actor_name": "Alice",
  "actor_model": "openai/gpt-4",
  "raw_prompt": "You are playing Liar's Dice...",
  "raw_model_response": "{\"action\": \"bid\", \"quantity\": 3, \"face\": 4}",
  "parsed_action": "bid",
  "parsed_quantity": 3,
  "parsed_face": 4,
  "utterance": "I'll bid 3 fours!",
  "response_time": 1.234,
  "token_usage": {
    "prompt_tokens": 150,
    "completion_tokens": 25,
    "total_tokens": 175
  }
}
```

**liar_resolution**
```json
{
  "type": "liar_resolution",
  "timestamp": 1640995205.789,
  "game_id": 1,
  "round_number": 1,
  "calling_player": "Bob",
  "target_player": "Alice", 
  "last_bid": {"quantity": 3, "face": 4},
  "actual_face_count": 2,
  "outcome": "success",
  "all_players_dice": {
    "Alice": [1, 2, 3, 4, 5],
    "Bob": [2, 3, 4, 5, 6]
  }
}
```

## Configuration

### Enabling Raw Logging

Raw logging is controlled by the `raw_logging_enabled` flag in `GameBatchRunner`:

```python
# During tournament setup
batch_runner = GameBatchRunner()
batch_runner.raw_logging_enabled = True  # Default: True

# Or via interactive setup
# "Enable raw tournament data capture for replay analysis? (y/n): y"
```

### Storage Options

- **Compression**: Optional gzip compression for JSONL files
- **Queue size**: Configurable event queue size (default: 10000)
- **Fallback**: Automatic synchronous writing if queue fills

```python
EventLogger(
    tournament_id="custom_id",
    base_path="./tournaments",
    compression=True,      # Enable gzip compression
    queue_maxsize=5000     # Smaller queue size
)
```

## Usage Examples

### Running a Tournament with Raw Logging

```python
from src.batch_runner import GameBatchRunner

# Create and configure tournament
runner = GameBatchRunner()
runner.raw_logging_enabled = True

# Run tournament - events will be automatically logged
runner.run_tournament()
```

### Accessing Raw Data

```python
import json
import pandas as pd

# Read tournament metadata
with open("tournament_20220101_120000/meta.json", "r") as f:
    meta = json.load(f)
    
print(f"Tournament: {meta['events'][0]['tournament_id']}")

# Read game events
game_events = []
with open("tournament_20220101_120000/games/game_0001.jsonl", "r") as f:
    for line in f:
        event = json.loads(line.strip())
        game_events.append(event)

# Convert to DataFrame for analysis
df = pd.DataFrame(game_events)
moves = df[df['type'] == 'move']
print(f"Total moves: {len(moves)}")
```

### Replay Analysis Example

```python
# Extract all moves from a game
def extract_moves(game_file):
    moves = []
    with open(game_file, 'r') as f:
        for line in f:
            event = json.loads(line.strip())
            if event['type'] == 'move':
                moves.append(event)
    return moves

# Analyze response times by model
moves = extract_moves("tournament_20220101_120000/games/game_0001.jsonl")
response_times = {}

for move in moves:
    model = move['actor_model']
    if model not in response_times:
        response_times[model] = []
    response_times[model].append(move['response_time'])

# Calculate average response times
for model, times in response_times.items():
    avg_time = sum(times) / len(times)
    print(f"{model}: {avg_time:.2f}s average")
```

## Storage Requirements

### Size Estimates

Based on typical tournament configurations:

- **Small tournament** (10 games, 2 models): ~1-5 MB
- **Medium tournament** (100 games, 4 models): ~50-200 MB  
- **Large tournament** (1000 games, 6 models): ~500-2000 MB

Per-event overhead:
- Tournament events: ~500 bytes each
- Game events: ~300 bytes each
- Round events: ~200 bytes each
- Move events: ~1-2 KB each (includes full prompts/responses)
- Liar resolution: ~500 bytes each

### Compression

Enabling gzip compression typically reduces file sizes by 60-80%:

```python
# Uncompressed: 1000 MB
# Compressed: 200-400 MB
EventLogger(compression=True)
```

## Validation and Testing

### Schema Validation

All events are validated against strict schemas:

```python
from src.event_schema import validate_event, validate_tournament_directory

# Validate single event
event = {...}
try:
    validate_event(event)
    print("Event is valid")
except EventValidationError as e:
    print(f"Validation error: {e}")

# Validate entire tournament directory
errors = validate_tournament_directory("tournament_20220101_120000")
if not any(errors.values()):
    print("All files are valid")
```

### Smoke Test

Run the smoke test to verify the system works without API calls:

```bash
python smoke_test_raw_logging.py
```

This test:
- Creates mock tournament data
- Validates file structure
- Checks event schemas
- Measures storage usage
- Verifies complete replay capability

### Unit Tests

```bash
python -m pytest tests/test_event_logger.py -v
```

Tests cover:
- EventLogger functionality
- Concurrent logging safety
- Queue overflow handling
- Event validation
- Factory pattern behavior

## Integration Points

The raw logging system integrates with existing tournament code at these points:

### GameBatchRunner
- `run_tournament()`: Tournament start/end events
- `setup_game()`: Game start events
- `_process_game_results()`: Game end events

### LiarsDice (Synchronous)
- `play_round()`: Round start/end events
- `get_player_bid()`: Move events
- `handle_liar_call()`: Liar resolution events

### AsyncGameRunner (Asynchronous)
- `play_round_async()`: Round start/end events
- `get_player_bid_async()`: Move events
- `play_single_game_async()`: Game coordination

## Performance Characteristics

### Non-blocking Design

- Events are queued and written by background thread
- Main game logic never blocks on I/O
- Queue overflow triggers synchronous fallback with warning

### Latency Impact

- Per-event overhead: < 1ms (queue insertion)
- Background writing: Batched for efficiency
- Memory usage: Bounded by queue size

### Thread Safety

- All operations are thread-safe
- Multiple games can log concurrently
- Single writer thread per tournament

## Troubleshooting

### Common Issues

**Queue overflow warnings**
```
Event logger queue full, falling back to synchronous write
```
- Increase `queue_maxsize` parameter
- Check disk I/O performance
- Reduce logging frequency if needed

**Validation errors**
```
Line 42: Invalid parsed_action: unknown_action
```
- Check event schema compliance
- Verify data types match requirements
- Ensure all required fields are present

**Missing events**
- Verify `raw_logging_enabled = True`
- Check EventLoggerFactory state
- Ensure proper logger cleanup

### Debug Mode

Enable debug logging for detailed diagnostics:

```python
import logging
logging.getLogger("event_logger").setLevel(logging.DEBUG)
```

## Future Enhancements

### Planned Features

1. **Parquet indexing**: Convert JSONL to Parquet for efficient queries
2. **Compression options**: Support for different compression algorithms
3. **Streaming analysis**: Real-time event processing capabilities
4. **Replay engine**: Deterministic tournament replay from raw events

### API Extensions

```python
# Planned API additions
logger.export_to_parquet("analysis.parquet")
logger.create_replay_engine()
logger.stream_events(callback=process_event)
```

## References

- [Event Schema Documentation](src/event_schema.py)
- [EventLogger Implementation](src/event_logger.py)  
- [Unit Tests](tests/test_event_logger.py)
- [Smoke Test](smoke_test_raw_logging.py)