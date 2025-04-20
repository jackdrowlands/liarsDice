# Liar's Dice Game with LLM Integration

This project implements a fully-featured Liar's Dice game with integration for multiple LLM models through OpenRouter API.

## Code Structure

The codebase has been organized into the following modules:

- **main.py**: Entry point for the application, handles main menu and program flow
- **src/**
  - **player.py**: Contains the `Player` base class
  - **ai_player.py**: Contains the `AIPlayer` class for LLM-powered players
  - **liars_dice.py**: Core game logic in the `LiarsDice` class
  - **batch_runner.py**: Tournament functionality in the `GameBatchRunner` and `AsyncGameRunner` classes
  - **utils.py**: Utility functions for UI and common operations

## Features

- Play Liar's Dice with various game modes:
  - Human players only
  - Mixed human and AI players
  - AI vs AI (model vs model)
- Run tournaments to pit different LLM models against each other
- Support for 50+ different models through OpenRouter
- Detailed statistics and performance metrics
- Auto-play mode for AI-only games
- **NEW**: Asynchronous game execution for improved performance
- **NEW**: Individual game saving with complete move history and LLM reasoning

## Dependencies

- Python 3.7+
- Required packages:
  - `requests` - For standard HTTP requests
  - `httpx` - For asynchronous HTTP requests (optional, only needed for async mode)
  - `matplotlib` and `numpy` - For visualization of tournament results

## Getting Started

1. Install the required packages:
   ```
   pip install requests httpx matplotlib numpy
   ```

2. Get an API key from [OpenRouter](https://openrouter.ai)

3. Run the game:
   ```
   python main.py
   ```

## Running Tournaments

The tournament mode allows you to:
- Run multiple games between different AI models
- Collect statistics on model performance
- Generate detailed reports on win rates and playing styles
- Compare models across providers (OpenAI, Anthropic, Google, etc.)
- Enable asynchronous execution for better performance
- Save complete game data including all moves, reasoning, and model responses

### Game Data Collection

When enabled, the system will save detailed data for each individual game:
- Complete move history with timestamps
- Player reasoning for each move
- Player utterances during the game
- Model responses and prompts
- Any invalid responses or errors that occurred
- Complete metrics for model performance

Games are saved in the `games/` directory with filenames like `game_1_20250420_132217.json`.

### Game Runner Modes

#### Threaded Execution (Default)

Uses Python's `concurrent.futures.ThreadPoolExecutor` to run multiple games concurrently in separate threads. Each game runs sequentially but multiple games can progress in parallel. Includes a built-in 30-minute timeout for each game to prevent indefinite hanging.

#### Asynchronous Execution

Uses Python's `asyncio` and `httpx` to enable true asynchronous execution:

1. Runs multiple games concurrently as asyncio tasks
2. Uses a single shared HTTP client for all API requests
3. Makes non-blocking API calls for better performance
4. Efficiently handles high volume API requests
5. Built-in 30-minute timeout per game to prevent stuck games from blocking the tournament

To enable async mode, select "y" when prompted during setup:

```
Use asynchronous game execution? (y/n): y
```

## Implementation Details

### AsyncGameRunner

The async implementation follows these principles:

1. **Asynchronous HTTP Client**: Uses `httpx.AsyncClient` for non-blocking API calls.
2. **Task-based Concurrency**: Each game runs as a separate asyncio task.
3. **Sequential Move Execution**: While games run in parallel, moves within each game remain sequential.
4. **Resource Efficiency**: Single shared HTTP client for all API requests.
5. **Error Handling**: Comprehensive exception handling to prevent task failures.

### Comparison with Threaded Execution

| Feature | Threaded Execution | Async Execution |
|---------|-------------------|----------------|
| Concurrency Model | Thread-based | Event loop-based |
| I/O Handling | Blocking | Non-blocking |
| Resource Usage | Higher (one thread per game) | Lower (single event loop) |
| HTTP Requests | Multiple clients | Single shared client |
| Performance | Good for CPU-bound tasks | Excellent for I/O-bound tasks |
| Scalability | Limited by thread overhead | Better for high-concurrency |

## Player Anonymization

To prevent bias in evaluation, all AI models are assigned random human names instead of showing their actual model names. This anonymization process:

1. Assigns a consistent human name to each model throughout a tournament
2. Makes it impossible to identify models during gameplay
3. Avoids influencing game outcomes based on model reputation
4. Creates a blind evaluation environment

The system includes 20 distinct human names like "Alex Morgan", "Blake Taylor", etc., that are mapped to models using a hashing function to ensure consistency within a tournament.

The real model identities are preserved in the tournament results for later analysis.