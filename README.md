# Liar's Dice Game with LLM Integration

This project implements a fully-featured Liar's Dice game with integration for multiple LLM models through OpenRouter API.

## Code Structure

The codebase has been organized into the following modules:

- **main.py**: Entry point for the application, handles main menu and program flow
- **src/**
  - **player.py**: Contains the `Player` base class
  - **ai_player.py**: Contains the `AIPlayer` class for LLM-powered players
  - **liars_dice.py**: Core game logic in the `LiarsDice` class
  - **batch_runner.py**: Tournament functionality in the `GameBatchRunner` class
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

## Dependencies

- Python 3.6+
- `requests` library for OpenRouter API integration

## Getting Started

1. Install the required packages:
   ```
   pip install requests
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

## LLM Personalities

Each AI model is assigned a unique personality trait that influences its playing style:

- Claude: "thoughtful and careful"
- GPT-4o: "calculated and adaptive"
- GPT-4: "analytical and strategic"
- GPT-3.5: "bold and unpredictable"
- Gemini/Palm: "creative and unexpected"
- Llama: "determined and focused"
- Mistral: "resourceful and practical"
- Command: "balanced and consistent"