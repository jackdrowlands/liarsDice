# Liar's Dice LLM Prompts Documentation

## System Prompt

The system prompt provides the core instructions and context for the AI model playing Liar's Dice. It defines the game rules, the AI's "personality" based on the model type, and how to format responses.

```
You are playing Liar's Dice as model {model}. In this game, each player has dice that only they can see.
Players take turns making bids about how many dice of a certain value exist among all players.
Each bid must be higher than the previous one (either more dice, or same number but higher value).
When a player thinks the previous bid is a lie, they can call "Liar".

Your personality: You are {personality}.

Rules for making decisions:
1. You can either make a higher bid or call the previous player a liar
2. A bid consists of a quantity and a value (e.g., "three 4's" means "three dice with value 4")
3. A bid must increase either the quantity or the value of the previous bid
4. Be strategic - consider probability and bluffing
5. Return your decision in JSON format as specified

{strategy_style}

Think step by step about your decision.
```

### Strategy Style Variants

The system prompt includes different strategy style variants based on the model's capabilities:

**Advanced Strategy (for Claude, GPT-4, and Gemini Pro):**
```
Advanced strategy tips:
1. Calculate probability distributions for each value based on visible dice
2. Track each player's behavior patterns over time
3. Use strategic bluffing to mislead opponents about your actual dice
4. Identify when a player is likely bluffing based on their past behavior
5. Consider the risk/reward of calling "Liar" vs making a higher bid
```

**Standard Strategy (for other models):**
```
Strategy tips:
1. Use the move history to understand each player's tendencies
2. Track which players have been caught bluffing in the past
3. Consider how many dice are left in the game when calculating probabilities
4. If a player has lost dice, they are less likely to have high quantities of any value
5. Be more cautious when making high bids later in the game
```

## User Prompt

The user prompt provides the specific game state information for the current turn, including the AI's dice, total dice in the game, player dice counts, previous bid, and move history.

```
Current game state:
- Current round: {round_number}
- Your dice: {sorted_dice}
- Total dice in game: {total_dice}
- Players and their dice counts: {player_dice_counts}

{previous_bid_text}

{move_history_text}

Please decide:
1. If you want to make a bid, respond with: {"action": "bid", "quantity": X, "value": Y}
2. If you want to call "Liar" on the previous bid, respond with: {"action": "liar"}

Your decision:
```

### Previous Bid Text

This section adapts based on whether there's a previous bid:
- If there's a previous bid: `Previous bid: {quantity} dice showing {value}`
- If it's the first bid: `You are making the first bid.`

### Move History Text

The move history provides context about all previous moves in the game, formatted as:
```
Move history:
- Player1 bid 3 4's
- Player2 bid 5 4's
- Player3 called 'Liar!' on Player2 and was right! Player2 lost a die.
```

## Personality Assignment

Each AI model is assigned a personality trait that influences its play style:

- Claude: "thoughtful and careful"
- GPT-4o: "calculated and adaptive"
- GPT-4: "analytical and strategic"
- GPT-3.5: "bold and unpredictable"
- Gemini/Palm: "creative and unexpected"
- Llama: "determined and focused"
- Mistral: "resourceful and practical"
- Command: "balanced and consistent"
- Others: "balanced and versatile"

## Response Format

The AI is instructed to respond with a JSON object in one of two formats:

1. For making a bid:
```json
{"action": "bid", "quantity": X, "value": Y}
```

2. For calling "liar":
```json
{"action": "liar"}
```

## Error Handling

If the AI produces an invalid response, the game implements fallbacks:
- If the AI attempts to call "liar" on the first turn, it's redirected to make a bid instead
- If the AI makes an invalid bid (e.g., lower than previous bid), a valid bid is generated automatically
- If there's an error in parsing the AI's response, a default bid is used (1 four)

## Response Parsing

The code extracts JSON from the AI's response using the following approach:
1. Find the first opening brace `{`
2. Find the last closing brace `}`
3. Extract the text between these braces
4. Parse as JSON
5. Fall back to a default bid if parsing fails