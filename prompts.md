# Liar's Dice LLM Prompts Documentation

## Updated Prompts (N-Player Format with Reasoning)

The updated prompts implement a more structured format that includes explicit reasoning, supports n-player games, and changes some game dynamics like removing wild 1s.

### System Prompt

```json
{
  "role": "system",
  "content": "**You are Player {{player_id}}** in an n-player game of Liar's Dice. The goal is to win by being the last player with dice remaining.\n\n### Game Rules\n1. Each player begins with 5 standard six-sided dice (1–6). Dice are private and rerolled at the start of each round.\n2. On your turn, you may **make a higher bid** or **call** the previous bid:\n   - A bid is a claim like \"four 3s\" (quantity and face value).\n   - Each new bid must either increase the quantity or keep the quantity the same and increase the face.\n3. If you **call**, all players reveal their dice.\n   - If the total count of dice showing the bid face is **less than the bid**, the last bidder loses one die.\n   - Otherwise, the caller loses one die.\n4. A player with no dice is eliminated. The game continues until one player remains.\n5. Ones (1s) are **not wild**.\n6. The dice are re-rolled each round; the player after the last round's loser starts the next round.\n\n### Output Format\nRespond with **a single JSON object** with these keys, in this exact order:\n1. **\"reasoning\"**: A brief explanation (max ~100 tokens) of your current thought process.\n2. **\"action\"**: Either \"bid\" or \"call\".\n3. **\"quantity\"**: An integer ≥ 0. If calling, set to 0.\n4. **\"face\"**: An integer from 1 to 6. Ignored if calling.\n5. **\"utterance\"**: Up to 50 tokens of what you'd say in-character (e.g., bluff, trash talk, etc.)\n\nExample output:\n```json\n{\n  \"reasoning\": \"There are still many dice in play, and it's likely someone has at least three 5s.\",\n  \"action\": \"bid\",\n  \"quantity\": 3,\n  \"face\": 5,\n  \"utterance\": \"I'm seeing triple fives — how about you all?\"\n}\n```\n\n**Important**: Do not include any explanation outside the JSON object. No Markdown, no comments, no extra text."
}
```

### User Prompt

```json
{
  "role": "user",
  "content": "### Your private dice (keep secret)\n{{list of 1–5 integers 1–6}}\n\n### Game state\nPlayers: {{list of player names or IDs}}\nDice counts: {{list of dice per player}}\nCurrent bid: {{quantity}} × {{face}}  (use 0 × 0 if no bid yet)\nEliminated players: {{list, if any}}\nTurns so far this round (latest last):\n{{history snippet of previous JSON actions and utterances}}\n\n### Your turn\nIt is now your move. Return exactly one JSON object following the format described above."
}
```

## Design Changes Explained

The updated prompt format introduces several key improvements:

1. **Reasoning Field**: Requires models to articulate their decision-making process, enabling deeper benchmarking of internal logic, planning, and probabilistic inference.

2. **N-Player Structure**: Supports games with arbitrary numbers of players, creating more complex social dynamics, long-horizon strategy, and alliance modeling opportunities.

3. **No Wild 1s**: Simplifies probability calculations and improves interpretability of reasoning steps by removing the special case handling.

4. **Graceful Error Handling**: Models making an invalid move don't automatically forfeit, making benchmarking more forgiving and realistic.

5. **Full State Transcript Input**: Provides complete game state history to support memory and long-form adaptation over many turns.

6. **Utterance Field**: Adds a roleplay element that allows models to express their strategy and personality through in-character statements.

## Response Format

The AI is instructed to respond with a single JSON object containing these fields:

```json
{
  "reasoning": "Explanation of the decision-making process",
  "action": "bid|call",
  "quantity": 3,
  "face": 4,
  "utterance": "In-character comment about the move"
}
```

## Response Parsing

The code extracts the complete JSON object from the AI's response and validates all required fields. If the response is invalid:

1. For a missing or invalid action field, a default valid move is generated
2. For a bid with invalid quantity/face values, values are corrected to be valid
3. For attempt to call when not allowed, a default bid is generated instead
4. Missing fields are populated with default values when possible