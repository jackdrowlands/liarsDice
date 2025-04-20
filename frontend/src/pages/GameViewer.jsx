import React, { useState, useEffect, useRef } from 'react';
import { useParams } from 'react-router-dom';

function GameViewer() {
  const { gameId } = useParams(); // Get gameId from URL
  const [gameState, setGameState] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');
  const pollIntervalRef = useRef(null);
  const [isSubmittingMove, setIsSubmittingMove] = useState(false);
  const [bidQuantity, setBidQuantity] = useState(1);
  const [bidValue, setBidValue] = useState(1);

  // --- Assume the user is controlling player named "Player 1" ---
  // TODO: Make this dynamic (e.g., pass player name as prop or use context)
  const HUMAN_PLAYER_NAME = "Player 1";
  // ---

  const fetchGameState = async (showLoading = false) => {
    // Don't set loading to true on subsequent polls
    // setLoading(true);
    setError('');
    try {
      const response = await fetch(`/api/status/${gameId}`);
      if (!response.ok) {
        const errData = await response.json();
        throw new Error(errData.detail || `Failed to fetch game state (${response.status})`);
      }
      const data = await response.json();
      setGameState(data.state);
    } catch (err) {
      console.error("Error fetching game state:", err);
      setError(err.message || 'Could not load game state.');
      // Stop polling on error
      if (pollIntervalRef.current) {
        clearInterval(pollIntervalRef.current);
        pollIntervalRef.current = null;
      }
    } finally {
      // Only set loading false on initial load
      if (showLoading || loading) setLoading(false);
    }
  };

  useEffect(() => {
    setLoading(true); // Set loading true on initial mount or gameId change
    fetchGameState(true); // Initial fetch, show loading

    // Set up polling
    pollIntervalRef.current = setInterval(() => {
      // Check if game is over before polling again
      setGameState(currentState => {
        if (currentState && currentState.game_over) {
          if (pollIntervalRef.current) {
            clearInterval(pollIntervalRef.current);
            pollIntervalRef.current = null;
          }
          return currentState; // Don't change state if game is over
        }
        // If game not over, fetch state without showing loading indicator
        fetchGameState(false);
        return currentState; // Return current state while fetch happens
      });
    }, 5000); // Poll every 5 seconds

    // Cleanup interval on component unmount or gameId change
    return () => {
      if (pollIntervalRef.current) {
        clearInterval(pollIntervalRef.current);
        pollIntervalRef.current = null;
      }
    };
  }, [gameId]); // Re-run effect if gameId changes

  if (loading) {
    return <div>Loading game state for {gameId}...</div>;
  }

  if (error) {
    return <div className="error-message">Error: {error}</div>;
  }

  if (!gameState) {
    return <div>Game not found or failed to load.</div>;
  }

  // Helper to display bids
  const formatBid = (bid) => {
    return bid ? `${bid[0]} x ${bid[1]}` : 'None';
  };

  return (
    <div>
      <h2>Game Viewer: {gameId.substring(0, 8)}...</h2>

      {gameState.game_over && (
        <h3 style={{ color: 'green' }}>Game Over! Winner: {gameState.winner}</h3>
      )}

      <p><strong>Round:</strong> {gameState.round_number}</p>
      <p><strong>Current Turn:</strong> {gameState.current_player} {!gameState.game_over && '(Thinking...)'}</p>
      <p><strong>Last Bid:</strong> {formatBid(gameState.last_bid)}</p>

      <h3>Players</h3>
      <ul>
        {gameState.players.map(player => (
          <li key={player.name}>
            {player.name}: {player.dice_count} dice
            {/* TODO: Display player's own dice if applicable */}
          </li>
        ))}
      </ul>

      {/* --- Human Action Controls --- */}
      {gameState && !gameState.game_over && gameState.current_player === HUMAN_PLAYER_NAME && (
        <div style={{ border: '1px solid blue', padding: '15px', marginTop: '15px' }}>
          <h4>Your Turn ({HUMAN_PLAYER_NAME})</h4>
          {error && <p className="error-message">{error}</p>} {/* Show move errors here */}
          <div>
            <label>
              Quantity:
              <input
                type="number"
                value={bidQuantity}
                onChange={(e) => setBidQuantity(parseInt(e.target.value, 10) || 1)}
                min="1"
                style={{ width: '60px', marginLeft: '5px', marginRight: '10px' }}
                disabled={isSubmittingMove}
              />
            </label>
            <label>
              Value:
              <select
                value={bidValue}
                onChange={(e) => setBidValue(parseInt(e.target.value, 10))}
                style={{ marginLeft: '5px', marginRight: '10px' }}
                disabled={isSubmittingMove}
              >
                {[1, 2, 3, 4, 5, 6].map(v => <option key={v} value={v}>{v}</option>)}
              </select>
            </label>
            <button
              onClick={() => handleMakeMove('bid')}
              disabled={isSubmittingMove}
            >
              Make Bid
            </button>
          </div>
          <div style={{ marginTop: '10px' }}>
            <button
              onClick={() => handleMakeMove('liar')}
              disabled={isSubmittingMove || !gameState.last_bid} // Can't call liar on first turn
            >
              Call Liar!
            </button>
          </div>
        </div>
      )}
      {/* --- End Human Action Controls --- */}

      <h3>Move History (Round {gameState.round_number})</h3>
      <ul style={{ maxHeight: '300px', overflowY: 'scroll', border: '1px solid #ccc', padding: '10px' }}>
        {gameState.move_history
          .filter(move => move.round === gameState.round_number) // Show only current round history for now
          .map((move, index) => (
            <li key={index}>
              <strong>{move.player}:</strong> {move.action}
              {move.action === 'bid' && ` (${move.quantity} x ${move.value})`}
              {/* TODO: Display reasoning/utterance if available */}
            </li>
          ))}
        {gameState.move_history.filter(move => move.round === gameState.round_number).length === 0 && (
          <li>No moves yet this round.</li>
        )}
      </ul>

      {/* Optionally show full history */}
      {/*
      <h3>Full Game History</h3>
      <pre style={{ maxHeight: '400px', overflowY: 'scroll', border: '1px solid #eee', padding: '10px', background: '#f9f9f9' }}>
        {JSON.stringify(gameState.move_history, null, 2)}
      </pre>
      */}
    </div>
  );

  // Function to handle submitting a move
  const handleMakeMove = async (actionType) => {
    setError(''); // Clear previous errors
    setIsSubmittingMove(true);

    let body = { game_id: gameId, action: actionType };
    if (actionType === 'bid') {
      // Basic validation (more robust validation needed based on game rules)
      const currentBid = [bidQuantity, bidValue];
      const lastBid = gameState.last_bid;
      if (lastBid) {
        if (currentBid[0] < lastBid[0] || (currentBid[0] === lastBid[0] && currentBid[1] <= lastBid[1])) {
          // Allow bidding same quantity but higher value
          if (!(currentBid[0] === lastBid[0] && currentBid[1] > lastBid[1])) {
            setError('Bid must be higher than the last bid.');
            setIsSubmittingMove(false);
            return;
          }
        }
      }
      if (currentBid[0] <= 0) {
        setError('Bid quantity must be positive.');
        setIsSubmittingMove(false);
        return;
      }

      body.quantity = bidQuantity;
      body.value = bidValue;
    }

    try {
      const response = await fetch('/api/make-move', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
      });

      if (!response.ok) {
        const errData = await response.json();
        throw new Error(errData.detail || `Failed to make move (${response.status})`);
      }

      // Success - fetch the new state immediately
      await fetchGameState(false); // Fetch without loading indicator

    } catch (err) {
      console.error("Error making move:", err);
      setError(err.message || 'Could not make move.');
    } finally {
      setIsSubmittingMove(false);
    }
  };
}

export default GameViewer;
