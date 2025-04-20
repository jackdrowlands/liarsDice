import React, { useState, useEffect } from 'react';
import { Link } from 'react-router-dom';

function GameControl() {
  const [players, setPlayers] = useState([{ name: '', model: '' }]); // { name: string, model: string }
  const [availableModels, setAvailableModels] = useState([]); // { id: string, name: string, ... }
  const [activeGames, setActiveGames] = useState([]); // [game_id_string]
  const [newGameId, setNewGameId] = useState(null);
  const [error, setError] = useState('');
  const [loadingModels, setLoadingModels] = useState(false);
  const [loadingGames, setLoadingGames] = useState(false);
  const [startingGame, setStartingGame] = useState(false);

  // Fetch available AI models
  useEffect(() => {
    setLoadingModels(true);
    fetch('/api/ai-models')
      .then(res => {
        if (!res.ok) {
          throw new Error('Failed to fetch AI models');
        }
        return res.json();
      })
      .then(data => {
        setAvailableModels(data.models || []);
        setError('');
      })
      .catch(err => {
        console.error("Error fetching models:", err);
        setError(err.message || 'Could not load AI models.');
        setAvailableModels([]); // Ensure it's an array on error
      })
      .finally(() => setLoadingModels(false));
  }, []);

  // Fetch active games
  const fetchActiveGames = () => {
    setLoadingGames(true);
    fetch('/api/games')
      .then(res => {
        if (!res.ok) {
          throw new Error('Failed to fetch active games');
        }
        return res.json();
      })
      .then(data => {
        setActiveGames(data.games || []);
        setError('');
      })
      .catch(err => {
        console.error("Error fetching games:", err);
        setError(err.message || 'Could not load active games.');
        setActiveGames([]); // Ensure it's an array on error
      })
      .finally(() => setLoadingGames(false));
  };

  useEffect(() => {
    fetchActiveGames();
  }, []); // Fetch on initial mount

  const handlePlayerNameChange = (index, name) => {
    const updatedPlayers = [...players];
    updatedPlayers[index].name = name;
    setPlayers(updatedPlayers);
  };

  const handlePlayerModelChange = (index, modelId) => {
    const updatedPlayers = [...players];
    updatedPlayers[index].model = modelId; // Store model ID, empty string means human
    setPlayers(updatedPlayers);
  };

  const addPlayer = () => {
    if (players.length < 6) { // Limit players (optional)
      setPlayers([...players, { name: '', model: '' }]);
    }
  };

  const removePlayer = (index) => {
    if (players.length > 1) { // Need at least one player
      const updatedPlayers = players.filter((_, i) => i !== index);
      setPlayers(updatedPlayers);
    }
  };

  const handleStartGame = () => {
    setError('');
    setNewGameId(null);

    // Validate player names
    if (players.some(p => !p.name.trim())) {
      setError('All players must have a name.');
      return;
    }
    const names = players.map(p => p.name.trim());
    if (new Set(names).size !== names.length) {
      setError('Player names must be unique.');
      return;
    }
    if (players.length < 2) {
      setError('You need at least 2 players to start a game.');
      return;
    }

    const humanPlayers = players.filter(p => !p.model).map(p => p.name.trim());
    const aiPlayers = players.filter(p => p.model).reduce((acc, p) => {
      acc[p.name.trim()] = p.model; // Map AI player name to model ID
      return acc;
    }, {});

    setStartingGame(true);
    fetch('/api/start-game', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ players: humanPlayers, ai_models: aiPlayers }),
    })
      .then(res => {
        if (!res.ok) {
          return res.json().then(errData => { throw new Error(errData.detail || 'Failed to start game'); });
        }
        return res.json();
      })
      .then(data => {
        setNewGameId(data.game_id);
        fetchActiveGames(); // Refresh active games list
        // Optionally reset player setup
        // setPlayers([{ name: '', model: '' }]);
      })
      .catch(err => {
        console.error("Error starting game:", err);
        setError(err.message || 'Could not start game.');
      })
      .finally(() => setStartingGame(false));
  };

  return (
    <div>
      <h2>Game Control</h2>

      {error && <p className="error-message">{error}</p>}

      <h3>Setup New Game</h3>
      {players.map((player, index) => (
        <div key={index} style={{ marginBottom: '10px', padding: '10px', border: '1px solid #ccc' }}>
          <label>
            Player {index + 1} Name:
            <input
              type="text"
              value={player.name}
              onChange={(e) => handlePlayerNameChange(index, e.target.value)}
              required
              style={{ marginLeft: '5px', marginRight: '10px' }}
            />
          </label>
          <label>
            Type:
            <select
              value={player.model}
              onChange={(e) => handlePlayerModelChange(index, e.target.value)}
              style={{ marginLeft: '5px', marginRight: '10px' }}
              disabled={loadingModels}
            >
              <option value="">Human</option>
              {loadingModels ? (
                <option disabled>Loading models...</option>
              ) : (
                availableModels.map(model => (
                  <option key={model.id} value={model.id}>
                    AI: {model.name || model.id} ({model.provider})
                  </option>
                ))
              )}
            </select>
          </label>
          {players.length > 1 && (
            <button onClick={() => removePlayer(index)} type="button">Remove</button>
          )}
        </div>
      ))}
      <button onClick={addPlayer} type="button" disabled={players.length >= 6}>Add Player</button>
      <button
        onClick={handleStartGame}
        type="button"
        disabled={startingGame || players.length < 2}
        style={{ marginLeft: '10px' }}
      >
        {startingGame ? 'Starting...' : 'Start Game'}
      </button>

      {newGameId && (
        <p style={{ marginTop: '15px', color: 'green' }}>
          New game started! ID: {newGameId}
          <Link to={`/viewer/${newGameId}`} style={{ marginLeft: '10px' }}>View Game</Link>
        </p>
      )}

      <hr style={{ margin: '20px 0' }} />

      <h3>Active Games</h3>
      {loadingGames ? (
        <p>Loading active games...</p>
      ) : activeGames.length > 0 ? (
        <ul>
          {activeGames.map(gameId => (
            <li key={gameId}>
              <Link to={`/viewer/${gameId}`}>{gameId}</Link>
            </li>
          ))}
        </ul>
      ) : (
        <p>No active games found.</p>
      )}
      <button onClick={fetchActiveGames} disabled={loadingGames}>Refresh Active Games</button>
    </div>
  );
}

export default GameControl;
