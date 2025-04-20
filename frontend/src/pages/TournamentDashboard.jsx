import React, { useState, useEffect } from 'react';
import { Link, useNavigate } from 'react-router-dom'; // Assuming react-router-dom is used

function TournamentDashboard() {
  const [tournaments, setTournaments] = useState([]); // Stores { id: string, details: object }
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const navigate = useNavigate(); // For redirecting after creation

  // State for creating a new tournament
  const [showCreateForm, setShowCreateForm] = useState(false);
  const [availableModels, setAvailableModels] = useState([]);
  const [selectedModelIds, setSelectedModelIds] = useState([]);
  const [totalGames, setTotalGames] = useState(10);
  const [modelsPerGame, setModelsPerGame] = useState(2);
  const [creating, setCreating] = useState(false);
  const [createError, setCreateError] = useState('');

  const fetchTournaments = async () => {
    setLoading(true);
    setError('');
    try {
      // 1. Fetch list of tournament IDs
      const listRes = await fetch('/api/tournaments');
      if (!listRes.ok) {
        throw new Error('Failed to fetch tournament list');
      }
      const listData = await listRes.json();
      const tournamentIds = listData.tournaments || [];

      if (tournamentIds.length === 0) {
        setTournaments([]);
        return;
      }

      // 2. Fetch details for each tournament
      const tournamentDetailsPromises = tournamentIds.map(async (id) => {
        const detailRes = await fetch(`/api/tournaments/${id}`);
        if (!detailRes.ok) {
          console.error(`Failed to fetch details for tournament ${id}`);
          return { id, details: null, error: true }; // Mark error for this specific tournament
        }
        const detailData = await detailRes.json();
        return { id, details: detailData };
      });

      const results = await Promise.all(tournamentDetailsPromises);
      setTournaments(results);

    } catch (err) {
      console.error("Error fetching tournaments:", err);
      setError(err.message || 'Could not load tournaments.');
      setTournaments([]);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchTournaments();
    fetchAvailableModels(); // Fetch models on mount too
  }, []); // Fetch on initial mount

  // Fetch available AI models for the create form
  const fetchAvailableModels = async () => {
    try {
      const res = await fetch('/api/ai-models');
      if (!res.ok) throw new Error('Failed to fetch AI models');
      const data = await res.json();
      setAvailableModels(data.models || []);
    } catch (err) {
      console.error("Error fetching models for create form:", err);
      setCreateError('Could not load AI models for selection.');
      setAvailableModels([]);
    }
  };

  const handleModelSelectionChange = (event) => {
    const { options } = event.target;
    const selectedIds = [];
    for (let i = 0, l = options.length; i < l; i += 1) {
      if (options[i].selected) {
        selectedIds.push(options[i].value);
      }
    }
    setSelectedModelIds(selectedIds);
  };

  const handleCreateTournament = async (event) => {
    event.preventDefault();
    setCreating(true);
    setCreateError('');

    if (selectedModelIds.length < 2) {
      setCreateError('Please select at least 2 models.');
      setCreating(false);
      return;
    }
    if (modelsPerGame < 2 || modelsPerGame > selectedModelIds.length) {
      setCreateError(`Models per game must be between 2 and ${selectedModelIds.length}.`);
      setCreating(false);
      return;
    }
    if (totalGames < 1) {
      setCreateError('Total games must be at least 1.');
      setCreating(false);
      return;
    }

    // Find full model info for selected IDs
    const selectedModelsData = availableModels.filter(m => selectedModelIds.includes(m.id));

    try {
      const res = await fetch('/api/tournaments', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          models: selectedModelsData, // Send full model info
          total_games: parseInt(totalGames, 10),
          models_per_game: parseInt(modelsPerGame, 10),
        }),
      });

      if (!res.ok) {
        const errData = await res.json();
        throw new Error(errData.detail || 'Failed to create tournament');
      }

      const data = await res.json();
      setShowCreateForm(false); // Hide form on success
      setSelectedModelIds([]); // Reset form
      setTotalGames(10);
      setModelsPerGame(2);
      fetchTournaments(); // Refresh the list
      // Optionally navigate to the new tournament's detail page
      // navigate(`/tournaments/${data.tournament_id}`);

    } catch (err) {
      console.error("Error creating tournament:", err);
      setCreateError(err.message || 'Could not create tournament.');
    } finally {
      setCreating(false);
    }
  };


  return (
    <div>
      <h2>Tournament Dashboard</h2>

      {error && <p className="error-message">{error}</p>}

      {/* --- Create New Tournament Section --- */}
      <button onClick={() => setShowCreateForm(!showCreateForm)} style={{ marginBottom: '15px' }}>
        {showCreateForm ? 'Cancel Create' : 'Create New Tournament'}
      </button>

      {showCreateForm && (
        <div style={{ border: '1px solid #eee', padding: '15px', marginBottom: '20px' }}>
          <h3>Create New Tournament</h3>
          {createError && <p className="error-message">{createError}</p>}
          <form onSubmit={handleCreateTournament}>
            <div>
              <label htmlFor="models">Select Models (at least 2):</label><br />
              <select
                id="models"
                multiple
                value={selectedModelIds}
                onChange={handleModelSelectionChange}
                required
                size={Math.min(availableModels.length, 10)} // Show multiple lines
                style={{ minWidth: '300px', marginBottom: '10px' }}
              >
                {availableModels.length > 0 ? (
                  availableModels.map(model => (
                    <option key={model.id} value={model.id}>
                      {model.name || model.id} ({model.provider})
                    </option>
                  ))
                ) : (
                  <option disabled>Loading models...</option>
                )}
              </select>
            </div>
            <div>
              <label htmlFor="totalGames">Total Games:</label>
              <input
                id="totalGames"
                type="number"
                value={totalGames}
                onChange={(e) => setTotalGames(e.target.value)}
                min="1"
                required
                style={{ marginLeft: '5px', marginBottom: '10px' }}
              />
            </div>
            <div>
              <label htmlFor="modelsPerGame">Models per Game:</label>
              <input
                id="modelsPerGame"
                type="number"
                value={modelsPerGame}
                onChange={(e) => setModelsPerGame(e.target.value)}
                min="2"
                max={selectedModelIds.length || 2} // Max is number of selected models
                required
                style={{ marginLeft: '5px', marginBottom: '10px' }}
              />
            </div>
            <button type="submit" disabled={creating || availableModels.length === 0}>
              {creating ? 'Creating...' : 'Create Tournament'}
            </button>
          </form>
        </div>
      )}
      {/* --- End Create New Tournament Section --- */}


      <button onClick={fetchTournaments} disabled={loading} style={{ marginBottom: '15px' }}>
        {loading ? 'Refreshing...' : 'Refresh List'}
      </button>

      <h3>Existing Tournaments</h3>
      {loading ? (
        <p>Loading tournaments...</p>
      ) : tournaments.length > 0 ? (
        <table>
          <thead>
            <tr>
              <th>ID</th>
              <th>Status</th>
              <th>Total Games</th>
              <th>Models per Game</th>
              <th>Models Involved</th>
              <th>Actions</th>
            </tr>
          </thead>
          <tbody>
            {tournaments.map(({ id, details, error: fetchError }) => (
              <tr key={id}>
                <td>{id.substring(0, 8)}...</td>
                {fetchError ? (
                  <td colSpan="5" style={{ color: 'red' }}>Error loading details</td>
                ) : details ? (
                  <>
                    <td>{details.status}</td>
                    <td>{details.settings?.total_games ?? 'N/A'}</td>
                    <td>{details.settings?.models_per_game ?? 'N/A'}</td>
                    <td>
                      {details.settings?.models?.map(m => m.name || m.id).join(', ') ?? 'N/A'}
                    </td>
                    <td>
                      <Link to={`/tournaments/${id}`}>View Details</Link>
                    </td>
                  </>
                ) : (
                  <td colSpan="5">Loading details...</td>
                )}
              </tr>
            ))}
          </tbody>
        </table>
      ) : (
        <p>No tournaments found.</p>
      )}
    </div>
  );
}

export default TournamentDashboard;
