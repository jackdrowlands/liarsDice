import React, { useState, useEffect, useRef } from 'react';
import { useParams, Link } from 'react-router-dom';

function TournamentDetail() {
  const { tournamentId } = useParams();
  const [tournamentData, setTournamentData] = useState(null);
  const [results, setResults] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');
  const [starting, setStarting] = useState(false);
  const pollIntervalRef = useRef(null);

  const fetchTournamentData = async (showLoading = false) => {
    if (showLoading) setLoading(true);
    setError('');
    try {
      const response = await fetch(`/api/tournaments/${tournamentId}`);
      if (!response.ok) {
        const errData = await response.json();
        throw new Error(errData.detail || `Failed to fetch tournament data (${response.status})`);
      }
      const data = await response.json();
      setTournamentData(data);

      // If completed, fetch results
      if (data.status === 'completed' && !results) {
        fetchResults();
      }

      // Stop polling if completed or errored
      if (data.status === 'completed' || data.status === 'error') { // Assuming an 'error' status might exist
        if (pollIntervalRef.current) {
          clearInterval(pollIntervalRef.current);
          pollIntervalRef.current = null;
        }
      }

    } catch (err) {
      console.error("Error fetching tournament data:", err);
      setError(err.message || 'Could not load tournament data.');
      if (pollIntervalRef.current) {
        clearInterval(pollIntervalRef.current);
        pollIntervalRef.current = null;
      }
    } finally {
      if (showLoading) setLoading(false);
    }
  };

  const fetchResults = async () => {
    try {
      const response = await fetch(`/api/tournaments/${tournamentId}/results`);
      if (!response.ok) {
        // Don't throw error if results just aren't ready yet
        if (response.status !== 404) {
          const errData = await response.json();
          throw new Error(errData.detail || `Failed to fetch results (${response.status})`);
        } else {
          console.log("Results not yet available.");
          setResults(null); // Ensure results are null if 404
        }
      } else {
        const data = await response.json();
        setResults(data);
      }
    } catch (err) {
      console.error("Error fetching results:", err);
      setError(prev => `${prev} (Error fetching results: ${err.message})`);
    }
  }

  useEffect(() => {
    fetchTournamentData(true); // Initial fetch

    // Set up polling only if status might be 'running' or 'pending' initially
    // We refine this based on the fetched status
    pollIntervalRef.current = setInterval(() => {
      setTournamentData(currentData => {
        if (currentData && (currentData.status === 'completed' || currentData.status === 'error')) {
          // Stop polling if already completed or errored
          if (pollIntervalRef.current) {
            clearInterval(pollIntervalRef.current);
            pollIntervalRef.current = null;
          }
          return currentData;
        }
        // Fetch data if potentially running
        fetchTournamentData(false);
        return currentData;
      });
    }, 8000); // Poll every 8 seconds

    // Cleanup interval
    return () => {
      if (pollIntervalRef.current) {
        clearInterval(pollIntervalRef.current);
        pollIntervalRef.current = null;
      }
    };
  }, [tournamentId]); // Re-run if ID changes

  const handleStartTournament = async () => {
    setStarting(true);
    setError('');
    try {
      const response = await fetch(`/api/tournaments/${tournamentId}/start`, { method: 'POST' });
      if (!response.ok) {
        const errData = await response.json();
        throw new Error(errData.detail || `Failed to start tournament (${response.status})`);
      }
      // Success - update state immediately and let polling take over
      setTournamentData(prev => ({ ...prev, status: 'running' }));
      // Ensure polling is active
      if (!pollIntervalRef.current) {
        pollIntervalRef.current = setInterval(() => fetchTournamentData(false), 8000);
      }

    } catch (err) {
      console.error("Error starting tournament:", err);
      setError(err.message || 'Could not start tournament.');
    } finally {
      setStarting(false);
    }
  };


  if (loading) {
    return <div>Loading tournament details for {tournamentId}...</div>;
  }

  if (error && !tournamentData) { // Show fatal error only if no data loaded at all
    return <div className="error-message">Error: {error}</div>;
  }

  if (!tournamentData) {
    return <div>Tournament not found or failed to load.</div>;
  }

  const { settings, status, leaderboard } = tournamentData;

  return (
    <div>
      <h2>Tournament: {tournamentId.substring(0, 8)}...</h2>
      {error && <p className="error-message">Note: {error}</p>} {/* Show non-fatal errors */}

      <p><strong>Status:</strong> {status}</p>

      {status === 'pending' && (
        <button onClick={handleStartTournament} disabled={starting}>
          {starting ? 'Starting...' : 'Start Tournament'}
        </button>
      )}

      <h3>Settings</h3>
      <p>Total Games: {settings?.total_games ?? 'N/A'}</p>
      <p>Models per Game: {settings?.models_per_game ?? 'N/A'}</p>
      <p>Models Involved: {settings?.models?.map(m => m.name || m.id).join(', ') ?? 'N/A'}</p>

      <h3>Leaderboard</h3>
      {leaderboard ? (
        <table>
          <thead>
            <tr>
              <th>Rank</th>
              <th>Model</th>
              <th>Win Rate</th>
              <th>Wins</th>
              <th>Games Played</th>
            </tr>
          </thead>
          <tbody>
            {Object.entries(leaderboard)
              .sort(([, a], [, b]) => (b.win_rate ?? 0) - (a.win_rate ?? 0) || (b.wins ?? 0) - (a.wins ?? 0)) // Sort by win rate, then wins
              .map(([modelId, stats], index) => (
                <tr key={modelId}>
                  <td>{index + 1}</td>
                  <td>{modelId}</td>
                  <td>{stats.win_rate !== undefined ? `${stats.win_rate.toFixed(1)}%` : 'N/A'}</td>
                  <td>{stats.wins ?? 'N/A'}</td>
                  <td>{stats.games_played ?? 'N/A'}</td>
                </tr>
              ))}
          </tbody>
        </table>
      ) : (
        <p>Leaderboard data not available yet.</p>
      )}

      {status === 'completed' && results && (
        <div>
          <h3>Final Results</h3>
          {/* TODO: Display more detailed results/stats from the 'results' object */}
          <p>Tournament finished. Check leaderboard above for final standings.</p>
          {/* Example: Displaying raw results for now */}
          <h4>Raw Results Data:</h4>
          <pre style={{ maxHeight: '400px', overflowY: 'scroll', border: '1px solid #eee', padding: '10px', background: '#f9f9f9' }}>
            {JSON.stringify(results, null, 2)}
          </pre>
        </div>
      )}
      {status === 'completed' && !results && (
        <p>Fetching final results...</p>
      )}

      <hr />
      <Link to="/tournaments">Back to Tournament List</Link>
    </div>
  );
}

export default TournamentDetail;
