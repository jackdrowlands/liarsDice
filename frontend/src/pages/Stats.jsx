import React, { useState, useEffect } from 'react';

function Stats() {
  const [statsData, setStatsData] = useState(null); // Expects { stats: { game_id: metrics_object } }
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');

  const fetchStats = async () => {
    setLoading(true);
    setError('');
    try {
      const response = await fetch('/api/stats');
      if (!response.ok) {
        const errData = await response.json();
        throw new Error(errData.detail || `Failed to fetch stats (${response.status})`);
      }
      const data = await response.json();
      setStatsData(data.stats || {}); // API returns { stats: {...} }
    } catch (err) {
      console.error("Error fetching stats:", err);
      setError(err.message || 'Could not load stats.');
      setStatsData(null);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchStats();
  }, []); // Fetch on initial mount

  return (
    <div>
      <h2>Game Statistics</h2>

      {error && <p className="error-message">Error: {error}</p>}

      <button onClick={fetchStats} disabled={loading} style={{ marginBottom: '15px' }}>
        {loading ? 'Refreshing...' : 'Refresh Stats'}
      </button>

      {loading ? (
        <p>Loading stats...</p>
      ) : statsData && Object.keys(statsData).length > 0 ? (
        <div>
          <p>Displaying raw metrics per game (backend aggregation needed for overall stats).</p>
          {Object.entries(statsData).map(([gameId, metrics]) => (
            <details key={gameId} style={{ marginBottom: '10px', border: '1px solid #ccc', padding: '10px' }}>
              <summary><strong>Game ID: {gameId.substring(0, 8)}...</strong></summary>
              <pre style={{ marginTop: '10px', background: '#f9f9f9', padding: '5px', overflowX: 'auto' }}>
                {JSON.stringify(metrics, null, 2)}
              </pre>
            </details>
          ))}
        </div>
      ) : (
        <p>No statistics data available.</p>
      )}
    </div>
  );
}

export default Stats;
