import { useState, useEffect, useCallback } from 'react';
import { getLatestPrediction, getTradeStats, fetchPrediction, acceptPrediction, rejectPrediction, setMode as apiSetMode } from '../api';
import { subscribe } from '../websocket';
import PredictionCard from '../components/PredictionCard';

export default function Dashboard({ mode, setMode }) {
  const [prediction, setPrediction] = useState(null);
  const [stats, setStats] = useState({ totalTrades: 0, wins: 0, losses: 0, winRate: 0, totalPnl: 0 });
  const [loading, setLoading] = useState(true);
  const [fetching, setFetching] = useState(false);
  const [actionLoading, setActionLoading] = useState(false);

  const loadData = useCallback(async () => {
    try {
      const [predRes, statsRes] = await Promise.all([
        getLatestPrediction().catch(() => ({ data: null })),
        getTradeStats().catch(() => ({ data: {} })),
      ]);
      if (predRes.data && predRes.data.id) setPrediction(predRes.data);
      if (statsRes.data) setStats(statsRes.data);
    } catch (e) {
      console.error('Failed to load dashboard:', e);
    }
    setLoading(false);
  }, []);

  useEffect(() => {
    loadData();

    // Listen for real-time prediction updates
    subscribe('/topic/predictions', (data) => {
      if (data && data.id) setPrediction(data);
    });

    subscribe('/topic/trades', () => {
      getTradeStats().then((res) => setStats(res.data)).catch(() => {});
    });

    // Refresh stats periodically
    const interval = setInterval(() => {
      getTradeStats().then((res) => setStats(res.data)).catch(() => {});
    }, 30000);

    return () => clearInterval(interval);
  }, [loadData]);

  const handleFetch = async () => {
    setFetching(true);
    try {
      const res = await fetchPrediction();
      if (res.data && res.data.id) setPrediction(res.data);
    } catch (e) {
      console.error('Fetch failed:', e);
    }
    setFetching(false);
  };

  const handleAccept = async (id) => {
    setActionLoading(true);
    try {
      await acceptPrediction(id);
      await loadData();
    } catch (e) {
      console.error('Accept failed:', e);
    }
    setActionLoading(false);
  };

  const handleReject = async (id) => {
    setActionLoading(true);
    try {
      await rejectPrediction(id);
      await loadData();
    } catch (e) {
      console.error('Reject failed:', e);
    }
    setActionLoading(false);
  };

  const handleModeChange = async (newMode) => {
    try {
      await apiSetMode(newMode);
      setMode(newMode);
    } catch (e) {
      console.error('Mode change failed:', e);
    }
  };

  if (loading) {
    return (
      <div className="loading-screen">
        <div className="spinner" style={{ width: 32, height: 32 }} />
        <span>Loading dashboard...</span>
      </div>
    );
  }

  return (
    <div>
      {/* Top bar */}
      <div className="top-bar">
        <h2>Dashboard</h2>
        <div className="mode-toggle">
          <button
            className={`mode-btn ${mode === 'MANUAL' ? 'active' : ''}`}
            onClick={() => handleModeChange('MANUAL')}
          >
            🎯 Manual
          </button>
          <button
            className={`mode-btn ${mode === 'AUTO' ? 'active' : ''}`}
            onClick={() => handleModeChange('AUTO')}
          >
            🤖 Auto
          </button>
        </div>
      </div>

      {/* Stats */}
      <div className="stats-grid">
        <div className="stat-card">
          <div className="stat-label">Total Trades</div>
          <div className="stat-value blue">{stats.totalTrades || 0}</div>
        </div>
        <div className="stat-card">
          <div className="stat-label">Win Rate</div>
          <div className="stat-value green">{stats.winRate || 0}%</div>
        </div>
        <div className="stat-card">
          <div className="stat-label">Wins / Losses</div>
          <div className="stat-value">
            <span className="green">{stats.wins || 0}</span>
            <span style={{ color: 'var(--text-muted)', margin: '0 4px' }}>/</span>
            <span className="red">{stats.losses || 0}</span>
          </div>
        </div>
        <div className="stat-card">
          <div className="stat-label">Total PnL</div>
          <div className={`stat-value ${(stats.totalPnl || 0) >= 0 ? 'green' : 'red'}`}>
            ${(stats.totalPnl || 0).toFixed(2)}
          </div>
        </div>
      </div>

      {/* Latest Prediction */}
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 16 }}>
        <h3 className="section-title" style={{ margin: 0 }}>Latest Prediction</h3>
        <button className="btn btn-primary" onClick={handleFetch} disabled={fetching}>
          {fetching ? <span className="spinner" /> : '🔄'} Fetch New
        </button>
      </div>

      {prediction ? (
        <PredictionCard
          prediction={prediction}
          mode={mode}
          onAccept={handleAccept}
          onReject={handleReject}
          actionLoading={actionLoading}
        />
      ) : (
        <div className="prediction-panel">
          <div className="empty-state">
            <div className="empty-state-icon">📡</div>
            <p>No predictions yet. Click "Fetch New" to get your first AI prediction.</p>
          </div>
        </div>
      )}
    </div>
  );
}
