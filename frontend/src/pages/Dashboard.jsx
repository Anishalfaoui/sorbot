import { useState, useEffect, useCallback } from 'react';
import {
  getPredictions,
  getTradeStats,
  fetchPrediction,
  fetchPredictionsForAllSymbols,
  acceptPrediction,
  rejectPrediction,
  setMode as apiSetMode,
} from '../api';
import { subscribe } from '../websocket';
import PredictionCard from '../components/PredictionCard';

export default function Dashboard({ mode, setMode }) {
  const [selectedSymbol, setSelectedSymbol] = useState('ALL');
  const [signalFilter, setSignalFilter] = useState('ALL');
  const [statusFilter, setStatusFilter] = useState('ALL');
  const [predictions, setPredictions] = useState([]);
  const [stats, setStats] = useState({ totalTrades: 0, wins: 0, losses: 0, winRate: 0, totalPnl: 0 });
  const [loading, setLoading] = useState(true);
  const [fetching, setFetching] = useState(false);
  const [actionLoading, setActionLoading] = useState(false);

  const loadPredictions = useCallback(async () => {
    try {
      const predRes = await getPredictions().catch(() => ({ data: [] }));
      const list = Array.isArray(predRes.data) ? predRes.data : [];
      setPredictions(list);
      return list;
    } catch (e) {
      console.error('Failed to load predictions:', e);
      return [];
    }
  }, []);

  const loadData = useCallback(async () => {
    try {
      const [, statsRes] = await Promise.all([
        loadPredictions(),
        getTradeStats().catch(() => ({ data: {} })),
      ]);
      if (statsRes.data) setStats(statsRes.data);
    } catch (e) {
      console.error('Failed to load dashboard:', e);
    }
    setLoading(false);
  }, []);

  useEffect(() => {
    const bootstrap = async () => {
      await loadData();
      const initial = await loadPredictions();
      if (initial.length === 0) {
        setFetching(true);
        try {
          await fetchPredictionsForAllSymbols();
          await loadData();
        } catch (e) {
          console.error('Bootstrap fetch failed:', e);
        }
        setFetching(false);
      }
    };

    bootstrap();

    // Listen for real-time prediction updates
    subscribe('/topic/predictions', (data) => {
      if (data && data.id) {
        setPredictions((prev) => {
          const existing = prev.filter((p) => p.id !== data.id);
          return [data, ...existing].sort((a, b) => new Date(b.timestamp) - new Date(a.timestamp));
        });
      }
    });

    subscribe('/topic/trades', () => {
      getTradeStats().then((res) => setStats(res.data)).catch(() => {});
    });

    // Refresh stats periodically
    const interval = setInterval(() => {
      getTradeStats().then((res) => setStats(res.data)).catch(() => {});
    }, 30000);

    return () => clearInterval(interval);
  }, [loadData, loadPredictions]);

  const handleFetch = async () => {
    setFetching(true);
    try {
      if (selectedSymbol === 'ALL') {
        await fetchPredictionsForAllSymbols();
      } else {
        await fetchPrediction(selectedSymbol);
      }
      await loadData();
    } catch (e) {
      console.error('Fetch failed:', e);
    }
    setFetching(false);
  };

  const handleSymbolChange = (symbol) => {
    setSelectedSymbol(symbol);
    localStorage.setItem('selectedSymbol', symbol);
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

  const filteredPredictions = predictions.filter((p) => {
    const symbolOk = selectedSymbol === 'ALL' || String(p.symbol || '').toUpperCase().replace(/[\/\s-]/g, '') === selectedSymbol;
    const signalOk = signalFilter === 'ALL' || p.signal === signalFilter;
    const statusOk = statusFilter === 'ALL' || p.tradeStatus === statusFilter;
    return symbolOk && signalOk && statusOk;
  });

  const availableStatuses = Array.from(new Set(predictions.map((p) => p.tradeStatus).filter(Boolean)));

  return (
    <div>
      {/* Top bar */}
      <div className="top-bar">
        <h2>Dashboard</h2>
        <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
          <select
            value={selectedSymbol}
            onChange={(e) => handleSymbolChange(e.target.value)}
            style={{
              background: 'var(--panel-2)',
              border: '1px solid var(--border)',
              color: 'var(--text-primary)',
              borderRadius: 8,
              padding: '8px 10px',
              fontSize: 12,
              fontWeight: 600,
            }}
          >
            <option value="ALL">All Symbols</option>
            <option value="BTCUSD">BTC/USD</option>
            <option value="EURUSD">EUR/USD</option>
            <option value="XAUUSD">XAU/USD</option>
          </select>
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

      {/* Predictions with filters */}
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 16 }}>
        <h3 className="section-title" style={{ margin: 0 }}>Predictions</h3>
        <div style={{ display: 'flex', gap: 8, alignItems: 'center', flexWrap: 'wrap' }}>
          <select
            value={signalFilter}
            onChange={(e) => setSignalFilter(e.target.value)}
            style={{
              background: 'var(--panel-2)',
              border: '1px solid var(--border)',
              color: 'var(--text-primary)',
              borderRadius: 8,
              padding: '8px 10px',
              fontSize: 12,
              fontWeight: 600,
            }}
          >
            <option value="ALL">All Signals</option>
            <option value="LONG">LONG</option>
            <option value="SHORT">SHORT</option>
          </select>
          <select
            value={statusFilter}
            onChange={(e) => setStatusFilter(e.target.value)}
            style={{
              background: 'var(--panel-2)',
              border: '1px solid var(--border)',
              color: 'var(--text-primary)',
              borderRadius: 8,
              padding: '8px 10px',
              fontSize: 12,
              fontWeight: 600,
            }}
          >
            <option value="ALL">All Status</option>
            {availableStatuses.map((status) => (
              <option key={status} value={status}>{status}</option>
            ))}
          </select>
          <button className="btn btn-primary" onClick={handleFetch} disabled={fetching}>
            {fetching ? <span className="spinner" /> : '🔄'} Fetch {selectedSymbol === 'ALL' ? 'All' : selectedSymbol}
          </button>
        </div>
      </div>

      {filteredPredictions.length > 0 ? (
        filteredPredictions.map((prediction) => (
          <PredictionCard
            key={prediction.id}
            prediction={prediction}
            mode={mode}
            onAccept={handleAccept}
            onReject={handleReject}
            actionLoading={actionLoading}
            compact
          />
        ))
      ) : (
        <div className="prediction-panel">
          <div className="empty-state">
            <div className="empty-state-icon">📡</div>
            <p>No predictions match current filters. Try clearing filters or fetching new predictions.</p>
          </div>
        </div>
      )}
    </div>
  );
}
