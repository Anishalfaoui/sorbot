import { useState, useEffect } from 'react';
import { getPredictions, fetchPrediction, acceptPrediction, rejectPrediction } from '../api';
import { subscribe } from '../websocket';
import PredictionCard from '../components/PredictionCard';

export default function Predictions({ mode }) {
  const [selectedSymbol, setSelectedSymbol] = useState(localStorage.getItem('selectedSymbol') || 'BTCUSD');
  const [predictions, setPredictions] = useState([]);
  const [loading, setLoading] = useState(true);
  const [fetching, setFetching] = useState(false);
  const [actionLoading, setActionLoading] = useState(false);

  const loadPredictions = async () => {
    try {
      const res = await getPredictions();
      setPredictions(res.data || []);
    } catch (e) {
      console.error('Failed to load predictions:', e);
    }
    setLoading(false);
  };

  useEffect(() => {
    loadPredictions();

    subscribe('/topic/predictions', () => {
      loadPredictions();
    });
  }, []);

  const handleFetch = async () => {
    setFetching(true);
    try {
      await fetchPrediction(selectedSymbol);
      await loadPredictions();
    } catch (e) {
      console.error('Fetch failed:', e);
    }
    setFetching(false);
  };

  const handleAccept = async (id) => {
    setActionLoading(true);
    try {
      await acceptPrediction(id);
      await loadPredictions();
    } catch (e) {
      console.error('Accept failed:', e);
    }
    setActionLoading(false);
  };

  const handleReject = async (id) => {
    setActionLoading(true);
    try {
      await rejectPrediction(id);
      await loadPredictions();
    } catch (e) {
      console.error('Reject failed:', e);
    }
    setActionLoading(false);
  };

  if (loading) {
    return (
      <div className="loading-screen">
        <div className="spinner" style={{ width: 32, height: 32 }} />
        <span>Loading predictions...</span>
      </div>
    );
  }

  return (
    <div>
      <div className="top-bar">
        <h2>Predictions History</h2>
        <div style={{ display: 'flex', gap: 10, alignItems: 'center' }}>
          <select
            value={selectedSymbol}
            onChange={(e) => {
              const v = e.target.value;
              setSelectedSymbol(v);
              localStorage.setItem('selectedSymbol', v);
            }}
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
            <option value="BTCUSD">BTC/USD</option>
            <option value="EURUSD">EUR/USD</option>
            <option value="XAUUSD">XAU/USD</option>
          </select>
          <button className="btn btn-primary" onClick={handleFetch} disabled={fetching}>
            {fetching ? <span className="spinner" /> : '🔄'} Fetch New
          </button>
        </div>
      </div>

      {predictions.length === 0 ? (
        <div className="prediction-panel">
          <div className="empty-state">
            <div className="empty-state-icon">📭</div>
            <p>No predictions yet. Fetch a new one to get started.</p>
          </div>
        </div>
      ) : (
        predictions.map((pred) => (
          <PredictionCard
            key={pred.id}
            prediction={pred}
            mode={mode}
            onAccept={handleAccept}
            onReject={handleReject}
            actionLoading={actionLoading}
            compact={true}
          />
        ))
      )}
    </div>
  );
}
