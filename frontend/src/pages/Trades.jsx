import { useState, useEffect } from 'react';
import { getTrades, getTradeStats, closePosition } from '../api';
import { subscribe } from '../websocket';

function formatDate(dateStr) {
  if (!dateStr) return '—';
  const d = new Date(dateStr);
  return d.toLocaleString('en-US', { month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit' });
}

function formatPrice(price) {
  if (price == null) return '—';
  return Number(price).toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 });
}

export default function Trades() {
  const [trades, setTrades] = useState([]);
  const [stats, setStats] = useState({});
  const [loading, setLoading] = useState(true);
  const [closing, setClosing] = useState(false);

  const loadData = async () => {
    try {
      const [tradesRes, statsRes] = await Promise.all([
        getTrades(),
        getTradeStats(),
      ]);
      setTrades(tradesRes.data || []);
      setStats(statsRes.data || {});
    } catch (e) {
      console.error('Failed to load trades:', e);
    }
    setLoading(false);
  };

  useEffect(() => {
    loadData();

    subscribe('/topic/trades', () => {
      loadData();
    });

    const interval = setInterval(loadData, 30000);
    return () => clearInterval(interval);
  }, []);

  const handleClose = async () => {
    setClosing(true);
    try {
      await closePosition();
      await loadData();
    } catch (e) {
      console.error('Close failed:', e);
    }
    setClosing(false);
  };

  if (loading) {
    return (
      <div className="loading-screen">
        <div className="spinner" style={{ width: 32, height: 32 }} />
        <span>Loading trades...</span>
      </div>
    );
  }

  return (
    <div>
      <div className="top-bar">
        <h2>Trade History</h2>
        <button className="btn btn-secondary" onClick={handleClose} disabled={closing}>
          {closing ? <span className="spinner" /> : '🔒'} Close Position
        </button>
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
          <div className="stat-label">Wins</div>
          <div className="stat-value green">{stats.wins || 0}</div>
        </div>
        <div className="stat-card">
          <div className="stat-label">Losses</div>
          <div className="stat-value red">{stats.losses || 0}</div>
        </div>
        <div className="stat-card">
          <div className="stat-label">Total PnL</div>
          <div className={`stat-value ${(stats.totalPnl || 0) >= 0 ? 'green' : 'red'}`}>
            ${formatPrice(stats.totalPnl || 0)}
          </div>
        </div>
      </div>

      {/* Trades Table */}
      <div className="table-container">
        <div className="table-header">
          <h3>Recent Trades</h3>
          <span style={{ fontSize: 12, color: 'var(--text-muted)' }}>{trades.length} trades</span>
        </div>

        {trades.length === 0 ? (
          <div className="empty-state">
            <div className="empty-state-icon">📋</div>
            <p>No trades yet.</p>
          </div>
        ) : (
          <div style={{ overflowX: 'auto' }}>
            <table>
              <thead>
                <tr>
                  <th>Date</th>
                  <th>Side</th>
                  <th>Entry</th>
                  <th>SL</th>
                  <th>TP</th>
                  <th>Qty</th>
                  <th>R:R</th>
                  <th>Mode</th>
                  <th>Status</th>
                  <th>PnL</th>
                </tr>
              </thead>
              <tbody>
                {trades.map((trade) => (
                  <tr key={trade.id}>
                    <td>{formatDate(trade.executedAt)}</td>
                    <td className={trade.side === 'LONG' ? 'td-green' : 'td-red'}>
                      {trade.side === 'LONG' ? '▲' : '▼'} {trade.side}
                    </td>
                    <td>${formatPrice(trade.entryPrice)}</td>
                    <td className="td-red">${formatPrice(trade.slPrice)}</td>
                    <td className="td-green">${formatPrice(trade.tpPrice)}</td>
                    <td>{trade.quantity ? trade.quantity.toFixed(6) : '—'}</td>
                    <td>{trade.riskReward ? trade.riskReward.toFixed(2) : '—'}</td>
                    <td>
                      <span style={{ color: trade.mode === 'AUTO' ? 'var(--green)' : 'var(--yellow)', fontSize: 12, fontWeight: 600 }}>
                        {trade.mode}
                      </span>
                    </td>
                    <td>
                      <span className={`status-badge status-${(trade.status || '').toLowerCase()}`}>
                        {trade.status}
                      </span>
                    </td>
                    <td className={trade.pnl != null ? (trade.pnl >= 0 ? 'td-green' : 'td-red') : ''}>
                      {trade.pnl != null ? `$${formatPrice(trade.pnl)}` : '—'}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </div>
  );
}
