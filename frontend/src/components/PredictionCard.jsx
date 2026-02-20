function formatPrice(price) {
  if (price == null) return '—';
  return Number(price).toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 });
}

function formatDate(dateStr) {
  if (!dateStr) return '—';
  const d = new Date(dateStr);
  return d.toLocaleString('en-US', {
    month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit', second: '2-digit',
  });
}

function getSignalClass(signal) {
  if (!signal) return '';
  if (signal === 'LONG') return 'long';
  if (signal === 'SHORT') return 'short';
  return 'no-trade';
}

function getStatusClass(status) {
  if (!status) return '';
  const s = status.toLowerCase().replace(/_/g, '-');
  if (['accepted', 'auto-executed'].includes(s)) return 'status-accepted';
  if (['rejected', 'failed', 'auto-failed', 'execution-failed'].includes(s)) return 'status-failed';
  if (s === 'pending' || s === 'auto-executing') return 'status-pending';
  return 'status-closed';
}

export default function PredictionCard({ prediction, mode, onAccept, onReject, actionLoading, compact }) {
  const p = prediction;
  const signalClass = getSignalClass(p.signal);
  const isPending = p.tradeStatus === 'PENDING';
  const showActions = mode === 'MANUAL' && isPending && p.signal !== 'NO_TRADE';

  return (
    <div className={`prediction-panel signal-${signalClass}`} style={compact ? { marginBottom: 16 } : {}}>
      {/* Header */}
      <div className="prediction-header">
        <div style={{ display: 'flex', alignItems: 'center', gap: 12, flexWrap: 'wrap' }}>
          <h3>{p.symbol || 'BTC-USD'}</h3>
          <span className={`signal-badge ${signalClass}`}>
            {p.signal === 'LONG' ? '▲' : p.signal === 'SHORT' ? '▼' : '⏸'}
            {' '}{p.signal || 'NO_TRADE'}
          </span>
          <span className={`status-badge ${getStatusClass(p.tradeStatus)}`}>
            {p.tradeStatus}
          </span>
        </div>
        <span style={{ fontSize: 12, color: 'var(--text-muted)' }}>
          {formatDate(p.timestamp)}
        </span>
      </div>

      {/* Main grid */}
      <div className="prediction-grid">
        <div className="pred-item">
          <span className="pred-item-label">Price</span>
          <span className="pred-item-value">${formatPrice(p.currentPrice)}</span>
        </div>
        <div className="pred-item">
          <span className="pred-item-label">Confidence</span>
          <span className="pred-item-value" style={{
            color: (p.confidencePct || 0) >= 70 ? 'var(--green)' :
                   (p.confidencePct || 0) >= 50 ? 'var(--yellow)' : 'var(--red)'
          }}>
            {p.confidencePct ? p.confidencePct.toFixed(1) : '—'}%
          </span>
        </div>
        <div className="pred-item">
          <span className="pred-item-label">Prob ▲ / ▼</span>
          <span className="pred-item-value">
            <span style={{ color: 'var(--green)' }}>{p.probabilityUp ? (p.probabilityUp * 100).toFixed(1) : '—'}%</span>
            {' / '}
            <span style={{ color: 'var(--red)' }}>{p.probabilityDown ? (p.probabilityDown * 100).toFixed(1) : '—'}%</span>
          </span>
        </div>

        {p.signal !== 'NO_TRADE' && (
          <>
            <div className="pred-item">
              <span className="pred-item-label">Stop Loss</span>
              <span className="pred-item-value" style={{ color: 'var(--red)' }}>
                ${formatPrice(p.slPrice)}
              </span>
            </div>
            <div className="pred-item">
              <span className="pred-item-label">Take Profit</span>
              <span className="pred-item-value" style={{ color: 'var(--green)' }}>
                ${formatPrice(p.tpPrice)}
              </span>
            </div>
            <div className="pred-item">
              <span className="pred-item-label">Risk:Reward</span>
              <span className="pred-item-value">
                {p.riskReward ? `1:${p.riskReward.toFixed(2)}` : '—'}
              </span>
            </div>
          </>
        )}

        <div className="pred-item">
          <span className="pred-item-label">ATR</span>
          <span className="pred-item-value">
            ${formatPrice(p.atr)} {p.atrPct ? `(${p.atrPct.toFixed(2)}%)` : ''}
          </span>
        </div>
      </div>

      {/* Indicators */}
      <div className="indicators-row">
        {p.trendDirection && (
          <span className="indicator-chip">
            <span className="label">Trend</span>
            <span style={{ color: p.trendDirection === 'bullish' ? 'var(--green)' : p.trendDirection === 'bearish' ? 'var(--red)' : 'var(--yellow)' }}>
              {p.trendDirection}
            </span>
          </span>
        )}
        {p.marketRegime && (
          <span className="indicator-chip">
            <span className="label">Regime</span> {p.marketRegime}
          </span>
        )}
        {p.rsi != null && (
          <span className="indicator-chip">
            <span className="label">RSI</span>
            <span style={{ color: p.rsi > 70 ? 'var(--red)' : p.rsi < 30 ? 'var(--green)' : 'var(--text-primary)' }}>
              {p.rsi.toFixed(1)}
            </span>
            {p.rsiZone && <span style={{ fontSize: 10, color: 'var(--text-muted)' }}>({p.rsiZone})</span>}
          </span>
        )}
        {p.adx != null && (
          <span className="indicator-chip">
            <span className="label">ADX</span> {p.adx.toFixed(1)}
            {p.adxInterpretation && <span style={{ fontSize: 10, color: 'var(--text-muted)' }}>({p.adxInterpretation})</span>}
          </span>
        )}
        {p.macdSignal && (
          <span className="indicator-chip">
            <span className="label">MACD</span>
            <span style={{ color: p.macdSignal === 'bullish' ? 'var(--green)' : p.macdSignal === 'bearish' ? 'var(--red)' : 'var(--text-primary)' }}>
              {p.macdSignal}
            </span>
          </span>
        )}
        {p.isSqueeze != null && (
          <span className="indicator-chip">
            <span className="label">Squeeze</span>
            {p.isSqueeze ? '🔴 Active' : '🟢 Off'}
          </span>
        )}
        {p.volumeRatio != null && (
          <span className="indicator-chip">
            <span className="label">Volume</span>
            <span style={{ color: p.volumeRatio > 1.5 ? 'var(--green)' : p.volumeRatio < 0.5 ? 'var(--red)' : 'var(--text-primary)' }}>
              {p.volumeRatio.toFixed(2)}x
            </span>
          </span>
        )}
        {p.htfOverallAlignment && (
          <span className="indicator-chip">
            <span className="label">HTF</span>
            <span style={{ color: p.htfOverallAlignment === 'aligned_long' ? 'var(--green)' : p.htfOverallAlignment === 'aligned_short' ? 'var(--red)' : 'var(--yellow)' }}>
              {p.htfOverallAlignment}
            </span>
          </span>
        )}
        {p.htf4hBias && (
          <span className="indicator-chip">
            <span className="label">4H</span> {p.htf4hBias}
          </span>
        )}
        {p.htf1dBias && (
          <span className="indicator-chip">
            <span className="label">1D</span> {p.htf1dBias}
          </span>
        )}
      </div>

      {/* Conclusion */}
      {p.conclusion && (
        <div className="conclusion-box">
          <strong>🧠 AI Conclusion:</strong>
          <br />
          {p.conclusion}
        </div>
      )}

      {/* Reject reason */}
      {p.rejectReason && (
        <div className="conclusion-box" style={{ borderColor: 'var(--red)', marginTop: 8 }}>
          <strong style={{ color: 'var(--red)' }}>⚠️ Reason:</strong> {p.rejectReason}
        </div>
      )}

      {/* Action buttons (manual mode) */}
      {showActions && (
        <div className="action-buttons">
          <button
            className="btn btn-accept"
            onClick={() => onAccept(p.id)}
            disabled={actionLoading}
          >
            {actionLoading ? <span className="spinner" /> : '✅'} Accept Trade
          </button>
          <button
            className="btn btn-reject"
            onClick={() => onReject(p.id)}
            disabled={actionLoading}
          >
            {actionLoading ? <span className="spinner" /> : '❌'} Reject
          </button>
        </div>
      )}
    </div>
  );
}

export { PredictionCard };
