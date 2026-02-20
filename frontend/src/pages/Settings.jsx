import { useState, useEffect } from 'react';
import { getSettings, setMode as apiSetMode, trainModel, getModelInfo, healthCheck } from '../api';

export default function Settings({ mode, setMode }) {
  const [settings, setSettings] = useState(null);
  const [modelInfo, setModelInfo] = useState(null);
  const [health, setHealth] = useState(null);
  const [training, setTraining] = useState(false);
  const [trainResult, setTrainResult] = useState(null);

  useEffect(() => {
    loadAll();
  }, []);

  const loadAll = async () => {
    try {
      const [settingsRes, modelRes, healthRes] = await Promise.all([
        getSettings().catch(() => ({ data: null })),
        getModelInfo().catch(() => ({ data: null })),
        healthCheck().catch(() => ({ data: null })),
      ]);
      if (settingsRes.data) setSettings(settingsRes.data);
      if (modelRes.data) setModelInfo(modelRes.data);
      if (healthRes.data) setHealth(healthRes.data);
    } catch (e) {
      console.error('Settings load error:', e);
    }
  };

  const handleModeChange = async (newMode) => {
    try {
      const res = await apiSetMode(newMode);
      setMode(newMode);
      setSettings(res.data);
    } catch (e) {
      console.error('Mode change failed:', e);
    }
  };

  const handleTrain = async () => {
    setTraining(true);
    setTrainResult(null);
    try {
      const res = await trainModel();
      setTrainResult(res.data);
      // Reload model info
      const modelRes = await getModelInfo();
      setModelInfo(modelRes.data);
    } catch (e) {
      setTrainResult({ error: e.message });
    }
    setTraining(false);
  };

  return (
    <div>
      <div className="top-bar">
        <h2>Settings</h2>
      </div>

      {/* Trading Mode */}
      <div className="settings-section">
        <h3>🎯 Trading Mode</h3>
        <div className="setting-row">
          <div className="setting-info">
            <h4>Current Mode</h4>
            <p>
              {mode === 'AUTO'
                ? 'Trades are executed automatically when the AI generates a signal.'
                : 'You will review each prediction and manually accept or reject trades.'}
            </p>
          </div>
          <div className="mode-toggle">
            <button
              className={`mode-btn ${mode === 'MANUAL' ? 'active' : ''}`}
              onClick={() => handleModeChange('MANUAL')}
            >
              Manual
            </button>
            <button
              className={`mode-btn ${mode === 'AUTO' ? 'active' : ''}`}
              onClick={() => handleModeChange('AUTO')}
            >
              Auto
            </button>
          </div>
        </div>
      </div>

      {/* AI Model */}
      <div className="settings-section">
        <h3>🧠 AI Model</h3>
        {modelInfo ? (
          <>
            <div className="prediction-grid">
              {modelInfo.model_loaded != null && (
                <div className="pred-item">
                  <span className="pred-item-label">Status</span>
                  <span className="pred-item-value" style={{ color: modelInfo.model_loaded ? 'var(--green)' : 'var(--red)' }}>
                    {modelInfo.model_loaded ? '✅ Loaded' : '❌ Not Loaded'}
                  </span>
                </div>
              )}
              {modelInfo.accuracy && (
                <div className="pred-item">
                  <span className="pred-item-label">Accuracy</span>
                  <span className="pred-item-value">{(modelInfo.accuracy * 100).toFixed(1)}%</span>
                </div>
              )}
              {modelInfo.auc && (
                <div className="pred-item">
                  <span className="pred-item-label">AUC</span>
                  <span className="pred-item-value">{modelInfo.auc.toFixed(3)}</span>
                </div>
              )}
              {modelInfo.f1 && (
                <div className="pred-item">
                  <span className="pred-item-label">F1 Score</span>
                  <span className="pred-item-value">{modelInfo.f1.toFixed(3)}</span>
                </div>
              )}
              {modelInfo.n_features && (
                <div className="pred-item">
                  <span className="pred-item-label">Features</span>
                  <span className="pred-item-value">{modelInfo.n_features}</span>
                </div>
              )}
              {modelInfo.train_rows && (
                <div className="pred-item">
                  <span className="pred-item-label">Training Rows</span>
                  <span className="pred-item-value">{modelInfo.train_rows.toLocaleString()}</span>
                </div>
              )}
            </div>

            <div style={{ marginTop: 16 }}>
              <button className="btn btn-primary" onClick={handleTrain} disabled={training}>
                {training ? (
                  <>
                    <span className="spinner" /> Training...
                  </>
                ) : (
                  '🔄 Retrain Model'
                )}
              </button>
            </div>

            {trainResult && (
              <div className="conclusion-box" style={{ marginTop: 12 }}>
                {trainResult.error ? (
                  <span style={{ color: 'var(--red)' }}>❌ {trainResult.error}</span>
                ) : (
                  <span style={{ color: 'var(--green)' }}>
                    ✅ Model retrained! Accuracy: {((trainResult.accuracy || 0) * 100).toFixed(1)}%,
                    AUC: {(trainResult.auc || 0).toFixed(3)}
                  </span>
                )}
              </div>
            )}
          </>
        ) : (
          <p style={{ color: 'var(--text-muted)' }}>Loading model info...</p>
        )}
      </div>

      {/* System Health */}
      <div className="settings-section">
        <h3>🏥 System Health</h3>
        {health ? (
          <div className="prediction-grid">
            <div className="pred-item">
              <span className="pred-item-label">Backend</span>
              <span className="pred-item-value" style={{ color: 'var(--green)' }}>
                ✅ {health.backend || 'Running'}
              </span>
            </div>
            <div className="pred-item">
              <span className="pred-item-label">AI Engine</span>
              <span className="pred-item-value" style={{
                color: health.aiEngine?.status === 'unreachable' ? 'var(--red)' : 'var(--green)'
              }}>
                {health.aiEngine?.status === 'unreachable' ? '❌ Offline' : '✅ Online'}
              </span>
            </div>
          </div>
        ) : (
          <p style={{ color: 'var(--text-muted)' }}>Loading health info...</p>
        )}
      </div>

      {/* About */}
      <div className="settings-section">
        <h3>ℹ️ About</h3>
        <div style={{ fontSize: 13, color: 'var(--text-secondary)', lineHeight: 1.8 }}>
          <p><strong>Sorbot AI Trading Engine v3.0</strong></p>
          <p>BTC/USD trading with XGBoost ML model, 122 features, multi-timeframe analysis.</p>
          <p>Architecture: React → Spring Boot → Python AI Engine → Binance Futures</p>
        </div>
      </div>
    </div>
  );
}
