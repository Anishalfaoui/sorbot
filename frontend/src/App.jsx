import { useState, useEffect } from 'react';
import { BrowserRouter as Router, Routes, Route, NavLink, useLocation } from 'react-router-dom';
import Dashboard from './pages/Dashboard';
import Predictions from './pages/Predictions';
import Trades from './pages/Trades';
import Settings from './pages/Settings';
import { getSettings } from './api';
import { connectWebSocket, subscribe, isConnected } from './websocket';

function Sidebar({ mode, wsConnected }) {
  return (
    <aside className="sidebar">
      <div className="sidebar-logo">
        <h1>SORBOT</h1>
        <span>AI Trading Engine</span>
      </div>
      <nav className="sidebar-nav">
        <NavLink to="/" className={({ isActive }) => `nav-link ${isActive ? 'active' : ''}`} end>
          <span className="nav-icon">📊</span> Dashboard
        </NavLink>
        <NavLink to="/predictions" className={({ isActive }) => `nav-link ${isActive ? 'active' : ''}`}>
          <span className="nav-icon">🤖</span> Predictions
        </NavLink>
        <NavLink to="/trades" className={({ isActive }) => `nav-link ${isActive ? 'active' : ''}`}>
          <span className="nav-icon">💰</span> Trades
        </NavLink>
        <NavLink to="/settings" className={({ isActive }) => `nav-link ${isActive ? 'active' : ''}`}>
          <span className="nav-icon">⚙️</span> Settings
        </NavLink>
      </nav>
      <div style={{ padding: '16px 20px', borderTop: '1px solid var(--border)' }}>
        <div style={{ fontSize: 12, color: 'var(--text-muted)', marginBottom: 6 }}>
          <span className={`connection-dot ${wsConnected ? 'connected' : 'disconnected'}`} />
          {wsConnected ? 'Live' : 'Offline'}
        </div>
        <div style={{ fontSize: 12 }}>
          Mode: <strong style={{ color: mode === 'AUTO' ? 'var(--green)' : 'var(--yellow)' }}>{mode}</strong>
        </div>
      </div>
    </aside>
  );
}

export default function App() {
  const [mode, setMode] = useState('MANUAL');
  const [wsConnected, setWsConnected] = useState(false);

  useEffect(() => {
    // Load settings
    getSettings()
      .then((res) => setMode(res.data?.mode || 'MANUAL'))
      .catch(() => {});

    // Connect WebSocket
    connectWebSocket(() => {
      setWsConnected(true);
    });

    // Listen for settings changes
    subscribe('/topic/settings', (data) => {
      if (data.mode) setMode(data.mode);
    });

    // Check connection periodically
    const interval = setInterval(() => {
      setWsConnected(isConnected());
    }, 3000);

    return () => clearInterval(interval);
  }, []);

  return (
    <Router>
      <div className="app-layout">
        <Sidebar mode={mode} wsConnected={wsConnected} />
        <main className="main-content">
          <Routes>
            <Route path="/" element={<Dashboard mode={mode} setMode={setMode} />} />
            <Route path="/predictions" element={<Predictions mode={mode} />} />
            <Route path="/trades" element={<Trades />} />
            <Route path="/settings" element={<Settings mode={mode} setMode={setMode} />} />
          </Routes>
        </main>
      </div>
    </Router>
  );
}
