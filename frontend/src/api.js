import axios from 'axios';

const API_BASE = '/api';

const api = axios.create({
  baseURL: API_BASE,
  timeout: 120000,
  headers: { 'Content-Type': 'application/json' },
});

// ── Predictions ──
export const fetchPrediction = () => api.post('/predictions/fetch');
export const getPredictions = () => api.get('/predictions');
export const getLatestPrediction = () => api.get('/predictions/latest');
export const acceptPrediction = (id) => api.post(`/predictions/${id}/accept`);
export const rejectPrediction = (id) => api.post(`/predictions/${id}/reject`);

// ── Trades ──
export const getTrades = () => api.get('/trades');
export const getOpenTrades = () => api.get('/trades/open');
export const getTradeStats = () => api.get('/trades/stats');
export const closePosition = () => api.post('/trades/close');

// ── Settings ──
export const getSettings = () => api.get('/settings');
export const setMode = (mode) => api.put('/settings/mode', null, { params: { mode } });

// ── Dashboard / Misc ──
export const getDashboard = () => api.get('/dashboard');
export const getAccountStatus = () => api.get('/account');
export const getModelInfo = () => api.get('/model');
export const trainModel = () => api.post('/train');
export const healthCheck = () => api.get('/health');

export default api;
