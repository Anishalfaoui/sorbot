import axios from 'axios';

const API_BASE = '/api';

const api = axios.create({
  baseURL: API_BASE,
  timeout: 120000,
  headers: { 'Content-Type': 'application/json' },
});

// ── Attach JWT token to every request ──
api.interceptors.request.use((config) => {
  const token = localStorage.getItem('token');
  if (token) {
    config.headers.Authorization = `Bearer ${token}`;
  }
  return config;
});

// ── Handle 401 responses (expired/invalid token) ──
api.interceptors.response.use(
  (response) => response,
  (error) => {
    if (error.response?.status === 401 || error.response?.status === 403) {
      // Don't redirect if we're already on an auth endpoint
      const url = error.config?.url || '';
      if (!url.includes('/auth/')) {
        localStorage.removeItem('token');
        localStorage.removeItem('user');
        window.location.reload();
      }
    }
    return Promise.reject(error);
  }
);

// ── Auth ──
export const login = (data) => api.post('/auth/login', data);
export const register = (data) => api.post('/auth/register', data);
export const getMe = () => api.get('/auth/me');

// ── Predictions ──
export const fetchPrediction = (symbol = 'BTCUSD') => api.post('/predictions/fetch', null, { params: { symbol } });
export const SUPPORTED_SYMBOLS = ['BTCUSD', 'EURUSD', 'XAUUSD'];
export const fetchPredictionsForAllSymbols = async () => {
  const settled = await Promise.allSettled(SUPPORTED_SYMBOLS.map((symbol) => fetchPrediction(symbol)));
  return settled;
};
export const getPredictions = () => api.get('/predictions');
export const getLatestPrediction = () => api.get('/predictions/latest');
export const acceptPrediction = (id) => api.post(`/predictions/${id}/accept`);
export const rejectPrediction = (id) => api.post(`/predictions/${id}/reject`);

// ── Trades ──
export const getTrades = () => api.get('/trades');
export const getOpenTrades = () => api.get('/trades/open');
export const getTradeStats = () => api.get('/trades/stats');
export const closePosition = () => api.post('/trades/close');
export const closeTradePosition = (tradeId) => api.post(`/trades/${tradeId}/close`);

// ── Settings ──
export const getSettings = () => api.get('/settings');
export const setMode = (mode) => api.put('/settings/mode', null, { params: { mode } });

// ── Dashboard / Misc ──
export const getDashboard = () => api.get('/dashboard');
export const getAccountStatus = (symbol = 'BTCUSD') => api.get('/account', { params: { symbol } });
export const getModelInfo = () => api.get('/model');
export const getModelInfoAll = () => api.get('/model/all');
export const trainModel = () => api.post('/train');
export const healthCheck = () => api.get('/health');

export default api;
