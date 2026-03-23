import { useState } from 'react';
import { login, register } from '../api';

export default function Login({ onLogin }) {
  const [isRegister, setIsRegister] = useState(false);
  const [form, setForm] = useState({ username: '', email: '', password: '' });
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);

  const handleChange = (e) => {
    setForm({ ...form, [e.target.name]: e.target.value });
    setError('');
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError('');
    setLoading(true);

    try {
      let res;
      if (isRegister) {
        res = await register(form);
      } else {
        res = await login({ username: form.username, password: form.password });
      }

      const data = res.data;
      if (data.token) {
        localStorage.setItem('token', data.token);
        localStorage.setItem('user', JSON.stringify({
          username: data.username,
          email: data.email,
          role: data.role,
          virtualBalance: data.virtualBalance,
        }));
        onLogin(data);
      } else {
        setError(data.message || 'Authentication failed');
      }
    } catch (err) {
      const msg = err.response?.data?.message || err.message || 'Something went wrong';
      setError(msg);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="auth-page">
      <div className="auth-container">
        <div className="auth-header">
          <h1 className="auth-logo">SORBOT</h1>
          <span className="auth-subtitle">AI Trading Engine</span>
        </div>

        <div className="auth-card">
          <h2>{isRegister ? 'Create Account' : 'Welcome Back'}</h2>
          <p className="auth-desc">
            {isRegister
              ? 'Register to start using Sorbot'
              : 'Sign in to your account'}
          </p>

          {error && <div className="auth-error">{error}</div>}

          <form onSubmit={handleSubmit} className="auth-form">
            <div className="form-group">
              <label htmlFor="username">Username</label>
              <input
                id="username"
                name="username"
                type="text"
                value={form.username}
                onChange={handleChange}
                placeholder="Enter your username"
                required
                autoComplete="username"
              />
            </div>

            {isRegister && (
              <div className="form-group">
                <label htmlFor="email">Email</label>
                <input
                  id="email"
                  name="email"
                  type="email"
                  value={form.email}
                  onChange={handleChange}
                  placeholder="Enter your email"
                  required
                  autoComplete="email"
                />
              </div>
            )}

            <div className="form-group">
              <label htmlFor="password">Password</label>
              <input
                id="password"
                name="password"
                type="password"
                value={form.password}
                onChange={handleChange}
                placeholder="Enter your password"
                required
                autoComplete={isRegister ? 'new-password' : 'current-password'}
              />
            </div>

            <button type="submit" className="btn btn-primary auth-submit" disabled={loading}>
              {loading ? (
                <><span className="spinner" /> {isRegister ? 'Creating...' : 'Signing in...'}</>
              ) : (
                isRegister ? 'Create Account' : 'Sign In'
              )}
            </button>
          </form>

          <div className="auth-switch">
            {isRegister ? (
              <>Already have an account?{' '}
                <button onClick={() => { setIsRegister(false); setError(''); }} className="auth-link">
                  Sign In
                </button>
              </>
            ) : (
              <>Don't have an account?{' '}
                <button onClick={() => { setIsRegister(true); setError(''); }} className="auth-link">
                  Create Account
                </button>
              </>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
