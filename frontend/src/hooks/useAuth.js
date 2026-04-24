import { useState, useEffect, useCallback } from 'react';
import api from '../utils/api';

export default function useAuth() {
  const [user, setUser] = useState(null);
  const [loading, setLoading] = useState(true);

  const isAuthenticated = !!localStorage.getItem('spotify_access_token');

  const fetchProfile = useCallback(async () => {
    if (!isAuthenticated) { setLoading(false); return; }
    try {
      const profile = await api.getProfile();
      setUser(profile);
    } catch { setUser(null); }
    setLoading(false);
  }, [isAuthenticated]);

  useEffect(() => { fetchProfile(); }, [fetchProfile]);

  const login = () => { window.location.href = 'http://localhost:5000/api/auth/login'; };

  const logout = () => {
    localStorage.removeItem('spotify_access_token');
    localStorage.removeItem('spotify_refresh_token');
    setUser(null);
    window.location.href = '/';
  };

  return { user, isAuthenticated, loading, login, logout };
}
