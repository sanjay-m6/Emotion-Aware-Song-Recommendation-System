const API_BASE = 'http://localhost:5000/api';

async function request(path, options = {}) {
  const headers = { 'Content-Type': 'application/json', ...options.headers };

  const res = await fetch(`${API_BASE}${path}`, { ...options, headers });

  if (!res.ok) {
    const err = await res.json().catch(() => ({ error: res.statusText }));
    throw new Error(err.error || 'Request failed');
  }

  return res.json();
}

export const api = {
  detectEmotion: (image) => request('/detect-emotion', { method: 'POST', body: JSON.stringify({ image }) }),
  chat: (message, history) => request('/chat', { method: 'POST', body: JSON.stringify({ message, history }) }),
  getRecommendations: (emotion, confidence = 0.8, limit = 10) => request(`/recommendations?emotion=${emotion}&confidence=${confidence}&limit=${limit}`),
  createPlaylist: (name, trackUris, description) => request('/playlist/create', { method: 'POST', body: JSON.stringify({ name, track_uris: trackUris, description }) }),
  saveTrack: (trackId) => request('/track/save', { method: 'POST', body: JSON.stringify({ track_id: trackId }) }),
  getMoodHistory: () => request('/mood-history'),
  getProfile: () => request('/auth/me'),
  health: () => fetch(`${API_BASE}/health`).then(r => r.json()),
};

export default api;
