import { useState, useEffect } from 'react';
import { Chart as ChartJS, ArcElement, Tooltip, Legend, CategoryScale, LinearScale, PointElement, LineElement } from 'chart.js';
import { Doughnut, Line } from 'react-chartjs-2';
import api from '../utils/api';

ChartJS.register(ArcElement, Tooltip, Legend, CategoryScale, LinearScale, PointElement, LineElement);

const EMOTION_COLORS = {
  anger: '#FF4444', surprise: '#FF8C00', contempt: '#8B4513', happy: '#FFD700',
  neutral: '#94A3B8', fear: '#9B59B6', sad: '#4169E1', disgust: '#228B22', shringara: '#FF69B4',
};

export default function MoodDashboard() {
  const [history, setHistory] = useState([]);
  const [stats, setStats] = useState({ total_detections: 0, emotion_counts: {}, dominant_emotion: 'neutral' });
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    api.getMoodHistory()
      .then(data => { setHistory(data.history || []); setStats(data.stats || stats); })
      .catch(() => {})
      .finally(() => setLoading(false));
  }, []);



  if (loading) {
    return <div className="dashboard page-enter"><p style={{ textAlign: 'center', color: 'var(--text-muted)' }}>Loading...</p></div>;
  }

  const counts = stats.emotion_counts || {};
  const emotions = Object.keys(counts);
  const values = Object.values(counts);

  const doughnutData = {
    labels: emotions.map(e => e.charAt(0).toUpperCase() + e.slice(1)),
    datasets: [{
      data: values,
      backgroundColor: emotions.map(e => EMOTION_COLORS[e] || '#94A3B8'),
      borderWidth: 2, borderColor: '#FFFFFF',
    }],
  };

  const timelineData = {
    labels: history.slice(-30).map((_, i) => i + 1),
    datasets: [{
      label: 'Confidence',
      data: history.slice(-30).map(h => h.confidence),
      borderColor: '#FF8A00',
      backgroundColor: 'rgba(255, 138, 0, 0.1)',
      fill: true, tension: 0.4, pointRadius: 3, pointBackgroundColor: '#FF8A00',
    }],
  };

  const chartOptions = {
    responsive: true, maintainAspectRatio: false,
    plugins: { legend: { position: 'bottom', labels: { padding: 16, usePointStyle: true, font: { family: 'Inter' } } } },
  };

  const lineOptions = {
    responsive: true, maintainAspectRatio: false,
    scales: {
      y: { min: 0, max: 1, grid: { color: '#E5E7EB' }, ticks: { font: { family: 'Inter' } } },
      x: { grid: { display: false }, ticks: { font: { family: 'Inter' } } },
    },
    plugins: { legend: { display: false } },
  };

  return (
    <div className="dashboard page-enter">
      <h1 className="dashboard__title">📊 Mood Insights</h1>

      <div className="stat-grid" style={{ marginBottom: 32 }}>
        <div className="stat-item">
          <div className="stat-item__value">{stats.total_detections}</div>
          <div className="stat-item__label">Total Detections</div>
        </div>
        <div className="stat-item">
          <div className="stat-item__value" style={{ textTransform: 'capitalize' }}>{stats.dominant_emotion}</div>
          <div className="stat-item__label">Dominant Mood</div>
        </div>
        <div className="stat-item">
          <div className="stat-item__value">{emotions.length}</div>
          <div className="stat-item__label">Unique Moods</div>
        </div>
      </div>

      <div className="dashboard__grid">
        <div className="dash-card">
          <div className="dash-card__title">Emotion Distribution</div>
          <div className="chart-container">
            {values.length > 0 ? <Doughnut data={doughnutData} options={chartOptions} /> : <p style={{ color: 'var(--text-muted)', textAlign: 'center', paddingTop: 60 }}>No data yet — detect some moods!</p>}
          </div>
        </div>

        <div className="dash-card">
          <div className="dash-card__title">Confidence Over Time</div>
          <div className="chart-container">
            {history.length > 0 ? <Line data={timelineData} options={lineOptions} /> : <p style={{ color: 'var(--text-muted)', textAlign: 'center', paddingTop: 60 }}>No data yet</p>}
          </div>
        </div>

        <div className="dash-card dash-card--full">
          <div className="dash-card__title">Recent Detections</div>
          {history.length > 0 ? (
            <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap' }}>
              {history.slice(-20).reverse().map((h, i) => (
                <span key={i} style={{
                  display: 'inline-block', padding: '6px 14px', borderRadius: 20,
                  background: `${EMOTION_COLORS[h.emotion]}15`, color: EMOTION_COLORS[h.emotion],
                  fontSize: '0.85rem', fontWeight: 500,
                }}>
                  {h.emotion} ({Math.round(h.confidence * 100)}%)
                </span>
              ))}
            </div>
          ) : <p style={{ color: 'var(--text-muted)' }}>No detections recorded this session</p>}
        </div>
      </div>
    </div>
  );
}
