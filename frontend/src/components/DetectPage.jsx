import { useState, useCallback, useEffect, useRef } from 'react';
import useWebcam from '../hooks/useWebcam';
import useEmotionDetection from '../hooks/useEmotionDetection';
import SongCard from './SongCard';
import api from '../utils/api';

function SkeletonCards() {
  return Array.from({ length: 5 }).map((_, i) => (
    <div className="song-card song-card--skeleton" key={i}>
      <div className="skeleton skeleton--art" />
      <div className="song-card__info">
        <div className="skeleton skeleton--text" style={{ width: '80%' }} />
        <div className="skeleton skeleton--text-sm" />
      </div>
    </div>
  ));
}

export default function DetectPage() {
  const webcam = useWebcam();
  const detection = useEmotionDetection();

  const [tracks, setTracks] = useState([]);
  const [loadingTracks, setLoadingTracks] = useState(false);
  const [aiExplanation, setAiExplanation] = useState(null);
  const [youtubeVideos, setYoutubeVideos] = useState([]);
  const [toasts, setToasts] = useState([]);
  const [autoDetect, setAutoDetect] = useState(false);
  const intervalRef = useRef(null);
  const lastEmotion = useRef(null);

  const addToast = useCallback((msg, type = 'success') => {
    const id = Date.now();
    setToasts(prev => [...prev, { id, msg, type }]);
    setTimeout(() => setToasts(prev => prev.filter(t => t.id !== id)), 3000);
  }, []);

  // Fetch recommendations when emotion changes
  const fetchRecs = useCallback(async (emotion, confidence) => {
    if (!emotion) return;
    setLoadingTracks(true);
    setAiExplanation(null);
    setYoutubeVideos([]);
    try {
      const data = await api.getRecommendations(emotion, confidence, 10);
      setTracks(data.tracks || []);
      setAiExplanation(data.explanation || null);
      setYoutubeVideos(data.youtube_videos || []);
    } catch (e) {
      addToast('Failed to load recommendations', 'error');
    }
    setLoadingTracks(false);
  }, [addToast]);

  // Auto-detection loop
  useEffect(() => {
    if (autoDetect && webcam.isActive) {
      intervalRef.current = setInterval(async () => {
        const frame = webcam.captureFrame();
        if (frame) {
          const result = await detection.detect(frame);
          if (result && result.face_found && result.emotion !== lastEmotion.current) {
            lastEmotion.current = result.emotion;
            fetchRecs(result.emotion, result.confidence);
          }
        }
      }, 2000);
    }
    return () => { if (intervalRef.current) clearInterval(intervalRef.current); };
  }, [autoDetect, webcam.isActive, webcam, detection, fetchRecs]);

  const handleSingleDetect = async () => {
    const frame = webcam.captureFrame();
    if (!frame) return;
    const result = await detection.detect(frame);
    if (result && result.face_found) {
      lastEmotion.current = result.emotion;
      fetchRecs(result.emotion, result.confidence);
    }
  };

  // Removed handleSave and handleCreatePlaylist since they require user login

  return (
    <div className="detect-page page-enter">
      <div className="detect-page__header">
        <h1 className="detect-page__title">🎭 Detect Your Mood</h1>
        <p className="detect-page__subtitle">Let AI read your emotion and find the perfect soundtrack</p>
      </div>

      <div className="detect-layout">
        {/* Left — Webcam */}
        <div className="webcam-panel">
          <div className="webcam-panel__video-wrap">
            {webcam.isActive ? (
              <>
                <video ref={webcam.videoRef} className="webcam-panel__video" autoPlay muted playsInline />
                <div className={`webcam-panel__glow ${detection.detecting ? 'webcam-panel__glow--active' : ''}`} />
                {detection.faceFound && detection.bbox && detection.frameSize && (
                  <div style={{
                    position: 'absolute',
                    border: '3px solid var(--orange)',
                    left: `${(detection.bbox[0] / detection.frameSize[0]) * 100}%`,
                    top: `${(detection.bbox[1] / detection.frameSize[1]) * 100}%`,
                    width: `${(detection.bbox[2] / detection.frameSize[0]) * 100}%`,
                    height: `${(detection.bbox[3] / detection.frameSize[1]) * 100}%`,
                    zIndex: 10,
                    borderRadius: '4px',
                    boxShadow: '0 0 10px var(--orange-glow)'
                  }}>
                    <span style={{
                      position: 'absolute', top: '-30px', left: '-3px', 
                      background: 'var(--orange)', color: '#fff', padding: '4px 8px',
                      fontSize: '12px', fontWeight: 'bold', borderRadius: '4px',
                      whiteSpace: 'nowrap'
                    }}>
                      {detection.emotionDisplay?.emoji} {detection.emotion?.toUpperCase()} ({Math.round(detection.confidence * 100)}%)
                    </span>
                  </div>
                )}
              </>
            ) : (
              <div className="webcam-panel__placeholder">
                <div className="webcam-panel__placeholder-icon">📷</div>
                <span>Camera is off</span>
              </div>
            )}
          </div>

          <div className="webcam-panel__controls">
            {!webcam.isActive ? (
              <button className="btn btn--primary" onClick={webcam.start}>🎥 Start Camera</button>
            ) : (
              <>
                <button className="btn btn--secondary btn--small" onClick={webcam.stop}>Stop</button>
                <button className="btn btn--primary btn--small" onClick={handleSingleDetect} disabled={detection.detecting}>
                  {detection.detecting ? 'Detecting...' : '📸 Detect Now'}
                </button>
                <button className={`btn btn--small ${autoDetect ? 'btn--primary' : 'btn--secondary'}`} onClick={() => setAutoDetect(!autoDetect)}>
                  {autoDetect ? '⏸ Auto' : '▶ Auto'}
                </button>
              </>
            )}
          </div>

          {webcam.error && <p style={{ padding: '12px 20px', color: '#EF4444', fontSize: '0.9rem' }}>{webcam.error}</p>}

          {detection.faceFound && detection.emotionDisplay && (
            <div className="webcam-panel__emotion">
              <div className="emotion-badge">
                <span className="emotion-badge__emoji">{detection.emotionDisplay.emoji}</span>
                <div>
                  <div>{detection.emotionDisplay.name}</div>
                  <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)', fontWeight: 400 }}>
                    {detection.emotionDisplay.meaning}
                  </div>
                </div>
              </div>
              <div className="confidence-bar">
                <div className="confidence-bar__track">
                  <div className="confidence-bar__fill" style={{ width: `${Math.round(detection.confidence * 100)}%` }} />
                </div>
                <div className="confidence-bar__label">{Math.round(detection.confidence * 100)}% confidence</div>
              </div>
            </div>
          )}

          {detection.detecting && (
            <div style={{ padding: '12px 20px' }}>
              <div className="detecting-spinner">
                <span className="detecting-spinner__dot" /><span className="detecting-spinner__dot" /><span className="detecting-spinner__dot" />
                <span>Analyzing expression...</span>
              </div>
            </div>
          )}

          {!detection.faceFound && detection.emotion === null && webcam.isActive && !detection.detecting && (
            <p style={{ padding: '12px 20px', color: 'var(--text-muted)', fontSize: '0.9rem' }}>
              Click "Detect Now" or enable Auto mode to begin
            </p>
          )}
        </div>

        {/* Right — Recommendations */}
        <div className="recs-panel">
          <div className="recs-panel__header">
            <h2 className="recs-panel__title">
              {detection.emotion ? `🎵 Songs for ${detection.emotion}` : '🎵 Recommendations'}
            </h2>
          </div>
          
          {aiExplanation && !loadingTracks && (
            <div style={{
              background: 'rgba(255, 138, 0, 0.1)',
              borderLeft: '4px solid #FF8A00',
              padding: '12px 16px',
              borderRadius: '4px',
              marginBottom: '20px',
              fontSize: '0.95rem',
              color: 'var(--text-primary)'
            }}>
              <strong>✨ AI Curator:</strong> {aiExplanation}
            </div>
          )}

          <div className="song-grid">
            {loadingTracks ? <SkeletonCards /> : tracks.length > 0 ? (
              tracks.map(t => <SongCard key={t.id} track={t} />)
            ) : (
              <div style={{ textAlign: 'center', padding: '40px 20px', color: 'var(--text-muted)' }}>
                <div style={{ fontSize: '3rem', marginBottom: '12px' }}>🎧</div>
                <p>Detect your mood to see personalized recommendations</p>
              </div>
            )}
          </div>

          {youtubeVideos.length > 0 && !loadingTracks && (
            <div className="youtube-panel" style={{ marginTop: '40px' }}>
              <div className="recs-panel__header">
                <h2 className="recs-panel__title" style={{ color: '#FF8A00' }}>
                  🎬 Therapy, Songs & Feel Good
                </h2>
              </div>
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))', gap: '20px' }}>
                {youtubeVideos.map(video => (
                  <div key={video.id} style={{ 
                    borderRadius: '16px', 
                    overflow: 'hidden', 
                    background: 'var(--surface-light)', 
                    border: '1px solid var(--border-color)', 
                    padding: '16px',
                    boxShadow: '0 8px 24px rgba(0,0,0,0.2)'
                  }}>
                    <h3 style={{ fontSize: '1rem', marginBottom: '12px', fontWeight: 600 }}>{video.title}</h3>
                    <div style={{ position: 'relative', paddingBottom: '56.25%', height: 0, overflow: 'hidden', borderRadius: '12px' }}>
                      <iframe 
                        src={`https://www.youtube.com/embed/${video.id}`} 
                        title={video.title}
                        frameBorder="0" 
                        allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" 
                        allowFullScreen
                        style={{ position: 'absolute', top: 0, left: 0, width: '100%', height: '100%' }}
                      />
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Toast notifications */}
      <div className="toast-container">
        {toasts.map(t => (
          <div key={t.id} className={`toast toast--${t.type}`}>{t.msg}</div>
        ))}
      </div>
    </div>
  );
}
