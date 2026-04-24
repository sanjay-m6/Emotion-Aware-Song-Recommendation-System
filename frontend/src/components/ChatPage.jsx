import { useState, useRef, useEffect } from 'react';
import SongCard from './SongCard';
import api from '../utils/api';

function SkeletonCards() {
  return Array.from({ length: 4 }).map((_, i) => (
    <div className="song-card song-card--skeleton" key={i}>
      <div className="skeleton skeleton--art" />
      <div className="song-card__info">
        <div className="skeleton skeleton--text" style={{ width: '80%' }} />
        <div className="skeleton skeleton--text-sm" />
      </div>
    </div>
  ));
}

export default function ChatPage() {
  const [messages, setMessages] = useState([
    { role: 'assistant', content: "Hi! I'm Emora. Tell me how you're feeling today, and I'll find the perfect Tamil or English songs for you!" }
  ]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const [tracks, setTracks] = useState([]);
  const [currentEmotion, setCurrentEmotion] = useState(null);
  const [youtubeVideos, setYoutubeVideos] = useState([]);
  
  const messagesEndRef = useRef(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const handleSend = async (e) => {
    e.preventDefault();
    if (!input.trim() || loading) return;

    const userMessage = input.trim();
    setInput('');
    setMessages(prev => [...prev, { role: 'user', content: userMessage }]);
    setLoading(true);

    try {
      // Pass a small history window (last 4 messages) to maintain context
      const history = messages.slice(-4);
      const data = await api.chat(userMessage, history);
      
      setMessages(prev => [...prev, { role: 'assistant', content: data.reply }]);
      if (data.tracks) {
        setTracks(data.tracks);
      }
      if (data.emotion) {
        setCurrentEmotion(data.emotion);
      }
      if (data.youtube_videos) {
        setYoutubeVideos(data.youtube_videos);
      }
    } catch (err) {
      setMessages(prev => [...prev, { role: 'assistant', content: "Oops, my brain disconnected for a second. Can you try saying that again?" }]);
    }
    
    setLoading(false);
  };

  return (
    <div className="detect-page page-enter">
      <div className="detect-page__header">
        <h1 className="detect-page__title">💬 Chat with Emora</h1>
        <p className="detect-page__subtitle">Describe your mood and get tailored Tamil & English song recommendations</p>
      </div>

      <div className="detect-layout">
        {/* Left — Chat Interface */}
        <div className="webcam-panel" style={{ display: 'flex', flexDirection: 'column', height: '600px', padding: 0 }}>
          <div style={{ flex: 1, overflowY: 'auto', padding: '24px', display: 'flex', flexDirection: 'column', gap: '16px' }}>
            {messages.map((msg, idx) => (
              <div key={idx} style={{
                alignSelf: msg.role === 'user' ? 'flex-end' : 'flex-start',
                maxWidth: '85%',
                background: msg.role === 'user' ? 'rgba(255, 138, 0, 0.2)' : 'rgba(255, 255, 255, 0.05)',
                border: msg.role === 'user' ? '1px solid rgba(255, 138, 0, 0.3)' : '1px solid rgba(255, 255, 255, 0.1)',
                padding: '12px 16px',
                borderRadius: msg.role === 'user' ? '16px 16px 4px 16px' : '16px 16px 16px 4px',
                lineHeight: 1.5,
                fontSize: '0.95rem',
              }}>
                {msg.role === 'assistant' && <div style={{ fontSize: '0.75rem', color: '#FF8A00', marginBottom: 4, fontWeight: 600 }}>✨ Emora</div>}
                {msg.content}
              </div>
            ))}
            {loading && (
              <div style={{
                alignSelf: 'flex-start',
                background: 'rgba(255, 255, 255, 0.05)',
                padding: '12px 16px',
                borderRadius: '16px 16px 16px 4px',
              }}>
                <div className="detecting-spinner" style={{ margin: 0, justifyContent: 'flex-start' }}>
                  <span className="detecting-spinner__dot" /><span className="detecting-spinner__dot" /><span className="detecting-spinner__dot" />
                </div>
              </div>
            )}
            <div ref={messagesEndRef} />
          </div>

          <form onSubmit={handleSend} style={{
            padding: '16px',
            borderTop: '1px solid rgba(255,255,255,0.05)',
            display: 'flex',
            gap: '12px',
            background: 'rgba(0,0,0,0.2)'
          }}>
            <input
              type="text"
              value={input}
              onChange={(e) => setInput(e.target.value)}
              placeholder="Type how you feel..."
              disabled={loading}
              style={{
                flex: 1,
                background: 'rgba(255,255,255,0.05)',
                border: '1px solid rgba(255,255,255,0.1)',
                color: '#fff',
                padding: '12px 16px',
                borderRadius: '8px',
                outline: 'none',
                fontFamily: 'Inter, sans-serif'
              }}
            />
            <button type="submit" className="btn btn--primary" disabled={loading || !input.trim()}>
              Send
            </button>
          </form>
        </div>

        {/* Right — Recommendations (persist while loading new ones) */}
        <div className="recs-panel">
          <div className="recs-panel__header">
            <h2 className="recs-panel__title">
              {currentEmotion ? `🎵 Songs for feeling ${currentEmotion}` : '🎵 Recommendations'}
              {loading && tracks.length > 0 && (
                <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)', fontWeight: 400, marginLeft: 8 }}>Updating...</span>
              )}
            </h2>
          </div>

          <div className="song-grid" style={{ opacity: loading && tracks.length > 0 ? 0.6 : 1, transition: 'opacity 0.3s ease' }}>
            {tracks.length > 0 ? (
              tracks.map(t => <SongCard key={t.id} track={t} />)
            ) : loading ? (
              <SkeletonCards />
            ) : (
              <div style={{ textAlign: 'center', padding: '40px 20px', color: 'var(--text-muted)' }}>
                <div style={{ fontSize: '3rem', marginBottom: '12px' }}>🎧</div>
                <p>Chat with Emora to get personalized Tamil and English songs</p>
              </div>
            )}
          </div>

          {youtubeVideos.length > 0 && (
            <div className="youtube-panel" style={{ marginTop: '40px', opacity: loading ? 0.6 : 1, transition: 'opacity 0.3s ease' }}>
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
    </div>
  );
}
