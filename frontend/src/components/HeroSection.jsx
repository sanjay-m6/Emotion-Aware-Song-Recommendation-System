import { useNavigate } from 'react-router-dom';

export default function HeroSection() {
  const navigate = useNavigate();

  return (
    <section className="hero page-enter">
      <div className="hero__bg-shape hero__bg-shape--1" />
      <div className="hero__bg-shape hero__bg-shape--2" />

      <h1 className="hero__title">
        Feel Your Mood.<br /><span>Hear Your Music.</span>
      </h1>

      <p className="hero__subtitle">
        AI-powered emotion detection meets Spotify. Let your face choose the
        perfect soundtrack — in real time.
      </p>

      <div className="hero__cta">
        <button className="btn btn--primary" onClick={() => navigate('/detect')}>
          🎭 Start Detection
        </button>
        <button className="btn btn--secondary" onClick={() => navigate('/detect')}>
          Learn More
        </button>
      </div>

      <div className="hero__features">
        <div className="hero__feature">
          <div className="hero__feature-icon">🧠</div>
          <span>AI Emotion Detection</span>
        </div>
        <div className="hero__feature">
          <div className="hero__feature-icon">🎵</div>
          <span>Spotify Integration</span>
        </div>
        <div className="hero__feature">
          <div className="hero__feature-icon">⚡</div>
          <span>Real-time Results</span>
        </div>
        <div className="hero__feature">
          <div className="hero__feature-icon">🎭</div>
          <span>Navarasa Framework</span>
        </div>
      </div>
    </section>
  );
}
