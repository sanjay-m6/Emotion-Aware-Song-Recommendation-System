import { useState } from 'react';

export default function SongCard({ track, onSave }) {
  const [playing, setPlaying] = useState(false);
  const [liked, setLiked] = useState(false);
  const [audio] = useState(() => track.preview_url ? new Audio(track.preview_url) : null);

  const togglePlay = () => {
    if (!audio) return;
    if (playing) { audio.pause(); setPlaying(false); }
    else {
      audio.play().catch(() => {});
      audio.onended = () => setPlaying(false);
      setPlaying(true);
    }
  };

  const handleLike = async () => {
    if (liked) return;
    setLiked(true);
    if (onSave) onSave(track.id);
  };

  return (
    <div className="song-card">
      {track.image || track.album_image ? (
        <img src={track.image || track.album_image} alt={track.album || track.album_name} className="song-card__art" loading="lazy" />
      ) : (
        <div className="song-card__art song-card__art--placeholder">🎵</div>
      )}

      <div className="song-card__info">
        <div className="song-card__name" title={track.name}>{track.name}</div>
        <div className="song-card__artist">{track.artist || track.artists}</div>
        <div className="song-card__album">{track.album || track.album_name}</div>
      </div>

      <div className="song-card__actions">
        {audio && (
          <button className={`icon-btn icon-btn--play`} onClick={togglePlay} aria-label={playing ? 'Pause' : 'Play preview'} title="Play 30s preview">
            {playing ? '⏸' : '▶'}
          </button>
        )}
        <button className={`icon-btn ${liked ? 'icon-btn--liked' : ''}`} onClick={handleLike} aria-label="Like song" title="Save to Liked Songs">
          {liked ? '❤️' : '🤍'}
        </button>
        {track.spotify_url && (
          <a href={track.spotify_url} target="_blank" rel="noopener noreferrer" className="icon-btn" aria-label="Open in Spotify" title="Open in Spotify">
            🔗
          </a>
        )}
      </div>
    </div>
  );
}
