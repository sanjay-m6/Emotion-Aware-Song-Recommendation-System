import { NavLink } from 'react-router-dom';

export default function Navbar() {

  return (
    <nav className="navbar" role="navigation" aria-label="Main navigation">
      <NavLink to="/" className="navbar__logo" aria-label="Emora Home">🎭 Emora</NavLink>

      <ul className="navbar__links">
        <li><NavLink to="/" className={({ isActive }) => `navbar__link ${isActive ? 'navbar__link--active' : ''}`} end>Home</NavLink></li>
        <li><NavLink to="/detect" className={({ isActive }) => `navbar__link ${isActive ? 'navbar__link--active' : ''}`}>Detect Mood</NavLink></li>
        <li><NavLink to="/chat" className={({ isActive }) => `navbar__link ${isActive ? 'navbar__link--active' : ''}`}>Chat Mode</NavLink></li>
        <li><NavLink to="/history" className={({ isActive }) => `navbar__link ${isActive ? 'navbar__link--active' : ''}`}>History</NavLink></li>
      </ul>

      <div className="navbar__right">
        {/* Auth removed by request */}
      </div>
    </nav>
  );
}
