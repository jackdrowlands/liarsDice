import React from 'react'
import { Routes, Route, Link } from 'react-router-dom'
import GameControl from './pages/GameControl'
import TournamentDashboard from './pages/TournamentDashboard'
import TournamentDetail from './pages/TournamentDetail'
import GameViewer from './pages/GameViewer'
import Stats from './pages/Stats'

const App = () => (
  <div>
    <nav>
      <ul>
        <li><Link to="/">Game Control</Link></li>
        <li><Link to="/tournaments">Tournaments</Link></li>
        <li><Link to="/stats">Stats</Link></li>
      </ul>
    </nav>
    <main>
      <Routes>
        <Route path="/" element={<GameControl />} />
        <Route path="/tournaments" element={<TournamentDashboard />} />
        <Route path="/tournaments/:tournamentId" element={<TournamentDetail />} />
        <Route path="/viewer/:gameId" element={<GameViewer />} />
        <Route path="/stats" element={<Stats />} />
      </Routes>
    </main>
  </div>
)

export default App
