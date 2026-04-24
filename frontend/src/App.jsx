import { BrowserRouter, Routes, Route, useSearchParams, useNavigate } from 'react-router-dom';
import { useEffect } from 'react';
import Navbar from './components/Navbar';
import HeroSection from './components/HeroSection';
import DetectPage from './components/DetectPage';
import ChatPage from './components/ChatPage';
import MoodDashboard from './components/MoodDashboard';
import Footer from './components/Footer';
import './index.css';

function AppRoutes() {
  return (
    <>
      <Navbar />
      <main>
        <Routes>
          <Route path="/" element={<HeroSection />} />
          <Route path="/detect" element={<DetectPage />} />
          <Route path="/chat" element={<ChatPage />} />
          <Route path="/history" element={<MoodDashboard />} />
        </Routes>
      </main>
      <Footer />
    </>
  );
}

export default function App() {
  return (
    <BrowserRouter>
      <AppRoutes />
    </BrowserRouter>
  );
}
