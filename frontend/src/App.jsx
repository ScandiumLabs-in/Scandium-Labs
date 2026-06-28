import React from 'react'
import { Routes, Route, Link, useLocation } from 'react-router-dom'
import Dashboard from './pages/Dashboard'
import Screening from './pages/Screening'
import Results from './pages/Results'
import ApiDocs from './pages/ApiDocs'

function NavLink({ to, children }) {
  const location = useLocation()
  const active = location.pathname === to
  return (
    <Link
      to={to}
      className={`px-4 py-2 rounded-lg text-sm font-medium transition ${
        active
          ? 'bg-indigo-600 text-white'
          : 'text-gray-300 hover:bg-gray-800 hover:text-white'
      }`}
    >
      {children}
    </Link>
  )
}

export default function App() {
  return (
    <div className="min-h-screen bg-gray-950 text-gray-100">
      <header className="border-b border-gray-800">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex items-center justify-between h-16">
            <div className="flex items-center gap-3">
              <img src="/scandium.svg" alt="Sc" className="w-8 h-8" />
              <Link to="/" className="text-xl font-bold bg-gradient-to-r from-indigo-400 to-cyan-400 bg-clip-text text-transparent">
                Scandium Labs
              </Link>
            </div>
            <nav className="flex items-center gap-2">
              <NavLink to="/">Dashboard</NavLink>
              <NavLink to="/screen">Screen Materials</NavLink>
              <NavLink to="/results">Results</NavLink>
              <NavLink to="/docs">API Docs</NavLink>
            </nav>
          </div>
        </div>
      </header>

      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        <Routes>
          <Route path="/" element={<Dashboard />} />
          <Route path="/screen" element={<Screening />} />
          <Route path="/results" element={<Results />} />
          <Route path="/docs" element={<ApiDocs />} />
        </Routes>
      </main>

      <footer className="border-t border-gray-800 py-6 mt-12">
        <div className="max-w-7xl mx-auto px-4 text-center text-sm text-gray-500">
          Scandium Labs — AI-Driven Solid Electrolyte Discovery Platform
        </div>
      </footer>
    </div>
  )
}
