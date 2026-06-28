import React, { useState, useEffect } from 'react'
import { healthCheck } from '../utils/api'

function StatCard({ label, value, color }) {
  return (
    <div className="bg-gray-900 border border-gray-800 rounded-xl p-6">
      <div className={`text-3xl font-bold ${color}`}>{value}</div>
      <div className="text-sm text-gray-400 mt-1">{label}</div>
    </div>
  )
}

export default function Dashboard() {
  const [health, setHealth] = useState(null)
  const [error, setError] = useState(null)

  useEffect(() => {
    healthCheck()
      .then(setHealth)
      .catch(e => setError(e.message))
  }, [])

  return (
    <div className="space-y-8">
      <div>
        <h1 className="text-3xl font-bold">Dashboard</h1>
        <p className="text-gray-400 mt-2">
          AI-Driven Solid Electrolyte Discovery Platform
        </p>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        <StatCard label="Model Status" value={health?.model_loaded ? 'Online' : 'Offline'} color={health?.model_loaded ? 'text-green-400' : 'text-red-400'} />
        <StatCard label="API Status" value={health?.status === 'healthy' ? 'Healthy' : 'Unhealthy'} color={health?.status === 'healthy' ? 'text-green-400' : 'text-red-400'} />
        <StatCard label="Materials Screened" value="0" color="text-indigo-400" />
        <StatCard label="Candidates Found" value="0" color="text-cyan-400" />
      </div>

      <div className="bg-gray-900 border border-gray-800 rounded-xl p-6">
        <h2 className="text-xl font-semibold mb-4">Quick Start</h2>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <div className="bg-gray-800 rounded-lg p-4">
            <div className="text-indigo-400 font-medium mb-2">1. Upload CIF</div>
            <p className="text-sm text-gray-400">Upload a crystal structure file (.cif, .poscar) for AI screening</p>
          </div>
          <div className="bg-gray-800 rounded-lg p-4">
            <div className="text-indigo-400 font-medium mb-2">2. AI Screening</div>
            <p className="text-sm text-gray-400">Our PINN-GNN model predicts ionic conductivity, stability, and more</p>
          </div>
          <div className="bg-gray-800 rounded-lg p-4">
            <div className="text-indigo-400 font-medium mb-2">3. Rank & Validate</div>
            <p className="text-sm text-gray-400">Pareto-ranked candidates with uncertainty estimates</p>
          </div>
        </div>
      </div>

      <div className="bg-gray-900 border border-gray-800 rounded-xl p-6">
        <h2 className="text-xl font-semibold mb-4">Supported Materials</h2>
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="text-gray-400 border-b border-gray-800">
                <th className="text-left py-2">Family</th>
                <th className="text-left py-2">Example</th>
                <th className="text-left py-2">Conductivity</th>
                <th className="text-left py-2">Stability</th>
              </tr>
            </thead>
            <tbody>
              <tr className="border-b border-gray-800">
                <td className="py-3">Sulfide</td>
                <td className="font-mono">Li₆PS₅Cl</td>
                <td className="text-green-400">10⁻³ S/cm</td>
                <td className="text-yellow-400">Moderate</td>
              </tr>
              <tr className="border-b border-gray-800">
                <td className="py-3">Oxide</td>
                <td className="font-mono">LLZO</td>
                <td className="text-yellow-400">10⁻⁴ S/cm</td>
                <td className="text-green-400">Excellent</td>
              </tr>
              <tr className="border-b border-gray-800">
                <td className="py-3">Halide</td>
                <td className="font-mono">Li₃YCl₆</td>
                <td className="text-green-400">10⁻³ S/cm</td>
                <td className="text-green-400">Good</td>
              </tr>
              <tr>
                <td className="py-3">LGPS</td>
                <td className="font-mono">Li₁₀GeP₂S₁₂</td>
                <td className="text-green-400">12×10⁻³ S/cm</td>
                <td className="text-red-400">Poor vs Li</td>
              </tr>
            </tbody>
          </table>
        </div>
      </div>
    </div>
  )
}
