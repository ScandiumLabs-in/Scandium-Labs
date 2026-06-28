import React, { useState, useEffect, useRef, useCallback } from 'react'
import { getJobStatus } from '../utils/api'
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell } from 'recharts'

function ConductivityChart({ data }) {
  if (!data || data.length === 0) return null

  const chartData = data.map((c, i) => ({
    name: c.material_id || `#${i + 1}`,
    conductivity: c.ionic_conductivity?.value || 0,
    rank: c.rank || i + 1,
  }))

  return (
    <div className="bg-gray-900 border border-gray-800 rounded-xl p-6">
      <h2 className="text-lg font-semibold mb-4">Conductivity Rankings</h2>
      <ResponsiveContainer width="100%" height={300}>
        <BarChart data={chartData}>
          <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
          <XAxis dataKey="name" stroke="#9CA3AF" tick={{ fontSize: 11 }} />
          <YAxis stroke="#9CA3AF" tick={{ fontSize: 11 }} />
          <Tooltip
            contentStyle={{ backgroundColor: '#1F2937', border: '1px solid #374151', borderRadius: '8px' }}
            labelStyle={{ color: '#F3F4F6' }}
          />
          <Bar dataKey="conductivity" radius={[4, 4, 0, 0]}>
            {chartData.map((entry, i) => (
              <Cell key={i} fill={entry.conductivity > 1e-3 ? '#10B981' : entry.conductivity > 1e-5 ? '#F59E0B' : '#EF4444'} />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  )
}

export default function Results() {
  const [jobId, setJobId] = useState('')
  const [jobData, setJobData] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const [history, setHistory] = useState([])
  const intervalRef = useRef(null)

  const pollJob = useCallback(async (id) => {
    try {
      const data = await getJobStatus(id)
      setJobData(data)
      if (data.status === 'completed') {
        clearInterval(intervalRef.current)
        setHistory(prev => [{ id, ...data }, ...prev])
      }
    } catch (e) {
      setError(e.message)
      clearInterval(intervalRef.current)
    }
  }, [])

  useEffect(() => {
    return () => clearInterval(intervalRef.current)
  }, [])

  function handleTrack() {
    if (!jobId.trim()) return
    setLoading(true)
    setError(null)
    setJobData(null)
    pollJob(jobId.trim())
    intervalRef.current = setInterval(() => pollJob(jobId.trim()), 3000)
    setLoading(false)
  }

  return (
    <div className="space-y-8">
      <div>
        <h1 className="text-3xl font-bold">Screening Results</h1>
        <p className="text-gray-400 mt-2">Track screening jobs and view candidate rankings</p>
      </div>

      <div className="bg-gray-900 border border-gray-800 rounded-xl p-6">
        <h2 className="text-lg font-semibold mb-4">Track Job</h2>
        <div className="flex gap-3">
          <input
            value={jobId}
            onChange={e => setJobId(e.target.value)}
            placeholder="Enter Job ID (UUID)"
            className="flex-1 bg-gray-800 border border-gray-700 rounded-lg px-4 py-2.5 text-sm"
          />
          <button
            onClick={handleTrack}
            disabled={!jobId.trim() || loading}
            className="bg-indigo-600 hover:bg-indigo-700 disabled:bg-gray-700 text-white rounded-lg px-6 py-2.5 font-medium transition"
          >
            {loading ? 'Loading...' : 'Track'}
          </button>
        </div>
      </div>

      {error && (
        <div className="bg-red-900/30 border border-red-800 rounded-xl p-4 text-red-300">
          {error}
        </div>
      )}

      {jobData && (
        <div className="bg-gray-900 border border-gray-800 rounded-xl p-6">
          <div className="flex items-center justify-between mb-4">
            <h2 className="text-lg font-semibold">Job Status</h2>
            <span className={`px-3 py-1 rounded-full text-xs font-medium ${
              jobData.status === 'completed' ? 'bg-green-900 text-green-300' :
              jobData.status === 'queued' ? 'bg-yellow-900 text-yellow-300' :
              'bg-blue-900 text-blue-300'
            }`}>
              {jobData.status}
            </span>
          </div>
          {jobData.progress != null && (
            <div className="mb-4">
              <div className="flex justify-between text-sm text-gray-400 mb-1">
                <span>Progress</span>
                <span>{jobData.progress.toFixed(0)}%</span>
              </div>
              <div className="w-full bg-gray-700 rounded-full h-2">
                <div className="bg-indigo-500 h-2 rounded-full transition-all duration-500" style={{ width: `${jobData.progress}%` }} />
              </div>
            </div>
          )}
          <div className="text-sm text-gray-400">
            {jobData.n_materials != null && <span>{jobData.completed_materials}/{jobData.n_materials} materials</span>}
          </div>

          {jobData.results && jobData.results.length > 0 && (
            <div className="mt-6">
              <h3 className="font-semibold mb-3">Top Candidates</h3>
              <div className="overflow-x-auto">
                <table className="w-full text-sm">
                  <thead>
                    <tr className="text-gray-400 border-b border-gray-800">
                      <th className="text-left py-2">Rank</th>
                      <th className="text-left py-2">Material</th>
                      <th className="text-left py-2">Conductivity</th>
                      <th className="text-left py-2">Eₐₕ (eV/atom)</th>
                      <th className="text-left py-2">Recommendation</th>
                    </tr>
                  </thead>
                  <tbody>
                    {jobData.results.map((c, i) => (
                      <tr key={i} className="border-b border-gray-800">
                        <td className="py-3 font-bold">{c.rank || i + 1}</td>
                        <td className="font-mono">{c.material_id || c.formula || `#${i + 1}`}</td>
                        <td className="font-mono text-green-400">{c.ionic_conductivity?.value?.toExponential(2) || '-'} S/cm</td>
                        <td className="font-mono">{c.energy_above_hull?.value?.toFixed(3) || '-'}</td>
                        <td>
                          <span className={`px-2 py-0.5 rounded text-xs ${
                            c.recommendation?.includes('HIGH') ? 'bg-green-900 text-green-300' :
                            c.recommendation?.includes('REJECT') ? 'bg-red-900 text-red-300' :
                            'bg-gray-700 text-gray-300'
                          }`}>
                            {c.recommendation || 'N/A'}
                          </span>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          )}

          {jobData.results && jobData.results.length > 0 && (
            <div className="mt-6">
              <ConductivityChart data={jobData.results} />
            </div>
          )}
        </div>
      )}

      {history.length > 0 && (
        <div className="bg-gray-900 border border-gray-800 rounded-xl p-6">
          <h2 className="text-lg font-semibold mb-4">Recent Jobs</h2>
          <div className="space-y-2">
            {history.map((h, i) => (
              <div key={i} className="flex items-center justify-between bg-gray-800 rounded-lg px-4 py-3">
                <span className="font-mono text-sm">{h.id}</span>
                <span className="text-xs text-gray-400">{h.status}</span>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  )
}
