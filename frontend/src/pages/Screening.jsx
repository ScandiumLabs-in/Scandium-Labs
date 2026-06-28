import React, { useState, useRef } from 'react'
import { uploadCif, screenMaterials } from '../utils/api'

export default function Screening() {
  const [file, setFile] = useState(null)
  const [materialIds, setMaterialIds] = useState('')
  const [temperature, setTemperature] = useState(300)
  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState(null)
  const [error, setError] = useState(null)
  const [jobId, setJobId] = useState(null)
  const fileRef = useRef()

  function handleCifUpload(e) {
    const f = e.target.files?.[0]
    if (f) setFile(f)
  }

  async function handleUpload() {
    if (!file) return
    setLoading(true)
    setError(null)
    setResult(null)
    try {
      const data = await uploadCif(file, temperature)
      setResult(data)
    } catch (e) {
      setError(e.message)
    }
    setLoading(false)
  }

  async function handleScreen() {
    const ids = materialIds.split(',').map(s => s.trim()).filter(Boolean)
    if (ids.length === 0) return
    setLoading(true)
    setError(null)
    setResult(null)
    try {
      const data = await screenMaterials(ids, [], temperature)
      setJobId(data.job_id)
      setResult(data)
    } catch (e) {
      setError(e.message)
    }
    setLoading(false)
  }

  function ConductivityBar({ value }) {
    if (value == null) return null
    const logVal = Math.log10(Math.max(value, 1e-10))
    const pct = Math.max(0, Math.min(100, (logVal + 5) / 7 * 100))
    const color = value > 1e-3 ? 'bg-green-500' : value > 1e-5 ? 'bg-yellow-500' : 'bg-red-500'
    return (
      <div className="w-full bg-gray-700 rounded-full h-2.5 mt-1">
        <div className={`h-2.5 rounded-full ${color}`} style={{ width: `${pct}%` }} />
      </div>
    )
  }

  function RecommendationBadge({ rec }) {
    if (!rec) return null
    const colors = {
      'HIGH PRIORITY': 'bg-green-900 text-green-300',
      'MEDIUM PRIORITY': 'bg-yellow-900 text-yellow-300',
      'LOW PRIORITY': 'bg-gray-700 text-gray-300',
      'REJECT': 'bg-red-900 text-red-300',
      'UNCERTAIN': 'bg-purple-900 text-purple-300',
    }
    const key = Object.keys(colors).find(k => rec.includes(k)) || 'LOW PRIORITY'
    return (
      <span className={`inline-block px-3 py-1 rounded-full text-xs font-medium ${colors[key]}`}>
        {rec}
      </span>
    )
  }

  return (
    <div className="space-y-8">
      <div>
        <h1 className="text-3xl font-bold">Screen Materials</h1>
        <p className="text-gray-400 mt-2">
          Upload crystal structures or enter MP IDs for AI-powered solid electrolyte screening
        </p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <div className="bg-gray-900 border border-gray-800 rounded-xl p-6">
          <h2 className="text-lg font-semibold mb-4">Upload CIF / POSCAR</h2>
          <div
            className="border-2 border-dashed border-gray-700 rounded-lg p-8 text-center cursor-pointer hover:border-indigo-500 transition"
            onClick={() => fileRef.current?.click()}
          >
            <input ref={fileRef} type="file" accept=".cif,.poscar,.vasp" onChange={handleCifUpload} hidden />
            {file ? (
              <div>
                <div className="text-indigo-400 text-lg mb-1">{file.name}</div>
                <div className="text-sm text-gray-500">{(file.size / 1024).toFixed(1)} KB</div>
              </div>
            ) : (
              <div>
                <div className="text-4xl mb-2 text-gray-600">+</div>
                <div className="text-gray-400">Drop CIF/POSCAR file or click to browse</div>
              </div>
            )}
          </div>
          <div className="mt-4 flex items-center gap-4">
            <label className="text-sm text-gray-400">Temperature (K):</label>
            <input
              type="number" value={temperature} onChange={e => setTemperature(Number(e.target.value))}
              className="bg-gray-800 border border-gray-700 rounded px-3 py-1.5 w-24 text-sm"
            />
          </div>
          <button
            onClick={handleUpload}
            disabled={!file || loading}
            className="mt-4 w-full bg-indigo-600 hover:bg-indigo-700 disabled:bg-gray-700 disabled:text-gray-500 text-white rounded-lg py-2.5 font-medium transition"
          >
            {loading ? 'Screening...' : 'Screen Single Material'}
          </button>
        </div>

        <div className="bg-gray-900 border border-gray-800 rounded-xl p-6">
          <h2 className="text-lg font-semibold mb-4">Batch Screening</h2>
          <label className="text-sm text-gray-400 mb-2 block">Materials Project IDs (comma-separated)</label>
          <textarea
            value={materialIds}
            onChange={e => setMaterialIds(e.target.value)}
            placeholder="mp-1234, mp-5678, mp-9012"
            rows={4}
            className="w-full bg-gray-800 border border-gray-700 rounded-lg px-3 py-2 text-sm resize-none"
          />
          <button
            onClick={handleScreen}
            disabled={!materialIds.trim() || loading}
            className="mt-4 w-full bg-cyan-700 hover:bg-cyan-800 disabled:bg-gray-700 disabled:text-gray-500 text-white rounded-lg py-2.5 font-medium transition"
          >
            {loading ? 'Submitting...' : 'Screen Batch'}
          </button>
        </div>
      </div>

      {error && (
        <div className="bg-red-900/30 border border-red-800 rounded-xl p-4 text-red-300">
          {error}
        </div>
      )}

      {result && (
        <div className="bg-gray-900 border border-gray-800 rounded-xl p-6">
          <h2 className="text-xl font-semibold mb-4">Results</h2>

          {result.formula && (
            <div className="mb-4">
              <div className="text-sm text-gray-400">Formula</div>
              <div className="text-lg font-mono">{result.formula}</div>
            </div>
          )}

          {result.ionic_conductivity && (
            <div className="mb-4">
              <div className="text-sm text-gray-400">Ionic Conductivity</div>
              <div className="text-lg font-bold">
                {result.ionic_conductivity.value?.toExponential(2)} S/cm
              </div>
              <ConductivityBar value={result.ionic_conductivity.value} />
              {result.ionic_conductivity.uncertainty && (
                <div className="text-xs text-gray-500 mt-1">
                  ± {result.ionic_conductivity.uncertainty.toExponential(1)} S/cm
                </div>
              )}
            </div>
          )}

          <div className="grid grid-cols-2 gap-4 mb-4">
            {result.formation_energy && (
              <div>
                <div className="text-sm text-gray-400">Formation Energy</div>
                <div className="font-mono">{result.formation_energy.value?.toFixed(3)} eV/atom</div>
              </div>
            )}
            {result.energy_above_hull && (
              <div>
                <div className="text-sm text-gray-400">Energy Above Hull</div>
                <div className="font-mono">{result.energy_above_hull.value?.toFixed(3)} eV/atom</div>
              </div>
            )}
            {result.activation_energy && (
              <div>
                <div className="text-sm text-gray-400">Activation Energy</div>
                <div className="font-mono">{result.activation_energy.value?.toFixed(2)} eV</div>
              </div>
            )}
            {result.band_gap && (
              <div>
                <div className="text-sm text-gray-400">Band Gap</div>
                <div className="font-mono">{result.band_gap.value?.toFixed(2)} eV</div>
              </div>
            )}
          </div>

          {result.recommendation && (
            <div className="flex items-center justify-between">
              <RecommendationBadge rec={result.recommendation} />
              {result.ood?.is_ood && (
                <span className="text-xs text-purple-400">Out-of-distribution material</span>
              )}
            </div>
          )}

          {jobId && (
            <div className="mt-4 text-sm text-gray-500">
              Job ID: <span className="font-mono text-indigo-400">{jobId}</span>
            </div>
          )}
        </div>
      )}
    </div>
  )
}
