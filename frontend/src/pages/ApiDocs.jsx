import React from 'react'

function Endpoint({ method, path, description, requestBody, response }) {
  const methodColors = { GET: 'text-green-400', POST: 'text-blue-400', DELETE: 'text-red-400' }
  return (
    <div className="bg-gray-900 border border-gray-800 rounded-xl p-6">
      <div className="flex items-center gap-3 mb-3">
        <span className={`font-mono text-sm font-bold ${methodColors[method] || 'text-gray-400'}`}>{method}</span>
        <code className="text-sm font-mono bg-gray-800 px-2 py-0.5 rounded">{path}</code>
      </div>
      <p className="text-sm text-gray-400 mb-3">{description}</p>
      {requestBody && (
        <div className="mb-3">
          <div className="text-xs text-gray-500 mb-1">Request Body:</div>
          <pre className="text-xs bg-gray-950 rounded-lg p-3 overflow-x-auto">{requestBody}</pre>
        </div>
      )}
      {response && (
        <div>
          <div className="text-xs text-gray-500 mb-1">Response:</div>
          <pre className="text-xs bg-gray-950 rounded-lg p-3 overflow-x-auto">{response}</pre>
        </div>
      )}
    </div>
  )
}

export default function ApiDocs() {
  return (
    <div className="space-y-8">
      <div>
        <h1 className="text-3xl font-bold">API Documentation</h1>
        <p className="text-gray-400 mt-2">
          REST API for the Scandium Labs solid electrolyte screening platform
        </p>
      </div>

      <div className="space-y-4">
        <Endpoint
          method="GET"
          path="/health"
          description="Check API and model health status"
          response={`{\n  "status": "healthy",\n  "model_loaded": true\n}`}
        />

        <Endpoint
          method="POST"
          path="/screen"
          description="Submit a batch screening job for multiple materials by MP ID"
          requestBody={`{\n  "material_ids": ["mp-1234", "mp-5678"],\n  "temperature": 300,\n  "top_k": 10\n}`}
          response={`{\n  "job_id": "uuid",\n  "status": "queued",\n  "created_at": "2026-01-01T00:00:00"\n}`}
        />

        <Endpoint
          method="POST"
          path="/screen/upload"
          description="Upload a CIF/POSCAR file for single-material screening (multipart/form-data)"
          response={`{\n  "formula": "Li6PS5Cl",\n  "spacegroup": 216,\n  "ionic_conductivity": {\n    "value": 1.43e-3,\n    "unit": "S/cm"\n  },\n  "formation_energy": {\n    "value": -2.71,\n    "unit": "eV/atom"\n  },\n  "recommendation": "HIGH PRIORITY — Excellent candidate"\n}`}
        />

        <Endpoint
          method="GET"
          path="/job/{job_id}"
          description="Poll the status and results of a screening job"
          response={`{\n  "job_id": "uuid",\n  "status": "completed",\n  "results": [...],\n  "top_k": [...]\n}`}
        />
      </div>

      <div className="bg-gray-900 border border-gray-800 rounded-xl p-6">
        <h2 className="text-lg font-semibold mb-4">Material Properties Predicted</h2>
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="text-gray-400 border-b border-gray-800">
                <th className="text-left py-2">Property</th>
                <th className="text-left py-2">Units</th>
                <th className="text-left py-2">Description</th>
              </tr>
            </thead>
            <tbody>
              <tr className="border-b border-gray-800"><td className="py-3 font-mono">ionic_conductivity</td><td>S/cm</td><td>Li⁺ ionic conductivity at room temperature</td></tr>
              <tr className="border-b border-gray-800"><td className="py-3 font-mono">formation_energy</td><td>eV/atom</td><td>Thermodynamic formation energy (more negative = more stable)</td></tr>
              <tr className="border-b border-gray-800"><td className="py-3 font-mono">energy_above_hull</td><td>eV/atom</td><td>Distance from convex hull (&lt;0.025 = stable)</td></tr>
              <tr className="border-b border-gray-800"><td className="py-3 font-mono">activation_energy</td><td>eV</td><td>Energy barrier for Li⁺ hopping (&lt;0.4 eV = fast)</td></tr>
              <tr className="border-b border-gray-800"><td className="py-3 font-mono">band_gap</td><td>eV</td><td>Electronic band gap</td></tr>
              <tr><td className="py-3 font-mono">recommendation</td><td>—</td><td>HIGH PRIORITY / MEDIUM / REJECT / UNCERTAIN</td></tr>
            </tbody>
          </table>
        </div>
      </div>
    </div>
  )
}
