import axios from 'axios'

const API_BASE = import.meta.env.VITE_API_URL || '/api'

let authToken = null

export function setAuthToken(token) {
  authToken = token
}

export function getAuthToken() {
  return authToken
}

async function request(method, path, data = null, opts = {}) {
  const headers = { 'Content-Type': 'application/json' }
  if (authToken) headers['Authorization'] = `Bearer ${authToken}`

  try {
    const res = await axios({
      method,
      url: `${API_BASE}${path}`,
      data: data && method !== 'get' ? data : undefined,
      params: method === 'get' ? data : undefined,
      headers,
      ...opts,
    })
    return res.data
  } catch (err) {
    const msg = err.response?.data?.detail || err.message
    throw new Error(msg)
  }
}

export function get(path, params) {
  return request('get', path, params)
}

export function post(path, data) {
  return request('post', path, data)
}

export function uploadCif(file, temperature = 300) {
  const formData = new FormData()
  formData.append('file', file)
  formData.append('temperature', String(temperature))

  const headers = { 'Content-Type': 'multipart/form-data' }
  if (authToken) headers['Authorization'] = `Bearer ${authToken}`

  return axios.post(`${API_BASE}/screen/upload`, formData, {
    headers,
    timeout: 60000,
  }).then(r => r.data)
}

export function screenMaterials(materialIds, formulas, temperature = 300) {
  const headers = {}
  if (authToken) headers['Authorization'] = `Bearer ${authToken}`
  return axios.post(
    `${API_BASE}/screen`,
    { material_ids: materialIds, formulas, temperature, top_k: 10 },
    { headers }
  ).then(r => r.data)
}

export function getJobStatus(jobId) {
  const headers = {}
  if (authToken) headers['Authorization'] = `Bearer ${authToken}`
  return axios.get(`${API_BASE}/job/${jobId}`, { headers }).then(r => r.data)
}

export function healthCheck() {
  return axios.get(`${API_BASE}/health`).then(r => r.data)
}
