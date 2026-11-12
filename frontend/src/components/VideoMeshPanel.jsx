import { useEffect, useMemo, useState } from 'react'
import MeshViewer from './MeshViewer.jsx'
import { getFrameImageUrl } from '../services/captureApi.js'
import { getMeshBatch, meshAssetUrl, startMeshBatch } from '../services/videoMeshApi.js'

const stem = (name) => name.replace(/\.[^.]+$/, '')

export default function VideoMeshPanel({ videoId, metadata, poseResults, viewpointData }) {
  const frames = metadata?.frames || []
  const poses = useMemo(() => Object.fromEntries((poseResults?.frames || []).map(p => [p.frame, p])), [poseResults])
  const viewpoints = useMemo(() => Object.fromEntries((viewpointData?.frames || []).map(p => [p.frame, p])), [viewpointData])
  const [selected, setSelected] = useState([])
  const [run, setRun] = useState(null)
  const [active, setActive] = useState('')
  const [error, setError] = useState('')
  const [viewerError, setViewerError] = useState('')
  const [starting, setStarting] = useState(false)

  useEffect(() => {
    setSelected(frames.filter(f => f.is_blurry === false && poses[stem(f.filename)]?.full_body_visible === true)
      .slice(0, 12).map(f => f.filename))
    setRun(null)
    setActive('')
    setError('')
  }, [videoId]) // A new capture starts a separate selection; pose results are already loaded.

  useEffect(() => {
    if (!run?.run_id || run.status !== 'processing') return undefined
    let cancelled = false
    let timer
    const poll = async () => {
      try {
        const next = await getMeshBatch(run.run_id)
        if (!cancelled) {
          setRun(next)
          setError('')
          if (next.status === 'processing') timer = setTimeout(poll, 2000)
        }
      } catch (err) {
        if (!cancelled) {
          setError(`Status unavailable; retrying: ${err.message}`)
          timer = setTimeout(poll, 4000)
        }
      }
    }
    timer = setTimeout(poll, 1000)
    return () => { cancelled = true; clearTimeout(timer) }
  }, [run?.run_id, run?.status])

  const busy = starting || run?.status === 'processing'
  const selectedFrames = frames.filter(f => selected.includes(f.filename))
  const blurry = selectedFrames.filter(f => f.is_blurry !== false).length
  const completed = run?.frames?.filter(f => f.status === 'completed') || []
  const current = completed.find(f => f.filename === active) || completed[0]
  const currentView = current ? viewpoints[stem(current.filename)] : null

  async function generate() {
    setStarting(true)
    setError('')
    setViewerError('')
    try {
      const next = await startMeshBatch(videoId, selected)
      setActive('')
      setRun(next)
    } catch (err) {
      setError(`Could not start reconstruction: ${err.message}`)
    } finally {
      setStarting(false)
    }
  }

  function toggle(name) {
    setSelected(previous => previous.includes(name)
      ? previous.filter(x => x !== name)
      : previous.length < 12 ? [...previous, name] : previous)
  }

  return (
    <section className="bg-white rounded-2xl border border-gray-200 shadow-sm p-5 space-y-4">
      <div>
        <h2 className="text-base font-semibold text-gray-800">Step 3 — 3D Mesh Reconstruction</h2>
        <p className="text-xs text-gray-500 mt-1">Individual 3D Meshes — Multi-view reconstruction pending.</p>
        <p className="text-xs text-gray-500 mt-1">Each mesh is an independent prediction, not a unified scan or validated body measurement.</p>
      </div>
      <p className="text-sm text-gray-700">{selected.length} selected (maximum 12). Select frames to review or deselect them below.</p>
      {selected.length > 0 && blurry * 2 >= selected.length && (
        <div role="alert" className="rounded-lg bg-amber-50 border border-amber-200 p-3 text-xs text-amber-800">
          {blurry === selected.length ? 'All selected frames are blurry or have unknown blur status.' : 'Most selected frames are blurry or have unknown blur status.'} Results may be unreliable; consider recording again.
        </div>
      )}
      <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 gap-2">
        {frames.map(frame => {
          const checked = selected.includes(frame.filename)
          const pose = poses[stem(frame.filename)]
          return (
            <label key={frame.filename} className="rounded-lg border border-gray-200 p-2 text-xs text-gray-700">
              <img src={getFrameImageUrl(videoId, frame.filename)} alt={frame.filename} className="w-full h-24 object-contain bg-gray-100 rounded" loading="lazy" />
              <span className="flex items-center gap-1 mt-1">
                <input type="checkbox" checked={checked} disabled={busy || (!checked && selected.length >= 12)} onChange={() => toggle(frame.filename)} />
                {frame.filename}
              </span>
              <span className="block text-gray-500">{frame.timestamp_sec?.toFixed(2) ?? '—'}s · {frame.is_blurry === false ? 'Not flagged blurry' : 'Blur warning'} · {pose?.full_body_visible ? 'Full body visible' : 'Full body unconfirmed'}</span>
            </label>
          )
        })}
      </div>
      <button onClick={generate} disabled={busy || selected.length === 0}
        className="px-4 py-2 rounded-xl bg-teal-600 hover:bg-teal-700 text-white text-sm font-semibold disabled:bg-gray-300 disabled:cursor-not-allowed">
        {busy ? 'Generating meshes…' : 'Generate 3D Meshes'}
      </button>
      {error && <p role="alert" className="text-xs text-red-700">{error}</p>}
      {run && (
        <div className="space-y-3">
          <p className="text-sm text-gray-700">Run {run.run_id.slice(0, 8)}: {run.status.replaceAll('_', ' ')} · {run.frames.filter(f => ['completed', 'failed'].includes(f.status)).length}/{run.frames.length} processed</p>
          <ul className="text-xs space-y-1">
            {run.frames.map(f => (
              <li key={f.filename} className="flex flex-wrap gap-2">
                <span className="font-medium">{f.filename}</span>
                <span className={f.status === 'failed' ? 'text-red-700' : 'text-gray-600'}>{f.status}</span>
                {f.detail && <span className="text-red-700">{f.detail}</span>}
              </li>
            ))}
          </ul>
          {completed.length > 0 && (
            <div className="space-y-3">
              <label className="text-sm text-gray-700">Mesh
                <select className="ml-2 border rounded p-1" value={current?.filename || ''} onChange={e => { setActive(e.target.value); setViewerError('') }}>
                  {completed.map(f => <option key={f.filename} value={f.filename}>{f.filename}</option>)}
                </select>
              </label>
              <p className="text-xs text-gray-500">
                {current.filename} · viewpoint hint: {currentView?.viewpoint || 'unknown'} · heuristic score: {currentView?.confidence ?? 'unavailable'} · {current.metadata?.vertices ?? '—'} vertices · {current.metadata?.faces ?? '—'} faces
              </p>
              {currentView?.reason && <p className="text-xs text-amber-700">Unvalidated viewpoint hint: {currentView.reason.replaceAll('_', ' ')}</p>}
              <div className="grid grid-cols-1 lg:grid-cols-2 gap-3">
                <img src={meshAssetUrl(run.run_id, 'frame', current.filename)} alt={`Source ${current.filename}`} className="w-full rounded-lg border object-contain bg-gray-100" />
                <MeshViewer key={`${run.run_id}-${current.filename}`} meshUrl={meshAssetUrl(run.run_id, 'mesh', current.filename)} onError={setViewerError} />
              </div>
              {viewerError && <p role="alert" className="text-xs text-red-700">Mesh viewer: {viewerError}</p>}
            </div>
          )}
        </div>
      )}
    </section>
  )
}
