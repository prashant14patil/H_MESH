/**
 * MeasurementModels.jsx
 * ---------------------
 * Model switch for a completed reconstruction: both models as Body Avatars side by side (the default), SHAPY, or
 * Pipeline (HMR2 + multi-view fit). The Pipeline view is rendered by the parent; this component renders the others.
 */
import { useEffect, useState } from 'react'
import MeshViewer from './MeshViewer.jsx'
import ModelAvatarCompare from './ModelAvatarCompare.jsx'
import { computeShapy, shapyMeshUrl } from '../services/shapyApi.js'
import { VIEWS, compareRows } from './measurementModels.js'

const fmt = (v, d) => (typeof v === 'number' && Number.isFinite(v) ? v.toFixed(d) : '—')

export function ModelSwitch({ view, onChange }) {
  return (
    <div role="tablist" aria-label="Measurement model" className="inline-flex rounded-lg border border-gray-300 overflow-hidden text-xs">
      {VIEWS.map(([id, label]) => (
        <button key={id} role="tab" aria-selected={view === id} onClick={() => onChange(id)}
          className={`px-3 py-1.5 ${view === id ? 'bg-teal-600 text-white' : 'bg-white text-gray-700 hover:bg-gray-50'}`}>
          {label}
        </button>
      ))}
    </div>
  )
}

/**
 * view: 'shapy' | 'compare'; pipeline: { measurements, measurements_cm? } from the run;
 * heightCm: current run height (re-fetches so SHAPY cm follow it); labels: site -> label.
 * The compare view loads the pipeline body itself, so pipelineMeshUrl is no longer needed.
 */
export function ShapyView({ runId, view, pipeline, heightCm, labels, onError }) {
  const [shapy, setShapy] = useState(null)
  const [loading, setLoading] = useState(false)
  const [failure, setFailure] = useState(null)

  useEffect(() => {
    let cancelled = false
    setLoading(true); setFailure(null)
    computeShapy(runId)
      .then((s) => { if (!cancelled) setShapy(s) })
      .catch((e) => { if (!cancelled) setFailure(e.message) })
      .finally(() => { if (!cancelled) setLoading(false) })
    return () => { cancelled = true }
  }, [runId, heightCm])

  if (failure) {
    return <p role="alert" className="text-sm text-red-700">SHAPY is not available for this run: {failure}</p>
  }
  if (!shapy) {
    return (
      <p className="text-xs text-gray-600" aria-live="polite">
        {loading ? 'Running SHAPY on the selected frames (about 30 s the first time; stored afterwards)…' : ''}
      </p>
    )
  }

  const sites = Object.keys(pipeline?.measurements || shapy.measurements || {})
  const { unit, rows } = compareRows(pipeline, shapy, sites)
  const d = unit === 'cm' ? 1 : 3
  const disagreeing = rows.filter((r) => r.disagree).length

  return (
    <div className="space-y-3 text-xs">
      {view === 'compare' ? (
        <ModelAvatarCompare runId={runId} shapy={shapy} />
      ) : (
        <figure className="space-y-1">
          <figcaption className="font-medium text-gray-700">SHAPY ({shapy.frames_used} frames)</figcaption>
          <MeshViewer meshUrl={shapyMeshUrl(runId)} upright onError={(m) => onError?.(`SHAPY mesh: ${m}`)} />
        </figure>
      )}

      <p className="font-medium text-gray-800">
        {view === 'compare' ? 'Pipeline vs SHAPY' : 'SHAPY predicted measurements'} ({unit}{unit === 'cm' ? `, from ${fmt(heightCm, 1)} cm height` : ''})
      </p>
      {unit !== 'cm' && (
        <p className="text-amber-800">Enter the subject’s height below to see centimetres.</p>
      )}
      <table className="min-w-[16rem] border-collapse">
        <thead>
          <tr className="text-left text-gray-500">
            <th className="pr-4 py-0.5 font-normal">Measurement</th>
            {view === 'compare' && <th className="pr-4 py-0.5 font-normal">Pipeline</th>}
            <th className="pr-4 py-0.5 font-normal">SHAPY</th>
            {view === 'compare' && <th className="py-0.5 font-normal">SHAPY − Pipeline</th>}
          </tr>
        </thead>
        <tbody>
          {rows.map((r) => (
            <tr key={r.site} className={`border-b border-gray-100 ${r.disagree && view === 'compare' ? 'bg-amber-50' : ''}`}>
              <td className="pr-4 py-0.5 text-gray-600">{labels[r.site] || r.site}</td>
              {view === 'compare' && <td className="pr-4 py-0.5 font-mono">{fmt(r.pipeline, d)}</td>}
              <td className="pr-4 py-0.5 font-mono font-medium text-gray-900">{fmt(r.shapy, d)}</td>
              {view === 'compare' && (
                <td className={`py-0.5 font-mono ${r.disagree ? 'text-amber-800 font-medium' : ''}`}>
                  {r.diff == null ? '—' : `${r.diff > 0 ? '+' : ''}${fmt(r.diff, d)}`}
                </td>
              )}
            </tr>
          ))}
        </tbody>
      </table>
      {view === 'compare' && (
        <p className="text-gray-600">
          {disagreeing === 0
            ? 'The two models agree at every site (within 3 cm).'
            : `${disagreeing} site(s) differ by more than 3 cm (highlighted). Agreement is not proof of accuracy; only a tape measure is.`}
        </p>
      )}
      <p className="text-gray-500">
        Both models use the same measurement A-pose and circumference definitions; only the body shape differs.
        SHAPY is a network trained on photos of people paired with their tape measurements; it
        estimates shape from each frame and its mirror image and takes the robust median, which made repeated
        recordings of the same person agree within about 0.5 cm, but like any learned estimate it pulls toward an
        average body. The pipeline fits one shape to the person’s outline in every view: specific to this person,
        but loose clothing reads as body. In the first tape check (loose clothing) SHAPY was within a few cm and the
        pipeline 4–14 cm too large, so SHAPY is shown first.
      </p>
    </div>
  )
}
