/**
 * UnifiedReconstructionPanel.jsx
 * ------------------------------
 * Simple end-to-end workflow for a processed 360° video:
 *   start → (confirm the person if several are detected) → progress →
 *   unified 3D body mesh with rotate / zoom / pan / reset → export.
 *
 * Consent is attested when the video is uploaded and re-verified by the server;
 * this panel never fakes or bypasses it.
 */
import { useCallback, useEffect, useRef, useState } from 'react'
import MeshViewer from './MeshViewer.jsx'
import {
  startReconstruction, getReconstruction, confirmPerson, reconstructionFileUrl,
} from '../services/reconstructionApi.js'

const POLL_MS = 1500

const STAGE_LABELS = {
  decode: 'Decoding video (orientation applied once)',
  detect: 'Detecting people',
  confirm: 'Waiting for subject confirmation',
  track: 'Tracking the subject',
  hmr2: 'HMR2 body inference',
  views: 'Estimating viewpoints',
  select: 'Selecting frames',
  masks: 'Segmenting the subject silhouette per view',
  fit: 'Fitting one shared body shape to keypoints and silhouettes',
  export: 'Exporting mesh and measuring circumferences',
  done: 'Done',
}

const ORDER = ['decode', 'detect', 'confirm', 'track', 'hmr2', 'views', 'select', 'masks', 'fit', 'export']

const MEASUREMENT_LABELS = {
  chest: 'Chest', waist: 'Waist', abdomen: 'Abdomen', hip: 'Hip',
  upper_arm_left: 'Upper arm (L)', upper_arm_right: 'Upper arm (R)', forearm_left: 'Forearm (L)',
  forearm_right: 'Forearm (R)', thigh_left: 'Thigh (L)', thigh_right: 'Thigh (R)', calf_left: 'Calf (L)', calf_right: 'Calf (R)',
}
const fmt = (v, d = 3) => (typeof v === 'number' && Number.isFinite(v) ? v.toFixed(d) : '—')

export default function UnifiedReconstructionPanel({ videoId }) {
  const [run, setRun] = useState(null)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState(null)
  const [meshInfo, setMeshInfo] = useState(null)
  const timer = useRef(null)

  const stop = useCallback(() => {
    if (timer.current) { clearInterval(timer.current); timer.current = null }
  }, [])

  const poll = useCallback((runId) => {
    stop()
    timer.current = setInterval(async () => {
      try {
        const r = await getReconstruction(runId)
        setRun(r)
        if (r.status === 'completed' || r.status === 'failed') stop()
      } catch (e) {
        setError(`Lost contact with the backend: ${e.message}`)
        stop()
      }
    }, POLL_MS)
  }, [stop])

  useEffect(() => stop, [stop])
  useEffect(() => { setRun(null); setError(null); setMeshInfo(null); stop() }, [videoId, stop])

  const start = async () => {
    setBusy(true); setError(null); setMeshInfo(null)
    try {
      const r = await startReconstruction(videoId)
      setRun(r)
      poll(r.run_id)
    } catch (e) {
      setError(e.message)
    } finally {
      setBusy(false)
    }
  }

  const choose = async (candidateId) => {
    setBusy(true); setError(null)
    try {
      setRun(await confirmPerson(run.run_id, candidateId))
      poll(run.run_id)
    } catch (e) {
      setError(e.message)
    } finally {
      setBusy(false)
    }
  }

  const running = run && (run.status === 'running' || run.status === 'awaiting_person_selection')
  const done = run?.status === 'completed'
  const result = run?.result
  const file = (name) => reconstructionFileUrl(run.run_id, name)

  return (
    <section className="bg-white rounded-2xl border border-teal-200 shadow-sm overflow-hidden" data-testid="unified-reconstruction">
      <div className="px-5 py-4 border-b border-gray-100">
        <h2 className="text-base font-semibold text-gray-800">Step 3 — Reconstruct one 3D body mesh</h2>
        <p className="text-xs text-gray-500 mt-0.5">
          Tracks the same person through the whole 360° video, estimates one shared body shape from all views and
          exports a single upright mesh. External body geometry only — not a Tanita BIA/BCM measurement.
        </p>
      </div>

      <div className="p-5 space-y-4">
        {!running && !done && (
          <button
            onClick={start}
            disabled={busy || !videoId}
            className="px-4 py-2 rounded-xl bg-teal-600 hover:bg-teal-700 disabled:bg-gray-300 text-white text-sm font-medium"
          >
            {run?.status === 'failed' ? 'Try again' : 'Reconstruct 3D body'}
          </button>
        )}

        {error && <p role="alert" className="text-sm text-red-700">{error}</p>}

        {run?.status === 'failed' && (
          <div role="alert" className="rounded-lg border border-red-200 bg-red-50 p-3 text-sm text-red-800">
            <p className="font-medium">Reconstruction stopped</p>
            <p className="mt-1">{run.error?.message}</p>
          </div>
        )}

        {running && run.status !== 'awaiting_person_selection' && (
          <div aria-live="polite">
            <div className="flex justify-between text-xs text-gray-600 mb-1">
              <span>{STAGE_LABELS[run.stage] || run.stage}</span>
              <span>{Math.round((run.progress || 0) * 100)}%</span>
            </div>
            <div className="h-2 rounded bg-gray-100 overflow-hidden">
              <div className="h-full bg-teal-500 transition-all" style={{ width: `${(run.progress || 0) * 100}%` }} />
            </div>
            <p className="mt-1 text-xs text-gray-500">{run.message}</p>
            <ol className="mt-2 flex flex-wrap gap-1 text-[10px] text-gray-500">
              {ORDER.map((s) => (
                <li key={s} className={`px-2 py-0.5 rounded-full border ${
                  s === run.stage ? 'bg-teal-50 border-teal-400 text-teal-700'
                    : ORDER.indexOf(s) < ORDER.indexOf(run.stage) ? 'bg-gray-50 border-gray-200 line-through' : 'border-gray-200'}`}>
                  {s}
                </li>
              ))}
            </ol>
          </div>
        )}

        {run?.status === 'awaiting_person_selection' && (
          <div>
            <p className="text-sm font-medium text-gray-800">{run.message}</p>
            <p className="text-xs text-gray-500 mb-2">Choose the person to reconstruct. Only this person will be tracked.</p>
            <img src={file('candidates_overview.jpg')} alt="Detected people, numbered" className="max-h-72 rounded border mb-3" />
            <div className="flex flex-wrap gap-3">
              {run.candidates.map((c) => (
                <button key={c.id} disabled={busy} onClick={() => choose(c.id)}
                  className="border rounded-lg p-2 hover:border-teal-500 text-xs text-left">
                  <img src={file(c.thumbnail)} alt={c.label} className="h-40 rounded mb-1" />
                  <span className="font-medium">{c.label}</span> — this is the subject
                </button>
              ))}
            </div>
          </div>
        )}

        {run?.warnings?.length > 0 && (
          <ul className="space-y-1" aria-label="Reconstruction warnings">
            {run.warnings.map((w, i) => (
              <li key={i} className="text-xs rounded border border-amber-300 bg-amber-50 text-amber-900 p-2">
                <span className="font-medium">{w.code.replace(/_/g, ' ')}:</span> {w.message}
              </li>
            ))}
          </ul>
        )}

        {done && result && (
          <div className="space-y-3">
            <MeshViewer meshUrl={file(result.mesh_obj)} upright onLoad={setMeshInfo}
              onError={(m) => setError(`Mesh viewer could not load the result: ${m}`)} />
            <dl className="grid grid-cols-2 sm:grid-cols-4 gap-2 text-xs">
              {/* OBJLoader de-indexes geometry, so its position count is 3x faces; report the file's vertex count. */}
              <Stat label="Vertices" value={result.vertex_count} />
              <Stat label="Faces" value={result.face_count} />
              <Stat label="Loaded in viewer" value={meshInfo ? `Yes (${meshInfo.faces} faces)` : 'Loading…'} />
              <Stat label="Frames used" value={`${result.frames_selected} of ${result.frames_tracked} tracked`} />
              <Stat label="Same person throughout" value={result.consistent_single_subject ? 'Yes' : 'Check warnings'} />
              <Stat label="Geometry valid" value={result.geometry_valid ? 'Yes' : 'No'} />
              <Stat label="View coverage" value={result.coverage_sufficient ? 'Front, back, both sides' : 'Incomplete'} />
              <Stat label="Scale" value="Relative / estimated (not metric)" />
              {result.silhouette_views != null && <Stat label="Silhouettes used" value={`${result.silhouette_views} of ${result.frames_selected} views`} />}
              {result.silhouette_iou_fit != null && <Stat label="Silhouette IoU (fitted views)" value={fmt(result.silhouette_iou_fit)} />}
              {result.held_out_iou != null && <Stat label="Silhouette IoU (held-out, cross-validated)"
                value={`${fmt(result.held_out_iou)} (previous method ${fmt(result.held_out_iou_previous_method)})`} />}
              {result.keypoint_error_pct_height != null && <Stat label="Landmark reprojection error" value={`${fmt(result.keypoint_error_pct_height, 2)}% of height`} />}
            </dl>
            {result.comparison && (
              <figure className="space-y-1">
                <figcaption className="text-xs text-gray-600">
                  Front / left / back / right: original frame · observed silhouette (red) vs fitted shared-shape mesh (green) · rendered mesh.
                  “Held-out IoU” comes from a shape fitted without that view.
                </figcaption>
                <img src={file(result.comparison)} alt="Original views compared with the fitted mesh" className="w-full rounded border" loading="lazy" />
              </figure>
            )}
            {result.turntable && (
              <figure className="flex flex-wrap gap-3 items-start">
                <img src={file(result.turntable)} alt="Rotating preview of the final mesh" className="h-64 rounded border" loading="lazy" />
                {result.measurements_overlay && <img src={file(result.measurements_overlay)} alt="Measurement locations on the mesh" className="h-64 rounded border" loading="lazy" />}
              </figure>
            )}
            {result.measurements && (
              <div className="text-xs space-y-1">
                <p className="font-medium text-gray-800">Circumferences from mesh cross-sections ({result.measurements_valid} valid)</p>
                <p className="text-amber-800">Relative values: tape circumference ÷ mesh height. No calibration or known-size reference exists, so these are not centimetres, and they follow the clothed outline.</p>
                <table className="min-w-[16rem] border-collapse">
                  <tbody>
                    {Object.entries(result.measurements).map(([key, value]) => (
                      <tr key={key} className="border-b border-gray-100">
                        <td className="pr-4 py-0.5 text-gray-600">{MEASUREMENT_LABELS[key] || key}</td>
                        <td className="py-0.5 font-mono">{value == null ? 'invalid section' : `${fmt(value)} × height`}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
            <div className="flex flex-wrap gap-2">
              <a className="px-3 py-2 rounded-lg bg-teal-600 text-white text-sm" href={file('final_body_mesh.obj')} download="final_body_mesh.obj">Export OBJ</a>
              <a className="px-3 py-2 rounded-lg border text-sm" href={file('final_body_mesh.ply')} download="final_body_mesh.ply">PLY</a>
              <a className="px-3 py-2 rounded-lg border text-sm" href={file('reconstruction_report.json')} target="_blank" rel="noreferrer">Report</a>
              <a className="px-3 py-2 rounded-lg border text-sm" href={file('selected_frames.json')} target="_blank" rel="noreferrer">Selected frames</a>
              <a className="px-3 py-2 rounded-lg border text-sm" href={file('tracking_report.json')} target="_blank" rel="noreferrer">Tracking report</a>
              {result.measurements && <a className="px-3 py-2 rounded-lg border text-sm" href={file('measurements.json')} target="_blank" rel="noreferrer">Measurements</a>}
              {result.measurements && <a className="px-3 py-2 rounded-lg border text-sm" href={file('measurement_rings.obj')} download="measurement_rings.obj">Measurement rings (OBJ)</a>}
              <button onClick={start} disabled={busy} className="px-3 py-2 rounded-lg border text-sm">Run again</button>
            </div>
            <p className="text-xs text-gray-500">
              The mesh is shown in a standard measurement A-pose. Its size is a relative SMPL estimate: no camera calibration
              or known-size reference was available, so no height or circumference is reported in centimetres, and nothing
              here measures body fat, muscle or other internal composition.
            </p>
          </div>
        )}
      </div>
    </section>
  )
}

function Stat({ label, value }) {
  return (
    <div className="rounded border border-gray-200 p-2">
      <dt className="text-gray-500">{label}</dt>
      <dd className="font-medium text-gray-800">{String(value)}</dd>
    </div>
  )
}
