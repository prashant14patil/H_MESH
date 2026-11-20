/**
 * ModelAvatarCompare.jsx
 * ----------------------
 * SHAPY and the pipeline as two Body Avatars: side by side on wide screens, stacked on phones, turning together,
 * each with its own girth cards. When the run has a tape check, both models are scored against the same tape:
 * sites within ±2 cm get the "tape-checked" colour on that body, and a table lists every taped site.
 */
import { useEffect, useMemo, useState } from 'react'
import segmentsJson from './smplSegments.json'
import { Stage } from './BodyAvatarPanel.jsx'
import {
  CALLOUTS, SITE_FULL_LABELS, STATUS, TAPE_TOLERANCE_CM, decodeLabels, formatValue, meanAbsError, parseObj,
  parseRings, scanValues, segmentStatus, siteStatus, tapeErrors, transferLabels,
} from './bodyAvatarModel.js'
import { getReconstruction, reconstructionFileUrl } from '../services/reconstructionApi.js'
import { shapyMeshUrl, shapyRingsUrl } from '../services/shapyApi.js'

const LABELS = decodeLabels(segmentsJson)
const TURNS = [['front', 'Front', 0], ['side', 'Side', Math.PI / 2], ['back', 'Back', Math.PI]]
const SPIN_STEP = 0.045 // radians every 100 ms: one turn in about 14 s; the avatar eases between targets

async function fetchText(url) {
  const res = await fetch(url)
  if (!res.ok) throw new Error(`HTTP ${res.status}`)
  return res.text()
}

/** Mesh + measurement rings of one model. Rings are optional: without them the cards still show, without dots. */
function useBody(meshUrl, ringsUrl, enabled) {
  const [body, setBody] = useState({ mesh: null, rings: {}, error: null })
  useEffect(() => {
    if (!enabled) return undefined
    let alive = true
    setBody({ mesh: null, rings: {}, error: null })
    Promise.all([fetchText(meshUrl), fetchText(ringsUrl).catch(() => '')])
      .then(([obj, rings]) => alive && setBody({ mesh: parseObj(obj), rings: parseRings(rings), error: null }))
      .catch((e) => alive && setBody({ mesh: null, rings: {}, error: e.message }))
    return () => { alive = false }
  }, [meshUrl, ringsUrl, enabled])
  return body
}

function modelStatus(scan, errs) {
  const statusOf = (site) => siteStatus(site, scan.values, errs)
  return { statusOf, segStatus: Object.fromEntries(CALLOUTS.map((c) => [c.segment, segmentStatus(c, statusOf)])) }
}

/** shapy: the SHAPY API result for this run ({ measurements, measurements_cm?, height_cm?, computed_at }). */
export default function ModelAvatarCompare({ runId, shapy }) {
  const [run, setRun] = useState(null)
  const [runError, setRunError] = useState(null)
  const [yawTarget, setYawTarget] = useState(0)
  const [turn, setTurn] = useState('front')
  const [mirror, setMirror] = useState(true)
  const [spin, setSpin] = useState(true)

  useEffect(() => {
    let alive = true
    getReconstruction(runId).then((r) => alive && setRun(r)).catch((e) => alive && setRunError(e.message))
    return () => { alive = false }
  }, [runId])

  useEffect(() => {
    if (!spin) return undefined
    const id = setInterval(() => setYawTarget((y) => y + SPIN_STEP), 100)
    return () => clearInterval(id)
  }, [spin])

  const result = run?.result
  // SHAPY files are rewritten when it recomputes; the version query keeps the browser from reusing old ones.
  const v = shapy?.computed_at ? `?v=${encodeURIComponent(shapy.computed_at)}` : ''
  const pipeBody = useBody(reconstructionFileUrl(runId, result?.mesh_obj || 'final_body_mesh.obj'),
    reconstructionFileUrl(runId, 'measurement_rings.obj'), Boolean(result))
  const shapyBody = useBody(shapyMeshUrl(runId) + v, shapyRingsUrl(runId) + v, Boolean(shapy))

  // SHAPY's body is SMPL-X (another vertex layout): borrow the segment colours from the pipeline body.
  const shapyLabels = useMemo(() => (shapyBody.mesh && pipeBody.mesh?.vertexCount === LABELS.length
    ? transferLabels(shapyBody.mesh, pipeBody.mesh, LABELS) : null), [shapyBody.mesh, pipeBody.mesh])

  const pipeScan = useMemo(() => (result ? scanValues(result) : null), [result])
  const shapyScan = useMemo(() => (shapy ? scanValues(shapy) : null), [shapy])
  const tape = result?.tape_validation && !result.tape_validation.excluded ? result.tape_validation.sites : null
  const pipeErrs = useMemo(() => (tape && pipeScan?.unit === 'cm' ? tapeErrors(pipeScan.values, tape) : null), [tape, pipeScan])
  const shapyErrs = useMemo(() => (tape && shapyScan?.unit === 'cm' ? tapeErrors(shapyScan.values, tape) : null), [tape, shapyScan])
  const pipeStatus = useMemo(() => (pipeScan ? modelStatus(pipeScan, pipeErrs) : null), [pipeScan, pipeErrs])
  const shapyStatus = useMemo(() => (shapyScan ? modelStatus(shapyScan, shapyErrs) : null), [shapyScan, shapyErrs])
  const common = pipeErrs && shapyErrs ? Object.keys(pipeErrs).filter((s) => s in shapyErrs) : []
  const pipeMae = meanAbsError(pipeErrs, common)
  const shapyMae = meanAbsError(shapyErrs, common)

  const turnTo = (key) => {
    const target = TURNS.find((t) => t[0] === key)[2]
    setSpin(false)
    setTurn(key)
    setYawTarget((prev) => prev + (((target - prev) % (2 * Math.PI)) + 3 * Math.PI) % (2 * Math.PI) - Math.PI)
  }
  const onDragTurn = (dy) => { setSpin(false); setTurn(null); setYawTarget((y) => y + dy) }

  if (runError) return <p role="alert" className="text-sm text-red-700">Could not load this run: {runError}</p>
  if (!result) return <p className="text-xs text-gray-600" aria-live="polite">Loading both bodies…</p>

  const columns = [
    { key: 'shapy', title: 'SHAPY', note: 'learned body shape', body: shapyBody, scan: shapyScan, status: shapyStatus,
      mae: shapyMae, labels: shapyLabels },
    { key: 'pipeline', title: 'Pipeline', note: 'fitted to your outline in every view', body: pipeBody, scan: pipeScan,
      status: pipeStatus, mae: pipeMae, labels: LABELS },
  ]

  return (
    <div className="space-y-3 text-xs" data-testid="model-avatar-compare">
      <div className="flex flex-wrap items-center gap-2">
        <div role="group" aria-label="Turn both bodies" className="inline-flex overflow-hidden rounded-lg border border-gray-300">
          {TURNS.map(([k, label]) => (
            <button key={k} type="button" onClick={() => turnTo(k)} aria-pressed={turn === k}
              className={`px-3 py-1.5 ${turn === k ? 'bg-teal-600 text-white' : 'bg-white text-gray-700 hover:bg-gray-50'}`}>
              {label}
            </button>
          ))}
        </div>
        <button type="button" onClick={() => setSpin((s) => !s)} aria-pressed={spin}
          className="rounded-lg border border-gray-300 bg-white px-3 py-1.5 text-gray-700 hover:bg-gray-50">
          {spin ? 'Pause turning' : 'Turn both'}
        </button>
        {pipeMae && shapyMae && (
          <span className="text-gray-700">
            Against your tape ({common.length} site{common.length === 1 ? '' : 's'}): SHAPY {shapyMae.cm.toFixed(1)} cm,
            pipeline {pipeMae.cm.toFixed(1)} cm mean error
          </span>
        )}
      </div>

      <div className="grid grid-cols-1 gap-4 sm:grid-cols-2">
        {columns.map((c) => (
          <section key={c.key} aria-label={`${c.title} body`} className="min-w-0">
            <h4 className="text-sm font-semibold text-gray-800">
              {c.title} <span className="font-normal text-gray-500">· {c.note}</span>
              {c.mae && <span className="font-normal text-gray-700"> · tape error {c.mae.cm.toFixed(1)} cm</span>}
            </h4>
            {c.scan && c.status ? (
              <Stage mesh={c.body.mesh} compareMesh={null} t={1} rings={c.body.rings} yawTarget={yawTarget}
                mirror={mirror} segStatus={c.status.segStatus} statusOf={c.status.statusOf} values={c.scan.values}
                unit={c.scan.unit} scan={c.scan} cmp={null} heightCm={c.scan.heightCm} labels={c.labels}
                error={c.body.error ? `${c.title} body could not load: ${c.body.error}` : null}
                onToggleMirror={() => setMirror((m) => !m)} onDragTurn={onDragTurn} />
            ) : <p className="mt-4 text-gray-600">Waiting for {c.title}…</p>}
          </section>
        ))}
      </div>

      {tape ? <TapeTable tape={tape} shapyErrs={shapyErrs} pipeErrs={pipeErrs} /> : (
        <p className="text-gray-500">
          Record a tape check for this run (“Accuracy check against a tape measure”) and each body marks the sites
          within ±{TAPE_TOLERANCE_CM} cm of the tape in the {STATUS.tape_ok.label.split(' (')[0].toLowerCase()} colour, so
          the closer model shows at a glance.
        </p>
      )}
    </div>
  )
}

function TapeTable({ tape, shapyErrs, pipeErrs }) {
  const sites = Object.keys(tape).filter((s) => Number.isFinite(tape[s]?.tape_cm))
  if (!sites.length) return null
  const cell = (e) => (e ? `${formatValue(e.model_cm, 'cm')} (${e.error_cm > 0 ? '+' : ''}${e.error_cm.toFixed(1)})` : '—')
  const closer = (s) => {
    const a = shapyErrs?.[s]?.error_cm, b = pipeErrs?.[s]?.error_cm
    if (!Number.isFinite(a) || !Number.isFinite(b) || Math.abs(a) === Math.abs(b)) return null
    return Math.abs(a) < Math.abs(b) ? 'shapy' : 'pipeline'
  }
  return (
    <table className="min-w-[18rem] border-collapse" aria-label="Both models against the tape">
      <thead>
        <tr className="text-left text-gray-500">
          <th className="pr-4 py-0.5 font-normal">Tape check</th>
          <th className="pr-4 py-0.5 font-normal">Tape</th>
          <th className="pr-4 py-0.5 font-normal">SHAPY (error)</th>
          <th className="py-0.5 font-normal">Pipeline (error)</th>
        </tr>
      </thead>
      <tbody>
        {sites.map((s) => (
          <tr key={s} className="border-b border-gray-100">
            <td className="pr-4 py-0.5 text-gray-600">{SITE_FULL_LABELS[s] || s}</td>
            <td className="pr-4 py-0.5 font-mono">{formatValue(tape[s].tape_cm, 'cm')}</td>
            <td className={`pr-4 py-0.5 font-mono ${closer(s) === 'shapy' ? 'font-semibold text-teal-700' : ''}`}>{cell(shapyErrs?.[s])}</td>
            <td className={`py-0.5 font-mono ${closer(s) === 'pipeline' ? 'font-semibold text-teal-700' : ''}`}>{cell(pipeErrs?.[s])}</td>
          </tr>
        ))}
      </tbody>
    </table>
  )
}
