/**
 * AssessmentsWorkspace.jsx
 * ------------------------
 * Assessment page: one assessment = one completed 360° camera reconstruction + optional BCM data, sharing the same
 * assessment ID and person code. Lists the history, shows the avatar and camera measurements (always available),
 * the BCM section (optional), both side by side per body segment, and comparison between assessments.
 *
 * Camera-derived values are tagged CAMERA (teal), BCM values BCM (indigo); DEMO DATA is amber and labelled.
 */
import { useCallback, useEffect, useMemo, useState } from 'react'
import BodyAvatarPanel from './BodyAvatarPanel.jsx'
import BcmSection, { Card, DeviceIntegrationCard, SourceTag } from './BcmSection.jsx'
import { SITE_FULL_LABELS, uncertaintyCm } from './bodyAvatarModel.js'
import { segmentRows, hasSegmental, formatChange, formatWhen, sourceBadge } from './bcmModel.js'
import {
  listAssessments, listPersons, getAssessment, setAssessmentPerson, compareAssessments, personHistory, getBcmSchema,
} from '../services/assessmentApi.js'
import { getReconstruction } from '../services/reconstructionApi.js'

const UNASSIGNED = '__unassigned__'

export default function AssessmentsWorkspace() {
  const [rows, setRows] = useState(null)
  const [persons, setPersons] = useState([])
  const [filter, setFilter] = useState('')
  const [selectedId, setSelectedId] = useState(null)
  const [error, setError] = useState(null)

  const refreshList = useCallback(async () => {
    try {
      const [a, p] = await Promise.all([listAssessments(), listPersons()])
      setRows(a); setPersons(p); setError(null)
      setSelectedId((id) => id ?? a[0]?.assessment_id ?? null)
    } catch (e) { setError(e.message); setRows([]) }
  }, [])
  useEffect(() => { refreshList() }, [refreshList])

  const visible = useMemo(() => (rows || []).filter((r) => !filter || (filter === UNASSIGNED ? !r.person_id : r.person_id === filter)), [rows, filter])

  return (
    <div className="grid grid-cols-1 gap-6 lg:grid-cols-[300px_1fr]">
      <aside className="space-y-4">
        <Card>
          <div className="flex items-center justify-between">
            <h2 className="text-sm font-semibold text-slate-900">Assessments</h2>
            <button type="button" onClick={refreshList} className="text-xs text-slate-500 hover:text-slate-800">Refresh</button>
          </div>
          <label className="mt-3 block text-xs text-slate-600">Person
            <select value={filter} onChange={(e) => setFilter(e.target.value)}
              className="mt-1 w-full rounded-lg border border-slate-200 bg-white px-2 py-1.5 text-sm">
              <option value="">All people</option>
              {persons.map((p) => <option key={p.person_id} value={p.person_id}>{p.person_id} ({p.assessments})</option>)}
              <option value={UNASSIGNED}>Not assigned</option>
            </select>
          </label>
          {error && <p className="mt-3 text-xs text-red-600">{error}</p>}
          {rows === null && <p className="mt-3 text-xs text-slate-400">Loading…</p>}
          {rows?.length === 0 && !error && (
            <p className="mt-3 text-xs text-slate-500">No completed 360° reconstructions yet. Run one in the “360° Video Capture” tab; it appears here as an assessment.</p>
          )}
          <ul className="mt-3 max-h-[60vh] space-y-1 overflow-y-auto" data-testid="assessment-list">
            {visible.map((r) => (
              <li key={r.assessment_id}>
                <button type="button" onClick={() => setSelectedId(r.assessment_id)}
                  className={`w-full rounded-lg px-3 py-2 text-left text-sm transition ${selectedId === r.assessment_id ? 'bg-teal-50 ring-1 ring-teal-200' : 'hover:bg-slate-50'}`}>
                  <div className="flex items-center justify-between gap-2">
                    <span className="font-medium text-slate-800">{formatWhen(r.assessed_at)}</span>
                    <BcmDot bcm={r.bcm} />
                  </div>
                  <div className="text-xs text-slate-500">
                    {r.person_id || 'No person'} · {r.assessment_id.slice(0, 8)} · {r.camera_unit === 'cm' ? `${r.height_cm} cm` : 'unscaled'}
                  </div>
                </button>
              </li>
            ))}
          </ul>
        </Card>
        <DeviceIntegrationCard />
      </aside>

      <div className="min-w-0">
        {selectedId
          ? <AssessmentDetail key={selectedId} id={selectedId} onChanged={refreshList} all={rows || []} />
          : rows?.length > 0 && <Card><p className="text-sm text-slate-500">Select an assessment.</p></Card>}
      </div>
    </div>
  )
}

function BcmDot({ bcm }) {
  if (!bcm?.available) return <span className="text-[10px] text-slate-400">camera only</span>
  if (bcm.is_demo) return <span className="rounded bg-amber-100 px-1.5 text-[10px] font-semibold text-amber-800">BCM · DEMO</span>
  return <span className="rounded bg-indigo-50 px-1.5 text-[10px] font-semibold text-indigo-700">BCM</span>
}

function AssessmentDetail({ id, onChanged, all }) {
  const [a, setA] = useState(null)
  const [run, setRun] = useState(null)
  const [error, setError] = useState(null)

  useEffect(() => {
    let alive = true
    Promise.all([getAssessment(id), getReconstruction(id)])
      .then(([asmt, r]) => { if (alive) { setA(asmt); setRun(r) } })
      .catch((e) => alive && setError(e.message))
    return () => { alive = false }
  }, [id])

  if (error) return <Card><p className="text-sm text-red-600">{error}</p></Card>
  if (!a) return <Card><p className="text-sm text-slate-400">Loading assessment…</p></Card>

  const onBcm = (rec) => { setA((x) => ({ ...x, bcm: rec })); onChanged() }
  const onPerson = (asmt) => { setA(asmt); onChanged() }

  return (
    <div className="space-y-5">
      <AssessmentHeader a={a} onPerson={onPerson} />
      {run?.result && <BodyAvatarPanel run={run} />}
      <div className="grid grid-cols-1 gap-5 xl:grid-cols-2">
        <CameraMeasurements camera={a.camera} />
        <BcmSection assessmentId={a.assessment_id} record={a.bcm} onChange={onBcm} />
      </div>
      <SideBySide camera={a.camera} bcm={a.bcm} />
      <HistoryCompare a={a} all={all} />
    </div>
  )
}

function AssessmentHeader({ a, onPerson }) {
  const [editing, setEditing] = useState(false)
  const [value, setValue] = useState(a.person_id || '')
  const [err, setErr] = useState(null)
  const save = async () => {
    setErr(null)
    try { onPerson(await setAssessmentPerson(a.assessment_id, value.trim())); setEditing(false) } catch (e) { setErr(e.message) }
  }
  const q = a.camera.quality
  return (
    <Card>
      <div className="flex flex-wrap items-start justify-between gap-4">
        <div>
          <p className="text-[11px] uppercase tracking-wider text-slate-400">Assessment</p>
          <h2 className="text-lg font-semibold text-slate-900">{formatWhen(a.assessed_at)}</h2>
          <p className="font-mono text-xs text-slate-400">{a.assessment_id}</p>
        </div>
        <div className="text-sm">
          <p className="text-[11px] uppercase tracking-wider text-slate-400">Person code</p>
          {editing ? (
            <div className="mt-1 flex gap-1">
              <input value={value} onChange={(e) => setValue(e.target.value)} maxLength={40} placeholder="S01" autoFocus
                className="w-28 rounded-lg border border-slate-200 px-2 py-1 text-sm" />
              <button type="button" onClick={save} disabled={!value.trim()} className="rounded-lg bg-slate-800 px-2 text-xs text-white disabled:bg-slate-300">Save</button>
              <button type="button" onClick={() => setEditing(false)} className="px-1 text-xs text-slate-500">Cancel</button>
            </div>
          ) : (
            <button type="button" onClick={() => setEditing(true)} className="mt-1 font-medium text-slate-800 hover:text-teal-700">
              {a.person_id || <span className="text-slate-400">Assign…</span>} <span className="text-xs text-slate-400">edit</span>
            </button>
          )}
          {err && <p className="mt-1 max-w-xs text-xs text-red-600">{err}</p>}
          <p className="mt-1 text-[11px] text-slate-400">A code, not a name. Links assessments of one person.</p>
        </div>
        <div className="flex flex-wrap gap-2 text-xs">
          <Chip label="Camera" ok value="available" />
          <Chip label="BCM" ok={!!a.bcm} value={a.bcm ? (a.bcm.source.is_demo ? 'DEMO DATA' : 'available') : 'not added'} warn={a.bcm?.source?.is_demo} />
          <Chip label="Scale" ok={a.camera.unit === 'cm'} value={a.camera.unit === 'cm' ? `${a.camera.height_cm} cm height` : 'relative only'} />
          {Number.isFinite(q.held_out_iou) && <Chip label="Held-out IoU" ok={q.held_out_iou >= 0.8} value={q.held_out_iou.toFixed(2)} />}
        </div>
      </div>
    </Card>
  )
}

function Chip({ label, value, ok, warn }) {
  const tone = warn ? 'bg-amber-50 text-amber-800 ring-amber-200' : ok ? 'bg-slate-50 text-slate-700 ring-slate-200' : 'bg-slate-50 text-slate-400 ring-slate-200'
  return <span className={`rounded-lg px-2 py-1 ring-1 ${tone}`}><span className="text-slate-400">{label}: </span>{value}</span>
}

function CameraMeasurements({ camera }) {
  const cm = camera.unit === 'cm'
  const values = cm ? camera.measurements_cm : camera.measurements_ratio
  const q = camera.quality
  return (
    <Card>
      <div className="flex items-center justify-between gap-2">
        <div className="flex items-center gap-2">
          <h3 className="text-base font-semibold text-slate-900">Body measurements</h3>
          <SourceTag kind="camera" />
        </div>
        <span className="text-xs text-slate-500">{cm ? `cm · height ${camera.height_cm} cm (supplied)` : '× height (no height supplied)'}</span>
      </div>
      <table className="mt-3 w-full text-sm">
        <thead><tr className="text-left text-[11px] uppercase tracking-wide text-slate-500">
          <th className="py-1 font-medium">Girth</th><th className="py-1 text-right font-medium">Value</th>
          {cm && <th className="py-1 text-right font-medium">± (est.)</th>}<th className="py-1 text-right font-medium">Status</th>
        </tr></thead>
        <tbody>
          {Object.entries(values || {}).map(([site, v]) => {
            const valid = camera.site_valid?.[site] !== false && Number.isFinite(v)
            const unc = cm ? uncertaintyCm(site, v, camera.relative_scale_uncertainty) : null
            return (
              <tr key={site} className="border-t border-slate-100">
                <td className="py-1 text-slate-700">{SITE_FULL_LABELS[site] || site}</td>
                <td className="py-1 text-right tabular-nums text-slate-900">{Number.isFinite(v) ? v.toFixed(cm ? 1 : 3) : '—'}</td>
                {cm && <td className="py-1 text-right tabular-nums text-slate-500">{unc ? unc.toFixed(1) : '—'}</td>}
                <td className={`py-1 text-right text-xs ${valid ? 'text-emerald-600' : 'text-orange-600'}`}>{valid ? 'valid' : 'unreliable'}</td>
              </tr>
            )
          })}
        </tbody>
      </table>
      <div className="mt-3 grid grid-cols-2 gap-2 text-xs text-slate-500 sm:grid-cols-4">
        <QualityStat k="Frames used" v={q.frames_selected} />
        <QualityStat k="Fit IoU" v={Number.isFinite(q.silhouette_iou_fit) ? q.silhouette_iou_fit.toFixed(2) : '—'} />
        <QualityStat k="Held-out IoU" v={Number.isFinite(q.held_out_iou) ? q.held_out_iou.toFixed(2) : '—'} />
        <QualityStat k="Keypoint error" v={Number.isFinite(q.keypoint_error_pct_height) ? `${q.keypoint_error_pct_height.toFixed(1)}% ht` : '—'} />
      </div>
      {q.warning_count > 0 && (
        <details className="mt-2 text-xs text-orange-700">
          <summary className="cursor-pointer">{q.warning_count} reconstruction warning(s)</summary>
          <ul className="mt-1 list-disc pl-4">{q.warnings.map((w, i) => <li key={i}>{w}</li>)}</ul>
        </details>
      )}
      <p className="mt-2 text-[11px] text-slate-400">External body shape from video. ± combines height-scale uncertainty with measured scan-to-scan repeatability.</p>
    </Card>
  )
}

function QualityStat({ k, v }) {
  return <div className="rounded-lg bg-slate-50 px-2 py-1"><div className="text-[10px] uppercase tracking-wide text-slate-400">{k}</div><div className="text-slate-700">{v ?? '—'}</div></div>
}

/** Each segment's camera girths next to the same segment's BCM composition (no merged or derived numbers). */
function SideBySide({ camera, bcm }) {
  const rows = segmentRows(camera, bcm)
  const seg = hasSegmental(bcm)
  const w = bcm?.values?.weight?.value
  const fat = bcm?.values?.body_fat_pct?.value
  return (
    <Card>
      <div className="flex flex-wrap items-center justify-between gap-2">
        <h3 className="text-base font-semibold text-slate-900">Camera and BCM by body segment</h3>
        {bcm && <span className={`rounded-full px-2 py-0.5 text-[11px] font-semibold ring-1 ${bcm.source.is_demo ? 'bg-amber-100 text-amber-800 ring-amber-300' : 'bg-indigo-50 text-indigo-700 ring-indigo-200'}`}>BCM: {sourceBadge(bcm).text}</span>}
      </div>
      <p className="mt-1 text-xs text-slate-500">
        Two independent measurements shown side by side: girths of the external shape from the camera, composition estimates from the BCM.
        {!bcm && ' No BCM data yet — the BCM column fills in once values are added.'}
        {bcm && !seg && ' This BCM record has no segmental values.'}
      </p>
      <div className="mt-3 grid grid-cols-2 gap-3 text-sm sm:grid-cols-4">
        <Headline tag="camera" k="Height" v={camera.height_cm ? `${camera.height_cm} cm` : '—'} hint={camera.height_cm ? 'supplied for scaling' : 'not supplied'} />
        <Headline tag="bcm" k="Weight" v={Number.isFinite(w) ? `${w.toFixed(1)} kg` : '—'} demo={bcm?.source?.is_demo} />
        <Headline tag="bcm" k="Body fat" v={Number.isFinite(fat) ? `${fat.toFixed(1)} %` : '—'} demo={bcm?.source?.is_demo} />
        <Headline tag="camera" k="Waist" v={Number.isFinite(camera.measurements_cm?.waist) ? `${camera.measurements_cm.waist.toFixed(1)} cm` : '—'} />
      </div>
      <div className="mt-4 overflow-x-auto">
        <table className="w-full min-w-[560px] text-sm">
          <thead>
            <tr className="text-left text-[11px] uppercase tracking-wide text-slate-500">
              <th className="py-1 font-medium">Segment</th>
              <th className="py-1 font-medium"><span className="text-teal-700">Camera</span> girths ({rows[0].unit === 'cm' ? 'cm' : '× height'})</th>
              <th className="py-1 text-right font-medium"><span className="text-indigo-700">BCM</span> fat %</th>
              <th className="py-1 text-right font-medium"><span className="text-indigo-700">BCM</span> fat kg</th>
              <th className="py-1 text-right font-medium"><span className="text-indigo-700">BCM</span> muscle kg</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((r) => (
              <tr key={r.key} className="border-t border-slate-100 align-top">
                <td className="py-1.5 font-medium text-slate-700">{r.label}</td>
                <td className="py-1.5 text-slate-700">
                  {r.camera.map((c) => (
                    <span key={c.site} className={`mr-3 inline-block ${c.valid ? '' : 'text-orange-600'}`}>
                      <span className="text-xs text-slate-400">{(SITE_FULL_LABELS[c.site] || c.site).replace(/ \([LR]\)/, '')} </span>
                      <span className="tabular-nums">{Number.isFinite(c.value) ? c.value.toFixed(r.unit === 'cm' ? 1 : 3) : '—'}</span>
                    </span>
                  ))}
                </td>
                {['fat_pct', 'fat_mass', 'muscle_mass'].map((k) => (
                  <td key={k} className={`py-1.5 text-right tabular-nums ${bcm?.source?.is_demo ? 'text-amber-800' : 'text-slate-900'}`}>
                    {Number.isFinite(r.bcm[k]) ? r.bcm[k].toFixed(1) : <span className="text-slate-300">—</span>}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </Card>
  )
}

function Headline({ tag, k, v, hint, demo }) {
  return (
    <div className={`rounded-xl px-3 py-2 ring-1 ${tag === 'camera' ? 'bg-teal-50/40 ring-teal-100' : demo ? 'bg-amber-50 ring-amber-200' : 'bg-indigo-50/40 ring-indigo-100'}`}>
      <div className="flex items-center justify-between text-[11px] uppercase tracking-wide text-slate-500">{k}
        <span className={tag === 'camera' ? 'text-teal-600' : demo ? 'text-amber-700' : 'text-indigo-600'}>{tag === 'camera' ? 'camera' : demo ? 'demo' : 'bcm'}</span>
      </div>
      <div className="text-lg font-semibold text-slate-900">{v}</div>
      {hint && <div className="text-[11px] text-slate-400">{hint}</div>}
    </div>
  )
}

const TREND_KEYS = ['weight', 'body_fat_pct', 'fat_mass', 'muscle_mass', 'total_body_water_pct', 'visceral_fat_rating']

function HistoryCompare({ a, all }) {
  const others = all.filter((r) => r.assessment_id !== a.assessment_id && (!a.person_id || r.person_id === a.person_id))
  const [otherId, setOtherId] = useState(others[0]?.assessment_id || '')
  const [cmp, setCmp] = useState(null)
  const [hist, setHist] = useState(null)
  const [schema, setSchema] = useState(null)
  const [err, setErr] = useState(null)
  const bcmKey = a.bcm?.updated_at || 'none'

  useEffect(() => { getBcmSchema().then(setSchema).catch(() => {}) }, [])
  useEffect(() => {
    if (!a.person_id) { setHist(null); return }
    personHistory(a.person_id).then(setHist).catch(() => setHist(null))
  }, [a.person_id, bcmKey])
  useEffect(() => {
    if (!otherId) { setCmp(null); return }
    const other = all.find((r) => r.assessment_id === otherId)
    // Always "earlier -> later" so a negative change means a decrease over time.
    const [before, after] = (other?.created_at || 0) < (all.find((r) => r.assessment_id === a.assessment_id)?.created_at || 0)
      ? [otherId, a.assessment_id] : [a.assessment_id, otherId]
    setErr(null)
    compareAssessments(before, after).then(setCmp).catch((e) => { setErr(e.message); setCmp(null) })
  }, [otherId, a.assessment_id, bcmKey, all])

  const label = (k) => schema?.parameters.find((p) => p.key === k)?.label || k
  const decimals = (k) => schema?.parameters.find((p) => p.key === k)?.decimals ?? 1

  return (
    <Card>
      <h3 className="text-base font-semibold text-slate-900">History and comparison</h3>
      {!a.person_id && <p className="mt-1 text-xs text-slate-500">Assign a person code to limit the history to this person. Showing all assessments.</p>}
      {others.length === 0 ? (
        <p className="mt-3 rounded-lg bg-slate-50 px-3 py-3 text-sm text-slate-500">
          This is the only assessment{a.person_id ? ` for ${a.person_id}` : ''}. Comparisons appear once there is another one.
        </p>
      ) : (
        <>
          <label className="mt-3 flex flex-wrap items-center gap-2 text-sm text-slate-600">Compare with
            <select value={otherId} onChange={(e) => setOtherId(e.target.value)} className="rounded-lg border border-slate-200 bg-white px-2 py-1 text-sm">
              {others.map((r) => (
                <option key={r.assessment_id} value={r.assessment_id}>
                  {formatWhen(r.assessed_at)} · {r.assessment_id.slice(0, 8)}{r.person_id ? ` · ${r.person_id}` : ''} · {r.bcm.available ? (r.bcm.is_demo ? 'BCM demo' : 'BCM') : 'camera only'}
                </option>
              ))}
            </select>
          </label>
          {err && <p className="mt-2 text-xs text-red-600">{err}</p>}
          {cmp && (
            <div className="mt-3 grid grid-cols-1 gap-4 xl:grid-cols-2">
              <div>
                <div className="flex items-center gap-2 text-xs font-semibold text-slate-600"><SourceTag kind="camera" />
                  {formatWhen(cmp.before.assessed_at)} → {formatWhen(cmp.after.assessed_at)} ({cmp.camera.unit === 'cm' ? 'cm' : '× height'})</div>
                {!cmp.same_person && <p className="mt-1 text-xs text-orange-700">Different or unassigned person codes: differences are not a change in one person.</p>}
                <table className="mt-2 w-full text-sm">
                  <CompareHead />
                  <tbody>
                    {cmp.camera.rows.map((r) => (
                      <tr key={r.site} className="border-t border-slate-100">
                        <td className="py-1 text-slate-700">{SITE_FULL_LABELS[r.site] || r.site}</td>
                        <td className="py-1 text-right tabular-nums text-slate-500">{fmt(r.before, cmp.camera.unit)}</td>
                        <td className="py-1 text-right tabular-nums text-slate-900">{fmt(r.after, cmp.camera.unit)}</td>
                        <td className="py-1 text-right tabular-nums font-medium text-slate-700">{formatChange(r.delta, cmp.camera.unit === 'cm' ? 1 : 3) ?? '—'}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
              <div>
                <div className="flex items-center gap-2 text-xs font-semibold text-slate-600"><SourceTag kind="bcm" />
                  {cmp.bcm.demo_involved && <span className="rounded bg-amber-100 px-1.5 text-[10px] font-semibold text-amber-800">INCLUDES DEMO DATA</span>}
                </div>
                {cmp.bcm.message && <p className="mt-2 rounded-lg bg-slate-50 px-3 py-3 text-sm text-slate-500">{cmp.bcm.message}</p>}
                {/* Rows only when both assessments have BCM data: one-sided values would be a table of dashes. */}
                {cmp.bcm.available && cmp.bcm.rows.length > 0 && (
                  <table className="mt-2 w-full text-sm">
                    <CompareHead />
                    <tbody>
                      {cmp.bcm.rows.filter((r) => r.delta !== null).map((r) => (
                        <tr key={r.key} className="border-t border-slate-100">
                          <td className="py-1 text-slate-700">{label(r.key)}</td>
                          <td className="py-1 text-right tabular-nums text-slate-500">{bcmNum(r.before, decimals(r.key))}</td>
                          <td className="py-1 text-right tabular-nums text-slate-900">{bcmNum(r.after, decimals(r.key))}</td>
                          <td className="py-1 text-right tabular-nums font-medium text-slate-700">{r.delta !== null ? `${formatChange(r.delta, decimals(r.key))} ${r.unit === 'rating' ? '' : r.unit}` : '—'}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                )}
                {cmp.bcm.available && <OneSidedNote rows={cmp.bcm.rows} label={label} />}
              </div>
            </div>
          )}
        </>
      )}

      {hist && (
        <div className="mt-5 border-t border-slate-100 pt-4">
          <p className="text-sm font-medium text-slate-700">BCM over time · {hist.person_id}</p>
          {!hist.bcm_trend_available ? (
            <p className="mt-1 text-xs text-slate-500">
              {hist.bcm_series.length === 0 ? 'No BCM records for this person, so no BCM trend is shown.'
                : 'Only one assessment has BCM data; a trend needs at least two.'} Camera history: {hist.camera_series.length} assessment(s).
            </p>
          ) : (
            <div className="mt-2 overflow-x-auto">
              <table className="w-full min-w-[480px] text-sm">
                <thead><tr className="text-left text-[11px] uppercase tracking-wide text-slate-500">
                  <th className="py-1 font-medium">Assessment</th>
                  {TREND_KEYS.map((k) => <th key={k} className="py-1 text-right font-medium">{label(k)}</th>)}
                </tr></thead>
                <tbody>
                  {hist.bcm_series.map((s) => (
                    <tr key={s.assessment_id} className={`border-t border-slate-100 ${s.is_demo ? 'text-amber-800' : 'text-slate-800'}`}>
                      <td className="py-1">{formatWhen(s.assessed_at)} {s.is_demo && <span className="rounded bg-amber-100 px-1 text-[10px] font-semibold">DEMO</span>}</td>
                      {TREND_KEYS.map((k) => <td key={k} className="py-1 text-right tabular-nums">{bcmNum(s.values[k], decimals(k))}</td>)}
                    </tr>
                  ))}
                </tbody>
              </table>
              <p className="mt-1 text-[11px] text-slate-400">Only assessments with BCM data are listed; camera-only assessments are not given BCM values.</p>
            </div>
          )}
        </div>
      )}
    </Card>
  )
}

/** BCM fields present in only one of the two records: listed by name, no change is computed for them. */
function OneSidedNote({ rows, label }) {
  const one = rows.filter((r) => r.delta === null)
  if (!one.length) return null
  const shared = rows.length - one.length
  return (
    <details className="mt-2 text-xs text-slate-500">
      <summary className="cursor-pointer">
        {shared === 0 ? 'The two BCM records share no parameters, so no change is shown. ' : ''}
        {one.length} parameter(s) recorded in only one assessment
      </summary>
      <p className="mt-1">{one.map((r) => label(r.key)).join(', ')}.</p>
    </details>
  )
}

function CompareHead() {
  return (
    <thead><tr className="text-left text-[11px] uppercase tracking-wide text-slate-500">
      <th className="py-1 font-medium" /><th className="py-1 text-right font-medium">Before</th>
      <th className="py-1 text-right font-medium">After</th><th className="py-1 text-right font-medium">Change</th>
    </tr></thead>
  )
}

function bcmNum(v, d = 1) {
  return Number.isFinite(v) ? v.toFixed(d) : '—'
}

function fmt(v, unit) {
  return Number.isFinite(v) ? v.toFixed(unit === 'cm' ? 1 : 3) : '—'
}
