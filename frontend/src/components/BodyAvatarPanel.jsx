/**
 * BodyAvatarPanel.jsx
 * -------------------
 * "Body Avatar" view of a completed reconstruction: the fitted mesh coloured by body segment, callout
 * cards with leader lines to where each girth was measured, Front / Side / Back turning, and an
 * optional comparison with an earlier scan (mesh morph + replay + change chips).
 *
 * Every number shown comes from the run's measurements (cm only when the run was scaled from a
 * supplied height). Segment colours describe measurement status or change versus repeatability,
 * never body composition.
 */
import { Suspense, useEffect, useMemo, useRef, useState } from 'react'
import { Canvas, useFrame, useThree } from '@react-three/fiber'
import { ContactShadows, RoundedBox } from '@react-three/drei'
import * as THREE from 'three'
import segmentsJson from './smplSegments.json'
import {
  CALLOUTS, SEGMENTS, SITE_LABELS, SITE_FULL_LABELS, STATUS, HEAD_COLOR, parseObj, parseRings, centroid,
  decodeLabels, scanValues, comparableValues, interpolateValues, uncertaintyCm, formatValue, formatDelta,
  siteStatus, changeStatus, segmentStatus, legendFor, hexToRgb, formatDate, formatDateTime, repeatabilityCoefficient,
} from './bodyAvatarModel.js'
import { listReconstructions, reconstructionFileUrl, getConditions, recordConditions } from '../services/reconstructionApi.js'

const LABELS = decodeLabels(segmentsJson)
const VIEWS = [['front', 'Front', 0], ['side', 'Side', Math.PI / 2], ['back', 'Back', Math.PI]]
// Card slots (fraction of stage height) per column, like a printed body-composition sheet.
const SLOTS = { left_arm: 0.24, right_arm: 0.24, trunk: 0.42, left_leg: 0.62, right_leg: 0.62 }

async function fetchText(url) {
  const res = await fetch(url)
  if (!res.ok) throw new Error(`HTTP ${res.status} for ${url.split('/').pop()}`)
  return res.text()
}

const DETAIL_SITES = [['chest', 'chest'], ['waist', 'waist'], ['abdomen', 'abdomen'], ['hip', 'hip']]

/** "chest 113.6 cm, waist 104.2 cm, ..." from the detailed girths (cm when the run was height-scaled). */
export function detailGirths(detailInfo, result) {
  const cm = result?.metric_scale?.measurements_cm_detailed
  const rel = detailInfo?.measurements || {}
  const parts = DETAIL_SITES.map(([k, label]) => {
    if (cm && cm[k] != null) return `${label} ${cm[k].toFixed(1)} cm`
    if (rel[k] != null) return `${label} ${rel[k].toFixed(3)}×height`
    return null
  }).filter(Boolean)
  return parts.length ? parts.join(', ') : null
}

export default function BodyAvatarPanel({ run }) {
  const runId = run?.run_id
  const result = run?.result
  const [mesh, setMesh] = useState(null)
  const [rings, setRings] = useState({})
  const [history, setHistory] = useState([])
  const [mode, setMode] = useState('scan')
  const [compareId, setCompareId] = useState(null)
  const [compareMesh, setCompareMesh] = useState(null)
  const [t, setT] = useState(1)
  const [view, setView] = useState('front')
  const [yawTarget, setYawTarget] = useState(0)
  const [mirror, setMirror] = useState(true)
  const [showKey, setShowKey] = useState(true)
  const [showAccuracy, setShowAccuracy] = useState(false)
  const [error, setError] = useState(null)
  // Surface detail (SMPL+D): same 6890-vertex body plus shared trunk offsets that keep local shape such as flank
  // fat, which the 10 SMPL shape coefficients smooth away. Shape only: the girth cards stay on the body model.
  const detailInfo = result?.surface_detail || null
  const [detail, setDetail] = useState(false)
  const meshFile = detail && detailInfo ? detailInfo.mesh_obj : (result?.mesh_obj || 'final_body_mesh.obj')

  useEffect(() => {
    if (!runId || !result) return
    let alive = true
    setError(null)
    Promise.all([fetchText(reconstructionFileUrl(runId, meshFile)),
      fetchText(reconstructionFileUrl(runId, 'measurement_rings.obj')).catch(() => '')])
      .then(([objText, ringText]) => {
        if (!alive) return
        const m = parseObj(objText)
        if (m.vertexCount !== LABELS.length) throw new Error(`expected an SMPL mesh (${LABELS.length} vertices), got ${m.vertexCount}`)
        setMesh(m)
        setRings(detail ? {} : parseRings(ringText))  // rings were cut on the body-model mesh
      })
      .catch((e) => alive && setError(`Avatar could not load the mesh: ${e.message}`))
    listReconstructions().then((rows) => alive && setHistory(rows)).catch(() => {})
    return () => { alive = false }
  }, [runId, result, meshFile, detail])

  const ownSource = history.find((r) => r.run_id === runId)?.source_key
  const choices = history.filter((r) => r.run_id !== runId)
  const compareRun = choices.find((r) => r.run_id === compareId) || null
  useEffect(() => {
    if (!compareRun) { setCompareMesh(null); return }
    let alive = true
    fetchText(reconstructionFileUrl(compareRun.run_id, compareRun.mesh_obj || 'final_body_mesh.obj'))
      .then((txt) => { if (alive) { setCompareMesh(parseObj(txt)); setT(1) } })
      .catch((e) => alive && setError(`Could not load the earlier scan: ${e.message}`))
    return () => { alive = false }
  }, [compareRun])

  const current = useMemo(() => ({ ...result, created_at: run?.created_at }), [result, run])
  const scan = scanValues(current)
  const comparing = mode === 'compare' && compareRun && compareMesh
  const cmp = comparing ? comparableValues(current, compareRun) : null
  const unit = cmp ? cmp.unit : scan.unit
  const shown = cmp ? interpolateValues(cmp.before, cmp.now, t) : scan.values
  // Excluded tape records (e.g. test entries) are never shown as a tape check.
  const tapeSites = result?.tape_validation && !result.tape_validation.excluded ? result.tape_validation.sites : null

  const statusOf = (site) => (cmp ? changeStatus(site, cmp.before[site], cmp.now[site]) : siteStatus(site, scan.values, tapeSites))
  const segStatus = Object.fromEntries(CALLOUTS.map((c) => [c.segment, segmentStatus(c, statusOf)]))

  const pickMode = (m) => {
    setMode(m)
    if (m === 'compare' && !compareId && choices.length) {
      // Prefer the newest scan of a different recording: a re-run of the same file is not a new measurement.
      const other = choices.find((r) => r.source_key && r.source_key !== ownSource) || choices[0]
      setCompareId(other.run_id)
    }
  }

  const turnTo = (key) => {
    const target = VIEWS.find((v) => v[0] === key)[2]
    setView(key)
    setYawTarget((prev) => prev + (((target - prev) % (2 * Math.PI)) + 3 * Math.PI) % (2 * Math.PI) - Math.PI)
  }

  if (!result) return null
  const heightCm = scan.heightCm
  const sameDay = compareRun && formatDate(compareRun.created_at) === formatDate(run?.created_at)
  const fmtWhen = sameDay ? formatDateTime : formatDate
  const beforeDate = compareRun ? fmtWhen(compareRun.created_at) : null

  return (
    <section className="rounded-[28px] bg-white shadow-[0_8px_30px_rgba(15,23,42,0.06)] border border-slate-100 p-4 sm:p-6"
      data-testid="body-avatar" aria-label="Body avatar">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <Segmented
          items={[['scan', 'This scan'], ['compare', compareRun ? `Since ${beforeDate}` : 'Compare']]}
          value={mode} onChange={pickMode} disabled={{ compare: choices.length === 0 }}
          title={{ compare: choices.length === 0 ? 'No earlier completed scan to compare with' : undefined }} />
        <Segmented items={VIEWS.map(([k, l]) => [k, l])} value={view} onChange={turnTo} />
        {detailInfo && (
          <Segmented items={[['body', 'Body model'], ['detail', 'Surface detail']]} value={detail ? 'detail' : 'body'}
            onChange={(k) => setDetail(k === 'detail')} />
        )}
      </div>
      {detail && detailInfo && (
        <p className="mt-2 text-sm text-slate-600">
          Surface detail keeps local trunk shape (for example side/flank fat) that the body model’s 10 shape numbers
          smooth away. It follows the outline seen in every view, so under loose clothing it also follows the garment.
          Girth cards and rings still show the body model.
          {detailGirths(detailInfo, result) && <> Detailed girths: {detailGirths(detailInfo, result)}.</>}
        </p>
      )}

      {mode === 'compare' && choices.length > 0 && (
        <label className="mt-3 flex flex-wrap items-center gap-2 text-sm text-slate-600">
          Compare with
          <select value={compareId || ''} onChange={(e) => setCompareId(e.target.value)}
            className="rounded-lg border border-slate-200 bg-white px-2 py-1 text-sm text-slate-800">
            {choices.map((r) => (
              <option key={r.run_id} value={r.run_id}>
                {formatDateTime(r.created_at)} · {r.run_id.slice(0, 8)}{r.height_cm ? ` · ${r.height_cm} cm` : ''}
                {r.source_key && r.source_key === ownSource ? ' (same video, re-run)' : ''}
              </option>
            ))}
          </select>
        </label>
      )}

      <Stage
        mesh={mesh} compareMesh={comparing ? compareMesh : null} t={t} rings={rings} yawTarget={yawTarget}
        mirror={mirror} segStatus={segStatus} values={shown} unit={unit} statusOf={statusOf}
        scan={scan} cmp={cmp} heightCm={heightCm} error={error}
        onToggleMirror={() => setMirror((m) => !m)} onDragTurn={(dy) => { setYawTarget((y) => y + dy); setView(null) }}
      />

      {comparing && (
        <Timeline t={t} setT={setT} from={beforeDate} to={fmtWhen(run?.created_at)} cmp={cmp} unit={unit} />
      )}

      <Collapsible title="Colour key" open={showKey} onToggle={() => setShowKey((s) => !s)}>
        <div className="flex flex-wrap items-center gap-x-5 gap-y-2 text-[15px] text-slate-600">
          {legendFor(cmp ? 'compare' : 'scan').map((k) => (
            <span key={k} className="inline-flex items-center gap-2">
              <span className="h-4 w-4 rounded-[5px]" style={{ background: STATUS[k].color }} />{STATUS[k].label}
            </span>
          ))}
          <span className="text-slate-500">
            {cmp ? 'Change between scans against the measured run-to-run repeatability (95%)'
              : 'Measurement status of each segment’s girths; not body composition'}
          </span>
        </div>
      </Collapsible>

      <Collapsible title="Accuracy & method" badge={heightCm ? 'CM' : 'RELATIVE'} hint="How sure each number is"
        open={showAccuracy} onToggle={() => setShowAccuracy((s) => !s)}>
        <AccuracyTable result={result} scan={scan} tapeSites={tapeSites} />
        <CaptureConditions runId={runId} />
      </Collapsible>
    </section>
  )
}

// ---------------------------------------------------------------- stage
function Stage({ mesh, compareMesh, t, rings, yawTarget, mirror, segStatus, values, unit, statusOf, scan, cmp,
  heightCm, error, onToggleMirror, onDragTurn }) {
  const wrap = useRef(null)
  const cardRefs = useRef({})
  const lineRefs = useRef({})
  const dotRefs = useRef({})
  const [sides, setSides] = useState({ left_arm: 'left', left_leg: 'left', right_arm: 'right', right_leg: 'right', trunk: 'right' })
  const drag = useRef(null)
  const compact = useCompact(wrap)

  const anchors = useMemo(() => Object.fromEntries(CALLOUTS.map((c) => [c.id, rings[c.primary] || null])), [rings])

  const onPointerDown = (e) => { drag.current = e.clientX; e.currentTarget.setPointerCapture?.(e.pointerId) }
  const onPointerMove = (e) => {
    if (drag.current == null) return
    onDragTurn((e.clientX - drag.current) * 0.01)
    drag.current = e.clientX
  }
  const onPointerUp = () => { drag.current = null }

  const cards = CALLOUTS.map((c) => (
    <CalloutCard key={c.id} c={c} side={sides[c.id]} top={SLOTS[c.id]} values={values} unit={unit} inline={compact}
      status={statusOf(c.primary)} cmp={cmp} scan={scan} innerRef={(el) => { cardRefs.current[c.id] = compact ? null : el }} />
  ))

  return (
    <>
    <div ref={wrap} className="relative mt-4 overflow-hidden rounded-[22px] select-none"
      style={{ height: compact ? 460 : 'min(78vh, 760px)', minHeight: compact ? 0 : 520,
        background: 'radial-gradient(120% 80% at 50% 30%, #ffffff 0%, #eef1f6 55%, #e4e8ef 100%)' }}>
      <div className="absolute inset-0 cursor-grab active:cursor-grabbing touch-pan-y"
        onPointerDown={onPointerDown} onPointerMove={onPointerMove} onPointerUp={onPointerUp} onPointerCancel={onPointerUp}>
        <Canvas camera={{ position: [0, 0.92, 3.6], fov: 32, near: 0.05, far: 50 }} shadows dpr={[1, 2]}
          gl={{ antialias: true, alpha: true }} onCreated={({ camera }) => camera.lookAt(0, 0.86, 0)}>
          <hemisphereLight args={['#ffffff', '#cfd6e0', 0.9]} />
          <directionalLight position={[2.5, 4, 3]} intensity={1.6} castShadow shadow-mapSize={[1024, 1024]} />
          <directionalLight position={[-3, 2.5, -2]} intensity={0.5} />
          <Suspense fallback={null}>
            {mesh && (
              <Avatar mesh={mesh} compareMesh={compareMesh} t={t} yawTarget={yawTarget} mirror={mirror}
                segStatus={segStatus} anchors={anchors} wrap={wrap} cardRefs={cardRefs} lineRefs={lineRefs}
                dotRefs={dotRefs} onSides={setSides} />
            )}
          </Suspense>
          <RoundedBox args={[0.64, 0.07, 0.44]} radius={0.025} smoothness={4} position={[0, -0.036, 0.02]} receiveShadow castShadow>
            <meshStandardMaterial color="#fbfcfd" roughness={0.35} />
          </RoundedBox>
          <ContactShadows position={[0, -0.072, 0]} opacity={0.38} scale={3} blur={2.6} far={2} />
        </Canvas>
      </div>

      {/* leader lines */}
      <svg className="pointer-events-none absolute inset-0 h-full w-full" aria-hidden="true">
        {CALLOUTS.map((c) => (
          <g key={c.id}>
            <line ref={(el) => { lineRefs.current[c.id] = el }} stroke="#94a3b8" strokeWidth="1.5" opacity="0" />
            <circle ref={(el) => { dotRefs.current[c.id] = el }} r="7" fill="#fff" stroke="#3b6fd6" strokeWidth="3" opacity="0" />
          </g>
        ))}
      </svg>

      {/* callout cards (overlay on wide screens; a grid under the stage on phones) */}
      {!compact && cards}

      {!mesh && !error && <p className="absolute inset-0 grid place-items-center text-sm text-slate-500">Loading avatar…</p>}
      {error && <p role="alert" className="absolute inset-x-4 top-4 rounded-lg bg-red-50 p-2 text-sm text-red-700">{error}</p>}

      <div className={`pointer-events-none absolute inset-x-4 bottom-3 flex items-end justify-between gap-3 text-slate-500 sm:inset-x-6 sm:bottom-4 ${compact ? 'text-xs' : 'text-[15px]'}`}>
        <button type="button" onClick={onToggleMirror} className="pointer-events-auto inline-flex items-center gap-2 hover:text-slate-700"
          title="Mirror view shows your left side on the left, like a mirror">
          <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" aria-hidden="true">
            <path d="M12 3v18M8 8l-4 4 4 4M16 8l4 4-4 4" strokeLinecap="round" strokeLinejoin="round" />
          </svg>
          Drag to turn · {mirror ? 'mirror view' : 'camera view'}
        </button>
        <span className="text-right">
          {heightCm ? `${heightCm} cm · scale from supplied height` : 'Relative scale · enter height for cm'}
        </span>
      </div>
    </div>
    {compact && <div className="mt-3 grid grid-cols-2 gap-2.5">{cards}</div>}
    </>
  )
}

/** True when the stage is too narrow for overlay cards beside the body. */
function useCompact(ref, limit = 640) {
  const [compact, setCompact] = useState(false)
  useEffect(() => {
    const el = ref.current
    if (!el || typeof ResizeObserver === 'undefined') return
    const ro = new ResizeObserver(([entry]) => setCompact(entry.contentRect.width < limit))
    ro.observe(el)
    return () => ro.disconnect()
  }, [ref, limit])
  return compact
}

function CalloutCard({ c, side, top, values, unit, status, cmp, scan, innerRef, inline = false }) {
  const v = values[c.primary]
  const unc = unit === 'cm' ? uncertaintyCm(c.primary, v, scan.relScaleUnc) : null
  const delta = cmp ? cmp.now[c.primary] - cmp.before[c.primary] : null
  const pos = side === 'left' ? { left: 'clamp(8px, 1.6vw, 22px)' } : { right: 'clamp(8px, 1.6vw, 22px)' }
  return (
    <div ref={innerRef} data-side={side}
      className={inline
        ? `rounded-2xl border border-slate-100 bg-white px-3 py-2 shadow-sm ${c.id === 'trunk' ? 'col-span-2' : ''}`
        : 'absolute w-[min(42vw,232px)] -translate-y-1/2 rounded-2xl border border-white bg-white/95 px-3 py-2 shadow-[0_6px_20px_rgba(15,23,42,0.08)] backdrop-blur transition-[left,right] duration-300 sm:px-4 sm:py-2.5'}
      style={inline ? { order: ['left_arm', 'right_arm', 'trunk', 'left_leg', 'right_leg'].indexOf(c.id) } : { top: `${top * 100}%`, ...pos }}>
      <div className="flex items-center justify-between gap-2">
        <p className="text-[11px] font-medium uppercase tracking-[0.12em] text-slate-500 sm:text-[13px]">{c.label}</p>
        <span className="h-2.5 w-2.5 rounded-full" style={{ background: STATUS[status].color }} title={STATUS[status].label} />
      </div>
      <p className="mt-0.5 leading-none text-slate-900">
        <span className="text-[22px] font-semibold tabular-nums sm:text-[27px]">{formatValue(v, unit)}</span>
        <span className="ml-1 text-sm text-slate-500">{unit === 'cm' ? 'cm' : '× H'}</span>
        {unc != null && <span className="ml-1.5 text-xs text-slate-400 tabular-nums">±{unc.toFixed(1)}</span>}
      </p>
      <p className="mt-1 text-[11px] text-slate-500 sm:text-xs">
        {SITE_LABELS[c.primary]}
        {cmp && Number.isFinite(delta) && (
          <span className="ml-1.5 font-medium tabular-nums text-slate-700">{formatDelta(delta, unit)}</span>
        )}
        {c.secondary.map((s) => (
          <span key={s}> · {SITE_LABELS[s]} <span className="tabular-nums text-slate-700">{formatValue(values[s], unit)}</span></span>
        ))}
      </p>
    </div>
  )
}

// ---------------------------------------------------------------- 3D avatar
function Avatar({ mesh, compareMesh, t, yawTarget, mirror, segStatus, anchors, wrap, cardRefs, lineRefs, dotRefs, onSides }) {
  const group = useRef()
  const inner = useRef()
  const yaw = useRef(yawTarget)
  const { camera, size } = useThree()
  const sidesRef = useRef({})

  const geometry = useMemo(() => {
    const g = new THREE.BufferGeometry()
    g.setAttribute('position', new THREE.BufferAttribute(mesh.positions.slice(), 3))
    g.setAttribute('color', new THREE.BufferAttribute(new Float32Array(mesh.vertexCount * 3), 3))
    g.setIndex(new THREE.BufferAttribute(mesh.indices, 1))
    g.computeVertexNormals()
    return g
  }, [mesh])
  useEffect(() => () => geometry.dispose(), [geometry])

  // Morph between the earlier and the current scan (same SMPL topology, so positions line up 1:1).
  useEffect(() => {
    const pos = geometry.attributes.position.array
    const a = compareMesh?.positions
    const b = mesh.positions
    for (let i = 0; i < pos.length; i++) pos[i] = a ? a[i] + (b[i] - a[i]) * t : b[i]
    geometry.attributes.position.needsUpdate = true
    geometry.computeVertexNormals()
  }, [geometry, mesh, compareMesh, t])

  useEffect(() => {
    const col = geometry.attributes.color.array
    const palette = SEGMENTS.map((s) => hexToRgb(s === 'head' ? HEAD_COLOR : STATUS[segStatus[s] || 'measured'].color))
    for (let i = 0; i < LABELS.length; i++) {
      const c = palette[LABELS[i]]
      col[i * 3] = c[0]; col[i * 3 + 1] = c[1]; col[i * 3 + 2] = c[2]
    }
    geometry.attributes.color.needsUpdate = true
  }, [geometry, segStatus])

  const v = new THREE.Vector3()
  useFrame((_, dt) => {
    if (!group.current || !inner.current) return
    yaw.current += (yawTarget - yaw.current) * Math.min(1, dt * 7)
    group.current.rotation.y = yaw.current
    group.current.updateMatrixWorld()
    const box = wrap.current?.getBoundingClientRect()
    if (!box) return
    const toPx = (p) => {
      v.set(p[0], p[1], p[2]).applyMatrix4(inner.current.matrixWorld)
      // Horizontal depth toward the camera (which sits on +z): height and lateral offset do not count.
      const depth = camera.position.z - v.z
      v.project(camera)
      return { x: (v.x + 1) / 2 * size.width, y: (1 - v.y) / 2 * size.height, depth }
    }
    // Screen span and depth of the trunk: a limb is dimmed only when it sits behind the trunk AND overlaps it.
    const trunkPts = (anchors.trunk || []).filter((_, k) => k % 3 === 0).map(toPx)
    const trunkDepth = trunkPts.length ? trunkPts.reduce((a, p) => a + p.depth, 0) / trunkPts.length : null
    const trunkMinX = trunkPts.length ? Math.min(...trunkPts.map((p) => p.x)) : 0
    const trunkMaxX = trunkPts.length ? Math.max(...trunkPts.map((p) => p.x)) : 0
    const screen = {}
    for (const c of CALLOUTS) {
      const ring = anchors[c.id]
      const line = lineRefs.current[c.id]
      const dot = dotRefs.current[c.id]
      const card = cardRefs.current[c.id]
      if (!ring || !line || !dot || !card) { line?.setAttribute('opacity', 0); dot?.setAttribute('opacity', 0); continue }
      // The ring point nearest the camera is where a tape would be seen from this side.
      let best = null
      for (let k = 0; k < ring.length; k += 2) {
        const p = toPx(ring[k])
        if (!best || p.depth < best.depth) best = p
      }
      const ctr = toPx(centroid(ring))
      screen[c.id] = ctr
      const hidden = c.id !== 'trunk' && trunkDepth != null && ctr.depth > trunkDepth && ctr.x > trunkMinX && ctr.x < trunkMaxX
      const r = card.getBoundingClientRect()
      const cx = card.dataset.side === 'left' ? r.right - box.left : r.left - box.left
      const cy = r.top - box.top + r.height / 2
      line.setAttribute('x1', cx); line.setAttribute('y1', cy)
      line.setAttribute('x2', best.x); line.setAttribute('y2', best.y)
      line.setAttribute('opacity', hidden ? 0.35 : 0.9)
      dot.setAttribute('cx', best.x); dot.setAttribute('cy', best.y)
      dot.setAttribute('opacity', hidden ? 0.45 : 1)
      dot.setAttribute('stroke', hidden ? '#94a3b8' : '#3b6fd6')
    }
    // Keep left and right cards on opposite columns, on the side their limb appears.
    if (screen.left_arm && screen.right_arm) {
      const leftOnLeft = screen.left_arm.x <= screen.right_arm.x
      const next = { left_arm: leftOnLeft ? 'left' : 'right', left_leg: leftOnLeft ? 'left' : 'right',
        right_arm: leftOnLeft ? 'right' : 'left', right_leg: leftOnLeft ? 'right' : 'left', trunk: 'right' }
      if (next.left_arm !== sidesRef.current.left_arm) { sidesRef.current = next; onSides(next) }
    }
  })

  return (
    <group ref={group}>
      <group ref={inner} scale={[mirror ? -1 : 1, 1, 1]}>
        <mesh geometry={geometry} castShadow>
          <meshStandardMaterial vertexColors roughness={0.5} metalness={0.02} side={THREE.DoubleSide} />
        </mesh>
      </group>
    </group>
  )
}

// ---------------------------------------------------------------- timeline
function Timeline({ t, setT, from, to, cmp, unit }) {
  const raf = useRef(null)
  const replay = () => {
    cancelAnimationFrame(raf.current)
    const start = performance.now()
    const step = (now) => {
      const k = Math.min(1, (now - start) / 2200)
      setT(k < 0.5 ? 2 * k * k : 1 - (-2 * k + 2) ** 2 / 2)
      if (k < 1) raf.current = requestAnimationFrame(step)
    }
    raf.current = requestAnimationFrame(step)
  }
  useEffect(() => () => cancelAnimationFrame(raf.current), [])
  const chips = [['waist', 'Waist'], ['hip', 'Hip'], ['chest', 'Chest']]
  const thigh = (vals) => (Number.isFinite(vals.thigh_left) && Number.isFinite(vals.thigh_right) ? (vals.thigh_left + vals.thigh_right) / 2 : null)
  return (
    <div className="mt-5 space-y-4">
      <div className="flex flex-wrap items-center gap-3 sm:flex-nowrap sm:gap-4">
        <span className="text-[15px] text-slate-500 whitespace-nowrap">{from}</span>
        <input type="range" min="0" max="1" step="0.001" value={t} onChange={(e) => setT(+e.target.value)}
          aria-label="Morph between the earlier and the current scan"
          className="h-2 min-w-[8rem] flex-1 cursor-pointer appearance-none rounded-full bg-slate-200 accent-[#3b6fd6]"
          style={{ background: `linear-gradient(90deg, #3b6fd6 ${t * 100}%, #e2e8f0 ${t * 100}%)` }} />
        <span className="text-[15px] text-slate-500 whitespace-nowrap">{to}</span>
        <button type="button" onClick={replay}
          className="inline-flex items-center gap-2 rounded-full bg-[#e8effc] px-4 py-2 text-[15px] font-medium text-[#2f5fc4] hover:bg-[#dce6fa]">
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.2" aria-hidden="true"><path d="M7 4l13 8-13 8z" strokeLinejoin="round" /></svg>
          Replay
        </button>
      </div>
      <div className="flex flex-wrap gap-2.5">
        {[...chips.map(([k, l]) => [l, cmp.before[k], cmp.now[k], k]), ['Thigh (avg)', thigh(cmp.before), thigh(cmp.now), 'thigh_left']]
          .map(([label, a, b, site]) => {
            const d = Number.isFinite(a) && Number.isFinite(b) ? b - a : null
            const status = changeStatus(site, a, b)
            return (
              <span key={label} className="rounded-xl bg-slate-100 px-3.5 py-2 text-[15px] text-slate-500">
                {label} <span className="font-medium text-slate-800 tabular-nums">{formatValue(a, unit)}</span>
                <span className="mx-1">→</span>
                <span className="font-medium text-slate-800 tabular-nums">{formatValue(b, unit)}{unit === 'cm' ? ' cm' : ''}</span>
                <span className="ml-1.5 font-semibold tabular-nums" style={{ color: status === 'steady' ? '#64748b' : status === 'smaller' ? '#2f5fc4' : '#a67c00' }}
                  title={`Repeatability (95%): ±${formatValue(repeatabilityCoefficient(site, b), unit)}`}>
                  {formatDelta(d, unit)}
                </span>
              </span>
            )
          })}
      </div>
    </div>
  )
}

// ---------------------------------------------------------------- accuracy
function AccuracyTable({ result, scan, tapeSites }) {
  const sites = Object.keys(result.measurements || {})
  return (
    <div className="space-y-2 text-sm text-slate-600">
      <div className="overflow-x-auto">
        <table className="w-full min-w-[30rem] border-collapse text-left">
          <thead className="text-xs uppercase tracking-wide text-slate-400">
            <tr>
              <th className="py-1 pr-3 font-medium">Site</th>
              <th className="py-1 pr-3 font-medium">{scan.unit === 'cm' ? 'Girth (cm)' : 'Girth (× height)'}</th>
              {scan.unit === 'cm' && <th className="py-1 pr-3 font-medium">± 1 SD</th>}
              <th className="py-1 pr-3 font-medium">Repeat ±95%</th>
              <th className="py-1 font-medium">Tape error</th>
            </tr>
          </thead>
          <tbody>
            {sites.map((s) => {
              const v = scan.values[s]
              const tape = tapeSites?.[s]
              return (
                <tr key={s} className="border-t border-slate-100">
                  <td className="py-1 pr-3">{SITE_FULL_LABELS[s] || s}</td>
                  <td className="py-1 pr-3 tabular-nums text-slate-800">{formatValue(v, scan.unit)}</td>
                  {scan.unit === 'cm' && <td className="py-1 pr-3 tabular-nums">{Number.isFinite(v) ? uncertaintyCm(s, v, scan.relScaleUnc).toFixed(1) : '—'}</td>}
                  <td className="py-1 pr-3 tabular-nums">{formatValue(repeatabilityCoefficient(s, v), scan.unit)}</td>
                  <td className="py-1 tabular-nums">{tape && Number.isFinite(tape.error_cm) ? `${formatDelta(tape.error_cm, 'cm')} cm` : 'not taped'}</td>
                </tr>
              )
            })}
          </tbody>
        </table>
      </div>
      <p className="text-xs leading-relaxed text-slate-500">
        Girths are plane sections of the fitted body mesh at the landmarks shown by the rings (horizontal for the
        trunk, perpendicular to the bone for limbs; tape = convex hull of the section). ± combines the supplied-height
        scale (±1 cm) with run-to-run repeatability measured on independent recordings of one subject — a first
        estimate, not a validated accuracy. Tape error is model − tape where the operator recorded tape values. The
        fit follows the clothed outline, so loose clothing can still overstate girths. External geometry only; nothing
        here measures body fat or muscle.
      </p>
    </div>
  )
}

// ---------------------------------------------------------------- capture conditions
const CLOTHING = [['minimal', 'Minimal'], ['tight', 'Tight'], ['loose', 'Loose'], ['unknown', 'Not recorded']]
const POSTURES = [['a_pose_standard', 'Standard A-pose'], ['arms_lower', 'Arms lower'], ['arms_higher', 'Arms higher'],
  ['relaxed_slouch', 'Relaxed / slouched'], ['other', 'Other']]

/** Tags this recording for the validation report (repeatability per condition, effect of each condition). */
function CaptureConditions({ runId }) {
  const [form, setForm] = useState({ subject_id: '', clothing: 'unknown', distance_m: '', posture: 'a_pose_standard' })
  const [saved, setSaved] = useState(null)
  const [msg, setMsg] = useState(null)
  useEffect(() => {
    let alive = true
    getConditions(runId).then((c) => {
      if (!alive || !c?.subject_id) return
      setSaved(c)
      setForm({ subject_id: c.subject_id, clothing: c.clothing, distance_m: c.distance_m ?? '', posture: c.posture })
    }).catch(() => {})
    return () => { alive = false }
  }, [runId])
  const set = (k) => (e) => setForm((f) => ({ ...f, [k]: e.target.value }))
  const save = async () => {
    setMsg(null)
    const distance = form.distance_m === '' ? null : Number(form.distance_m)
    try {
      const rec = await recordConditions(runId, { ...form, subject_id: form.subject_id.trim(), distance_m: distance })
      setSaved(rec)
      setMsg('Saved — included in scripts/validation_report.py')
    } catch (e) {
      setMsg(e.message)
    }
  }
  const field = 'rounded-lg border border-slate-200 bg-white px-2 py-1 text-sm text-slate-800'
  return (
    <div className="mt-4 border-t border-slate-100 pt-3">
      <p className="text-sm font-medium text-slate-700">Capture conditions of this recording</p>
      <p className="text-xs text-slate-500">
        Used to measure repeatability and the effect of clothing, camera distance and posture. Use a subject code, not a name.
      </p>
      <div className="mt-2 flex flex-wrap items-end gap-2 text-xs text-slate-600">
        <label className="flex flex-col gap-1">Subject code
          <input value={form.subject_id} onChange={set('subject_id')} placeholder="S01" maxLength={40} className={`${field} w-24`} />
        </label>
        <label className="flex flex-col gap-1">Clothing
          <select value={form.clothing} onChange={set('clothing')} className={field}>
            {CLOTHING.map(([v, l]) => <option key={v} value={v}>{l}</option>)}
          </select>
        </label>
        <label className="flex flex-col gap-1">Camera distance (m)
          <input type="number" min="0.5" max="10" step="0.1" value={form.distance_m} onChange={set('distance_m')} className={`${field} w-24`} />
        </label>
        <label className="flex flex-col gap-1">Posture
          <select value={form.posture} onChange={set('posture')} className={field}>
            {POSTURES.map(([v, l]) => <option key={v} value={v}>{l}</option>)}
          </select>
        </label>
        <button type="button" onClick={save} disabled={!form.subject_id.trim()}
          className="rounded-lg bg-slate-800 px-3 py-1.5 text-sm text-white disabled:bg-slate-300">
          {saved ? 'Update' : 'Save'}
        </button>
      </div>
      {msg && <p className="mt-1 text-xs text-slate-500" role="status">{msg}</p>}
    </div>
  )
}

// ---------------------------------------------------------------- bits
function Segmented({ items, value, onChange, disabled = {}, title = {} }) {
  return (
    <div role="tablist" className="inline-flex rounded-full bg-slate-100 p-1.5">
      {items.map(([k, label]) => (
        <button key={k} type="button" role="tab" aria-selected={value === k} disabled={disabled[k]} title={title[k]}
          onClick={() => onChange(k)}
          className={`rounded-full px-4 py-1.5 text-[15px] transition sm:px-6 sm:py-2 sm:text-[17px] ${value === k
            ? 'bg-white font-medium text-slate-900 shadow-sm' : 'text-slate-500 hover:text-slate-700 disabled:opacity-40'}`}>
          {label}
        </button>
      ))}
    </div>
  )
}

function Collapsible({ title, badge, hint, open, onToggle, children }) {
  return (
    <div className="mt-4 rounded-2xl border border-dashed border-slate-200 px-4 py-3 sm:px-6 sm:py-4">
      <button type="button" onClick={onToggle} aria-expanded={open} className="flex w-full items-center justify-between gap-3 text-left">
        <span className="flex items-center gap-3">
          <span className="text-[17px] font-medium text-slate-800 sm:text-lg">{title}</span>
          {badge && <span className="rounded-full bg-amber-50 px-3 py-0.5 text-xs font-semibold tracking-[0.14em] text-amber-700">{badge}</span>}
        </span>
        <span className="flex items-center gap-3 text-[15px] text-slate-500">
          {hint && <span className="hidden sm:inline">{hint}</span>}
          <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"
            className={`transition-transform ${open ? '' : 'rotate-180'}`} aria-hidden="true"><path d="M6 15l6-6 6 6" strokeLinecap="round" strokeLinejoin="round" /></svg>
        </span>
      </button>
      {open && <div className="mt-3">{children}</div>}
    </div>
  )
}
