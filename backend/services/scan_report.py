"""Self-contained HTML results page for one completed 360-degree scan (``scan_report.html`` in the run folder).

Reads only files the pipeline already wrote, so it can be regenerated for any past run:

    from backend.services import scan_report; scan_report.write_html(run_dir, baseline_dir=None)

Images are referenced relatively (they sit next to the page). Every value is HTML-escaped.
"""
from __future__ import annotations

import html
import json
import time
from pathlib import Path

SITES = ["chest", "waist", "abdomen", "hip", "upper_arm_left", "upper_arm_right", "forearm_left", "forearm_right",
         "thigh_left", "thigh_right", "calf_left", "calf_right"]
IMAGES = [("final_mesh_views.jpg", "Unified mesh: front, side, back, side"),
          ("mesh_turntable.gif", "Turntable"),
          ("measurements_overlay.jpg", "Measurement planes"),
          ("view_comparison.jpg", "Fitted body over observed frames"),
          ("contact_sheet.jpg", "Selected frames across the rotation"),
          ("silhouette_masks.jpg", "Subject silhouettes"),
          ("orientation_check.jpg", "Orientation check")]


def _load(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _e(v) -> str:
    return html.escape("" if v is None else str(v))


def _num(v, fmt="{:.3f}"):
    return "-" if not isinstance(v, (int, float)) else fmt.format(v)


def _badge(status: str) -> str:
    cls = {"pass": "ok", "good": "ok", "consistent": "ok", "warn": "warn", "acceptable_with_warnings": "warn",
           "usable_with_warnings": "warn", "fail": "bad", "insufficient": "bad", "unreliable": "bad"}.get(status, "")
    return f'<span class="badge {cls}">{_e(status)}</span>'


def _value(v) -> str:
    if isinstance(v, dict):
        return "<br>".join(f"<code>{_e(k)}</code>: {_e(json.dumps(x) if isinstance(x, (list, dict)) else x)}"
                           for k, x in v.items() if x is not None)
    return _e(v)


def _checks_table(report: dict | None, title: str) -> str:
    if not report:
        return f"<section><h2>{_e(title)}</h2><p>Not available for this run.</p></section>"
    rows = "".join(f"<tr><td>{_e(k.replace('_', ' '))}</td><td>{_badge(c.get('status'))}</td>"
                   f"<td class='small'>{_value(c.get('value'))}</td><td>{_e(c.get('feedback', ''))}</td></tr>"
                   for k, c in report.get("checks", {}).items())
    extra = report.get("accuracy_statement") or report.get("note") or ""
    return (f"<section><h2>{_e(title)} {_badge(report.get('verdict'))}</h2>"
            f"<div class='scroll'><table><thead><tr><th>Check</th><th>Status</th><th>Value</th><th>Action</th></tr></thead>"
            f"<tbody>{rows}</tbody></table></div><p class='note'>{_e(extra)}</p></section>")


def _rel(meas: dict | None, site: str):
    m = ((meas or {}).get("measurements") or {}).get(site) or {}
    return m.get("tape_circumference_over_height") if m.get("valid", True) else None


def _cm(scaled: dict | None, site: str):
    return (((scaled or {}).get("measurements") or {}).get(site) or {}).get("tape_circumference_cm")


def build_html(run_dir: Path, baseline_dir: Path | None = None) -> str:
    run_dir = Path(run_dir)
    report = _load(run_dir / "reconstruction_report.json") or {}
    status = _load(run_dir / "status.json") or {}
    meas = _load(run_dir / "measurements.json")
    scaled = _load(run_dir / "measurements_cm.json")
    shapy = _load(run_dir / "shapy_measurements.json")
    cq = _load(run_dir / "capture_quality_report.json")
    rq = _load(run_dir / "reconstruction_quality_report.json")
    tape = _load(run_dir / "tape_validation.json")
    base_meas = _load(baseline_dir / "measurements.json") if baseline_dir else None
    base_report = (_load(baseline_dir / "reconstruction_report.json") or {}) if baseline_dir else {}
    height_cm = (scaled or {}).get("height_cm")
    factor = (scaled or {}).get("cm_per_mesh_unit")
    shapy_scale = None
    if shapy and height_cm and shapy.get("mesh_height"):
        shapy_scale = height_cm / shapy["mesh_height"]

    # --- measurements -------------------------------------------------------------------------------------
    head = ["Site", "Pipeline (x height)"]
    if height_cm:
        head.append("Pipeline cm")
    head.append("Keypoint-only (x height)")
    if shapy:
        head.append("SHAPY (x height)")
        if shapy_scale:
            head.append("SHAPY cm")
    if base_meas:
        head += ["Baseline run (x height)", "Change"]
    if tape:
        head += ["Tape cm", "Error cm"]
    rows = []
    stature = ["Height", "1 (by definition)"]
    if height_cm:
        stature.append(f"{height_cm:.1f} (supplied, not measured)")
    stature.append("1")
    if shapy:
        stature.append("1")
        if shapy_scale:
            stature.append(f"{height_cm:.1f} (supplied)")
    if base_meas:
        stature += ["1", ""]
    if tape:
        stature += ["", ""]
    rows.append(stature)
    prev = (meas or {}).get("previous_method_keypoint_only") or {}
    for site in SITES:
        r = _rel(meas, site)
        row = [site.replace("_", " "), _num(r)]
        if height_cm:
            row.append(_num(_cm(scaled, site), "{:.1f}"))
        row.append(_num(prev.get(site)))
        if shapy:
            s = _rel(shapy, site)
            row.append(_num(s))
            if shapy_scale:
                sm = ((shapy.get("measurements") or {}).get(site) or {}).get("tape_circumference")
                row.append(_num(sm * shapy_scale if isinstance(sm, (int, float)) else None, "{:.1f}"))
        if base_meas:
            b = _rel(base_meas, site)
            row += [_num(b), _num((r - b) / b * 100 if isinstance(r, float) and isinstance(b, float) and b else None,
                                  "{:+.1f}%")]
        if tape:
            t = (tape.get("sites") or {}).get(site) or {}
            row += [_num(t.get("tape_cm"), "{:.1f}"), _num(t.get("error_cm"), "{:+.1f}")]
        rows.append(row)
    mtable = ("<div class='scroll'><table><thead><tr>" + "".join(f"<th>{_e(h)}</th>" for h in head) + "</tr></thead><tbody>"
              + "".join("<tr>" + "".join(f"<td>{_e(c)}</td>" for c in r) + "</tr>" for r in rows) + "</tbody></table></div>")
    scale_note = (f"Centimetres use the operator-supplied barefoot height {height_cm:.1f} cm "
                  f"({factor:.2f} cm per mesh unit). Height is therefore an input, not a measurement."
                  if height_cm else
                  "No height or calibration was supplied, so values are relative (fraction of body height) only. "
                  "Re-run with --height-cm to get centimetres.")
    tape_note = (f"Tape values recorded ({_e(tape.get('clothing'))} clothing). Note: {_e(tape.get('note'))}"
                 if tape else "No independent tape measurements recorded: measurement accuracy is NOT validated.")

    # --- reconstruction method comparison -------------------------------------------------------------------
    held = (report.get("multiview_optimisation", {}).get("held_out_validation", {}).get("results") or {})
    fit_sum = report.get("silhouette_fitting", {}).get("summary_by_method") or {}
    mrows = "".join(
        f"<tr><td>{_e(k)}</td><td>{_num(v.get('iou_mean'))}</td><td>{_num(v.get('protrusion_frac_mean'), '{:.3f}')}</td>"
        f"<td>{_num((fit_sum.get(k) or {}).get('iou_mean'))}</td><td>{_num(v.get('keypoint_error_pct_height'), '{:.2f}')}</td></tr>"
        for k, v in held.items())
    methods = ("<div class='scroll'><table><thead><tr><th>Method</th><th>Held-out IoU</th><th>Held-out protrusion</th>"
               "<th>Fit IoU</th><th>Held-out keypoint err %H</th></tr></thead><tbody>" + mrows + "</tbody></table></div>")

    # --- before / after -------------------------------------------------------------------------------------
    compare = ""
    if baseline_dir:
        bv, nv = base_report.get("validation_summary", {}), report.get("validation_summary", {})
        keys = ["frames_extracted", "frames_tracked", "frames_in_final_fit", "frames_excluded",
                "silhouette_iou_mean_final_fit", "landmark_reprojection_error_pct_height",
                "circumference_sections_valid"]
        crow = "".join(f"<tr><td>{_e(k)}</td><td>{_e(_fmt(bv.get(k)))}</td><td>{_e(_fmt(nv.get(k)))}</td></tr>" for k in keys)
        bh = (base_report.get("multiview_optimisation", {}).get("held_out_validation", {}).get("results") or {})
        key = "silhouette_keypoint_shared_shape (new)"
        crow += (f"<tr><td>held-out IoU (fitted shape)</td><td>{_num((bh.get(key) or {}).get('iou_mean'))}</td>"
                 f"<td>{_num((held.get(key) or {}).get('iou_mean'))}</td></tr>")
        crow += (f"<tr><td>total time (s)</td><td>{_num(base_report.get('timings_sec', {}).get('total'), '{:.0f}')}</td>"
                 f"<td>{_num(report.get('timings_sec', {}).get('total'), '{:.0f}')}</td></tr>")
        crow += (f"<tr><td>capture-quality report</td><td>{'yes' if (baseline_dir / 'capture_quality_report.json').is_file() else 'no'}</td>"
                 f"<td>{'yes' if cq else 'no'}</td></tr>")
        compare = (f"<section><h2>Before / after</h2><p>Baseline run <code>{_e(baseline_dir.name)}</code> vs this run.</p>"
                   "<div class='scroll'><table><thead><tr><th>Metric</th><th>Baseline</th><th>This run</th></tr></thead><tbody>"
                   + crow + "</tbody></table></div></section>")

    # --- timings / resources --------------------------------------------------------------------------------
    t = report.get("timings_sec") or {}
    gpu = report.get("gpu") or {}
    trow = "".join(f"<tr><td>{_e(k)}</td><td>{_num(v, '{:.1f}')}</td></tr>" for k, v in t.items())
    res = {"GPU": gpu.get("device_name") or "none (CPU)", "peak GPU memory allocated (MiB)": gpu.get("peak_memory_allocated_mb"),
           "peak GPU memory reserved (MiB)": gpu.get("peak_memory_reserved_mb"),
           "process peak RSS (MiB)": gpu.get("process_peak_rss_mb"), "torch": gpu.get("torch")}
    rrow = "".join(f"<tr><td>{_e(k)}</td><td>{_e(v)}</td></tr>" for k, v in res.items())

    imgs = "".join(f"<figure><img src='{_e(name)}' alt='{_e(cap)}' loading='lazy'><figcaption>{_e(cap)}</figcaption></figure>"
                   for name, cap in IMAGES if (run_dir / name).is_file())
    warnings = "".join(f"<li><code>{_e(w.get('code'))}</code> {_e(w.get('message'))}</li>"
                       for w in (report.get("warnings") or status.get("warnings") or []))
    limits = "".join(f"<li>{_e(x)}</li>" for x in report.get("limitations", []))
    vs = report.get("validation_summary", {})
    created = status.get("created_at")
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>360 Body Scan</title>
<style>
:root{{--bg:#f7f7f5;--fg:#1d1d1b;--muted:#666;--card:#fff;--line:#ddd;--ok:#1f7a3a;--warn:#a06100;--bad:#b3261e}}
@media (prefers-color-scheme: dark){{:root{{--bg:#151514;--fg:#ececea;--muted:#a0a09c;--card:#1f1f1d;--line:#3a3a37;--ok:#5cc27a;--warn:#e0a640;--bad:#f07167}}}}
body{{background:var(--bg);color:var(--fg);font:15px/1.5 system-ui,sans-serif;margin:0;padding:16px}}
main{{max-width:1180px;margin:auto}} section{{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:16px;margin:16px 0}}
h1{{font-size:1.6rem;margin:.2em 0}} h2{{font-size:1.15rem;margin:0 0 .6em}} .muted,.note{{color:var(--muted)}} .note{{font-size:.85rem}}
table{{border-collapse:collapse;width:100%}} th,td{{border-bottom:1px solid var(--line);padding:6px 8px;text-align:left;vertical-align:top}}
th{{font-weight:600}} .small{{font-size:.8rem}} .scroll{{overflow-x:auto}} code{{font-size:.85em}}
.badge{{display:inline-block;padding:1px 8px;border-radius:10px;font-size:.8rem;border:1px solid currentColor}}
.ok{{color:var(--ok)}} .warn{{color:var(--warn)}} .bad{{color:var(--bad)}}
.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(320px,1fr));gap:12px}} figure{{margin:0}}
img{{max-width:100%;border:1px solid var(--line);border-radius:4px;background:#fff}} figcaption{{font-size:.85rem;color:var(--muted)}}
.kpis{{display:flex;flex-wrap:wrap;gap:16px}} .kpis div{{min-width:140px}} .kpis b{{display:block;font-size:1.2rem}}
</style></head><body><main>
<h1>360° body scan</h1>
<p class="muted">Run <code>{_e(report.get('run_id'))}</code> · recording <code>{_e(report.get('video_id'))}</code>
{(' · ' + _e(time.strftime('%Y-%m-%d %H:%M', time.localtime(created)))) if created else ''}</p>
<section><div class="kpis">
<div><span class="muted">Capture quality</span><b>{_badge((cq or {}).get('verdict', 'n/a'))}</b></div>
<div><span class="muted">Reconstruction</span><b>{_badge((rq or {}).get('verdict', 'n/a'))}</b></div>
<div><span class="muted">Frames used / decoded</span><b>{_e(vs.get('frames_in_final_fit'))} / {_e(vs.get('frames_extracted'))}</b></div>
<div><span class="muted">Sections valid</span><b>{_e(vs.get('circumference_sections_valid'))}</b></div>
<div><span class="muted">Scale</span><b>{'cm (from height)' if height_cm else 'relative'}</b></div>
<div><span class="muted">Total time</span><b>{_num(t.get('total'), '{:.0f}')} s</b></div>
</div><p class="note">{_e(report.get('scope'))}</p></section>
<section><h2>Mesh and diagnostics</h2><div class="grid">{imgs}</div>
<p class="note">Files: final_body_mesh.obj / .ply (relative units){', final_body_mesh_metric.obj (metres)' if height_cm else ''}, measurement_rings.obj.</p></section>
<section><h2>Measurements</h2>{mtable}<p class="note">{_e(scale_note)} {tape_note}
Pipeline = HMR2 initialisation + one shared SMPL shape fitted to all views. Keypoint-only = previous method on the same views.
SHAPY = independent shape regressor, same A-pose and section definitions.</p></section>
{compare}
{_checks_table(cq, 'Capture quality')}
{_checks_table(rq, 'Reconstruction quality')}
<section><h2>Shape methods on held-out views</h2>{methods}
<p class="note">IoU is scored against the clothed outline and rewards inflation; judge with protrusion and proportions too.</p></section>
<section><h2>Processing time and resources</h2><div class="grid"><div class="scroll"><table><tbody>{trow}</tbody></table></div>
<div class="scroll"><table><tbody>{rrow}</tbody></table></div></div></section>
<section><h2>Warnings</h2><ul>{warnings or '<li>None</li>'}</ul></section>
<section><h2>Known limitations</h2><ul>{limits}</ul></section>
</main></body></html>"""


def _fmt(v):
    return f"{v:.3f}" if isinstance(v, float) else v


def write_html(run_dir: Path, baseline_dir: Path | None = None) -> Path:
    out = Path(run_dir) / "scan_report.html"
    out.write_text(build_html(Path(run_dir), Path(baseline_dir) if baseline_dir else None), encoding="utf-8")
    return out
