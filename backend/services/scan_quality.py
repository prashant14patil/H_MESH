"""Capture-quality and reconstruction-quality assessment for a 360-degree body scan.

Pure NumPy/OpenCV (no model inference), so every rule here is unit-testable. The reconstruction pipeline
(``body_reconstruction``) feeds it the data it already has - tracked boxes, keypoint confidences, HMR2
headings, sharpness, silhouette checks - plus cheap per-frame image statistics computed here.

Three questions are answered separately:

1. Rotation: how did the subject turn (direction, speed, total angle) and which per-frame headings are
   inconsistent with that turn (HMR2 front/back flips)?
2. Capture quality: is the recording good enough to reconstruct from (coverage, resolution, lighting,
   visibility, camera stability, useful frames)? Each failed check carries an actionable instruction.
3. Reconstruction quality: given the fit diagnostics, how far can the mesh be trusted? This never claims
   absolute accuracy - that needs independent tape or 3D-scan validation.
"""
from __future__ import annotations

import math

import cv2
import numpy as np

PASS, WARN, FAIL = "pass", "warn", "fail"
_RANK = {PASS: 0, WARN: 1, FAIL: 2}

# ---------------------------------------------------------------- thresholds
COVERAGE_BIN_DEG = 15.0
HEADING_OUTLIER_DEG = 60.0          # heading this far from its temporal neighbours = inconsistent with the turn
TRAJECTORY_WINDOW = 3               # neighbours on each side used to predict a frame's heading
GAP_WARN_DEG, GAP_FAIL_DEG = 45.0, 90.0
ROTATION_GOOD_DEG, ROTATION_MIN_DEG = 330.0, 270.0
TURN_TIME_MIN_SEC, TURN_TIME_MAX_SEC = 8.0, 90.0
SUBJECT_PX_WARN, SUBJECT_PX_FAIL = 600.0, 350.0
LUMA_LOW, LUMA_HIGH = 55.0, 205.0
CLIP_WARN, CLIP_FAIL = 0.05, 0.15
CONTRAST_WARN = 22.0
LUMA_CV_WARN = 0.15                 # lighting changes this much as the subject turns
BG_SHIFT_WARN, BG_SHIFT_FAIL = 0.01, 0.03     # background motion as a fraction of image width
BOX_HEIGHT_CV_WARN, BOX_HEIGHT_CV_FAIL = 0.04, 0.08
# Drift of anatomical anchors (ankle midpoint height, hip midpoint x) as a fraction of the subject's height.
# Box edges are NOT used: arms entering/leaving the box and the lowest stepping foot move them without any drift.
FEET_DRIFT_WARN, FEET_DRIFT_FAIL = 0.05, 0.10
CENTER_DRIFT_WARN = 0.15
USABLE_FRAC_WARN, USABLE_FRAC_FAIL = 0.6, 0.35
SELECTED_GOOD, SELECTED_MIN = 12, 6
MASKS_GOOD, MASKS_MIN = 8, 4
BLURRY_FRAC_WARN = 0.3


def wrap_deg(a):
    return (np.asarray(a, dtype=np.float64) + 180.0) % 360.0 - 180.0


def _circ_mean_deg(angles) -> float:
    r = np.radians(np.asarray(angles, dtype=np.float64))
    return float(np.degrees(np.arctan2(np.sin(r).mean(), np.cos(r).mean())))


def _check(status, value, feedback=None, **extra) -> dict:
    out = {"status": status, "value": value}
    if feedback and status != PASS:
        out["feedback"] = feedback
    out.update(extra)
    return out


def _worst(statuses) -> str:
    statuses = list(statuses)
    return max(statuses, key=_RANK.get) if statuses else PASS


def _finite(values) -> np.ndarray:
    a = np.asarray([v for v in values if v is not None], dtype=np.float64)
    return a[np.isfinite(a)]


# ---------------------------------------------------------------- rotation
def rotation_trajectory(times, yaws, window: int = TRAJECTORY_WINDOW,
                        outlier_deg: float = HEADING_OUTLIER_DEG) -> dict:
    """Rotation of the subject from per-frame headings ordered in time.

    Each heading is compared with the circular mean of its ``window`` neighbours on each side. A frame more
    than ``outlier_deg`` away cannot be explained by a steady turn (typically an HMR2 front/back flip, which
    is ~180 deg). Outliers are replaced by their neighbour prediction before the sequence is unwrapped, so a
    single flip cannot fake or hide a full turn.
    """
    t = np.asarray(times, dtype=np.float64)
    y = wrap_deg(yaws)
    n = len(y)
    if n == 0:
        return {"frames": 0, "outliers": [], "residual_deg": [], "total_rotation_deg": 0.0}
    order = np.argsort(t, kind="stable")
    t, y = t[order], y[order]
    pred = np.full(n, np.nan)
    resid = np.zeros(n)
    for i in range(n):
        nb = [j for j in range(max(0, i - window), min(n, i + window + 1)) if j != i]
        if len(nb) < 2:
            continue
        pred[i] = _circ_mean_deg(y[nb])
        resid[i] = float(wrap_deg(y[i] - pred[i]))
    outlier = np.abs(resid) > outlier_deg
    clean = np.where(outlier & np.isfinite(pred), pred, y)
    unwrapped = np.degrees(np.unwrap(np.radians(clean)))
    steps = np.diff(unwrapped) if n > 1 else np.zeros(0)
    total = float(unwrapped[-1] - unwrapped[0]) if n > 1 else 0.0
    direction = 0 if abs(total) < 1e-6 else int(np.sign(total))
    span = float(t[-1] - t[0]) if n > 1 else 0.0
    speed = float(np.polyfit(t, unwrapped, 1)[0]) if n > 2 and span > 0 else None
    back_steps = int(np.sum(steps * direction < -10.0)) if direction else 0
    max_step_i = int(np.argmax(np.abs(steps))) if len(steps) else None
    inv = np.empty(n, dtype=int)
    inv[order] = np.arange(n)
    return {
        "frames": n,
        "method": f"circular mean of {window} neighbours each side; outlier if |residual| > {outlier_deg:.0f} deg",
        "total_rotation_deg": abs(total),
        "direction": {1: "counter_clockwise_heading_increasing", -1: "clockwise_heading_decreasing",
                      0: "none"}[direction],
        "mean_speed_deg_per_sec": abs(speed) if speed is not None else None,
        "estimated_seconds_per_full_turn": (360.0 / abs(speed)) if speed else None,
        "duration_sec": span,
        "max_step_deg": float(abs(steps[max_step_i])) if max_step_i is not None else 0.0,
        "max_step_between_times_sec": ([float(t[max_step_i]), float(t[max_step_i + 1])]
                                       if max_step_i is not None else None),
        "reversal_steps": back_steps,
        # Per-frame values in the caller's ORIGINAL order.
        "residual_deg": [float(r) for r in resid[inv]],
        "outlier_mask": [bool(o) for o in outlier[inv]],
        "outliers": [int(order[i]) for i in np.flatnonzero(outlier)],
        "cleaned_yaw_deg": [float(c) for c in wrap_deg(clean)[inv]],
    }


def angular_coverage(yaws, bin_deg: float = COVERAGE_BIN_DEG) -> dict:
    """Which headings were seen: covered bins, missing arcs and the largest circular gap between views."""
    y = np.sort(wrap_deg(_finite(yaws)))
    nbins = int(round(360.0 / bin_deg))
    if len(y) == 0:
        return {"views": 0, "bins": nbins, "bins_covered": 0, "coverage_fraction": 0.0, "largest_gap_deg": 360.0,
                "largest_gap_centre_deg": None, "missing_arcs_deg": [[-180.0, 180.0]], "views_by_label": {}}
    occupied = np.zeros(nbins, dtype=bool)
    occupied[((y + 180.0) // bin_deg).astype(int) % nbins] = True
    gaps = np.diff(np.concatenate([y, [y[0] + 360.0]]))
    g = int(np.argmax(gaps))
    arcs = []
    if not occupied.all():
        start = int(np.argmax(occupied))  # rotate so the scan starts on an occupied bin
        run = None
        for k in range(1, nbins + 1):
            b = (start + k) % nbins
            if not occupied[b] and run is None:
                run = b
            elif occupied[b] and run is not None:
                arcs.append([float(wrap_deg(-180.0 + run * bin_deg)), float(wrap_deg(-180.0 + b * bin_deg))])
                run = None
    labels = {"front": 0, "right_side": 0, "back": 0, "left_side": 0}
    for a in y:
        if abs(a) <= 35:
            labels["front"] += 1
        elif abs(a) >= 145:
            labels["back"] += 1
        elif a > 0:
            labels["right_side"] += 1
        else:
            labels["left_side"] += 1
    return {"views": int(len(y)), "bins": nbins, "bin_deg": bin_deg, "bins_covered": int(occupied.sum()),
            "coverage_fraction": float(occupied.mean()), "largest_gap_deg": float(gaps[g]),
            "largest_gap_centre_deg": float(wrap_deg(y[g] + gaps[g] / 2.0)),
            "missing_arcs_deg": arcs, "views_by_label": labels}


def describe_heading(yaw: float) -> str:
    a = float(wrap_deg(yaw))
    if abs(a) <= 35:
        return "front"
    if abs(a) >= 145:
        return "back"
    side = "right side" if a > 0 else "left side"
    if abs(a) < 70:
        return f"front-{side}"
    if abs(a) > 110:
        return f"back-{side}"
    return side


# ---------------------------------------------------------------- per-frame image statistics
def lighting_stats(bgr, box) -> dict | None:
    """Luminance of the subject box: mean, contrast (std) and the clipped share (crushed or blown)."""
    x0, y0, x1, y1 = [int(round(v)) for v in box]
    crop = bgr[max(0, y0):max(0, y1), max(0, x0):max(0, x1)]
    if crop.size == 0:
        return None
    scale = min(1.0, 256.0 / max(crop.shape[0], 1))
    if scale < 1.0:
        crop = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    g = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY).astype(np.float32)
    return {"mean": float(g.mean()), "std": float(g.std()),
            "dark_frac": float((g <= 8).mean()), "bright_frac": float((g >= 247).mean())}


class BackgroundMotion:
    """Global background shift relative to the first frame, subject box masked out (phase correlation).

    A fixed camera gives ~0 shift; a hand-held or bumped camera gives a drifting or jittering shift.
    """

    def __init__(self, width: int = 320):
        self.width = width
        self.ref = None
        self.window = None

    def _prep(self, bgr, box):
        h, w = bgr.shape[:2]
        s = self.width / float(w)
        g = cv2.resize(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY), (self.width, max(8, int(round(h * s))))).astype(np.float32)
        x0, y0, x1, y1 = box
        pad = 0.08 * (y1 - y0)
        xa, xb = int(max(0, (x0 - pad) * s)), int(min(g.shape[1], (x1 + pad) * s))
        ya, yb = int(max(0, (y0 - pad) * s)), int(min(g.shape[0], (y1 + pad) * s))
        g[ya:yb, xa:xb] = float(np.median(g))
        return g, s

    def shift_frac(self, bgr, box) -> float | None:
        """|shift| / image width, or None for the reference frame."""
        g, _ = self._prep(bgr, box)
        if self.ref is None:
            self.ref = g
            self.window = cv2.createHanningWindow((g.shape[1], g.shape[0]), cv2.CV_32F)
            return None
        if g.shape != self.ref.shape:
            return None
        (dx, dy), _resp = cv2.phaseCorrelate(self.ref, g, self.window)
        return float(math.hypot(dx, dy) / g.shape[1])


# ---------------------------------------------------------------- capture assessment
def assess_capture(*, image_wh, frames_decoded: int, tracked_boxes, tracking: dict, usable_frames: int,
                   anchors=None,
                   lighting: list, bg_shift: list, trajectory: dict, coverage_usable: dict,
                   coverage_selected: dict, selected: int, usable_masks: int | None, blurry: int,
                   hmr2_frames: int) -> dict:
    """Score the recording and return actionable feedback for each check that is not a pass.

    ``tracked_boxes``: xyxy boxes of the tracked subject in time order. ``anchors``: optional per tracked frame
    (hip_mid_x, ankle_mid_y) in pixels (None where the joints are not confident). ``tracking``: frames_tracked,
    identity_switches, other_person_detections. ``lighting``: per tracked frame dicts from
    :func:`lighting_stats`. ``bg_shift``: per-frame background shift fractions (None for the reference).
    """
    W, H = image_wh
    checks = {}

    # --- angular coverage -------------------------------------------------------------------------------
    gap = coverage_usable.get("largest_gap_deg", 360.0)
    rot = trajectory.get("total_rotation_deg", 0.0)
    gap_status = PASS if gap <= GAP_WARN_DEG else WARN if gap <= GAP_FAIL_DEG else FAIL
    rot_status = PASS if rot >= ROTATION_GOOD_DEG else WARN if rot >= ROTATION_MIN_DEG else FAIL
    missing = [f"{describe_heading((a + b) / 2)} ({a:.0f} to {b:.0f} deg)"
               for a, b in coverage_usable.get("missing_arcs_deg", [])]
    fb = []
    if rot_status != PASS:
        fb.append(f"Only {rot:.0f} deg of rotation was measured: turn a full 360 deg (finish facing the camera "
                  "again) before stopping the recording.")
    if gap_status != PASS:
        centre = coverage_usable.get("largest_gap_centre_deg")
        fb.append(f"No usable view within a {gap:.0f} deg arc around the "
                  f"{describe_heading(centre) if centre is not None else 'body'}; turn slowly and keep the whole "
                  "body in frame through that part of the rotation.")
    checks["angular_coverage"] = _check(
        _worst([gap_status, rot_status]),
        {"total_rotation_deg": round(rot, 1), "largest_gap_deg": round(gap, 1),
         "bins_covered": f"{coverage_usable.get('bins_covered', 0)}/{coverage_usable.get('bins', 24)}",
         "selected_views_by_label": coverage_selected.get("views_by_label"), "missing_arcs": missing},
        " ".join(fb), thresholds={"largest_gap_deg_warn_fail": [GAP_WARN_DEG, GAP_FAIL_DEG],
                                  "rotation_deg_good_min": [ROTATION_GOOD_DEG, ROTATION_MIN_DEG]})

    # --- rotation speed / steadiness ----------------------------------------------------------------------
    turn = trajectory.get("estimated_seconds_per_full_turn")
    n_out = len(trajectory.get("outliers", []))
    n_traj = max(trajectory.get("frames", 0), 1)
    status, fb = PASS, []
    if turn is not None and turn < TURN_TIME_MIN_SEC:
        status = WARN
        fb.append(f"The turn took about {turn:.0f} s; rotate more slowly (about 15-30 s per full turn) so every "
                  "angle is captured sharply.")
    elif turn is not None and turn > TURN_TIME_MAX_SEC:
        status = WARN
        fb.append(f"The turn would take about {turn:.0f} s; a steadier, continuous 15-30 s turn reduces pose "
                  "changes between views.")
    if trajectory.get("reversal_steps", 0) > max(2, 0.1 * n_traj):
        status = WARN
        fb.append("The rotation reversed direction several times; turn in one direction only.")
    if n_out > 0.15 * n_traj:
        status = WARN
        fb.append(f"{n_out} frames had headings inconsistent with a steady turn (front/back confusion); "
                  "use plain, close-fitting clothing and even lighting so front and back are distinguishable.")
    checks["rotation_steadiness"] = _check(
        status, {"seconds_per_full_turn": None if turn is None else round(turn, 1),
                 "direction": trajectory.get("direction"), "reversal_steps": trajectory.get("reversal_steps"),
                 "heading_outliers": n_out, "max_step_deg": round(trajectory.get("max_step_deg", 0.0), 1)},
        " ".join(fb))

    # --- resolution -----------------------------------------------------------------------------------------
    heights = np.array([b[3] - b[1] for b in tracked_boxes], dtype=float) if len(tracked_boxes) else np.zeros(0)
    med_h = float(np.median(heights)) if len(heights) else 0.0
    status = PASS if med_h >= SUBJECT_PX_WARN else WARN if med_h >= SUBJECT_PX_FAIL else FAIL
    checks["resolution"] = _check(
        status, {"image_wh": [int(W), int(H)], "subject_height_px_median": round(med_h),
                 "subject_height_frac": round(med_h / H, 3) if H else None},
        f"The body spans only {med_h:.0f} px; move the camera closer (body filling ~70-85% of the frame "
        "height, head and feet visible) or record at a higher resolution (1080p or more).",
        thresholds={"subject_px_warn_fail": [SUBJECT_PX_WARN, SUBJECT_PX_FAIL]})

    # --- lighting -------------------------------------------------------------------------------------------
    lit = [s for s in lighting if s]
    if lit:
        mean = np.array([s["mean"] for s in lit])
        std = np.array([s["std"] for s in lit])
        clip = np.array([s["dark_frac"] + s["bright_frac"] for s in lit])
        m, c, k = float(np.median(mean)), float(np.median(std)), float(np.median(clip))
        cv = float(mean.std() / max(mean.mean(), 1e-6))
        st, fb = [], []
        if m < LUMA_LOW:
            st.append(WARN if m >= 0.6 * LUMA_LOW else FAIL)
            fb.append("The subject is underexposed; add light in front of the person (diffuse, from the camera side).")
        elif m > LUMA_HIGH:
            st.append(WARN if m <= LUMA_HIGH + 0.5 * (255 - LUMA_HIGH) else FAIL)
            fb.append("The subject is overexposed; reduce direct light or lower the camera exposure.")
        if k > CLIP_WARN:
            st.append(WARN if k <= CLIP_FAIL else FAIL)
            fb.append(f"{k:.0%} of the body region is clipped to black/white; avoid backlight (windows behind the "
                      "person) and harsh spot lights.")
        if c < CONTRAST_WARN:
            st.append(WARN)
            fb.append("Low contrast between the person and the scene; use a plain background that differs from the "
                      "clothing colour.")
        if cv > LUMA_CV_WARN:
            st.append(WARN)
            fb.append("Brightness changes strongly as the person turns; light the scene evenly from several sides.")
        checks["lighting"] = _check(_worst(st), {"subject_luma_median": round(m, 1), "contrast_std_median": round(c, 1),
                                                 "clipped_frac_median": round(k, 4), "luma_variation_cv": round(cv, 3)},
                                    " ".join(fb), thresholds={"luma_ok": [LUMA_LOW, LUMA_HIGH],
                                                              "clipped_warn_fail": [CLIP_WARN, CLIP_FAIL]})
    else:
        checks["lighting"] = _check(FAIL, None, "No tracked subject frames to assess lighting.")

    # --- body visibility ------------------------------------------------------------------------------------
    n_tr = max(tracking.get("frames_tracked", 0), 1)
    frac = usable_frames / n_tr
    status = PASS if frac >= USABLE_FRAC_WARN else WARN if frac >= USABLE_FRAC_FAIL else FAIL
    fb = ("Many frames were unusable (body cut off by the frame edge or limbs hidden); keep head and feet "
          "inside the frame with some margin, arms slightly away from the body, and nothing in front of the person.")
    st = [status]
    if tracking.get("identity_switches", 0) or tracking.get("other_person_detections", 0) > 0.2 * n_tr:
        st.append(WARN)
        fb += " Other people were visible during the recording; only the subject should be in view."
    checks["body_visibility"] = _check(
        _worst(st), {"usable_of_tracked": f"{usable_frames}/{tracking.get('frames_tracked', 0)}",
                     "usable_fraction": round(frac, 3), "identity_switches": tracking.get("identity_switches", 0),
                     "other_person_detections": tracking.get("other_person_detections", 0),
                     "tracked_of_decoded": f"{tracking.get('frames_tracked', 0)}/{frames_decoded}"}, fb)

    # --- camera consistency ---------------------------------------------------------------------------------
    sh = _finite(bg_shift)
    st, fb = [], []
    med_shift = float(np.median(sh)) if len(sh) else 0.0
    p90_shift = float(np.percentile(sh, 90)) if len(sh) else 0.0
    if len(sh):
        st.append(PASS if p90_shift <= BG_SHIFT_WARN else WARN if p90_shift <= BG_SHIFT_FAIL else FAIL)
        if st[-1] != PASS:
            fb.append("The background moves between frames, so the camera was not fixed; mount it on a tripod "
                      "and do not touch it during the scan.")
    box_cv = feet_drift = cx_drift = None
    if len(heights) >= 3:
        box_cv = float(heights.std() / max(heights.mean(), 1e-6))
        ref = float(np.median(heights))
        pts = np.asarray([a for a in (anchors or []) if a is not None and all(np.isfinite(a))], dtype=float)
        if len(pts) < 3:  # fall back to box edges (over-reports drift)
            b = np.asarray(tracked_boxes, dtype=float)
            pts = np.stack([(b[:, 0] + b[:, 2]) / 2.0, b[:, 3]], 1)
        feet_drift = float((np.percentile(pts[:, 1], 95) - np.percentile(pts[:, 1], 5)) / ref)
        cx_drift = float((np.percentile(pts[:, 0], 95) - np.percentile(pts[:, 0], 5)) / ref)
        st.append(PASS if box_cv <= BOX_HEIGHT_CV_WARN else WARN if box_cv <= BOX_HEIGHT_CV_FAIL else FAIL)
        if st[-1] != PASS:
            fb.append("The person's apparent size changes during the scan (camera zoom or the person stepping "
                      "toward/away); keep a fixed zoom and turn on one spot (mark it on the floor).")
        st.append(PASS if feet_drift <= FEET_DRIFT_WARN else WARN if feet_drift <= FEET_DRIFT_FAIL else FAIL)
        if st[-1] != PASS:
            fb.append("The foot line moves up/down in the image; keep the camera height and tilt fixed and the "
                      "person standing on the same spot.")
        if cx_drift > CENTER_DRIFT_WARN:
            st.append(WARN)
            fb.append("The person drifts sideways while turning; rotate in place on a floor marker.")
    checks["camera_consistency"] = _check(
        _worst(st), {"background_shift_median_frac_width": round(med_shift, 4),
                     "background_shift_p90_frac_width": round(p90_shift, 4),
                     "subject_height_cv": None if box_cv is None else round(box_cv, 4),
                     "ankle_line_drift_frac_subject_height": None if feet_drift is None else round(feet_drift, 4),
                     "hip_horizontal_drift_frac_subject_height": None if cx_drift is None else round(cx_drift, 4),
                     "anchor_source": "keypoints" if anchors and sum(a is not None for a in anchors) >= 3
                     else "box_edges"},
        " ".join(fb), thresholds={"bg_shift_warn_fail": [BG_SHIFT_WARN, BG_SHIFT_FAIL],
                                  "subject_height_cv_warn_fail": [BOX_HEIGHT_CV_WARN, BOX_HEIGHT_CV_FAIL],
                                  "feet_drift_warn_fail": [FEET_DRIFT_WARN, FEET_DRIFT_FAIL]})

    # --- enough useful frames -------------------------------------------------------------------------------
    st, fb = [PASS if selected >= SELECTED_GOOD else WARN if selected >= SELECTED_MIN else FAIL], []
    if st[0] != PASS:
        fb.append(f"Only {selected} distinct, sharp viewpoints were usable (want {SELECTED_GOOD}+); record a slower, "
                  "complete turn.")
    if usable_masks is not None:
        st.append(PASS if usable_masks >= MASKS_GOOD else WARN if usable_masks >= MASKS_MIN else FAIL)
        if st[-1] != PASS:
            fb.append(f"Only {usable_masks} reliable silhouettes; use a plain, uncluttered background that contrasts "
                      "with the clothing.")
    if hmr2_frames and blurry > BLURRY_FRAC_WARN * hmr2_frames:
        st.append(WARN)
        fb.append(f"{blurry} frames were rejected as blurry; add light (shorter exposure) or turn more slowly.")
    checks["useful_frames"] = _check(_worst(st), {"selected_views": selected, "reliable_silhouettes": usable_masks,
                                                  "blurry_rejected": blurry, "hmr2_frames": hmr2_frames}, " ".join(fb),
                                     thresholds={"selected_good_min": [SELECTED_GOOD, SELECTED_MIN],
                                                 "silhouettes_good_min": [MASKS_GOOD, MASKS_MIN]})

    worst = _worst(c["status"] for c in checks.values())
    verdict = {PASS: "good", WARN: "acceptable_with_warnings", FAIL: "insufficient"}[worst]
    return {"verdict": verdict, "checks": checks,
            "feedback": [c["feedback"] for c in checks.values() if c.get("feedback")],
            "note": "Thresholds are engineering defaults for a fixed phone/webcam setup, not validated cut-offs. "
                    "'insufficient' means re-recording is recommended; the mesh may still be produced."}


# ---------------------------------------------------------------- reconstruction assessment
def assess_reconstruction(report: dict) -> dict:
    """Trust indicators for the unified mesh from ``reconstruction_report.json`` (no ground truth involved)."""
    vs = report.get("validation_summary", {})
    sf = report.get("silhouette_fitting", {})
    mo = report.get("multiview_optimisation", {})
    new = sf.get("summary_by_method", {}).get("silhouette_keypoint_shared_shape (new)", {})
    held = (mo.get("held_out_validation", {}).get("results", {})
            .get("silhouette_keypoint_shared_shape (new)", {}))
    betas = np.asarray(report.get("shared_shape", {}).get("betas") or [0.0], dtype=float)
    checks = {}
    geo = report.get("geometry_validation", {})
    checks["mesh_geometry"] = _check(
        PASS if geo.get("valid") and geo.get("watertight", True) else FAIL,
        {k: geo.get(k) for k in ("valid", "watertight", "upright", "facing_plus_z", "connected_components",
                                 "vertex_count", "face_count")},
        "The exported mesh failed topology/orientation validation.")
    kp = vs.get("landmark_reprojection_error_pct_height")
    checks["keypoint_reprojection"] = _check(
        PASS if kp is not None and kp <= 2.0 else WARN if kp is not None and kp <= 4.0 else FAIL,
        None if kp is None else round(kp, 2),
        "2D joints are poorly explained by the fitted poses; check tracking and viewpoint conflicts.",
        unit="% of body height (median)")
    hiou = held.get("iou_mean")
    checks["held_out_silhouette"] = _check(
        PASS if hiou is not None and hiou >= 0.78 else WARN if hiou is not None and hiou >= 0.70 else FAIL,
        None if hiou is None else round(hiou, 3),
        "The shared shape generalises poorly to views it was not fitted on.",
        note="IoU vs the CLOTHED outline on unseen views; rewards inflation, so it is never used alone.")
    prot = new.get("protrusion_frac_mean")
    checks["outline_protrusion"] = _check(
        PASS if prot is not None and prot <= 0.03 else WARN if prot is not None and prot <= 0.05 else FAIL,
        None if prot is None else round(prot, 4),
        "The body extends outside its own clothed outline (physically impossible): masks or cameras are off.")
    side = ((sf.get("proportion_check") or {}).get("side_depth") or {}).get("median_ratio") or []
    front = ((sf.get("proportion_check") or {}).get("front_back_width") or {}).get("median_ratio") or []
    over = [v for v in list(side) + list(front) if v is not None and v > 1.05]
    checks["proportions"] = _check(
        PASS if not over else WARN, {"side_depth_ratio": side, "front_width_ratio": front},
        "Model torso width/depth exceeds the observed outline at some level.",
        note="model/observed torso width per level; <=1 expected under clothing")
    bmax = float(np.abs(betas).max())
    checks["shape_plausibility"] = _check(
        PASS if bmax <= 2.5 else WARN if bmax <= 3.0 else FAIL, round(bmax, 2),
        "Shape coefficients are far outside the body-shape population (loose clothing or mask errors).",
        unit="max |beta| in population SD")
    meas = report.get("measurements", {})
    ok, tot = meas.get("valid", 0), meas.get("total", 0)
    checks["measurement_sections"] = _check(PASS if tot and ok == tot else WARN if ok >= 0.75 * max(tot, 1) else FAIL,
                                            f"{ok}/{tot}", "Some circumference cross-sections were invalid.")
    worst = _worst(c["status"] for c in checks.values())
    return {"verdict": {PASS: "consistent", WARN: "usable_with_warnings", FAIL: "unreliable"}[worst],
            "checks": checks,
            "feedback": [c["feedback"] for c in checks.values() if c.get("feedback")],
            "accuracy_statement": "These are internal-consistency checks. Absolute measurement accuracy is NOT "
                                  "established without independent tape or 3D-scan validation."}
