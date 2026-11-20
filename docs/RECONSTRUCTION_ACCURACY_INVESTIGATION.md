# Reconstruction accuracy investigation: root cause, fix and validation

Date: 18–19 Nov 2026 · Scope: unified 360° reconstruction (`backend/services/body_reconstruction.py`, `smpl_multiview.py`) · Status: **root cause fixed and validated on silhouette consistency; absolute 3D accuracy NOT established (no real tape or 3D scan yet)**

## 1. Summary

| Question | Finding |
|---|---|
| Where is the error? | In the silhouette objective of the shared-shape fit (`smpl_multiview._silhouette_loss`). Image loading, segmentation, tracking, yaw estimation and camera alignment were tested and are not the cause. |
| What was wrong? | The "body must stay inside its outline" term saturated (Geman-McClure) and was averaged over all 6890 vertices. The "fill the outline" term was averaged over ~400 contour points, so it was about 17× stronger. The body therefore grew past its own clothed outline. |
| Why did results swing between versions? | Every version kept the broken term and changed only the shape-prior weight to compensate: 0.02 (original, never committed), then 0 (committed: inflated), then 0.2 (pre-fix working copy: shape pinned to HMR2, profile depth about 20% short). |
| Are the problems in section 2 of the brief one bug or several? | One common cause. Inflated proportions, shallow side-view depth, the generic/smooth look and inconsistent view contributions all come from the same objective plus prior trade-off. |
| What was changed? | A non-saturating inside term on the same scale as the coverage term. The HMR2-jitter prior is replaced by a weak population prior. New consistency metrics (protrusion, front/side proportion check) and warnings. Every run records its exact fit configuration. |
| Did the 3D body improve? | **Objectively yes in one respect:** protrusion outside the clothed outline, which is physically impossible, fell from 3.9% to 1.6%. Profile depth now matches the observed outline (0.96–0.99) instead of 0.79–0.91. **Whether the resulting girths are closer to the truth is not established.** That needs real tape (see §7). |

## 2. Pipeline audit (what was checked, and the evidence)

| Stage | Code | Check | Result |
|---|---|---|---|
| Frame decode and orientation | `decode_frames` | `orientation_check.jpg`; head above feet in 100% of frames | OK |
| Person detection and tracking | `person_tracking.py` | identity events; single subject | OK |
| Segmentation | `subject_masks.py` (Mask R-CNN GPU + GrabCut + MediaPipe vote) | `silhouette_masks.jpg`, agreement 0.91–0.94 on all 24 views | OK; masks follow the clothed outline as intended |
| Yaw / view identification | `heading_yaw_deg`, `verify_view` | yaw drift between HMR2 init and final fit | ≤ 1.5° on every view: **not a cause** |
| Camera model | `crop_cam_to_full`, HMR2 focal | initial keypoint reprojection check; per-view camera stays consistent | OK. Focal is HMR2's assumption (a limitation, not the cause) |
| Multi-view fusion | `fit_multiview` (one shared beta vector) | single shape for all views | Correct by construction |
| **Silhouette objective** | `_silhouette_loss` | protrusion; controlled refits; synthetic ground truth | **Root cause** (§3) |
| Shape prior | `BETA_PRIOR_WEIGHT`, `beta_sigma` | Mahalanobis change from HMR2 | σ is HMR2's frame-to-frame jitter (floor 0.25), so "0.2" is very strong: shape moved 0.13 σ, i.e. silhouettes were ignored |
| Per-view pose | stage B body pose | same shape, keypoint-only pose | Pose absorbs ~0.045 IoU per fit view (explains the fit vs held-out gap, §4) |
| Final mesh / export | `canonical_mesh`, `export_outputs` | OBJ re-read and validated; measured from the file | OK. Shared betas in a fixed A-pose (per-view pose not carried, by design) |
| Measurements | `body_measurements.measure_body` | 12/12 sections valid | OK. Inherits whatever the shape is |

## 3. Root cause evidence

### 3.1 Controlled refits (same saved inputs, same GPU code; recordings B = `8ba5ec5f`, A = `38895355`)

4-fold cross-validated; "side depth" = model/observed torso width in profile views, abdomen–hip levels.

| Config | Held-out IoU B / A | Fit IoU | Fit protrusion | Side depth | max \|β\| | Hip / height B / A |
|---|---|---|---|---|---|---|
| Original: legacy term, prior 0.02 | .805 / .812 | .859 / .850 | 3.9 / 4.2% | 0.85–0.98 | 1.1 | .623 / .587 |
| Committed: legacy, prior 0 | — | .867 / .878 | 7.7 / 7.2% | **1.08–1.12** | 2.6 / 2.8 | .741 / .720 |
| Pre-fix working copy: legacy, prior 0.2 | .782 / .796 | .829 / .830 | 3.4 / 3.9% | **0.79–0.91** | 1.5 / 1.7 | .574 / .558 |
| Fixed term, no prior | .818 / .835 | .888 / .891 | 1.8 / 1.8% | 0.92–1.01 | 1.6 / 2.2 | .666 / .658 |
| **Fixed term + population prior 0.05 (new default)** | .816 / .830 | .885 / .886 | **1.6 / 1.7%** | 0.92–1.00 | **0.9 / 0.6** | .656 / .646 |
| Fixed + per-view pose locked | .790 / .814 | .804 / .826 | 2.5 / 2.5% | 0.79–0.95 | **3.8** / 2.6 | .641 / .627 |

- Prior 0 with the legacy term pushes the body about 10% *outside* its own clothed outline in profile: physically impossible.
- Prior 0.2 removes that by pinning the shape to HMR2's estimate. Silhouettes no longer change it (0.13 σ), and profile depth drops 10–20% *inside* the outline. That produces the generic, shallow torso.
- The fixed term keeps the body inside the outline with no prior needed to suppress inflation. A third recording (`2e0b075f`, checked independently by the validation session) shows the same pattern: protrusion 1.3–1.4% vs 3.4–4.1%, with no repeatability cost.
- The original run `eb02e080` (code never committed) is reproduced by the legacy term at prior 0.02: hip/H 0.587 vs 0.581, fit IoU 0.850 vs 0.848.

### 3.2 Synthetic ground truth (known 3D shape; `backend/tests/test_multiview_shape_fusion.py`)

| Case | Legacy term | Fixed term | Truth |
|---|---|---|---|
| Exact outlines: width / depth coefficient | +1.05 / −0.85 | +1.02 / −0.91 | +1 / −1 |
| Outline grown 25 px ("loose garment"): width / depth | +2.02 / +0.77 | +1.88 / +0.43 | +1 / −1 |

The fixed term is closer to the truth in both cases. Under a uniformly loose garment **both** inflate: silhouettes cannot see under clothing, and no objective change fixes that.

## 4. Metric audit

| Metric | How it is computed | What it can't measure |
|---|---|---|
| Fit IoU | Rasterised mesh vs mask on the fitted views, with each view's pose and camera optimised on that very silhouette | Inflated by per-view pose: same shape with keypoint-only pose scores ~0.045 lower |
| Held-out IoU | 4 interleaved folds over heading; shape from the other folds, held-out pose from keypoints only; held-out silhouettes never used (verified in `fit_unified`: `hv.contours = None`) | Genuinely held out. But it is scored against the **clothed** outline, so it rewards a body that fills loose clothing. This is how prior 0 was once chosen "by cross-validation" |
| Fit vs held-out gap | Difference of the two | Mostly per-view pose absorption (pre-fix: same-shape keypoint-pose IoU .783 ≈ reported held-out .782). Locking pose closes the gap but makes the shape absorb pose error instead (β up to 3.8 SD) |
| Final mesh | Metrics use the same shared betas as the exported mesh; per-view pose is not exported | Correspond to the exported shape |
| **Not measurable by any of these** | — | Whether the body under loose clothing is correct; absolute girths in cm |

New metrics added because of this: `protrusion_frac` (body outside the outline, always an error), `uncovered_frac` (outline left empty, expected under clothing), and `proportion_check` (model/observed torso width per level, frontal = width, profile = depth).

## 5. End-to-end runs (same video `777541cd`, same code, only the configuration differs)

| | Original config (`776042ea`) | New default (`268f55fd`) |
|---|---|---|
| Fit IoU | 0.859 | 0.885 |
| Held-out IoU | 0.805 | 0.815 |
| Fit protrusion | 3.9% (warning raised) | 1.6% |
| Held-out protrusion | 4.7% | 5.9% |
| Side depth, model/observed (chest → hip) | 1.02, 0.93, 0.98, 0.94 | 0.99, 0.96, 0.96, 0.98 |
| Hip / waist ÷ height | 0.623 / 0.529 | 0.654 / 0.560 |
| Geometry valid / sections | yes / 12 of 12 | yes / 12 of 12 |

All four configurations, run end to end on the same video (`scripts/compare_reconstructions.py`; cm assume 179 cm height):

| | Original (`776042ea`) | Pre-fix working copy (`e552b3db`) | Fixed + HMR2 prior 0.2 (`0f761c80`) | **Fixed default (`268f55fd`)** |
|---|---|---|---|---|
| Held-out IoU | 0.805 | 0.782 | 0.783 | **0.815** |
| Fit protrusion | 3.9% | 3.4% | **1.3%** | 1.6% |
| Side depth, model/observed | 0.94–1.02 | 0.81–0.98 | 0.87–0.93 | 0.96–0.99 |
| Change from HMR2 shape (σ) | 1.19 | 0.13 | 0.18 | 2.12 |
| Chest / waist / abdomen / hip (cm) | 108.8 / 94.7 / 98.3 / 111.5 | 104.0 / 87.9 / 90.8 / 102.7 | 104.0 / 87.7 / 90.6 / 103.1 | 114.2 / 100.2 / 104.0 / 117.1 |

Per view (fit / held-out IoU), default: front 0.868 / 0.820, left 0.904 / 0.826, back 0.859 / 0.839, right 0.889 / 0.809. The default beats all three alternatives on every view.

### 5.1 Model-free check: multi-view visual hull (`scripts/visual_hull_girths.py`)

At each torso height the slice is kept only where it lies inside the silhouette in all 24 views (1 view of tolerance). Its convex perimeter is the girth of the **clothed** body seen by the cameras: an upper bound that uses neither SMPL nor SHAPY. Default fit, at 179 cm:

| Level | Clothed hull girth / depth | Mesh girth / depth | Mesh/hull | Reading |
|---|---|---|---|---|
| Hip (buttocks) | 122.7 / 33.9 cm | 117.2 / 32.2 cm | 0.96 | Mesh nearly fills the loose cargo trousers, so it carries trouser ease |
| Abdomen | 117.0 / 32.9 cm | 103.9 / 28.8 cm | 0.89 | Mesh inside the T-shirt drape |
| Waist | 114.3 / 32.6 cm | 100.7 / 29.3 cm | 0.88 | Shirt hangs flat over the lumbar curve and loose at the sides (visible in `hull_slices_*.png`) |
| Chest | depth 30.8 cm (girth not measurable: hanging arms and sleeves join the slice) | depth 30.0 cm | — | Shirt touches front and back, so depth is reliable; mesh fills it |

No fitted girth exceeds the clothed hull. The hull shows clearly where clothing adds girth: the hip (loose trousers) and the waist sides and back (drape).

### 5.2 Hypothesis tested and rejected: loose trouser legs inflate the torso

Down-weighting coverage on the outline between hip and ankle keypoints to 0.3 (as already done for hair and shoes) changed the girths by −1.0 cm (hip) to +1.8 cm (waist), and held-out IoU 0.816 → 0.817. The trouser legs are **not** what drives the torso size, so the option was not kept.

### 5.3 Against SHAPY (screenshot, same video, same `measure_body` code and A-pose)

SHAPY is a single-image regressor trained on model-agency photos. It sees neither the 360° outlines nor this subject's profile, and it gives chest 106.1, waist 88.5, abdomen 90.8, hip 105.0 cm. The same measurement code runs on both, so the 8–13 cm differences are body shape, not plane placement. SHAPY's waist of 88.5 cm would need a near-circular ~28 cm body section inside a clothed section 38 × 33 cm. That is possible only under a very loose shirt, so SHAPY is probably low here. Neither model is ground truth.

### 5.4 Best estimate of real tape (reasoned, NOT measured) — SUPERSEDED, see §5.8

> **Correction (20 Nov).** Real tape (waist 88–90, abdomen 95–100, hip 98–103 cm) shows these ranges were
> about 8–10 cm too high: loose clothing adds about 25 cm to the outline at waist and hip, far more than the
> 6–12 cm ease assumed below. The ANSUR "typical hip ≈ 108 cm" was also too high, because it was conditioned
> on girths that clothing had already inflated. Kept for the record only.

Body = clothed hull minus clothing allowance (fabric plus ease or drape), cross-checked with the mesh:

| Site | Pipeline (default) | SHAPY | Estimated real tape | Basis |
|---|---|---|---|---|
| Chest | 114.3 | 106.1 | 106–114 | Clothed depth 30.8 cm (≈ 29.5–30 cm body); width uncertain because of the arms |
| Waist | 100.3 | 88.5 | 95–103 | Hull 114 with lateral and lumbar drape; mesh depth 29.3 inside clothed 32.6 |
| Abdomen | 104.0 | 90.8 | 98–106 | Hull 117; shirt touches the belly front |
| Hip | 117.1 | 105.0 | 110–116 | Hull 122.7 minus loose-trouser ease (≈ 6–12 cm) |

These ranges are an engineering estimate for planning, not a validation result. Only tape can confirm them.

Per-view images: each run's `view_comparison.jpg` (photo | red observed vs green fitted outline | render; front, left, back, right) and `final_mesh_views.jpg` (0/90/180/270°). Side-by-side composites: `outputs/accuracy_study/regression_fix/` (`scripts/compare_reconstructions.py`).

## 5.5 Local trunk shape (side / flank fat): why it was missing, and the surface-detail stage

**Observation.** The subject's flank ("love handle") fat is visible in real life but absent from the mesh.

**Is it in the data?** Yes. Radial comparison of the multi-view visual hull with the mesh at waist height (+0.06 to +0.10 H above the pelvis): the observed outline extends beyond the mesh by 2.3–3.1 cm at the back-left flank, 0.9–1.2 cm at the back-right flank, 0.8–1.7 cm at the sides, and only 0.1–0.3 cm at the front.

**Is the model removing it?** Yes, by construction. A synthetic, smooth 2 cm flank bulge on the SMPL body is reproduced only **3%** by the best 10-coefficient shape change: it adds 4.3 cm of waist girth, the SMPL version 0.3 cm. SMPL's 10 coefficients are whole-body modes, so localised fat is smoothed away. SHAPY uses the same kind of 10-coefficient model (SMPL-X), so it cannot show it either.

**Fix: shared trunk surface offsets (SMPL+D)** (`smpl_multiview.fit_surface_offsets`, pipeline stage in `body_reconstruction.export_surface_detail`):
- One per-vertex offset along the rest-pose normal, **shared by all views**, so it is 3D rather than per-view.
- Restricted to the trunk (pelvis/buttocks to chest). With the upper legs included, flared cargo trousers added about 2 cm of thigh "girth", so legs and arms keep the SMPL shape.
- Smoothed by a mesh Laplacian and bounded to ±6 cm. Shape, pose and cameras are frozen; the inside-the-outline term still applies.
- Synthetic ground truth: a 2.5 cm flank bulge is recovered and the front of the trunk stays unchanged (`test_surface_offsets_recover_a_flank_bulge_that_betas_cannot`).
- Cross-validation on run `268f55fd`: held-out IoU 0.816 → 0.818 (trunk only; 0.824 with legs, rejected for the trouser reason above). Offsets fitted without the held-out views improve the fit on those views, so they describe real 3D shape, not per-view noise.

End-to-end run `6303a2d5` (same video, cm at 179 cm):

| | Body model | + surface detail | Δ |
|---|---|---|---|
| Fit IoU / protrusion | 0.885 / 1.6% | 0.890 / 1.6% | — |
| Chest | 114.3 | 113.5 | −0.7 |
| Waist | 100.3 | 104.2 | +3.9 |
| Abdomen | 104.0 | 108.7 | +4.7 |
| Hip | 117.1 | 118.8 | +1.7 |
| Thigh / calf / upper arm | 60.5 / 41.6 / 35.7 | 60.8 / 41.6 / 35.8 | ≈ 0 |

Offsets: 5th / 50th / 95th percentile −0.4 / +0.1 / +1.0 cm. The added volume sits as a band across the lower back and flanks at waist height (`outputs/accuracy_study/surface_detail/offset_map_6303a2d5.jpg`).

**Caveat.** Against a loose T-shirt the detail follows the outer surface: fat **and** garment drape. The body-model girths therefore stay the primary measurements, and both sets are exported (`measurements.json`, `measurements_detailed.json`, cm versions after height scaling). Tape on one subject, or a tight-clothing recording, decides which to report. Disable with `RECON_SURFACE_DETAIL=0`.

## 5.6 Systematic error sweep (19 Nov): what was tested, what was kept

All tests refit run `6303a2d5` (same video) with the pipeline's stage A+B and the male shape prior, scored with
4-fold held-out IoU. cm values at 179 cm stature.

| # | Suspected error source | Evidence | Verdict |
|---|---|---|---|
| 1 | Camera model: HMR2's fixed 42,857 px focal vs the iPad Pro's ~1,400 px wide lens (subject at 2.2–2.4 m) | Real focal recovers a 2.27 m distance, but girths move ≤ 0.6 cm and held-out IoU is 0.8145 vs 0.8181 | Not a meaningful error; unchanged |
| 2 | Hip keypoints: detector's "hip" sits ~4.3 cm further out per side than SMPL's hip joint | Hip weight ×0.25 / ×0: held-out IoU 0.8196 / 0.8191 (noise); hip **grows** 116.0 → 116.8 | Not the cause; unchanged |
| 3 | Shoulder keypoints (1.5–2 cm residual) | Shoulders ×0.5: held-out 0.8199, every girth +1 cm | Unchanged |
| 4 | Mask edge bias | Image-edge search at 64k boundary points: median offset +0.07–0.08 cm in every region (`scripts/mask_edge_bias.py`). Low contrast on the torso (white shirt, white wall) | Not an error |
| 5 | Height reference: the A-pose mesh (legs 5° apart) is 0.27% shorter than stature (feet together) | Measured | **Fixed**: `stature_height` / `restate_to_stature` |
| 6 | Measurement placement | Chest 0.730 / navel 0.604 of stature vs ANSUR II 0.735 / 0.602; hip plane 2.3 cm low. Leg spread 0–5° and plane height change the hip ≤ 0.4 cm | Not an error |
| 7 | Loose trousers inflate the hip | Visual hull: mesh fills 96% of the clothed hip; ANSUR II: hip +2.5 SD given chest and navel waist | **Main remaining error**, see below |
| 7a | Body weight as a volume constraint | 105 / 95 / 85 kg: held-out IoU 0.807 / 0.806 / 0.796 (vs 0.818); volume went to or came from where the outline has slack (the trousers): at 105 kg hip 121.8 cm (+4.2 SD) | **Rejected** as a constraint; kept as a **mass check** (`mass_check`) |
| 10 | Plausibility against 4,082 men / 1,986 women (ANSUR II) | Conditional girth regression; `scripts/ansur_check.py` | **Added** to every height-scaled run (`anthropometric_check.py`, warning `girth_out_of_line_with_population`) |

The hip remains the out-of-line girth: men with this chest (114 cm), navel waist (104 cm) and stature typically have
a hip of 108 ± 3.4 cm, against 117 cm reconstructed. It cannot be fixed from this recording: the trousers hide the
pelvis outline in every view, and neither weight nor keypoints carry the missing information. A tape measurement
or a recording in fitted clothing is required.

## 5.7 Repeatability and multi-recording fusion (`scripts/fuse_recordings.py`)

The same person was filmed in three independent recordings: A = `38895355` (video 35e61e9b), B = `6303a2d5`
(777541cd), C = `2e0b075f` (f304bffd). Each was refitted with the current stage A+B (male prior), and then ONE
shape was fitted to all 71 views, with each view keeping its own pose and camera. cm at 179 cm.

| | Chest | Waist | Abdomen | Hip | Thigh | Calf | Upper arm |
|---|---|---|---|---|---|---|---|
| A alone | 113.0 | 98.9 | 101.6 | 113.2 | 59.6 | 40.9 | 35.4 |
| B alone | 115.6 | 102.2 | 104.5 | 115.6 | 60.9 | 41.5 | 36.9 |
| C alone | 114.4 | 100.6 | 103.0 | 114.0 | 60.0 | 41.1 | 35.9 |
| **Fused (71 views)** | **114.5** | **100.8** | **103.3** | **114.4** | **60.2** | **41.2** | **36.2** |
| SD across A/B/C | 1.3 | 1.7 | 1.5 | 1.2 | 0.7 | 0.3 | 0.8 |

- **Repeatability** between recordings is 1.2–1.7 cm SD at the torso and 0.3–0.8 cm on the limbs. Fusing three recordings reduces the random part by about √3 (≈ 0.7–1.0 cm at the torso). Systematic error common to all recordings, such as the same loose trousers, is not reduced.
- **Leave-one-recording-out** silhouette IoU on the left-out recording (its poses and cameras from keypoints only):

  | Left out | Shape from the other two | Its own shape |
  |---|---|---|
  | A | 0.8351 | 0.8329 |
  | B | 0.8162 | 0.8190 |
  | C | 0.8399 | 0.8398 |

  The mean difference is ≈ 0: a shape from different recordings explains a new recording as well as its own fit. The shape is a property of the person, not overfitted to one video.
- **Fused plausibility (ANSUR II, male):** chest +0.1 SD, abdomen −1.1 SD, hip +1.8 SD (114.4 cm vs a typical 108.3 cm). Within the flag limit, but still on the high side, consistent with the trousers.
- **Population bias check:** the typical hip for this chest, waist and stature is 108.1 / 110.1 / 108.3 / 108.4 cm in ANSUR's White / Black / Hispanic / Asian men. The result does not depend on the reference group (ANSUR has no separate South Asian group).
- The fused body is exported as `outputs/accuracy_study/fused_recordings/fused_body_mesh_metric.obj` (metres, A-pose, plain face); numbers are in `fused_recordings.json`.

## 5.8 Real tape: configuration chosen by ground truth (`scripts/tape_config_study.py`)

Tape, self-measured by the subject on 20 Nov: waist 88–90, abdomen 95–100, hip 98–103 cm (chest not measured).
The stature of 179 cm is not yet confirmed barefoot, so it was held fixed for every configuration; a 2 cm error
moves every girth by about 1.1%. Eleven configurations were refitted on all three recordings (A/B/C). Error is the
distance to the tape range (0 inside it):

| Configuration | Mean error | Waist / Abdomen / Hip (mean of A, B, C) |
|---|---|---|
| D0: HMR2 prior off (the 18–19 Nov default) | **8.3 cm** | 100.6 / 103.0 / 114.3 |
| IN: outline only as upper limit, no HMR2 prior | 7.3 | 78.9 / 82.2 / 100.4 |
| C025: coverage ×0.25 | 3.5 | 85.6 / 88.5 / 104.6 |
| IN_P02: outline as upper limit + HMR2 prior 0.2 | 3.5 | 84.9 / 87.7 / 101.4 |
| HMR2 shape alone (no silhouette fit) | 3.4 | 85.1 / 87.8 / 100.3 |
| **P02 / P02m: HMR2 prior 0.2 (± male prior)** | **2.9** | 85.7–85.8 / 88.5–88.6 / 101.5–101.9 |
| P02(m) + trunk surface detail | 1.6–1.7 | 89.6–89.8 / 91.1–91.3 / 102.6–103.0 |
| SHAPY (recordings B and A only; not run on C) | 2.1 / 2.2 (B / A) | 88.3 / 90.6 / 105.0 (mean of B, A) |

For a like-for-like comparison with SHAPY on recordings B and A only, P02m scores 1.7 / 3.9 cm (mean 2.8 cm). SHAPY
is closer at the waist and shares the abdomen shortfall (about 4 cm under); its hip is 2 cm over the range.

Findings:
1. **The 18–19 Nov default was wrong.** Prior 0 was chosen by held-out silhouette IoU, which is scored against the clothed outline and rewards a body that fills loose clothes. Every girth came out 10–14 cm too large. Silhouette IoU must not be used to select fit settings under clothing.
2. **Under loose clothing the accuracy comes from HMR2's learned shape.** HMR2 alone is within about 1 cm of the best fit; the silhouettes mostly describe the garment. Restoring the HMR2 prior (0.2) halves the error compared with any silhouette-driven variant.
3. **Decision rule fixed before the results:** adopt the principled "outline as upper limit + learned prior" (IN_P02) only if it beats the fallback on all recordings. It did not (3.5 vs 2.9 cm), so the fallback was adopted: `RECON_BETA_PRIOR` default 0.2, male/female prior kept (it changes size < 0.5 cm and fixes fat placement).
4. **Surface detail is not made primary**, although it scores best. It works by filling the garment, which here happens to offset HMR2's undersizing; on the outline-only variants it overshot the hip by 4–7 cm. One subject in one outfit cannot justify it. It stays a secondary display.
5. **Remaining error:** the abdomen is 4–8 cm under tape in every learned variant (HMR2 underestimates belly projection); the waist is 2–4 cm under on two of three recordings; the hip is inside the range.
6. **What the ANSUR consistency check cannot see:** clothing inflated all girths together, so their relationships stayed normal (hip only +1.7–1.9 SD). It flags local outliers, not uniform inflation.

7. **Drape-contact coverage (tested, rejected):** the outline pulls only at convex points where fabric rests on the body; straight hanging spans act only as an upper bound. Thresholds were fixed a priori (turn over ~8 cm of outline, full weight at 15°); 36–37% of outline points counted as contact. Mean error 2.88 vs 2.87 cm, and no girth moved more than 0.1 cm. With the HMR2 prior on, the silhouette barely shapes the body, so the abdomen shortfall belongs to HMR2's shape estimate, not to how silhouettes are used.
8. **Fusion and repeatability with the new default** (`outputs/accuracy_study/fused_recordings_prior02.json`): fused waist 86.1 / abdomen 88.9 / hip 101.8 / chest 103.0 cm. Between-recording SD: chest 1.0, waist 1.7, abdomen 1.7, hip 1.4 cm, about the same as before. Leave-one-recording-out IoU difference (shape from the other recordings minus its own): +0.008 / −0.010 / +0.000. ANSUR consistency z: chest −0.1, abdomen −0.6, hip +0.6.

9. **Anterior-belly contact detail (tested, not wired in)** (`scripts/belly_contact_study.py`): surface offsets restricted to forward-facing trunk vertices between pelvis and spine2, pushed only by convex outline points. Thresholds were fixed a priori. Mean distance to tape 2.87 → 2.73 cm; abdomen +0.2 to +0.3 cm on every recording; offsets ≤ 0.25 cm (95th percentile). The pre-registered rule was met, but the rule had no minimum effect size, and the gain is far below tape resolution (±2.5 cm) and between-recording repeatability (1.7 cm). Not added to the pipeline. Likely reason: in the A-pose profile views the hanging forearm and hand sit in front of the belly, so the belly-level outline is mostly arm. Worth retesting on a tight-clothing recording.
10. **SHAPY side by side** (same tape metric; recordings B and A only): SHAPY 2.1 / 2.2 cm vs pipeline 1.7 / 3.9 cm; averaging the two gives 1.8 / 2.7 cm (mean 2.25). Two recordings are not enough to switch methods; SHAPY shares the abdomen shortfall (about 4 cm) and is 2 cm over at the hip.

Caveat: one subject, self-measured ranges, one outfit. This ranks configurations; it does not establish population accuracy.

## 6. Changes made

| File | Change |
|---|---|
| `backend/services/smpl_multiview.py` | `_silhouette_loss`: Charbonnier inside term summed per contour sample (`FitConfig.sil_outside_form`, `"legacy"` kept). `FitConfig.fit_body_pose` option. `silhouette_metrics` returns protrusion/uncovered. New `torso_width_ratios`. |
| `backend/services/body_reconstruction.py` | Defaults: `RECON_SIL_OUTSIDE=charbonnier`, `RECON_BETA_PRIOR=0`, `RECON_POP_PRIOR=0.05`, `RECON_FIT_BODY_POSE=1`. `proportion_check`; warnings `body_protrudes_outside_outline`, `body_depth_exceeds_profile`, `per_view_overfit`. Report records `multiview_optimisation.fit_config`. |
| `backend/tests/test_multiview_shape_fusion.py` | 5 new tests (synthetic depth recovery, loose-outline inflation, metric splits, proportion ratios, pose lock). 65/65 pass. |
| `scripts/compare_reconstructions.py` | Reproducible side-by-side figures and a metrics table for any set of runs. |
| `scripts/visual_hull_girths.py` | Model-free clothed girth upper bound per torso level from the multi-view silhouettes, with optional arm carving. |

Reproduce earlier behaviour (kept for comparison):
- original: `RECON_SIL_OUTSIDE=legacy RECON_BETA_PRIOR=0.02 RECON_POP_PRIOR=0.002`
- committed: `... RECON_BETA_PRIOR=0`
- pre-fix working copy: `... RECON_BETA_PRIOR=0.2`

Hardware: unchanged. RTX 4070 Laptop (8 GB) used about 3.6 GB peak. HMR2, detection, Mask R-CNN and the optimiser run on CUDA; GrabCut, MediaPipe and rasterisation are CPU-only. No new dependencies.

## 7. Remaining limitations

1. **Absolute accuracy is not established.** There is no real tape or 3D scan. The tape records in runs `44c10580` and `2e0b075f` are test entries.
2. **The shape-prior choice is unresolved.** The fixed term with HMR2 prior 0 gives girths about 14% larger than with prior 0.2 (hip/H ≈ 0.65 vs ≈ 0.57). Profile depth of 0.96 means "nearly fills the T-shirt over the abdomen"; 0.80 means "20% inside it". Both are physically possible, and only tape can decide. The new default is better at matching the clothed outline. It is **not** shown to be more accurate in centimetres.
3. **Loose clothing.** The hips and thighs follow the loose cargo trousers (visible in the back view), so those girths are likely overestimated. This holds for every silhouette-based setting.
4. **Model capacity.** 10 SMPL shape coefficients, gender-neutral model, no per-vertex refinement: local shape (abdomen contour, chest) stays generic. Hands and feet are SMPL template geometry (shoes are down-weighted).
5. **Held-out protrusion of 5–6%** comes from keypoint-only pose on unseen views.
6. **Camera intrinsics** are HMR2's assumed focal length.

## 8. Recommended next engineering step

Collect tape measurements (stadiometer height plus the 12 sites in `MEASUREMENT_ACCURACY_REPEATABILITY_REPORT.md` §2) on at least 3 subjects, including **one tight or minimal-clothing recording**. Then run the fixed term at `RECON_BETA_PRIOR` 0 and 0.2 on the same recordings (`scripts/measurement_accuracy_study.py`, or full runs grouped by `fit_config`), and pick the prior by tape error, not IoU. If loose-clothing bias dominates after that, the next technical step is clothing-aware silhouette weighting: down-weight coverage where the garment is loose, using a clothing-segmentation model, and keep the inside constraint everywhere.
