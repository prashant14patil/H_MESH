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

### 5.4 Best estimate of real tape (reasoned, NOT measured)

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
