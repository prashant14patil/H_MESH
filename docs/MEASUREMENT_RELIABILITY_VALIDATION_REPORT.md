# Measurement reliability validation report

Date: 19 Nov 2026 · Scope: body circumferences from the unified 360° reconstruction (current defaults: Charbonnier outside term, population prior 0.05, HMR2 prior 0, per-view pose) · Status: **precision and robustness measured on one subject (3 recordings); accuracy against tape NOT established: no real tape data exists**

## 1. Summary

| Question | Answer |
|---|---|
| Is accuracy in cm known? | **No.** No real tape measurements exist; the only tape records on disk are test entries (excluded). No accuracy figure in this report is a tape result. |
| Is the result repeatable? | Yes, on one subject. Between three independent recordings: within-subject SD 0.8–1.6% per site, repeatability coefficient (RC, 95%) 0.7–4.3 cm. |
| Do the two measurement models agree? | **No.** At the same height the pipeline is 5–12% larger than SHAPY at every site (waist +9–12 cm, hip +10–12 cm). Both are repeatable, so they can't both be accurate. Only tape can decide. |
| Are rings at anatomically meaningful places? | Mostly. Torso levels fall in plausible bands. Three definitional problems were found: the **forearm** maximum always lands on its search edge; the **thigh** ring sits 0.5–1.1% of stature above the tape protocol (+1.4–2.3%, about +1 cm); and the **chest** depends on the arm angle of the measurement pose (0.3–1.6 cm per 10°). |
| Which capture conditions matter? | MULTIVIEW_SUMMARY |
| Can bad measurements be flagged automatically? | Yes, for internal consistency. A new per-measurement grade (good/fair/poor/unavailable) with reasons and a ±1 SD precision estimate is computed from the run's own evidence (§6). It describes precision, not accuracy. |

## 2. Audit of the latest changes

| Item | Finding |
|---|---|
| Silhouette fix (Charbonnier outside term) | In place and default. Protrusion is 1.6–1.8% on all three recordings under the study harness (was 3.4–7.8% with the legacy term). |
| Default shape prior | HMR2 prior 0, population prior 0.05. This **raises girths about 10–14%** relative to the prior-0.2 numbers in `MEASUREMENT_ACCURACY_REPEATABILITY_REPORT.md` §8.2 (e.g. hip 114 vs 100 cm at 176 cm). The prior choice is still unresolved and is the largest single source of uncertainty in absolute girths. |
| Repeatability figures used in the UI | `REPEATABILITY_SD_PCT` was measured under prior 0.2. Under the current defaults Sw is similar (0.8–1.6% vs 0.9–2.4%), so the UI's ± is still of the right order, but it is not from the current configuration. |
| `docs/RECONSTRUCTION_ACCURACY_INVESTIGATION.md` | Untracked, and §5 still contains a literal `PENDING_RUNS` placeholder. |
| Determinism | A CPU refit reproduces the saved GPU run (`9e126c35`) to ≤ 0.002 × height (≤ 0.3%) at every site. Runs `268f55fd` and `9e126c35` (same video and config) give identical girths. |
| Concurrent work | Another session is adding capture-quality checks (`scan_quality.py`: lighting, background motion, rotation trajectory, coverage) to `body_reconstruction.py`. This work reuses its verdicts rather than duplicating them. |

## 3. Measurement locations (anatomical audit)

Tool: `scripts/measurement_location_audit.py` (CPU). Output: `outputs/accuracy_study/location_audit.json`. Bodies audited: 5 fitted shapes (3 recordings, several configurations) and 5 synthetic SMPL shapes (mean, ±2 SD on β1 and β2). Ring height is the ring centre ÷ stature (floor = 0).

| Site | Ring height ÷ stature (fitted bodies) | Search position | Finding |
|---|---|---|---|
| Chest | 0.732–0.744 | interior (0.37–0.46 of range) | Plausible (nipple level ≈ 0.72–0.73 H in adult males). **Pose-sensitive:** the ring is in the armpit region that the A-pose compresses (edge lengths down to 0.78 of upright). See below. |
| Waist | 0.647–0.660 | interior | Narrowest torso girth. Sits about 2–4% H above a typical natural-waist height (≈ 0.62–0.63 H). Plausible, but slightly high. |
| Abdomen | 0.605–0.615 (spine1 joint) | fixed | Close to navel height (≈ 0.60 H). |
| Hip | 0.491–0.497 | interior (0.50–0.58) | Plausible (buttock level ≈ 0.50 H). On the synthetic β2 −2 SD shape the maximum hits the **pelvis-joint edge** (0.533 H), so for some shapes "hip" can read the lower belly. Now flagged automatically. |
| Upper arm | 0.744–0.752, bone midpoint | fixed | The SMPL shoulder joint is about 3–4 cm below the acromion the tape protocol uses, so the ring sits about 1.5–2 cm distal of the protocol point. The arm is nearly cylindrical there (major/minor 1.04), so the effect is small. |
| **Forearm** | 0.688–0.694 | **edge (12%) in 10/10 bodies** | SMPL's forearm narrows steadily from elbow to wrist. There is no muscle-belly maximum, so "fullest girth" is always the first plane. The value moves only 0.5 cm between 12% and 25% of the forearm length (28.7 → 28.2 cm), so the bias is small. This is a model limitation, noted on every result. |
| **Thigh** | 0.376–0.381 (hip–knee joint midpoint) | fixed | The tape protocol is halfway from the groin crease to the top of the kneecap. Crotch found on the mesh at 0.446–0.450 H; kneecap top approximated at knee joint + 0.025 H. The ring is 0.5–1.1% H **above** the protocol point, so the girth reads **+1.4 to +2.3%** (about +0.8 to +1.4 cm) on realistic fits. Girth changes about 3% per 1% H along the thigh, so ±1 cm of tape placement is itself worth about ±1.7%. |
| Calf | 0.211–0.214 | interior (0.36) | Plausible (calf max ≈ 0.20–0.22 H). |

Reference heights are approximate adult population proportions, used only to catch gross mislocation, not as accuracy evidence.

**Cross-section geometry.** Hull ÷ contour (concavity) is 1.000–1.006 on every realistic fit: the SMPL torso has no buttock cleft or chest concavity deep enough for the tape-versus-contour distinction to matter (≤ 0.6%). Depth ÷ width is chest 0.79–0.84, waist 0.85–0.88, hip 0.74–0.77: elliptical, plausible values. Limb sections are close to round (upper arm 1.04, calf 1.01), except the forearm (1.39, flattened as expected).

**Measurement pose (chest).** The same fitted body, arms lowered from horizontal:

| Arm angle below horizontal | 0° (T) | 30° | 45° (used) | 55° | 65° |
|---|---|---|---|---|---|
| Chest, fitted body `9e126c35` (cm @176) | 118.4 (edge) | 113.1 | 112.3 | 112.2 | 112.0 |
| Chest, SMPL mean body | 111.7 (edge) | 106.1 | 104.4 | 103.3 | 102.8 |

The waist, upper arm and limbs change by < 0.4 cm across 30–65°. The tape protocol measures with the arms relaxed at the sides, which SMPL can't represent without the arm merging into the torso. The chest therefore carries an unquantified pose-definition bias of the order of 1–3 cm, to be resolved with tape.

## 4. Geometric validation

| Check | Result |
|---|---|
| Mesh topology | Watertight, one component, Euler number 2, no degenerate faces, upright, facing +z (existing `validate_mesh`, all runs). |
| Height normalisation | The A-pose measurement mesh is 0.12–0.27% shorter than the same body upright (legs abducted 5°). Scaling to a supplied barefoot height therefore inflates girths by the same 0.1–0.3% (≤ 0.4 cm at the hip). Negligible next to the other terms. A ±1 cm stature error moves every girth by ±0.57%. |
| Surface distortion from posing | Lowering the arms to 45° compresses edges near the chest ring (5th percentile 0.78) and the upper-arm ring (median 0.976). Waist, hip, thigh, forearm and calf rings are undistorted (median 1.000–1.004, 5–95% within 0.94–1.05). Only the chest, and to a small degree the upper arm, are sensitive to skinning artefacts. |
| Shape plausibility | Current-default fits stay within \|β\| ≤ 1.0 SD. Older configurations on disk reach 2.8 SD (legacy term, no prior). |
| Proportions vs observed outline | Side depth model/observed 0.96–0.99, front width ≤ 1: the body is inside its outline at every level (no protrusion). Loose garments mean the outline is not the body, so this can't confirm girths. |
| Consistency across views | Per-view silhouette agreement is exact-rasterised on all 24 views (fit IoU 0.885). Held-out views (shape fitted without them) give IoU 0.815 and protrusion 5–6%, from keypoint-only pose on unseen views. |

## 5. Multi-view reliability

**Method.** `scripts/reliability_study.py` refits each recording from its saved fit inputs, with the pipeline's current default settings (including the post-fit IoU view exclusion and refit), after a controlled degradation of what the fit consumes: masks, keypoints, available views and initial pose/camera. It isolates the fitting stage. HMR2 and segmentation are **not** re-run on degraded pixels: "blur" models what a blurred frame does to the outline and keypoint precision, not what HMR2 would predict from it. Three recordings × 17 conditions; random conditions use 2 seeds; 75 fits on CPU (a CPU refit reproduces the GPU run to ≤ 0.3%). Each change is relative to the same recording's undegraded fit and is judged against the **between-recording repeatability** of the undegraded fits: Sw = within-subject SD, RC = 2.77 × Sw.

**Repeatability (current defaults, 3 recordings, one subject; n = 3, provisional)**

| Site | Mean cm @176 | Sw % | RC % | RC cm |
|---|---|---|---|---|
| Chest | 111.3 | 1.19 | 3.3 | 3.7 |
| Waist | 97.3 | 1.61 | 4.5 | 4.3 |
| Abdomen | 100.9 | 1.55 | 4.3 | 4.3 |
| Hip | 114.3 | 0.85 | 2.4 | 2.7 |
| Upper arm | 34.5 | 1.62 | 4.5 | 1.5 |
| Forearm | 28.6 | 0.93 | 2.6 | 0.7 |
| Thigh | 58.8 | 1.01 | 2.8 | 1.6 |
| Calf | 40.5 | 0.80 | 2.2 | 0.9 |

**Condition effects.** Worst |change| % over recordings and seeds; mean signed change in brackets. ▲ = exceeds RC.

| Brief condition | Proxy | Chest | Waist | Hip | Thigh | Upper arm | Verdict |
|---|---|---|---|---|---|---|---|
| Rotation speed (fewer distinct views) | 16 views | 0.9 (−0.5) | 1.2 (−0.6) | 0.8 (−0.6) | 0.9 (−0.6) | 1.3 | negligible |
| | 12 views | 0.7 (+0.4) | 0.9 (+0.5) | 0.6 (+0.3) | 0.7 (+0.4) | 1.0 | negligible |
| | 8 views (45° apart) | 0.7 (+0.2) | 0.7 (+0.2) | 0.8 (+0.1) | 0.8 (0.0) | 0.9 | negligible |
| | 6 views (60° apart) | 1.3 (−1.1) | 1.9 (−1.6) | 1.3 (−1.0) | 2.0 (−1.3) | 2.6 | small **systematic shrink**, within RC |
| Blurry frames | 25% of views smeared 12 px + keypoint noise | 0.5 | 0.6 | 0.8 | 1.0 | 0.8 | negligible |
| | 50% of views | 2.2 (+1.5) | 3.0 (+2.0) | 1.8 (+0.9) | 2.4 (+1.3) | 3.4 | inflates 1–2%, within RC |
| Temporary occlusion | hip band hidden in 25% of views | 7.3 (−3.2) ▲ | 11.3 (−4.6) ▲ | 8.1 (−3.9) ▲ | 10.5 (−4.9) ▲ | 13.6 ▲ | **severe shrink** |
| | chest band hidden in 25% of views | 9.5 (−6.6) ▲ | 15.6 (−10.3) ▲ | 7.1 (−4.2) ▲ | 8.9 (−5.2) ▲ | 13.7 ▲ | **severe shrink** |
| Insufficient coverage | no back views (90° sector) | 0.4 | 0.8 | 0.6 | 0.7 | 0.3 | negligible |
| | no left-side views (90° sector) | 0.4 | 0.8 | 0.9 | 1.0 | 0.5 | negligible |
| Lighting (segmentation boundary) | random ±6 px per view | 3.1 (−2.2) | 4.0 (−3.1) | 2.0 (−1.1) | 2.7 (−1.7) | 4.2 | shrink 1–3%, near RC |
| | shadow bleed: all masks +4 px | 4.0 (+3.6) ▲ | 5.5 (+4.7) ▲ | 3.2 (+2.8) ▲ | 3.7 (+3.6) ▲ | 5.7 ▲ | **systematic inflation** |
| | dark edges: all masks −4 px | 5.5 (−4.7) ▲ | 7.3 (−6.5) ▲ | 3.9 (−2.8) ▲ | 5.2 (−4.1) ▲ | 7.8 ▲ | **systematic shrink** |
| Not rotating in place | ±10% distance + ±15% lateral drift | 0.3 | 0.5 | 0.3 | 0.3 | 0.5 | negligible |
| Different standing pose | per-view joint init ±10° | 1.1 | 1.7 | 0.8 | 0.8 | 1.4 | small (initialisation only) |
| Heading (yaw) error | per-view heading init ±15° | 2.4 (−1.2) | 3.0 (−1.5) | 2.5 (−1.2) ▲ | 2.8 (−1.4) ▲ | 3.7 | moderate shrink |

Full table (all 12 sites): `outputs/accuracy_study/reliability/summary.md`.

**What matters, ranked**

1. **Mask boundary accuracy dominates.** A uniform 4 px boundary shift (about 0.4% of the subject's image height) moves girths 3–7%, roughly **1–1.5% of girth per pixel**, in the direction of the shift. Under-segmentation is worse than over-segmentation (−6.5% vs +4.7% at the waist). The non-saturating "stay inside" term pulls hard toward any outline that is too small. The same term is what removed the protrusion bug, so the trade-off is deliberate, but it makes segmentation quality the main accuracy lever.
2. **Occlusion is severe**, and the current safeguard does not hold (next subsection).
3. **Heading error** of ±15° in initialisation shrinks girths 1–2.5%.
4. **Insensitive:** 8 or more views, a missing 90° sector, drift and scale change while turning, rotation speed down to 8 distinct views, and blur in a quarter of the views. The capture protocol can be relaxed on these axes. It can't be relaxed on lighting or segmentation.

**Repeatability of the fit itself (jackknife).** Shapes fitted on 3 of the 4 heading-interleaved folds (the pipeline's CV folds) spread with an SD of 0.48–1.00% (mean ≈ 0.6 × Sw at every site). The delete-a-group jackknife standard error from this (≈ √3 × population SD of the fold estimates ≈ 0.9 × Sw) is about equal to the between-recording SD. That suggests **which views happen to be sampled explains most of the recording-to-recording variation** on this subject.

**Does the fold spread detect degradations?** Fold SD %, recording B:

| Condition | Chest | Waist | Hip | Thigh | × clean |
|---|---|---|---|---|---|
| Clean | 0.52 | 0.76 | 0.34 | 0.43 | 1 |
| Occlusion left in the fit | 6.6 | 11.2 | 4.2 | 5.3 | **10–15** |
| 8 views only | 2.4 | 3.0 | 2.5 | 2.6 | 4–6 (folds have 6 views) |
| Dark edges −4 px (girths −6%) | 0.80 | 1.28 | 0.65 | 0.89 | **1.5** |
| Random ±6 px | 0.39 | 0.53 | 0.38 | 0.50 | < 1 |

Fold spread catches views that **disagree with each other** (occlusion, a few bad masks). It is blind to **errors shared by all views** (a systematic boundary shift). No internal-consistency check can see those; only calibration against tape, or a segmentation accuracy check, can.

EXCLUSION_SECTION

## 6. Automatic reliability indicators (implemented)

Code: `backend/services/measurement_quality.py`, called from `body_reconstruction.export_outputs` → `<run>/measurement_quality.json` (also summarised in `reconstruction_report.json` and the run result). Backfill for older runs: `scripts/measurement_quality_report.py RUN [--folds]`. Tests: `backend/tests/test_measurement_quality.py` (10 pass).

Each girth gets a **grade** (good / fair / poor / unavailable), the **reasons**, and a **±1 SD precision** estimate (in % and, when the height is known, in cm). The grade is the worst of the following signals, each of which the evidence above supports:

| Signal | Rule | Evidence |
|---|---|---|
| Run-level verdicts | Reconstruction `unreliable` or capture `insufficient` → poor. `usable_with_warnings` / `acceptable_with_warnings` → at most fair. **Missing verdict → at most fair.** | Uses `scan_quality` from the concurrent work. A first version treated missing evidence as a pass and graded an inflated legacy run (hip 127 cm @176) "good"; fixed and tested. |
| Cross-section | Invalid → unavailable. A min/max search ending on the edge of its zone → fair (forearm exempt, noted as a model limitation). | Location audit, §3. |
| Fold stability | SD of the girth across the 4 CV-fold shapes: > 2.5 × typical → fair; > 5 × typical → poor. | §5: clean ≤ 1.6×, 8-view captures 4–6×, occlusion left in 10–15×. |
| Outline evidence (torso) | Model/observed width (frontal) or depth (profile) at the site's level: > 1.05 → poor (body outside its outline); < 0.85 → fair (loose clothing: girth set by the shape model, not observed); no evidence → fair. | Protrusion is physically impossible. A loose outline can't constrain the body. |
| Coverage | Usable-silhouette views < 8 or largest heading gap > 90° → fair; < 6 views or gap > 120° → poor. | §5: 8 views and a 90° gap are negligible; 6 views gives a −1 to −2% systematic shrink; > 90° untested. |

Precision ±1 SD = √(Sw² + max(0, fold SD² − typical fold SD²)): the measured between-recording SD, inflated when this run is less stable than usual, plus ±1 cm of height (d/H) when centimetres are shown.

**On the existing runs** (backfill without fold data: these runs predate saving the fold shapes): current-default runs `9e126c35` and `268f55fd` → all 12 sites good; ±0.3–1.7 cm (forearm and calf ±0.3–0.5, torso ±1.2–1.7). Legacy-configuration runs `2e0b075f` and `38895355` → all poor (reconstruction verdict unreliable: protrusion not recorded, shape coefficients 2.7–2.8 SD). Original-configuration run `776042ea` → capped at fair (protrusion 3.9%).

**What it can't detect:** errors shared by every view, i.e. a systematic segmentation-boundary bias (§5: −6% girths gave 1.5× fold spread, below threshold), loose clothing that is uniformly loose, and the shape-prior choice. The grades describe precision and internal consistency; **they are not accuracy**, and the files say so.

## 7. Pipeline vs SHAPY (no ground truth)

Same height (176 cm); pipeline = current defaults (study baselines); SHAPY = stored results (v2, 23–24 frames, flip test-time augmentation).

| Site | Pipeline A | SHAPY A | Pipeline B | SHAPY B | Pipeline C | Pipeline − SHAPY |
|---|---|---|---|---|---|---|
| Chest | 109.9 | 104.3 | 112.5 | 104.3 | 111.5 | +5.6 … +8.2 |
| Waist | 95.6 | 86.6 | 98.8 | 87.0 | 97.5 | +9.0 … +11.8 |
| Abdomen | 99.3 | 88.9 | 102.5 | 89.3 | 101.0 | +10.4 … +13.2 |
| Hip | 113.7 | 103.3 | 115.4 | 103.2 | 113.7 | +10.4 … +12.2 |
| Upper arm L | 34.1 | 32.2 | 35.2 | 32.3 | 34.5 | +1.9 … +2.9 |
| Forearm L | 28.3 | 27.9 | 28.8 | 27.9 | 28.5 | +0.4 … +0.9 |
| Thigh L | 58.6 | 53.9 | 59.6 | 54.0 | 58.6 | +4.7 … +5.6 |
| Calf L | 40.4 | 37.9 | 41.0 | 37.9 | 40.5 | +2.5 … +3.1 |

- SHAPY is extremely repeatable here (A vs B within 0.4 cm). Repeatability is not accuracy: SHAPY regresses from single crops and is known to pull toward average bodies.
- The pipeline follows the clothed outline. The gap is largest at the waist, abdomen and hip, which are under a loose T-shirt and cargo trousers, and smallest at the forearm, which is bare. That pattern fits clothing inflation in the pipeline, SHAPY shrinkage toward the mean, or both. **It does not tell which.**
- The pipeline under the older prior-0.2 setting (chest about 101, waist 85, hip 100 cm) is close to SHAPY. Tape on this subject would therefore discriminate between the settings as well as between the models.
- SHAPY has not been computed for recording C (needs the GPU, which was busy).

## 8. Ground-truth validation workflow (implemented; no data yet)

Code: `backend/services/ground_truth_validation.py`, CLI `scripts/ground_truth_report.py`, templates `docs/validation/sessions_template.csv` and `tape_template.csv`, tests `backend/tests/test_ground_truth_validation.py` (7 pass).

**Protocol per subject**

1. Record the 360° video. Run the reconstruction and compute SHAPY (`POST /runs/{id}/shapy`). Write the run IDs into the sessions sheet **before** any tape is taken, so the model values can't be adjusted toward the tape.
2. Measure barefoot height with a stadiometer.
3. Take tape measurements at the 12 sites (protocol in `body_measurements.TAPE_PROTOCOL`, minimal or tight clothing for the primary analysis): one trained measurer, inelastic tape, **two readings per site**, and a third when they differ by > 1.0 cm (torso) or > 0.5 cm (limbs). The report flags missing thirds.
4. Run `scripts/ground_truth_report.py --sessions … --tape …`.

**What it reports**

- Per subject and site: tape, pipeline, SHAPY, error (model − tape) in cm and %, absolute errors, and the pipeline quality grade.
- Per site and method: n, bias, SD, MAE, MAPE, 95% limits of agreement, worst error, and target met / not met / n too small.
- The measurer's technical error of measurement (TEM, from the duplicate readings), which is the floor below which no method can be shown to be accurate.
- A paired comparison: which model is closer to the tape per subject.

Model centimetres are computed as girth ÷ mesh height × **stadiometer** height, straight from each run's files, so a mistyped height in the app can't leak in. Statistics are marked claimable only from n ≥ 5 subjects.

**Rules it enforces:** tape is the only reference. Pipeline and SHAPY are never compared as truth against each other. With no tape rows the report states "no accuracy can be stated" and prints nothing else. The acceptance targets (±2 cm torso, ±1 cm limbs, applied to |bias| and MAE) are proposals to be agreed, not derived from data.

**Minimum dataset to answer the open questions:** ≥ 5 subjects of varied build. At least 2 recordings each, one in tight or minimal clothing and one in everyday loose clothing. Run with both `RECON_BETA_PRIOR=0` and `0.2` on the same recordings, so the prior is chosen by tape error.

## 9. Remaining limitations

1. **No ground truth.** Nothing here measures accuracy in centimetres. Every "error" is a change relative to another fit, or a disagreement between models.
2. **One subject, three recordings, one outfit (loose).** Repeatability (n = 3) and the condition effects may differ on other bodies, other clothing and other cameras. Treat the Sw/RC values and quality thresholds as provisional.
3. **Degradations are proxies.** Masks, keypoints and views were perturbed. HMR2 and segmentation were not re-run on degraded video, so real blur, low light or occlusion can also corrupt HMR2's pose and heading, which these tests only partly represent (the heading-initialisation test).
4. **Systematic segmentation bias is undetectable internally** and is the largest measured sensitivity (about 1–1.5% of girth per pixel of boundary shift). Neither fold spread nor protrusion flags a uniform shift.
5. **Absolute girths depend on the unresolved shape-prior choice:** about 10–14% between `RECON_BETA_PRIOR` 0 and 0.2, larger than any capture effect except occlusion and boundary bias. Pipeline and SHAPY differ by up to 13 cm at the abdomen.
6. **Clothing.** The silhouette is the clothed outline. Torso and thigh girths under loose garments are an upper bound shaped by the model prior, not a measurement.
7. **Model capacity and definitions.** 10 SMPL shape coefficients, gender-neutral model (a sex-specific option is being added by the concurrent work), no muscle relief (forearm), LBS compression at the axilla (chest pose-sensitivity 1–3 cm), and a thigh ring about 1 cm above the tape protocol point.
8. **Camera intrinsics** are HMR2's assumed focal length. Scale comes only from the supplied stature (±1 cm → ±0.57%).

## 10. Recommended next phase

**Phase: ground-truth collection and calibration**, in this order. Each step is gated on the previous one's evidence.

1. **Collect tape data with the workflow in §8.** At least 5 subjects of varied build, 2 recordings each (tight or minimal clothing, and everyday clothing), stadiometer height, two tape readings per site. Freeze run IDs (and SHAPY) before measuring. Also run each recording with `RECON_BETA_PRIOR=0` and `0.2`.
2. **Decide from tape:** (a) the shape prior, (b) pipeline vs SHAPY per site, and (c) whether a per-site linear bias correction is justified. Apply (c) only if bias is consistent across subjects and larger than measurer TEM, and validate it by leave-one-subject-out.
3. **Segmentation accuracy, the largest measured lever.** Hand-annotate subject outlines on about 50 frames across lighting conditions and measure the boundary bias in pixels of the current mask stack (Mask R-CNN + GrabCut + MediaPipe). At about 1–1.5% of girth per pixel, a 2 px bias is already a 2–3% girth error. If a bias exists, correct it (boundary refinement, or a fitted boundary offset), or add a capture-side control: lighting check, plain background.
4. **Resolve the measurement definitions with tape:** the thigh plane at the tape-protocol point (crotch on the mesh → kneecap), a fixed forearm level, and the chest measured at a pose-insensitive level or arm angle. Change a definition only where tape shows it reduces error.
5. **Clothing.** If tape shows loose-clothing bias dominates, add clothing-aware silhouette weighting (garment segmentation, down-weight coverage on loose regions, keep the inside constraint), or require fitted clothing in the capture protocol.
6. **Replace the provisional constants** (`measurement_quality.REPEATABILITY_SD_PCT`, typical fold SD, thresholds, and the UI's `REPEATABILITY_SD_PCT`) with values pooled over all subjects, once ≥ 5 subjects × ≥ 3 recordings exist (`scripts/validation_report.py`).

Not recommended before tape data exists: further objective or prior tuning chosen by IoU, held-out IoU or SHAPY agreement. None of these can tell which girths are correct.

## Appendix: reproduce

```
.venv\Scripts\python.exe scripts/measurement_location_audit.py 9e126c35 268f55fd 38895355 2e0b075f 4fe46ed4 --synthetic --output outputs/accuracy_study/location_audit.json
set CUDA_VISIBLE_DEVICES= & .venv\Scripts\python.exe scripts/reliability_study.py 38895355 --output outputs/accuracy_study/reliability/38895355.json   (and 9e126c35, 2e0b075f)
.venv\Scripts\python.exe scripts/summarize_reliability.py outputs/accuracy_study/reliability/*.json --markdown outputs/accuracy_study/reliability/summary.md --json outputs/accuracy_study/reliability/summary.json
.venv\Scripts\python.exe scripts/ground_truth_report.py --sessions <sessions.csv> --tape <tape.csv> --markdown outputs/validation/ground_truth.md
```
