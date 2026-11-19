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

MULTIVIEW_SECTION

## 6. Automatic reliability indicators (implemented)

MQ_SECTION

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

LIMITATIONS

## 10. Recommended next phase

NEXT_PHASE

## Appendix: reproduce

```
.venv\Scripts\python.exe scripts/measurement_location_audit.py 9e126c35 268f55fd 38895355 2e0b075f 4fe46ed4 --synthetic --output outputs/accuracy_study/location_audit.json
set CUDA_VISIBLE_DEVICES= & .venv\Scripts\python.exe scripts/reliability_study.py 38895355 --output outputs/accuracy_study/reliability/38895355.json   (and 9e126c35, 2e0b075f)
.venv\Scripts\python.exe scripts/summarize_reliability.py outputs/accuracy_study/reliability/*.json --markdown outputs/accuracy_study/reliability/summary.md --json outputs/accuracy_study/reliability/summary.json
.venv\Scripts\python.exe scripts/ground_truth_report.py --sessions <sessions.csv> --tape <tape.csv> --markdown outputs/validation/ground_truth.md
```
