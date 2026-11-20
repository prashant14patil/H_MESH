# Measurement reliability validation report

Date: 19–20 Nov 2026 · Scope: body circumferences from the unified 360° reconstruction · Current defaults (since 20 Nov): Charbonnier outside term, **HMR2 shape prior 0.2**, population prior 0.05, per-view pose, **protrusion-based view exclusion** · Status: **precision and robustness measured (one subject, 3 recordings, 2 configurations). Preliminary tape exists for one subject at 3 sites (self-measured ranges, loose clothing). Population accuracy is NOT established.**

## 1. Summary

| Question | Answer |
|---|---|
| Is accuracy in cm known? | **Only preliminarily.** One subject self-measured waist 88–90, abdomen 95–100 and hip 98–103 cm, in loose clothing (`RECONSTRUCTION_ACCURACY_INVESTIGATION.md` §5.8). Against the range midpoints, the current-default run `ff1c226a` is waist −1.4, abdomen −7.1 and hip +2.5 cm (§8). The setting this study started on (HMR2 prior 0) was 11–17 cm too large and has been replaced. One subject in one outfit ranks configurations; it doesn't establish accuracy. |
| Is the result repeatable? | Yes, on one subject. Current default, three independent recordings: within-subject SD 0.9–2.3% per site, repeatability coefficient (RC, 95%) 0.7 cm (forearm) to 4.9 cm (abdomen). |
| Which capture conditions matter? | **Under the current default, essentially none of those tested.** Fewer views, blur, occlusion, lighting/boundary shifts, drift, missing coverage and heading error all move girths ≤ 1.3% (most ≤ 0.5%), below the between-recording SD. With the HMR2 prior on, the silhouettes barely shape the body. The recording-to-recording spread comes from **HMR2's shape estimate per recording**, and the remaining accuracy error (abdomen −4 to −8 cm) is HMR2's too. Under the previous prior-0 setting, segmentation boundary bias (1–1.5% per pixel) and occlusion (−3 to −16%) dominated (§5.3). |
| Do the two models agree? | **Now closely at the torso; not on the limbs.** Pipeline vs SHAPY at 179 cm: waist −0.7 / −3.7, abdomen −0.2 / −3.3, hip −2.0 / −5.0 cm (recordings B / A). Pipeline limbs are 1–5 cm smaller than SHAPY's (§7). Neither is ground truth. Images: `outputs/model_comparison/`. |
| Are rings at anatomically meaningful places? | Mostly. Three definitional issues: the **forearm** maximum always lands on its search edge (≈ 0.5 cm effect); the **thigh** ring sits ≈ 1 cm above the tape-protocol point (+1.4–2.3%); the **chest** depends on the measurement-pose arm angle (1–3 cm) (§3). |
| Can unreliable measurements be flagged automatically? | Yes, for internal consistency. Each run now writes `measurement_quality.json` with a per-girth grade, reasons and a ±1 SD precision, calibrated per configuration (§6). It is live in the pipeline. **Demonstrated blind spot:** the old prior-0 run that was 11–17 cm too large against tape graded *good*, because uniform clothing inflation is consistent across views. |
| What changed in the code? | (1) View exclusion by protrusion outliers instead of IoU (§5.2; adopted as the default). (2) Per-measurement quality and precision from the CV-fold shapes (§6). (3) A search-edge flag on every girth. (4) A tape vs pipeline vs SHAPY ground-truth workflow (§8). (5) Study and figure tools (appendix). |

## 2. Audit of the latest changes

| Item | Finding |
|---|---|
| Silhouette fix (Charbonnier outside term) | In place. Protrusion 1.5–1.8% on all recordings (was 3.4–7.8% with the legacy term). |
| Shape prior | Changed during this work by the concurrent session, from 0 to **0.2**, on the basis of the first real tape. This study independently reproduces that setting's girths: waist / abdomen / hip at 179 cm, mean of 3 recordings, 86.0 / 88.8 / 101.8 cm vs their 85.7 / 88.5 / 101.5 cm. |
| View exclusion | Replaced by the protrusion rule from this study (§5.2), committed by the concurrent session together with this report's draft. |
| Stature scaling | The 0.27% A-pose shortening found here (§4) is now corrected in the pipeline (`stature_height`). |
| UI repeatability (`bodyAvatarModel.REPEATABILITY_SD_PCT`) | Measured earlier under prior 0.2 (legacy term). It matches this study's current-default Sw closely (e.g. waist 2.0 vs 1.98%, hip 1.5 vs 1.56%, thigh 1.8–1.9 vs 1.89%). No change needed. |
| Determinism | A CPU refit reproduces the saved GPU run to ≤ 0.3% at every site. Same video and config gives identical girths. |
| Concurrent work | Capture-quality checks (`scan_quality.py`), sex-specific shape prior, volume/weight check, SHAPY as the default measurement view. This work consumes `scan_quality`'s verdicts rather than duplicating them. |

## 3. Measurement locations (anatomical audit)

Tool: `scripts/measurement_location_audit.py` (CPU). Output: `outputs/accuracy_study/location_audit.json`. Bodies: 5 fitted shapes (3 recordings, several configurations) and 5 synthetic SMPL shapes (mean, ±2 SD on β1 and β2). Ring height = ring centre ÷ stature.

| Site | Ring height ÷ stature | Search position | Finding |
|---|---|---|---|
| Chest | 0.732–0.744 | interior | Plausible (nipple level ≈ 0.72–0.73 H). **Pose-sensitive** (below). |
| Waist | 0.647–0.660 | interior | Narrowest torso girth. About 2–4% H above a typical natural-waist height (≈ 0.62–0.63 H): plausible, slightly high. |
| Abdomen | 0.605–0.615 (spine1 joint) | fixed | Close to navel height (≈ 0.60 H). |
| Hip | 0.491–0.497 | interior | Plausible (≈ 0.50 H). On β2 −2 SD the maximum hits the **pelvis-joint edge**, so for some shapes "hip" can read the lower belly. Now flagged. |
| Upper arm | 0.744–0.752 | fixed | The SMPL shoulder joint is ≈ 3–4 cm below the acromion, so the ring sits ≈ 1.5–2 cm distal of the protocol point. Small effect (near-cylindrical). |
| **Forearm** | 0.688–0.694 | **edge in 10/10 bodies** | SMPL has no forearm muscle belly, so "fullest girth" is always the plane nearest the elbow. 12% → 25% of forearm length changes it by only 0.5 cm. Model limitation, noted on every result. |
| **Thigh** | 0.376–0.381 | fixed | Protocol: halfway from the groin crease (crotch found on the mesh at 0.446–0.450 H) to the top of the kneecap (≈ knee joint + 0.025 H). The ring is 0.5–1.1% H above that point, so it reads **+1.4 to +2.3%** (≈ +1 cm). Girth changes ≈ 3% per 1% H along the thigh, so tape placement itself matters as much. |
| Calf | 0.211–0.214 | interior | Plausible (≈ 0.20–0.22 H). |

Reference heights are approximate adult proportions, used only to catch gross mislocation.

**Cross-sections.** Hull ÷ contour is 1.000–1.006, so the tape-vs-contour distinction is negligible on SMPL. Depth ÷ width: chest 0.79–0.84, waist 0.85–0.88, hip 0.74–0.77. Limbs near-round, except the forearm (1.39).

**Measurement pose (chest).** Same fitted body, arms lowered from horizontal (cm @176):

| Arm angle below horizontal | 0° (T) | 30° | 45° (used) | 55° | 65° |
|---|---|---|---|---|---|
| Fitted body (prior 0) | 118.4 (edge) | 113.1 | 112.3 | 112.2 | 112.0 |
| SMPL mean body | 111.7 (edge) | 106.1 | 104.4 | 103.3 | 102.8 |

Other sites change < 0.4 cm across 30–65°. The tape is taken with the arms at the sides, which SMPL can't represent without the arm merging into the torso. The chest therefore carries a pose-definition uncertainty of ≈ 1–3 cm.

## 4. Geometric validation

| Check | Result |
|---|---|
| Mesh topology | Watertight, one component, Euler number 2, no degenerate faces, upright, facing +z (all runs). |
| Height normalisation | The A-pose mesh is 0.12–0.27% shorter than the same body upright. Scaling by it inflated girths by the same fraction; now corrected in the pipeline. ±1 cm of stature = ±0.56% on every girth. |
| Posing distortion | Arms at 45° compress edges near the chest ring (5th percentile 0.78) and slightly at the upper arm (median 0.976). Waist, hip, thigh, forearm and calf rings are undistorted (median 1.000–1.004). |
| Shape plausibility | Current-default fits stay well inside the population (\|β\| ≤ 1 SD). Legacy configurations reached 2.8 SD. |
| Proportions vs outline | No protrusion at any torso level. Under loose clothing the model is 0.79–0.88 of the outline width/depth at the waist/abdomen (current default), i.e. inside the garment, as expected. |
| Views | Fit IoU 0.80–0.89 against the clothed outline, held-out 0.75–0.82. IoU is not an accuracy measure under clothing (it selected the wrong prior once). |

## 5. Multi-view reliability

**Method.** `scripts/reliability_study.py` refits each recording from its saved fit inputs after a controlled degradation of what the fit consumes: masks, keypoints, available views, initial pose/camera. HMR2 and segmentation are **not** re-run on degraded video, so the degradations are proxies (e.g. "blur" = smeared outline plus keypoint noise). Each change is relative to the same recording's undegraded fit, judged against between-recording repeatability (Sw = within-subject SD, RC = 2.77 Sw). Run twice:

- **(a) current default:** prior 0.2, protrusion exclusion. 3 recordings × 12 conditions (2 seeds where random) + jackknife. `outputs/accuracy_study/reliability_p02/`.
- **(b) previous default:** prior 0, IoU exclusion. 3 × 17 conditions. `outputs/accuracy_study/reliability/`.

### 5.1 Current default (prior 0.2)

**Repeatability, 3 recordings, cm @179 (n = 3, provisional)**

| Site | Mean | Sw % | RC % | RC cm |
|---|---|---|---|---|
| Chest | 102.9 | 1.02 | 2.8 | 2.9 |
| Waist | 86.0 | 1.98 | 5.5 | 4.7 |
| Abdomen | 88.8 | 2.00 | 5.6 | 4.9 |
| Hip | 101.8 | 1.56 | 4.3 | 4.4 |
| Upper arm | 29.2 | 2.33 | 6.5 | 1.9 |
| Forearm | 27.0 | 0.91 | 2.5 | 0.7 |
| Thigh | 51.0 | 1.89 | 5.2 | 2.7 |
| Calf | 37.3 | 1.12 | 3.1 | 1.2 |

**Condition effects: worst |change| % over recordings and seeds**

| Brief condition | Proxy | Chest | Waist | Hip | Thigh | Upper arm | Verdict |
|---|---|---|---|---|---|---|---|
| Rotation speed | 8 views | 0.3 | 0.4 | 0.6 | 0.6 | 0.6 | negligible |
| | 6 views | 0.1 | 0.1 | 0.4 | 0.4 | 0.1 | negligible |
| Blurry frames | 50% of views | 0.0 | 0.1 | 0.3 | 0.3 | 0.2 | negligible |
| Occlusion | hip band, 25% of views | 0.0 | 0.1 | 0.1 | 0.1 | 0.1 | negligible (occluded views excluded) |
| | chest band, 25% of views | 0.3 | 0.5 | 0.4 | 0.5 | 0.6 | negligible |
| Coverage | no back views | 0.7 | 1.2 | 0.7 | 0.9 | 1.3 | small, below Sw |
| Lighting | random ±6 px | 0.1 | 0.2 | 0.1 | 0.1 | 0.2 | negligible |
| | shadow bleed +4 px | 0.0 | 0.1 | 0.1 | 0.1 | 0.0 | negligible |
| | dark edges −4 px | 0.2 | 0.3 | 0.1 | 0.1 | 0.3 | negligible |
| Not rotating in place | ±10% distance, ±15% drift | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | negligible |
| Heading error | init ±15° | 0.0 | 0.1 | 0.1 | 0.1 | 0.1 | negligible |

All 12 sites: `outputs/accuracy_study/reliability_p02/summary.md`. The fold jackknife spread is 0.08–0.31%, about 0.1 × Sw.

**Interpretation.** With the HMR2 prior at 0.2 the shape is held near HMR2's multi-frame estimate, and the silhouettes add little. That makes the fit **robust to capture quality**. It also means repeatability (Sw 1–2.3%) and the remaining tape error (abdomen 4–8 cm under) are set by HMR2's per-recording shape estimate, which no capture-side measure tested here changes. The silhouettes still matter in one place: they cap the body inside the outline (no protrusion).

Not tested: real changes in standing pose between recordings (only pose *initialisation* was perturbed). Re-run HMR2 on degraded frames was also not tested, and that is where blur, low light or occlusion would act under this default.

### 5.2 View exclusion: why the old rule failed, and the fix (implemented, default)

After the first fit the pipeline drops unreliable views and refits. The old rule (IoU < max(0.75, median − 0.10)) failed under occlusion. Occluded masks shrink the whole body, so the clean views become the ones with low IoU (down to 0.69), while occluded views span 0.46–0.83. The rule dropped clean views and kept occluded ones.

**Protrusion separates them**, because clothing only ever enlarges an outline. New rule (`body_reconstruction.select_view_exclusions`): exclude views whose per-view protrusion exceeds max(4%, median + 3 MAD), refit, repeat up to 3 rounds. An IoU floor of 0.60 is kept for unusable masks (no clean view fell below 0.69), and at least 6 silhouettes are always kept. The old rule is still available as `RECON_VIEW_EXCLUSION=iou`.

Evidence under the prior-0 fit, where occlusion is most damaging (12 cases = 3 recordings × hip/chest band × 2 seeds; `outputs/accuracy_study/view_exclusion_iter*.json`):

| | Old IoU rule | Iterated protrusion rule |
|---|---|---|
| Cases with worst-site change ≤ 1.5% | 0 / 12 | **10 / 12** |
| Median / max worst-site change | 7.7% / 15.6% | **0.7%** / 8.9% |
| Occluded views found / clean views wrongly excluded | partial / up to 7 per case | 70 of 72 / **0** |

- No view was excluded on clean data, under ±6 px or ±4 px boundary shifts, or under 50% blur.
- One case (C, chest band) missed 2 occluded views and was worse than the old rule (−8.9% vs −5.3%). **The quality indicator flags it:** fold spread 5–10× typical, so 6 sites grade *poor*.
- Under the current default (prior 0.2) occlusion with this rule moves girths ≤ 0.6% (§5.1).

### 5.3 Previous default (prior 0): what a silhouette-driven fit is sensitive to

Kept because any future change that gives silhouettes more weight (tight-clothing mode, clothing-aware weighting) would bring these sensitivities back. Worst |change| %, with the old IoU exclusion:

| Condition | Chest | Waist | Hip | Thigh | Verdict |
|---|---|---|---|---|---|
| 6 views | 1.3 | 1.9 | 1.3 | 2.0 | −1 to −2% systematic |
| 50% blur | 2.2 | 3.0 | 1.8 | 2.4 | +1 to +2% |
| Occlusion (hip / chest band) | 7.3 / 9.5 | 11.3 / 15.6 | 8.1 / 7.1 | 10.5 / 8.9 | **severe** (fixed by §5.2) |
| Random ±6 px boundary | 3.1 | 4.0 | 2.0 | 2.7 | −1 to −3% |
| Shadow bleed +4 px | 4.0 | 5.5 | 3.2 | 3.7 | **+3 to +5%** |
| Dark edges −4 px | 5.5 | 7.3 | 3.9 | 5.2 | **−3 to −7%** |
| Heading init ±15° | 2.4 | 3.0 | 2.5 | 2.8 | −1 to −2.5% |
| ≥ 8 views, missing 90° sector, drift, pose init | ≤ 1.7 | | | | negligible |

At prior 0, a systematic segmentation boundary shift moves girths ≈ 1–1.5% per pixel. Fold spread can't see it (1.5× typical for a −6% shift), because every view shifts together.

## 6. Automatic reliability indicators (implemented, live)

Code: `backend/services/measurement_quality.py`, called from `body_reconstruction.export_outputs` → `<run>/measurement_quality.json` (summarised in the report and run result). Backfill: `scripts/measurement_quality_report.py RUN [--folds]`. Tests: `backend/tests/test_measurement_quality.py` (11 pass).

Each girth gets a **grade** (good / fair / poor / unavailable), **reasons**, and a **±1 SD precision**. The grade is the worst of:

| Signal | Rule | Evidence |
|---|---|---|
| Run-level verdicts | `scan_quality` reconstruction `unreliable` / capture `insufficient` → poor. Warnings → at most fair. **Missing verdict → at most fair.** | A first version treated missing evidence as a pass and graded an inflated legacy run "good"; fixed and tested. |
| Cross-section | Invalid → unavailable. Search ended on its zone edge → fair (forearm exempt, noted). | §3 |
| Fold stability | SD of the girth across the 4 CV-fold shapes (each fitted on 3/4 of the views, already computed by the pipeline): > 2.5 × typical → fair; > 5 × → poor. | Clean ≤ 1.8× typical. Occlusion left in the fit 10–15×. Residual-miss case 5–10×. |
| Outline evidence (torso) | Model/observed width or depth at the site's level: > 1.05 → poor (protrusion); < 0.85 → fair ("girth set by the shape model, not observed"); missing → fair. | Physical constraint; loose outlines can't measure the body. |
| Coverage | < 8 usable silhouettes or a heading gap > 90° → fair; < 6 or > 120° → poor. | §5: 8 views and a 90° gap negligible; wider gaps untested. |

**Calibration per configuration.** Typical fold SD and Sw depend strongly on the shape prior: the fold SD is ≈ 10× smaller at prior 0.2. The constants are therefore tabulated per HMR2 prior weight (0.2 and 0) and chosen from the run's recorded `fit_config`. Other configurations fall back to the default values and are flagged `calibrated: false`. Precision ±1 SD = √(Sw² + max(0, fold SD² − typical²)), plus ±1 cm of stature when shown in cm.

**On real runs.** Current-default run `ff1c226a`: 10 sites *good*; **waist and abdomen *fair*** ("outline much wider than the body: girth set by the shape model"). That is consistent with the tape: those girths come from HMR2's prior, not from the image. Legacy-configuration runs → *poor*.

**What it can't detect (demonstrated).** Errors shared by every view: uniform clothing inflation, systematic segmentation bias, a biased shape prior. The prior-0 run `9d237499`, 11–17 cm above the tape, graded *good*. The grades describe precision and internal consistency; **they are not accuracy**, and the files say so.

## 7. Pipeline vs SHAPY (current default; same stature, 179 cm)

Figure: `outputs/model_comparison/pipeline_vs_shapy_A_B.png` (front / side / back of both bodies with the measured rings; `scripts/render_model_comparison.py`). Pipeline = current default (prior 0.2) refit; SHAPY = stored result for the same video.

| Site | A: Pipeline | A: SHAPY | B: Pipeline | B: SHAPY | C: Pipeline | Tape (self, ranges) |
|---|---|---|---|---|---|---|
| Chest | 101.9 | 106.0 | 104.0 | 106.1 | 102.9 | — |
| Waist | 84.4 | 88.1 | 87.8 | 88.5 | 85.8 | 88–90 |
| Abdomen | 87.1 | 90.4 | 90.6 | 90.8 | 88.6 | 95–100 |
| Hip | 100.0 | 105.0 | 103.0 | 105.0 | 102.4 | 98–103 |
| Upper arm L | 28.5 | 32.8 | 29.9 | 32.8 | 29.2 | — |
| Forearm L | 26.6 | 28.4 | 27.1 | 28.4 | 26.8 | — |
| Thigh L | 50.3 | 54.9 | 52.1 | 54.9 | 51.5 | — |
| Calf L | 36.9 | 38.6 | 37.8 | 38.6 | 37.4 | — |

- At the torso the two models now differ by 0.2–5 cm (they differed by 6–13 cm under prior 0). **Both fall short at the abdomen** (≈ 5–10 cm under the tape range), so that error is shared, likely a learned-shape limitation for belly projection.
- At the waist SHAPY is inside or at the edge of the range; the pipeline is 0.2–3.6 cm under. At the hip the pipeline is inside the range; SHAPY is 2 cm over.
- Pipeline limbs are 1.3–4.6 cm smaller than SHAPY's (upper arm 3–4 cm, thigh 3–5 cm). There is no limb tape yet, so it isn't known which is right.
- SHAPY is very repeatable (A vs B ≤ 0.4 cm). The pipeline varies 2–3.5 cm between recordings. Repeatability is not accuracy.
- SHAPY has not been computed for recording C (needs the GPU).

## 8. Ground-truth validation workflow (implemented; first data entered)

Code: `backend/services/ground_truth_validation.py`, CLI `scripts/ground_truth_report.py`, templates `docs/validation/*_template.csv`, tests `backend/tests/test_ground_truth_validation.py` (7 pass).

**Protocol per subject**

1. Record. Run the reconstruction and SHAPY, and write the run IDs into the sessions sheet **before** taping, so model values can't be tuned to the tape.
2. Measure stadiometer height, barefoot.
3. Tape the 12 sites (`body_measurements.TAPE_PROTOCOL`): one trained measurer, inelastic tape, **two readings per site**, and a third if they differ > 1.0 cm (torso) / 0.5 cm (limbs).
4. Run `scripts/ground_truth_report.py --sessions … --tape …`.

**What it reports**

- Per subject and site: tape, pipeline, SHAPY, error (model − tape) in cm and %, absolute errors, and the pipeline quality grade.
- Per site and method: n, bias, SD, MAE, MAPE, 95% limits of agreement, worst error, and target met / not met / n too small (claimable from n ≥ 5).
- Measurer TEM from the duplicate readings.
- Paired pipeline-vs-SHAPY absolute error.

Model cm = girth ÷ mesh height × stadiometer height, read from each run's files. Tape is the only reference; with no tape it states "no accuracy can be stated". Targets (±2 cm torso, ±1 cm limbs) are proposals.

**First data (S01, 20 Nov).** Self-measured ranges, loose clothing, stature 179 cm (not confirmed barefoot). The range ends were entered as the two readings (`data/validation/S01_*.csv`, output `outputs/validation/ground_truth_S01.md`):

| Site | Tape (midpoint) | Pipeline, current default (`ff1c226a`) | Error | Old prior-0 run (`9d237499`) | SHAPY (`9d237499`) |
|---|---|---|---|---|---|
| Waist | 89.0 | 87.6 | −1.4 cm | 100.3 (+11.3) | 88.5 (−0.5) |
| Abdomen | 97.5 | 90.4 | −7.1 cm | 104.0 (+6.5) | 90.8 (−6.7) |
| Hip | 100.5 | 103.0 | +2.5 cm | 117.1 (+16.6) | 105.0 (+4.5) |

The report flags that **every reading pair exceeds the re-measure tolerance**: the implied measurer TEM is 1.4–3.5 cm, as large as the errors being judged. The next tape session must follow the protocol (two independent readings, trained measurer) before any per-cm conclusion is drawn.

## 9. Remaining limitations

1. **Accuracy rests on one subject, three sites, self-measured ranges, loose clothing.** It is enough to reject the prior-0 setting (10–17 cm off), not to state accuracy.
2. **Repeatability and all thresholds come from one subject (n = 3 recordings)** and are provisional.
3. **Under the current default, accuracy is set by HMR2's learned shape:** abdomen 4–8 cm under on every recording, waist 0–4 cm under. No capture-side measure tested here changes that.
4. **Degradations are proxies.** HMR2 and segmentation were not re-run on degraded video, and that is where blur, low light and occlusion act under the current default.
5. **Internal checks can't see shared errors:** uniform clothing inflation, systematic segmentation bias, prior bias. Demonstrated in §6.
6. **Limbs:** pipeline and SHAPY differ by 1–5 cm, with no limb tape.
7. **Definitions:** forearm edge (≈ 0.5 cm), thigh ≈ 1 cm above protocol, chest pose-sensitivity 1–3 cm.
8. **Scale** comes only from the supplied stature (±1 cm → ±0.56%). Camera intrinsics are HMR2's assumption.

## 10. Recommended next phase: ground truth at protocol quality

1. **Tape properly:** ≥ 5 subjects of varied build, trained measurer, two readings per site at all 12 sites, stadiometer height. Each subject recorded **twice: tight/minimal and everyday clothing.** Freeze run IDs and SHAPY first. This is the single most valuable step: today's measurer error (1.4–3.5 cm) is as large as the model error.
2. **Tight-clothing recordings are the key experiment.** With an honest outline the silhouettes could correct HMR2's abdomen shortfall, but that brings back the prior-0 sensitivities (§5.3), which are then governed by segmentation quality. Compare prior 0.2 vs a lower prior on tight clothing by tape error.
3. **Decide per site between pipeline, SHAPY or their average** by tape, including the limbs, where they differ by 1–5 cm. Consider a per-site bias correction only if it is consistent across subjects and larger than measurer TEM (validate leave-one-subject-out).
4. **Re-run HMR2 on degraded frames** (real blur, low light, partial occlusion) to measure capture sensitivity where it now lives.
5. **Measurement definitions:** move the thigh plane to the protocol point (crotch on the mesh → kneecap) and fix the forearm level, only if tape shows less error.
6. **Pool the calibration constants** (`measurement_quality.CALIBRATION`) over all subjects once ≥ 5 × 3 recordings exist (`scripts/validation_report.py`).

Not recommended: choosing fit settings by IoU, held-out IoU or model agreement. None of them can tell which girths are correct (they chose prior 0 once).

## Appendix: reproduce

```
.venv\Scripts\python.exe scripts/measurement_location_audit.py 9e126c35 268f55fd 38895355 2e0b075f 4fe46ed4 --synthetic --output outputs/accuracy_study/location_audit.json
set CUDA_VISIBLE_DEVICES= & .venv\Scripts\python.exe scripts/reliability_study.py 9e126c35 --beta-prior 0.2 --exclusion protrusion --output outputs/accuracy_study/reliability_p02/9e126c35.json   (also 38895355, 2e0b075f; omit both flags for the prior-0 study)
.venv\Scripts\python.exe scripts/summarize_reliability.py outputs/accuracy_study/reliability_p02/*.json --height-cm 179 --markdown outputs/accuracy_study/reliability_p02/summary.md
set CUDA_VISIBLE_DEVICES= & .venv\Scripts\python.exe scripts/view_exclusion_study.py 9e126c35 38895355 2e0b075f --device cpu --rules iou protrusion_iter none --output outputs/accuracy_study/view_exclusion_iter.json
.venv\Scripts\python.exe scripts/measurement_quality_report.py RUN_ID --height-cm 179 [--folds]
.venv\Scripts\python.exe scripts/ground_truth_report.py --sessions data/validation/S01_sessions.csv --tape data/validation/S01_tape.csv --markdown outputs/validation/ground_truth_S01.md
.venv\Scripts\python.exe scripts/render_model_comparison.py --height-cm 179 --case "Recording B=study:outputs/accuracy_study/reliability_p02/9e126c35.json,shapy:9d237499" --tape waist=88-90 abdomen=95-100 hip=98-103 --output outputs/model_comparison/pipeline_vs_shapy.png
```
